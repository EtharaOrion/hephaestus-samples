"""Routed fused mixture-of-experts layer, forward and backward.

Pipeline per call:

  scores      s = sigmoid(x @ router_w) in float32 together with the
              order-preserving integer selection key, in one kernel over [T, E]
  selection   top-A of the packed keys (score bits shifted over the inverted
              expert index), so a tie resolves to the lower expert index
  dispatch    the T*A (token, expert) pairs are ordered by expert with a sort
              of (expert, pair index) keys; segment bounds and per-expert tile
              counts come from one metadata kernel
  experts     gate/up projection, SwiGLU, down projection and the weighted
              combine run as tiled GEMMs over the expert-ordered pairs, with
              the token gather folded into the operand addressing
  backward    the same decomposition transposed: the down projection's input
              gradient fused with the SwiGLU backward and the per-pair score
              cotangent, then the gate/up input gradient, the two weight
              gradients and the router gradients
"""

import torch
import triton
import triton.language as tl


def _p2(n, lo=8):
    return 1 << max((lo - 1).bit_length(), (max(int(n), 1) - 1).bit_length())


# ---------------------------------------------------------------- primitives

@triton.jit
def _dot(a, b, acc, LA: tl.constexpr, LB: tl.constexpr):
    """a @ b accumulated into acc, with LA / LB bfloat16 limbs per operand.

    One limb is a plain bfloat16 tensor-core dot and holds an operand that is
    already exactly representable in bfloat16.  Two limbs carry a float32
    operand as an exact hi + lo bfloat16 pair, so the operand reaches the
    tensor cores with about sixteen mantissa bits; the product is re-formed
    from the cross terms, dropping only lo*lo.
    """
    ah = a.to(tl.bfloat16)
    bh = b.to(tl.bfloat16)
    acc = tl.dot(ah, bh, acc)
    if LA == 2:
        al = (a.to(tl.float32) - ah.to(tl.float32)).to(tl.bfloat16)
        acc = tl.dot(al, bh, acc)
    if LB == 2:
        bl = (b.to(tl.float32) - bh.to(tl.float32)).to(tl.bfloat16)
        acc = tl.dot(ah, bl, acc)
    return acc


@triton.jit
def _seg(TS, pid, E, EPAD: tl.constexpr):
    """Expert whose tile range contains tile `pid`; E if `pid` is past the end."""
    idx = tl.arange(0, EPAD)
    v = tl.load(TS + idx + 1, mask=idx < E, other=0x7FFFFFFF)
    return tl.sum((v <= pid).to(tl.int32))


@triton.jit
def _lower_bound(KEY, N, tgt, NSHIFT: tl.constexpr, STEPS: tl.constexpr):
    lo = tl.zeros(tgt.shape, dtype=tl.int32)
    hi = tl.full(tgt.shape, N, tl.int32)
    for _ in range(STEPS):
        mid = (lo + hi) // 2
        k = tl.load(KEY + mid, mask=mid < N, other=0)
        less = ((k >> NSHIFT).to(tl.int32) < tgt) & (mid < N)
        lo = tl.where(less, mid + 1, lo)
        hi = tl.where(less, hi, mid)
    return lo


@triton.jit
def _zero_kernel(P, n, BLK: tl.constexpr):
    o = tl.program_id(0) * BLK + tl.arange(0, BLK)
    tl.store(P + o, tl.zeros((BLK,), dtype=tl.float32), mask=o < n)


# ------------------------------------------------------------------ routing

@triton.jit
def _router_kernel(X, RW, RB, S, KEY, T, D, E,
                   SHIFT: tl.constexpr, BT: tl.constexpr, BE: tl.constexpr,
                   BK: tl.constexpr, IEEE: tl.constexpr):
    rt = tl.program_id(0) * BT + tl.arange(0, BT)
    re = tl.program_id(1) * BE + tl.arange(0, BE)
    tm = rt < T
    em = re < E
    acc = tl.zeros((BT, BE), dtype=tl.float32)
    for k0 in range(0, D, BK):
        rk = k0 + tl.arange(0, BK)
        km = rk < D
        a = tl.load(X + rt[:, None] * D + rk[None, :],
                    mask=tm[:, None] & km[None, :], other=0.0)
        b = tl.load(RW + rk[:, None] * E + re[None, :],
                    mask=km[:, None] & em[None, :], other=0.0)
        if IEEE == 1:
            acc = tl.dot(a, b, acc, input_precision="ieee")
        else:
            acc = tl.dot(a, b, acc)
    s = tl.sigmoid(acc)
    off = rt[:, None] * E + re[None, :]
    msk = tm[:, None] & em[None, :]
    tl.store(S + off, s, mask=msk)
    biased = s + tl.load(RB + re, mask=em, other=0.0)[None, :]
    biased = tl.where(biased == 0.0, 0.0, biased)
    ib = biased.to(tl.int32, bitcast=True).to(tl.int64)
    u = tl.where(ib >= 0, ib ^ 0x80000000, (~ib) & 0xFFFFFFFF)
    tl.store(KEY + off, (u << SHIFT) | (E - 1 - re).to(tl.int64)[None, :],
             mask=msk)


@triton.jit
def _meta1_kernel(SEL, S, W, PKEY, T, E, A,
                  NSHIFT: tl.constexpr, BT: tl.constexpr, AP: tl.constexpr):
    rt = tl.program_id(0) * BT + tl.arange(0, BT)
    ra = tl.arange(0, AP)
    msk = (rt < T)[:, None] & (ra < A)[None, :]
    i = rt[:, None] * A + ra[None, :]
    sel = tl.load(SEL + i, mask=msk, other=0)
    sv = tl.load(S + rt[:, None] * E + sel, mask=msk, other=0.0)
    tot = tl.sum(tl.where(msk, sv, 0.0), axis=1)
    tl.store(W + i, sv / tot[:, None], mask=msk)
    key = (sel << NSHIFT) | i
    tl.store(PKEY + i, key.to(PKEY.dtype.element_ty), mask=msk)


@triton.jit
def _meta2_kernel(PKEY, WIN, ROWS, WOUT, POS, N, A,
                  NSHIFT: tl.constexpr, BLK: tl.constexpr):
    j = tl.program_id(0) * BLK + tl.arange(0, BLK)
    m = j < N
    k = tl.load(PKEY + j, mask=m, other=0)
    i = (k & ((1 << NSHIFT) - 1)).to(tl.int32)
    tl.store(ROWS + j, i // A, mask=m)
    tl.store(WOUT + j, tl.load(WIN + i, mask=m, other=0.0), mask=m)
    tl.store(POS + i, j.to(tl.int32), mask=m)


@triton.jit
def _scan_kernel(PKEY, BND, TS, N, E, NSHIFT: tl.constexpr,
                 EPAD: tl.constexpr, NBM: tl.constexpr, STEPS: tl.constexpr):
    e = tl.arange(0, EPAD)
    lo = tl.where(e < E, _lower_bound(PKEY, N, e, NSHIFT, STEPS), N)
    nxt = tl.where(e + 1 < E, _lower_bound(PKEY, N, e + 1, NSHIFT, STEPS), N)
    tl.store(BND + e, lo, mask=e <= E)
    cnt = tl.where(e < E, nxt - lo, 0)
    for b in tl.static_range(NBM):
        nt = (cnt + ((32 << b) - 1)) // (32 << b)
        tl.store(TS + b * (E + 1) + e, tl.cumsum(nt) - nt, mask=e < E)
        tl.store(TS + b * (E + 1) + E, tl.sum(nt))


@triton.jit
def _dsel_kernel(DWV, POS, SEL, S, W, DLOG, T, E, A,
                 BT: tl.constexpr, AP: tl.constexpr):
    rt = tl.program_id(0) * BT + tl.arange(0, BT)
    ra = tl.arange(0, AP)
    msk = (rt < T)[:, None] & (ra < A)[None, :]
    i = rt[:, None] * A + ra[None, :]
    j = tl.load(POS + i, mask=msk, other=0)
    dw = tl.load(DWV + j, mask=msk, other=0.0)
    w = tl.load(W + i, mask=msk, other=0.0)
    sel = tl.load(SEL + i, mask=msk, other=0)
    s = tl.load(S + rt[:, None] * E + sel, mask=msk, other=0.0)
    ssum = tl.sum(tl.where(msk, s, 0.0), axis=1)
    ds = (dw - tl.sum(tl.where(msk, dw * w, 0.0), axis=1)[:, None]) / ssum[:, None]
    tl.store(DLOG + rt[:, None] * E + sel, ds * s * (1.0 - s), mask=msk)


# ------------------------------------------------------------------ forward

@triton.jit
def _gemm1_kernel(X, W1, GU, ACTW, ROWS, WGT, TS, BND, D, F, E,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                  EPAD: tl.constexpr, LA: tl.constexpr, LB: tl.constexpr):
    """GU = x[rows] @ w1[e], ACTW = combine_weight * silu(gate) * up."""
    pid_m = tl.program_id(1)
    e = _seg(TS, pid_m, E, EPAD)
    if e < E:
        lo = tl.load(BND + e)
        hi = tl.load(BND + e + 1)
        rm = lo + (pid_m - tl.load(TS + e)) * BM + tl.arange(0, BM)
        mm = rm < hi
        rows = tl.load(ROWS + rm, mask=mm, other=0).to(tl.int64)
        rn = tl.program_id(0) * BN + tl.arange(0, BN)
        nm = rn < F
        F2 = 2 * F
        xp = X + rows[:, None] * D
        wg = W1 + e.to(tl.int64) * D * F2 + rn[None, :]
        accg = tl.zeros((BM, BN), dtype=tl.float32)
        accu = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, D, BK):
            rk = k0 + tl.arange(0, BK)
            km = rk < D
            a = tl.load(xp + rk[None, :], mask=mm[:, None] & km[None, :],
                        other=0.0)
            bmsk = km[:, None] & nm[None, :]
            o = rk[:, None] * F2
            accg = _dot(a, tl.load(wg + o, mask=bmsk, other=0.0), accg, LA, LB)
            accu = _dot(a, tl.load(wg + o + F, mask=bmsk, other=0.0), accu,
                        LA, LB)
        w = tl.load(WGT + rm, mask=mm, other=0.0)
        act = (accg * tl.sigmoid(accg)) * accu * w[:, None]
        om = mm[:, None] & nm[None, :]
        go = rm.to(tl.int64)[:, None] * F2 + rn[None, :]
        ty = GU.dtype.element_ty
        tl.store(GU + go, accg.to(ty), mask=om)
        tl.store(GU + go + F, accu.to(ty), mask=om)
        tl.store(ACTW + rm.to(tl.int64)[:, None] * F + rn[None, :],
                 act.to(ty), mask=om)


@triton.jit
def _gemm2_kernel(ACTW, W2, Y, TS, BND, F, D, E,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                  EPAD: tl.constexpr, LA: tl.constexpr, LB: tl.constexpr):
    pid_m = tl.program_id(1)
    e = _seg(TS, pid_m, E, EPAD)
    if e < E:
        lo = tl.load(BND + e)
        hi = tl.load(BND + e + 1)
        rm = lo + (pid_m - tl.load(TS + e)) * BM + tl.arange(0, BM)
        mm = rm < hi
        rn = tl.program_id(0) * BN + tl.arange(0, BN)
        nm = rn < D
        ap = ACTW + rm.to(tl.int64)[:, None] * F
        wp = W2 + e.to(tl.int64) * F * D + rn[None, :]
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, F, BK):
            rk = k0 + tl.arange(0, BK)
            km = rk < F
            a = tl.load(ap + rk[None, :], mask=mm[:, None] & km[None, :],
                        other=0.0)
            b = tl.load(wp + rk[:, None] * D, mask=km[:, None] & nm[None, :],
                        other=0.0)
            acc = _dot(a, b, acc, LA, LB)
        tl.store(Y + rm.to(tl.int64)[:, None] * D + rn[None, :],
                 acc.to(Y.dtype.element_ty), mask=mm[:, None] & nm[None, :])


@triton.jit
def _combine_kernel(Y, POS, OUT, T, D, A,
                    BT: tl.constexpr, BD: tl.constexpr):
    rt = tl.program_id(0) * BT + tl.arange(0, BT)
    rd = tl.program_id(1) * BD + tl.arange(0, BD)
    tm = rt < T
    dm = rd < D
    acc = tl.zeros((BT, BD), dtype=tl.float32)
    for a in range(A):
        j = tl.load(POS + rt * A + a, mask=tm, other=0).to(tl.int64)
        acc += tl.load(Y + j[:, None] * D + rd[None, :],
                       mask=tm[:, None] & dm[None, :], other=0.0)
    tl.store(OUT + rt[:, None] * D + rd[None, :],
             acc.to(OUT.dtype.element_ty), mask=tm[:, None] & dm[None, :])


# ----------------------------------------------------------------- backward

@triton.jit
def _dactgu_kernel(DY, W2, GU, DGU, DWP, ROWS, WGT, TS, BND, F, D, E, N,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                   EPAD: tl.constexpr, LA: tl.constexpr, LB: tl.constexpr):
    """dact = dy[rows] @ w2[e]^T, then the SwiGLU backward and the partial
    score cotangent <dact, act> in the epilogue."""
    pid_m = tl.program_id(1)
    e = _seg(TS, pid_m, E, EPAD)
    if e < E:
        lo = tl.load(BND + e)
        hi = tl.load(BND + e + 1)
        rm = lo + (pid_m - tl.load(TS + e)) * BM + tl.arange(0, BM)
        mm = rm < hi
        rows = tl.load(ROWS + rm, mask=mm, other=0).to(tl.int64)
        pid_n = tl.program_id(0)
        rn = pid_n * BN + tl.arange(0, BN)
        nm = rn < F
        dyp = DY + rows[:, None] * D
        wp = W2 + e.to(tl.int64) * F * D + rn[:, None] * D
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, D, BK):
            rk = k0 + tl.arange(0, BK)
            km = rk < D
            a = tl.load(dyp + rk[None, :], mask=mm[:, None] & km[None, :],
                        other=0.0)
            b = tl.load(wp + rk[None, :], mask=nm[:, None] & km[None, :],
                        other=0.0)
            acc = _dot(a, tl.trans(b), acc, LA, LB)
        F2 = 2 * F
        go = rm.to(tl.int64)[:, None] * F2 + rn[None, :]
        om = mm[:, None] & nm[None, :]
        g = tl.load(GU + go, mask=om, other=0.0).to(tl.float32)
        u = tl.load(GU + go + F, mask=om, other=0.0).to(tl.float32)
        sg = tl.sigmoid(g)
        silu = g * sg
        w = tl.load(WGT + rm, mask=mm, other=0.0)
        dact = acc * w[:, None]
        ty = DGU.dtype.element_ty
        tl.store(DGU + go, (dact * u * (sg + silu * (1.0 - sg))).to(ty), mask=om)
        tl.store(DGU + go + F, (dact * silu).to(ty), mask=om)
        tl.store(DWP + pid_n * N + rm,
                 tl.sum(tl.where(om, acc * silu * u, 0.0), axis=1), mask=mm)


@triton.jit
def _dwp_kernel(DWP, OUT, N, NF, BLK: tl.constexpr):
    j = tl.program_id(0) * BLK + tl.arange(0, BLK)
    m = j < N
    acc = tl.zeros((BLK,), dtype=tl.float32)
    for k in range(NF):
        acc += tl.load(DWP + k * N + j, mask=m, other=0.0)
    tl.store(OUT + j, acc, mask=m)


@triton.jit
def _dxs_kernel(DGU, W1, DXP, TS, BND, F, D, E,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                EPAD: tl.constexpr, LA: tl.constexpr, LB: tl.constexpr):
    pid_m = tl.program_id(1)
    e = _seg(TS, pid_m, E, EPAD)
    if e < E:
        lo = tl.load(BND + e)
        hi = tl.load(BND + e + 1)
        rm = lo + (pid_m - tl.load(TS + e)) * BM + tl.arange(0, BM)
        mm = rm < hi
        rn = tl.program_id(0) * BN + tl.arange(0, BN)
        nm = rn < D
        F2 = 2 * F
        gp = DGU + rm.to(tl.int64)[:, None] * F2
        wp = W1 + e.to(tl.int64) * D * F2 + rn[:, None] * F2
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, F2, BK):
            rk = k0 + tl.arange(0, BK)
            km = rk < F2
            a = tl.load(gp + rk[None, :], mask=mm[:, None] & km[None, :],
                        other=0.0)
            b = tl.load(wp + rk[None, :], mask=nm[:, None] & km[None, :],
                        other=0.0)
            acc = _dot(a, tl.trans(b), acc, LA, LB)
        tl.store(DXP + rm.to(tl.int64)[:, None] * D + rn[None, :],
                 acc.to(DXP.dtype.element_ty), mask=mm[:, None] & nm[None, :])


@triton.jit
def _dxr_kernel(DXP, POS, DLOG, RW, DX, T, D, E, A,
                BT: tl.constexpr, BD: tl.constexpr, EPAD: tl.constexpr,
                LA: tl.constexpr, LB: tl.constexpr):
    rt = tl.program_id(0) * BT + tl.arange(0, BT)
    rd = tl.program_id(1) * BD + tl.arange(0, BD)
    tm = rt < T
    dm = rd < D
    acc = tl.zeros((BT, BD), dtype=tl.float32)
    for a in range(A):
        j = tl.load(POS + rt * A + a, mask=tm, other=0).to(tl.int64)
        acc += tl.load(DXP + j[:, None] * D + rd[None, :],
                       mask=tm[:, None] & dm[None, :], other=0.0)
    re = tl.arange(0, EPAD)
    em = re < E
    dl = tl.load(DLOG + rt[:, None] * E + re[None, :],
                 mask=tm[:, None] & em[None, :], other=0.0)
    rw = tl.load(RW + rd[:, None] * E + re[None, :],
                 mask=dm[:, None] & em[None, :], other=0.0)
    acc = _dot(dl, tl.trans(rw), acc, LA, LB)
    tl.store(DX + rt[:, None] * D + rd[None, :],
             acc.to(DX.dtype.element_ty), mask=tm[:, None] & dm[None, :])


@triton.jit
def _dw1_kernel(X, DGU, DW1, ROWS, BND, D, F, E,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                LA: tl.constexpr, LB: tl.constexpr):
    F2 = 2 * F
    e = tl.program_id(2)
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    mm = rm < D
    nm = rn < F2
    lo = tl.load(BND + e)
    K = tl.load(BND + e + 1) - lo
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        km = rk < K
        rows = tl.load(ROWS + lo + rk, mask=km, other=0).to(tl.int64)
        a = tl.load(X + rows[:, None] * D + rm[None, :],
                    mask=km[:, None] & mm[None, :], other=0.0)
        b = tl.load(DGU + (lo + rk).to(tl.int64)[:, None] * F2 + rn[None, :],
                    mask=km[:, None] & nm[None, :], other=0.0)
        acc = _dot(tl.trans(a), b, acc, LA, LB)
    tl.store(DW1 + e.to(tl.int64) * D * F2 + rm[:, None] * F2 + rn[None, :],
             acc.to(DW1.dtype.element_ty), mask=mm[:, None] & nm[None, :])


@triton.jit
def _dw2_kernel(ACTW, DY, DW2, ROWS, BND, F, D, E,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                LA: tl.constexpr, LB: tl.constexpr):
    e = tl.program_id(2)
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    mm = rm < F
    nm = rn < D
    lo = tl.load(BND + e)
    K = tl.load(BND + e + 1) - lo
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        km = rk < K
        rows = tl.load(ROWS + lo + rk, mask=km, other=0).to(tl.int64)
        a = tl.load(ACTW + (lo + rk).to(tl.int64)[:, None] * F + rm[None, :],
                    mask=km[:, None] & mm[None, :], other=0.0)
        b = tl.load(DY + rows[:, None] * D + rn[None, :],
                    mask=km[:, None] & nm[None, :], other=0.0)
        acc = _dot(tl.trans(a), b, acc, LA, LB)
    tl.store(DW2 + e.to(tl.int64) * F * D + rm[:, None] * D + rn[None, :],
             acc.to(DW2.dtype.element_ty), mask=mm[:, None] & nm[None, :])


@triton.jit
def _drw_kernel(X, DLOG, DRW, T, D, E,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                LA: tl.constexpr, LB: tl.constexpr):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    mm = rm < D
    nm = rn < E
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, T, BK):
        rk = k0 + tl.arange(0, BK)
        km = rk < T
        a = tl.load(X + rk[:, None] * D + rm[None, :],
                    mask=km[:, None] & mm[None, :], other=0.0)
        b = tl.load(DLOG + rk[:, None] * E + rn[None, :],
                    mask=km[:, None] & nm[None, :], other=0.0)
        acc = _dot(tl.trans(a), b, acc, LA, LB)
    tl.store(DRW + rm[:, None] * E + rn[None, :],
             acc.to(DRW.dtype.element_ty), mask=mm[:, None] & nm[None, :])


# ------------------------------------------------------------------- driver

_BMS = (32, 64, 128, 256)


def _routing(x, router_w, router_b, A):
    T, D = x.shape
    E = router_w.shape[1]
    N = T * A
    dev = x.device
    shift = max(10, (E - 1).bit_length())
    s = torch.empty(T, E, device=dev, dtype=torch.float32)
    key = torch.empty(T, E, device=dev, dtype=torch.int64)
    be = min(_p2(E, 16), 64)
    _router_kernel[(triton.cdiv(T, 64), triton.cdiv(E, be))](
        x, router_w, router_b, s, key, T, D, E, SHIFT=shift, BT=64, BE=be,
        BK=64, IEEE=1 if x.dtype == torch.float32 else 0,
        num_warps=4, num_stages=3)
    sel = torch.topk(key, A, dim=-1, largest=True, sorted=True).indices

    nshift = max(1, (N - 1).bit_length())
    kdt = torch.int32 if (E << nshift) < 2 ** 31 else torch.int64
    w = torch.empty(T, A, device=dev, dtype=torch.float32)
    pkey = torch.empty(N, device=dev, dtype=kdt)
    _meta1_kernel[(triton.cdiv(T, 64),)](
        sel, s, w, pkey, T, E, A, NSHIFT=nshift, BT=64, AP=_p2(A), num_warps=4)
    pkey = torch.sort(pkey)[0]

    rows = torch.empty(N, device=dev, dtype=torch.int32)
    wg = torch.empty(N, device=dev, dtype=torch.float32)
    pos = torch.empty(T, A, device=dev, dtype=torch.int32)
    _meta2_kernel[(triton.cdiv(N, 256),)](
        pkey, w, rows, wg, pos, N, A, NSHIFT=nshift, BLK=256, num_warps=4)
    bnd = torch.empty(E + 1, device=dev, dtype=torch.int32)
    ts = torch.empty(len(_BMS), E + 1, device=dev, dtype=torch.int32)
    _scan_kernel[(1,)](pkey, bnd, ts, N, E, NSHIFT=nshift, EPAD=_p2(E + 1),
                       NBM=len(_BMS), STEPS=nshift + 1, num_warps=8)
    return s, sel, w, rows, wg, pos, bnd, ts


def _cfg(lsum):
    if lsum >= 5:
        return dict(BM=64, BN=64, BK=32, num_warps=4, num_stages=3)
    return dict(BM=128, BN=64, BK=64, num_warps=8, num_stages=3)


def _forward(x, w1, w2, rows, wg, pos, bnd, ts, A, E, li):
    T, D = x.shape
    F = w1.shape[2] // 2
    N = rows.shape[0]
    dev, dt = x.device, x.dtype
    epad = _p2(E + 1)
    f32 = torch.float32
    c = _cfg(li + li)
    bm, bn = c["BM"], c["BN"]
    tsb = ts[_BMS.index(bm)]
    gu = torch.empty(N, 2 * F, device=dev, dtype=f32)
    actw = torch.empty(N, F, device=dev, dtype=f32)
    _gemm1_kernel[(triton.cdiv(F, bn), triton.cdiv(N, bm) + E)](
        x, w1, gu, actw, rows, wg, tsb, bnd, D, F, E, EPAD=epad, LA=li, LB=li,
        BM=bm, BN=bn, BK=c["BK"], num_warps=c["num_warps"],
        num_stages=c["num_stages"])
    y = torch.empty(N, D, device=dev, dtype=f32)
    c2 = _cfg(2 + li)
    bm2, bn2 = c2["BM"], c2["BN"]
    _gemm2_kernel[(triton.cdiv(D, bn2), triton.cdiv(N, bm2) + E)](
        actw, w2, y, ts[_BMS.index(bm2)], bnd, F, D, E, EPAD=epad, LA=2, LB=li,
        BM=bm2, BN=bn2, BK=c2["BK"], num_warps=c2["num_warps"],
        num_stages=c2["num_stages"])
    out = torch.empty(T, D, device=dev, dtype=dt)
    _combine_kernel[(triton.cdiv(T, 32), triton.cdiv(D, 128))](
        y, pos, out, T, D, A, BT=32, BD=128, num_warps=4, num_stages=2)
    return out, gu, actw


def _backward(dy, x, router_w, w1, w2, s, sel, w, rows, wg, pos, bnd, ts,
              gu, actw, A, E, li):
    T, D = x.shape
    F = w1.shape[2] // 2
    N = rows.shape[0]
    dev, dt = x.device, x.dtype
    epad = _p2(E + 1)
    f32 = torch.float32
    ca = _cfg(li + li)
    bm, bn = ca["BM"], ca["BN"]
    nf = triton.cdiv(F, bn)
    dgu = torch.empty(N, 2 * F, device=dev, dtype=f32)
    dwp = torch.empty(nf, N, device=dev, dtype=f32)
    _dactgu_kernel[(nf, triton.cdiv(N, bm) + E)](
        dy, w2, gu, dgu, dwp, rows, wg, ts[_BMS.index(bm)], bnd, F, D, E, N,
        EPAD=epad, LA=li, LB=li, BM=bm, BN=bn, BK=ca["BK"],
        num_warps=ca["num_warps"], num_stages=ca["num_stages"])
    dwv = torch.empty(N, device=dev, dtype=f32)
    _dwp_kernel[(triton.cdiv(N, 256),)](dwp, dwv, N, nf, BLK=256, num_warps=4)
    dlog = torch.empty(T, E, device=dev, dtype=f32)
    _zero_kernel[(triton.cdiv(T * E, 1024),)](dlog, T * E, BLK=1024,
                                              num_warps=4)
    _dsel_kernel[(triton.cdiv(T, 64),)](dwv, pos, sel, s, w, dlog, T, E, A,
                                        BT=64, AP=_p2(A), num_warps=4)
    cs = _cfg(2 + li)
    bms, bns = cs["BM"], cs["BN"]
    dxp = torch.empty(N, D, device=dev, dtype=f32)
    _dxs_kernel[(triton.cdiv(D, bns), triton.cdiv(N, bms) + E)](
        dgu, w1, dxp, ts[_BMS.index(bms)], bnd, F, D, E, EPAD=epad, LA=2,
        LB=li, BM=bms, BN=bns, BK=cs["BK"], num_warps=cs["num_warps"],
        num_stages=cs["num_stages"])
    dx = torch.empty(T, D, device=dev, dtype=dt)
    _dxr_kernel[(triton.cdiv(T, 32), triton.cdiv(D, 128))](
        dxp, pos, dlog, router_w, dx, T, D, E, A, BT=32, BD=128,
        EPAD=_p2(E, 16), LA=2, LB=li, num_warps=4, num_stages=2)
    dw1 = torch.empty_like(w1)
    dw2 = torch.empty_like(w2)
    cw = _cfg(2 + li)
    wbm, wbn, wbk = cw["BM"], cw["BN"], cw["BK"]
    _dw1_kernel[(triton.cdiv(D, wbm), triton.cdiv(2 * F, wbn), E)](
        x, dgu, dw1, rows, bnd, D, F, E, BM=wbm, BN=wbn, BK=wbk, LA=li, LB=2,
        num_warps=cw["num_warps"], num_stages=cw["num_stages"])
    _dw2_kernel[(triton.cdiv(F, wbm), triton.cdiv(D, wbn), E)](
        actw, dy, dw2, rows, bnd, F, D, E, BM=wbm, BN=wbn, BK=wbk, LA=2,
        LB=li, num_warps=cw["num_warps"], num_stages=cw["num_stages"])
    drw = torch.empty(D, E, device=dev, dtype=dt)
    _drw_kernel[(triton.cdiv(D, 64), triton.cdiv(E, 32))](
        x, dlog, drw, T, D, E, BM=64, BN=32, BK=64, LA=li, LB=2,
        num_warps=4, num_stages=3)
    return dx, drw, dw1, dw2


class _FusedMoe(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, router_w, router_b, w1, w2, A):
        A = int(A)
        E = router_w.shape[1]
        if not 1 <= A <= E:
            raise ValueError(f"A={A} out of range for E={E}")
        x = x.contiguous()
        router_w = router_w.contiguous()
        w1 = w1.contiguous()
        w2 = w2.contiguous()
        li = 2 if x.dtype == torch.float32 else 1
        s, sel, w, rows, wg, pos, bnd, ts = _routing(
            x, router_w, router_b.contiguous(), A)
        out, gu, actw = _forward(x, w1, w2, rows, wg, pos, bnd, ts, A, E, li)
        ctx.save_for_backward(x, router_w, w1, w2, s, sel, w, rows, wg, pos,
                              bnd, ts, gu, actw)
        ctx.meta = (A, E, li)
        return out

    @staticmethod
    def backward(ctx, dy):
        (x, router_w, w1, w2, s, sel, w, rows, wg, pos, bnd, ts, gu,
         actw) = ctx.saved_tensors
        A, E, li = ctx.meta
        dx, drw, dw1, dw2 = _backward(
            dy.contiguous(), x, router_w, w1, w2, s, sel, w, rows, wg, pos,
            bnd, ts, gu, actw, A, E, li)
        return dx, drw, None, dw1, dw2, None


def fused_moe(
    x: torch.Tensor,         # [T, D] float32 or bfloat16
    router_w: torch.Tensor,  # [D, E] same dtype as x
    router_b: torch.Tensor,  # [E] float32, selection-only bias, no gradient
    w1: torch.Tensor,        # [E, D, 2F]  [:, :, :F] gate, [:, :, F:] up
    w2: torch.Tensor,        # [E, F, D]
    A: int,                  # experts per token
) -> torch.Tensor:           # [T, D] same dtype as x
    """Entry point. Signature and semantics are fixed; the body is yours."""
    return _FusedMoe.apply(x, router_w, router_b, w1, w2, A)
