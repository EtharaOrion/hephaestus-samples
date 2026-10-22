"""Routed fused mixture-of-experts layer -- forward and backward in Triton.

Shape of the computation, per call:

  * a router kernel forms the float32 logits, the sigmoid scores S[T, E] and
    an order-preserving int64 key per (token, expert);
  * top-A over the keys picks the experts, argsort of the [T*A] expert ids
    groups the routes by expert;
  * a device-built tile table maps a flat tile id to (expert, tile within
    expert), so every expert GEMM is one launch over ragged row groups whose
    rows are gathered through the routing tables;
  * the expert GEMMs, the SwiGLU, the combine and the whole backward run as
    tiled Triton kernels over that layout.

The combine weight is folded into the activation in the first GEMM's
epilogue: the second GEMM then emits already-weighted rows, the combine is a
plain gather-sum, and no later stage divides by a combine weight.  Operands
reach the tensor cores by one of three routes -- a float32 dot, a three-pass
TF32 accumulation, or an exact bfloat16 hi/lo limb pair -- chosen per GEMM in
_PREC.
"""

import torch
import triton
import triton.language as tl


def _cdiv(a, b):
    return -(-a // b)


def _p2(n, lo=16):
    v = 1
    while v < n:
        v <<= 1
    return max(v, lo)


# --------------------------------------------------------------------------
# operand splitting: an fp32 value as an exact bf16 hi/lo pair
# --------------------------------------------------------------------------


@triton.jit
def _hi(v):
    return v.to(tl.bfloat16)


@triton.jit
def _lo(v, h):
    return (v - h.to(tl.float32)).to(tl.bfloat16)


@triton.jit
def _dot2(a, b, acc, LA: tl.constexpr, LB: tl.constexpr, TF: tl.constexpr):
    """acc += a @ b.

    TF == 2 issues a plain float32 dot, TF == 1 the three-pass TF32
    accumulation, and TF == 0 splits the operands into bfloat16 limbs, LA of
    a and LB of b; one limb reproduces a value that is already bfloat16.
    """
    if TF == 2:
        return tl.dot(a, b, acc, input_precision="ieee")
    if TF == 1:
        return tl.dot(a, b, acc, input_precision="tf32x3")
    ah = _hi(a)
    bh = _hi(b)
    acc = tl.dot(ah, bh, acc)
    if LB > 1:
        acc = tl.dot(ah, _lo(b, bh), acc)
    if LA > 1:
        acc = tl.dot(_lo(a, ah), bh, acc)
    return acc


# --------------------------------------------------------------------------
# routing
# --------------------------------------------------------------------------


@triton.jit
def _router_kernel(X, RW, RB, S, KEY, T, D, E,
                   sx0, sx1, sr0, sr1,
                   BT: tl.constexpr, BE: tl.constexpr, BK: tl.constexpr,
                   SHIFT: tl.constexpr, FP32: tl.constexpr):
    pt = tl.program_id(0)
    pe = tl.program_id(1)
    rt = pt * BT + tl.arange(0, BT)
    re = pe * BE + tl.arange(0, BE)
    mt = rt < T
    me = re < E
    acc = tl.zeros((BT, BE), dtype=tl.float32)
    for k0 in range(0, D, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < D
        a = tl.load(X + rt[:, None].to(tl.int64) * sx0 + rk[None, :] * sx1,
                    mask=mt[:, None] & mk[None, :], other=0.0)
        b = tl.load(RW + rk[:, None] * sr0 + re[None, :] * sr1,
                    mask=mk[:, None] & me[None, :], other=0.0)
        if FP32:
            acc = tl.dot(a, b, acc, input_precision="ieee")
        else:
            acc = tl.dot(a, b, acc)
    s = tl.sigmoid(acc)
    rb = tl.load(RB + re, mask=me, other=0.0)
    biased = s + rb[None, :]
    biased = tl.where(biased == 0.0, 0.0, biased)
    ib = biased.to(tl.int32, bitcast=True).to(tl.int64)
    u = tl.where(ib >= 0, ib ^ 2147483648, (~ib) & 4294967295)
    key = (u << SHIFT) | (E - 1 - re)[None, :].to(tl.int64)
    off = rt[:, None].to(tl.int64) * E + re[None, :]
    m = mt[:, None] & me[None, :]
    tl.store(S + off, s, mask=m)
    tl.store(KEY + off, key, mask=m)


@triton.jit
def _wnorm_kernel(S, SEL, W, SSUM, T, E, A,
                  BT: tl.constexpr, BA: tl.constexpr):
    rt = tl.program_id(0) * BT + tl.arange(0, BT)
    ra = tl.arange(0, BA)
    mt = rt < T
    ma = ra < A
    m = mt[:, None] & ma[None, :]
    p = rt[:, None].to(tl.int64) * A + ra[None, :]
    sel = tl.load(SEL + p, mask=m, other=0)
    s = tl.load(S + rt[:, None].to(tl.int64) * E + sel, mask=m, other=0.0)
    ssum = tl.sum(s, axis=1)
    tl.store(W + p, s / ssum[:, None], mask=m)
    tl.store(SSUM + rt, ssum, mask=mt)


@triton.jit
def _count_kernel(FLAT, CNT, N, BLK: tl.constexpr):
    i = tl.program_id(0) * BLK + tl.arange(0, BLK)
    m = i < N
    e = tl.load(FLAT + i, mask=m, other=0)
    tl.atomic_add(CNT + e, 1, mask=m)


@triton.jit
def _meta_kernel(ORDER, WFLAT, RID, INV, WGT, N, A, BLK: tl.constexpr):
    i = tl.program_id(0) * BLK + tl.arange(0, BLK)
    m = i < N
    p = tl.load(ORDER + i, mask=m, other=0)
    tl.store(RID + i, (p // A).to(tl.int32), mask=m)
    tl.store(INV + p, i.to(tl.int32), mask=m)
    tl.store(WGT + i, tl.load(WFLAT + p, mask=m, other=0.0), mask=m)



@triton.jit
def _offs_kernel(CNT, EOFF, TS, E, BM: tl.constexpr, BE: tl.constexpr):
    e = tl.arange(0, BE)
    m = e < E
    c = tl.load(CNT + e, mask=m, other=0)
    tl.store(EOFF + e, tl.cumsum(c, axis=0) - c, mask=m)
    n = (c + (BM - 1)) // BM
    tl.store(TS + e, tl.cumsum(n, axis=0) - n, mask=m)
    tl.store(TS + E, tl.sum(n, axis=0))


@triton.jit
def _tile_kernel(TS, TILE_E, TILE_M, E, MAXT,
                 BE: tl.constexpr, BLK: tl.constexpr):
    i = tl.program_id(0) * BLK + tl.arange(0, BLK)
    mi = i < MAXT
    e = tl.arange(0, BE)
    me = e < E
    ts = tl.load(TS + e, mask=me, other=0)
    tot = tl.load(TS + E)
    cnt = tl.sum(((ts[None, :] <= i[:, None]) & me[None, :]).to(tl.int32),
                 axis=1)
    eid = cnt - 1
    st = tl.load(TS + tl.maximum(eid, 0), mask=mi, other=0)
    tl.store(TILE_E + i, tl.where(i < tot, eid, -1), mask=mi)
    tl.store(TILE_M + i, i - st, mask=mi)


# --------------------------------------------------------------------------
# forward expert compute
# --------------------------------------------------------------------------


@triton.jit
def _gemm1_kernel(X, W1, RID, WGT, TILE_E, TILE_M, EOFF, CNT, G, U, WACT,
                  D, F, F2, sx0, sx1,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                  LX: tl.constexpr, LW: tl.constexpr, TF: tl.constexpr):
    pid = tl.program_id(0)
    e = tl.load(TILE_E + pid)
    if e < 0:
        return
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    mn = rn < F
    base = tl.load(EOFF + e)
    rm = base + tl.load(TILE_M + pid) * BM + tl.arange(0, BM)
    mm = rm < base + tl.load(CNT + e)
    rid = tl.load(RID + rm, mask=mm, other=0).to(tl.int64)
    wb = W1 + e.to(tl.int64) * (D * F2)
    accg = tl.zeros((BM, BN), dtype=tl.float32)
    accu = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, D, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < D
        a = tl.load(X + rid[:, None] * sx0 + rk[None, :] * sx1,
                    mask=mm[:, None] & mk[None, :], other=0.0)
        bmask = mk[:, None] & mn[None, :]
        wp = wb + rk[:, None] * F2 + rn[None, :]
        bg = tl.load(wp, mask=bmask, other=0.0)
        bu = tl.load(wp + F, mask=bmask, other=0.0)
        if TF:
            accg = tl.dot(a, bg, accg, input_precision="tf32x3")
            accu = tl.dot(a, bu, accu, input_precision="tf32x3")
        else:
            ah = _hi(a)
            bgh = _hi(bg)
            buh = _hi(bu)
            accg = tl.dot(ah, bgh, accg)
            accu = tl.dot(ah, buh, accu)
            if LW > 1:
                accg = tl.dot(ah, _lo(bg, bgh), accg)
                accu = tl.dot(ah, _lo(bu, buh), accu)
            if LX > 1:
                al = _lo(a, ah)
                accg = tl.dot(al, bgh, accg)
                accu = tl.dot(al, buh, accu)
    o = rm[:, None].to(tl.int64) * F + rn[None, :]
    m = mm[:, None] & mn[None, :]
    tl.store(G + o, accg, mask=m)
    tl.store(U + o, accu, mask=m)
    wa = accg * tl.sigmoid(accg) * accu * tl.load(WGT + rm, mask=mm,
                                                  other=0.0)[:, None]
    tl.store(WACT + o, wa, mask=m)


@triton.jit
def _gemm2_kernel(WACT, W2, TILE_E, TILE_M, EOFF, CNT, YP, F, D,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                  LA: tl.constexpr, LW: tl.constexpr, TF: tl.constexpr):
    pid = tl.program_id(0)
    e = tl.load(TILE_E + pid)
    if e < 0:
        return
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    mn = rn < D
    base = tl.load(EOFF + e)
    rm = base + tl.load(TILE_M + pid) * BM + tl.arange(0, BM)
    mm = rm < base + tl.load(CNT + e)
    rm64 = rm.to(tl.int64)
    wb = W2 + e.to(tl.int64) * (F * D)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, F, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < F
        a = tl.load(WACT + rm64[:, None] * F + rk[None, :],
                    mask=mm[:, None] & mk[None, :], other=0.0)
        b = tl.load(wb + rk[:, None] * D + rn[None, :],
                    mask=mk[:, None] & mn[None, :], other=0.0)
        acc = _dot2(a, b, acc, LA, LW, TF)
    tl.store(YP + rm64[:, None] * D + rn[None, :], acc,
             mask=mm[:, None] & mn[None, :])


@triton.jit
def _combine_kernel(YP, INV, Y, T, D, A,
                    BT: tl.constexpr, BN: tl.constexpr):
    rt = tl.program_id(0) * BT + tl.arange(0, BT)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    mt = rt < T
    m = mt[:, None] & (rn < D)[None, :]
    acc = tl.zeros((BT, BN), dtype=tl.float32)
    p = rt.to(tl.int64) * A
    for a in range(A):
        i = tl.load(INV + p + a, mask=mt, other=0).to(tl.int64)
        acc += tl.load(YP + i[:, None] * D + rn[None, :], mask=m, other=0.0)
    tl.store(Y + rt[:, None].to(tl.int64) * D + rn[None, :],
             acc.to(Y.dtype.element_ty), mask=m)


# --------------------------------------------------------------------------
# backward
# --------------------------------------------------------------------------


@triton.jit
def _dact_kernel(DY, W2, G, U, RID, WGT, TILE_E, TILE_M, EOFF, CNT,
                 DGU, PART, N, D, F, F2, sd0,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                 LD: tl.constexpr, LW: tl.constexpr, TF: tl.constexpr):
    pid = tl.program_id(0)
    e = tl.load(TILE_E + pid)
    if e < 0:
        return
    pn = tl.program_id(1)
    rn = pn * BN + tl.arange(0, BN)
    mn = rn < F
    base = tl.load(EOFF + e)
    rm = base + tl.load(TILE_M + pid) * BM + tl.arange(0, BM)
    mm = rm < base + tl.load(CNT + e)
    rm64 = rm.to(tl.int64)
    rid = tl.load(RID + rm, mask=mm, other=0).to(tl.int64)
    wb = W2 + e.to(tl.int64) * (F * D)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, D, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < D
        a = tl.load(DY + rid[:, None] * sd0 + rk[None, :],
                    mask=mm[:, None] & mk[None, :], other=0.0)
        # w2 is contracted over its output axis, so the tile is read along
        # the contiguous axis and transposed in registers
        b = tl.load(wb + rn[:, None] * D + rk[None, :],
                    mask=mn[:, None] & mk[None, :], other=0.0)
        acc = _dot2(a, tl.trans(b), acc, LD, LW, TF)
    o = rm64[:, None] * F + rn[None, :]
    m = mm[:, None] & mn[None, :]
    g = tl.load(G + o, mask=m, other=0.0)
    u = tl.load(U + o, mask=m, other=0.0)
    sg = tl.sigmoid(g)
    si = g * sg
    tl.store(PART + pn.to(tl.int64) * N + rm, tl.sum(acc * (si * u), axis=1),
             mask=mm)
    d = acc * tl.load(WGT + rm, mask=mm, other=0.0)[:, None]
    ob = rm64[:, None] * F2 + rn[None, :]
    tl.store(DGU + ob, d * u * (sg + si * (1.0 - sg)), mask=m)
    tl.store(DGU + ob + F, d * si, mask=m)


@triton.jit
def _dwsum_kernel(PART, DWG, N, NP, BLK: tl.constexpr):
    i = tl.program_id(0) * BLK + tl.arange(0, BLK)
    m = i < N
    s = tl.zeros((BLK,), dtype=tl.float32)
    for p in range(NP):
        s += tl.load(PART + p.to(tl.int64) * N + i, mask=m, other=0.0)
    tl.store(DWG + i, s, mask=m)


@triton.jit
def _rbwd_kernel(S, SEL, W, SSUM, INV, DWG, DLOG, T, E, A,
                 BT: tl.constexpr, BA: tl.constexpr):
    rt = tl.program_id(0) * BT + tl.arange(0, BT)
    ra = tl.arange(0, BA)
    mt = rt < T
    m = mt[:, None] & (ra < A)[None, :]
    p = rt[:, None].to(tl.int64) * A + ra[None, :]
    i = tl.load(INV + p, mask=m, other=0).to(tl.int64)
    g = tl.load(DWG + i, mask=m, other=0.0)
    w = tl.load(W + p, mask=m, other=0.0)
    ss = tl.load(SSUM + rt, mask=mt, other=1.0)[:, None]
    ds = (g - tl.sum(g * w, axis=1)[:, None]) / ss
    sel = tl.load(SEL + p, mask=m, other=0)
    o = rt[:, None].to(tl.int64) * E + sel
    s = tl.load(S + o, mask=m, other=0.0)
    tl.store(DLOG + o, ds * s * (1.0 - s), mask=m)


@triton.jit
def _dxs_kernel(DGU, W1, TILE_E, TILE_M, EOFF, CNT, DXS, D, F2,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                LG: tl.constexpr, LW: tl.constexpr, TF: tl.constexpr):
    pid = tl.program_id(0)
    e = tl.load(TILE_E + pid)
    if e < 0:
        return
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    mn = rn < D
    base = tl.load(EOFF + e)
    rm = base + tl.load(TILE_M + pid) * BM + tl.arange(0, BM)
    mm = rm < base + tl.load(CNT + e)
    rm64 = rm.to(tl.int64)
    wb = W1 + e.to(tl.int64) * (D * F2)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, F2, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < F2
        a = tl.load(DGU + rm64[:, None] * F2 + rk[None, :],
                    mask=mm[:, None] & mk[None, :], other=0.0)
        b = tl.load(wb + rn[:, None] * F2 + rk[None, :],
                    mask=mn[:, None] & mk[None, :], other=0.0)
        acc = _dot2(a, tl.trans(b), acc, LG, LW, TF)
    tl.store(DXS + rm64[:, None] * D + rn[None, :], acc,
             mask=mm[:, None] & mn[None, :])


@triton.jit
def _dx_kernel(DXS, DLOG, RW, INV, DX, T, D, E, A, sr0, sr1,
               BT: tl.constexpr, BN: tl.constexpr, BE: tl.constexpr,
               LL: tl.constexpr, LR: tl.constexpr, TF: tl.constexpr):
    rt = tl.program_id(0) * BT + tl.arange(0, BT)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    mt = rt < T
    mn = rn < D
    m = mt[:, None] & mn[None, :]
    acc = tl.zeros((BT, BN), dtype=tl.float32)
    p = rt.to(tl.int64) * A
    for a in range(A):
        i = tl.load(INV + p + a, mask=mt, other=0).to(tl.int64)
        acc += tl.load(DXS + i[:, None] * D + rn[None, :], mask=m, other=0.0)
    for k0 in range(0, E, BE):
        rk = k0 + tl.arange(0, BE)
        mk = rk < E
        dl = tl.load(DLOG + rt[:, None].to(tl.int64) * E + rk[None, :],
                     mask=mt[:, None] & mk[None, :], other=0.0)
        b = tl.load(RW + rn[:, None] * sr0 + rk[None, :] * sr1,
                    mask=mn[:, None] & mk[None, :], other=0.0)
        acc = _dot2(dl, tl.trans(b), acc, LL, LR, TF)
    tl.store(DX + rt[:, None].to(tl.int64) * D + rn[None, :],
             acc.to(DX.dtype.element_ty), mask=m)


@triton.jit
def _dw1_kernel(X, DGU, RID, EOFF, CNT, DW1, D, F2, sx0, sx1,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                LX: tl.constexpr, LG: tl.constexpr, TF: tl.constexpr):
    e = tl.program_id(2)
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    mm = rm < D
    mn = rn < F2
    i0 = tl.load(EOFF + e)
    n = tl.load(CNT + e)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, n, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < n
        row = (i0 + rk).to(tl.int64)
        rid = tl.load(RID + row, mask=mk, other=0).to(tl.int64)
        # gathered rows are read along their contiguous axis and transposed
        # in registers: the token index carries the long stride, so putting
        # it on the tile's fast axis would serialise every lane
        a = tl.load(X + rid[:, None] * sx0 + rm[None, :] * sx1,
                    mask=mk[:, None] & mm[None, :], other=0.0)
        b = tl.load(DGU + row[:, None] * F2 + rn[None, :],
                    mask=mk[:, None] & mn[None, :], other=0.0)
        acc = _dot2(tl.trans(a), b, acc, LX, LG, TF)
    tl.store(DW1 + e.to(tl.int64) * (D * F2) + rm[:, None].to(tl.int64) * F2
             + rn[None, :], acc.to(DW1.dtype.element_ty),
             mask=mm[:, None] & mn[None, :])


@triton.jit
def _dw2_kernel(WACT, DY, RID, EOFF, CNT, DW2, F, D, sd0,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                LA: tl.constexpr, LD: tl.constexpr, TF: tl.constexpr):
    e = tl.program_id(2)
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    mm = rm < F
    mn = rn < D
    i0 = tl.load(EOFF + e)
    n = tl.load(CNT + e)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, n, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < n
        row = (i0 + rk).to(tl.int64)
        rid = tl.load(RID + row, mask=mk, other=0).to(tl.int64)
        a = tl.load(WACT + row[:, None] * F + rm[None, :],
                    mask=mk[:, None] & mm[None, :], other=0.0)
        b = tl.load(DY + rid[:, None] * sd0 + rn[None, :],
                    mask=mk[:, None] & mn[None, :], other=0.0)
        acc = _dot2(tl.trans(a), b, acc, LA, LD, TF)
    tl.store(DW2 + e.to(tl.int64) * (F * D) + rm[:, None].to(tl.int64) * D
             + rn[None, :], acc.to(DW2.dtype.element_ty),
             mask=mm[:, None] & mn[None, :])


@triton.jit
def _drw_kernel(X, DLOG, DRW, T, D, E, sx0, sx1, CH,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                LX: tl.constexpr, LL: tl.constexpr, TF: tl.constexpr):
    ps = tl.program_id(2)
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    mm = rm < D
    mn = rn < E
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    lo = ps * CH
    hi = tl.minimum(lo + CH, T)
    for k0 in range(lo, hi, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < hi
        a = tl.load(X + rk[:, None].to(tl.int64) * sx0 + rm[None, :] * sx1,
                    mask=mk[:, None] & mm[None, :], other=0.0)
        b = tl.load(DLOG + rk[:, None].to(tl.int64) * E + rn[None, :],
                    mask=mk[:, None] & mn[None, :], other=0.0)
        acc = _dot2(tl.trans(a), b, acc, LX, LL, TF)
    tl.store(DRW + ps.to(tl.int64) * (D * E) + rm[:, None].to(tl.int64) * E
             + rn[None, :], acc, mask=mm[:, None] & mn[None, :])


@triton.jit
def _drw_reduce_kernel(SRC, DST, n, NP, BLK: tl.constexpr):
    i = tl.program_id(0) * BLK + tl.arange(0, BLK)
    m = i < n
    s = tl.zeros((BLK,), dtype=tl.float32)
    for p in range(NP):
        s += tl.load(SRC + p.to(tl.int64) * n + i, mask=m, other=0.0)
    tl.store(DST + i, s.to(DST.dtype.element_ty), mask=m)


# --------------------------------------------------------------------------
# launch configuration
# --------------------------------------------------------------------------

# Limbs kept of a computed float32 operand entering a bfloat16 dot.  Two
# limbs reproduce the operand to better than float32 rounding; one limb is
# exact only for a value that is already bfloat16.
_LIMB = {"gemm2": 2, "dxs": 2, "dw1": 2, "dw2": 2}

# Operand route per GEMM, indexed by whether the inputs are float32: 0 splits
# into bfloat16 limbs, 1 accumulates in three TF32 passes, 2 issues a float32
# dot.  Measured on the full forward+backward, the float32 rows split between
# routes 1 and 2.
_PREC = {
    "gemm1": {0: 0, 1: 2},
    "gemm2": {0: 0, 1: 2},
    "dact": {0: 0, 1: 1},
    "dxs": {0: 0, 1: 1},
    "dw1": {0: 0, 1: 2},
    "dw2": {0: 0, 1: 2},
    "dx": {0: 0, 1: 2},
    "drw": {0: 0, 1: 2},
}

# Dispatch GEMMs: (column block, depth block, warps, stages).  Weight-gradient
# GEMMs: a ladder of (row, column, depth, warps, stages), largest tile first.
_TUNE = {
    "rowblock": {0: 128, 1: 128},
    "gemm1": {0: (64, 64, 8, 3), 1: (64, 32, 8, 3)},
    "gemm2": {0: (256, 64, 8, 3), 1: (128, 32, 8, 3)},
    "dact": {0: (64, 64, 8, 4), 1: (64, 32, 8, 3)},
    "dxs": {0: (256, 64, 8, 3), 1: (128, 32, 8, 3)},
    # weight-gradient ladders, largest tile first
    "dw1": {0: [(256, 64, 64, 8, 3), (128, 128, 64, 8, 3),
                (128, 64, 64, 8, 3), (64, 64, 64, 4, 3), (32, 64, 64, 4, 3)],
            1: [(256, 64, 32, 8, 3), (128, 128, 32, 8, 3),
                (128, 64, 32, 8, 3), (64, 64, 32, 4, 3), (32, 64, 32, 4, 3)]},
    "dw2": {0: [(128, 256, 64, 8, 3), (128, 128, 64, 8, 3),
                (128, 64, 64, 8, 3), (64, 64, 64, 4, 3), (32, 64, 64, 4, 3)],
            1: [(128, 256, 32, 8, 3), (128, 128, 32, 8, 3),
                (128, 64, 32, 8, 3), (64, 64, 32, 4, 3), (32, 64, 32, 4, 3)]},
    "drw": {0: [(64, 64, 64, 4, 3)], 1: [(64, 64, 32, 4, 3)]},
}

_SMS = torch.cuda.get_device_properties(0).multi_processor_count
# blocks per SM a weight-gradient grid must reach before a larger tile is
# preferred; the float32 tiles are heavier, so they want a finer grid
_OCC = {0: 2, 1: 8}


def _rowblock(q, rows_per_expert):
    """Row block for the dispatch GEMMs, shrunk when experts are small."""
    b = _TUNE["rowblock"][q]
    while b > 32 and b >= 2 * rows_per_expert:
        b >>= 1
    return b


def _dcfg(name, q, n, k):
    """Column/depth blocks of a dispatch GEMM, clipped to the real extents."""
    bn, bk, w, s = _TUNE[name][q]
    return min(bn, _p2(n)), min(bk, _p2(k)), w, s


def _wcfg(name, q, m, n, e):
    """Blocks of a per-expert weight-gradient GEMM.

    The grid is (m/BM) * (n/BN) * E blocks with no reduction across the row
    dimension, so a large tile on a small output leaves most of the machine
    idle.  Walk the ladder from the largest tile down and take the first that
    still puts _OCC[q] blocks on every SM.
    """
    for bm, bn, bk, w, s in _TUNE[name][q]:
        bm, bn = min(bm, _p2(m)), min(bn, _p2(n))
        if _cdiv(m, bm) * _cdiv(n, bn) * e >= _OCC[q] * _SMS:
            return bm, bn, bk, w, s
    bm, bn, bk, w, s = _TUNE[name][q][-1]
    return min(bm, _p2(m)), min(bn, _p2(n)), bk, w, s


class _FusedMoe(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, router_w, router_b, w1, w2, A):
        A = int(A)
        T, D = x.shape
        E = router_w.shape[1]
        F2 = w1.shape[2]
        F = F2 // 2
        N = T * A
        dev = x.device
        q = 1 if x.dtype == torch.float32 else 0
        x = x.contiguous()
        router_w = router_w.contiguous()
        w1 = w1.contiguous()
        w2 = w2.contiguous()

        S = torch.empty(T, E, device=dev, dtype=torch.float32)
        KEY = torch.empty(T, E, device=dev, dtype=torch.int64)
        BE = min(_p2(E), 128)
        _router_kernel[(_cdiv(T, 64), _cdiv(E, BE))](
            x, router_w, router_b, S, KEY, T, D, E,
            x.stride(0), x.stride(1), router_w.stride(0), router_w.stride(1),
            BT=64, BE=BE, BK=min(64, _p2(D)),
            SHIFT=max(1, (E - 1).bit_length()),
            FP32=bool(q), num_warps=4, num_stages=3)

        sel = torch.topk(KEY, A, dim=-1, largest=True, sorted=True).indices

        W = torch.empty(T, A, device=dev, dtype=torch.float32)
        SSUM = torch.empty(T, device=dev, dtype=torch.float32)
        BA = _p2(A, 1)
        BTW = max(1, min(64, 8192 // BA))
        _wnorm_kernel[(_cdiv(T, BTW),)](S, sel, W, SSUM, T, E, A,
                                        BT=BTW, BA=BA, num_warps=4)

        flat = sel.view(-1)
        order = torch.argsort(flat, stable=True)

        CNT = torch.zeros(E, device=dev, dtype=torch.int32)
        _count_kernel[(_cdiv(N, 512),)](flat, CNT, N, BLK=512, num_warps=4)
        RID = torch.empty(N, device=dev, dtype=torch.int32)
        INV = torch.empty(N, device=dev, dtype=torch.int32)
        WGT = torch.empty(N, device=dev, dtype=torch.float32)
        _meta_kernel[(_cdiv(N, 512),)](order, W, RID, INV, WGT, N, A,
                                       BLK=512, num_warps=4)

        BM = _rowblock(q, max(1, N // E))
        EOFF = torch.empty(E, device=dev, dtype=torch.int32)
        TS = torch.empty(E + 1, device=dev, dtype=torch.int32)
        _offs_kernel[(1,)](CNT, EOFF, TS, E, BM=BM, BE=_p2(E, 1), num_warps=4)
        MT = _cdiv(N, BM) + E
        TE = torch.empty(MT, device=dev, dtype=torch.int32)
        TM = torch.empty(MT, device=dev, dtype=torch.int32)
        _tile_kernel[(_cdiv(MT, 64),)](TS, TE, TM, E, MT,
                                       BE=_p2(E, 1), BLK=64, num_warps=4)

        G = torch.empty(N, F, device=dev, dtype=torch.float32)
        U = torch.empty(N, F, device=dev, dtype=torch.float32)
        WACT = torch.empty(N, F, device=dev, dtype=torch.float32)
        bn, bk, nw, ns = _dcfg("gemm1", q, F, D)
        _gemm1_kernel[(MT, _cdiv(F, bn))](
            x, w1, RID, WGT, TE, TM, EOFF, CNT, G, U, WACT, D, F, F2,
            x.stride(0), x.stride(1), BM=BM, BN=bn, BK=bk,
            LX=1 + q, LW=1 + q, TF=_PREC["gemm1"][q], num_warps=nw, num_stages=ns)

        YP = torch.empty(N, D, device=dev, dtype=torch.float32)
        bn, bk, nw, ns = _dcfg("gemm2", q, D, F)
        _gemm2_kernel[(MT, _cdiv(D, bn))](
            WACT, w2, TE, TM, EOFF, CNT, YP, F, D,
            BM=BM, BN=bn, BK=bk, LA=_LIMB["gemm2"], LW=1 + q, TF=_PREC["gemm2"][q], num_warps=nw, num_stages=ns)

        y = torch.empty(T, D, device=dev, dtype=x.dtype)
        BNC = min(128, _p2(D))
        _combine_kernel[(_cdiv(T, 32), _cdiv(D, BNC))](
            YP, INV, y, T, D, A, BT=32, BN=BNC, num_warps=4, num_stages=2)

        ctx.save_for_backward(x, router_w, w1, w2, S, sel, W, SSUM, G, U,
                              WACT, RID, INV, WGT, EOFF, CNT, TE, TM)
        ctx.meta = (T, D, E, F, F2, A, N, q, MT, BM)
        return y

    @staticmethod
    def backward(ctx, dy):
        (x, router_w, w1, w2, S, sel, W, SSUM, G, U, WACT, RID, INV, WGT,
         EOFF, CNT, TE, TM) = ctx.saved_tensors
        T, D, E, F, F2, A, N, q, MT, BM = ctx.meta
        dev = x.device
        dy = dy.contiguous()

        DGU = torch.empty(N, F2, device=dev, dtype=torch.float32)
        bn, bk, nw, ns = _dcfg("dact", q, F, D)
        NP = _cdiv(F, bn)
        PART = torch.empty(NP, N, device=dev, dtype=torch.float32)
        _dact_kernel[(MT, NP)](
            dy, w2, G, U, RID, WGT, TE, TM, EOFF, CNT, DGU, PART,
            N, D, F, F2, dy.stride(0), BM=BM, BN=bn, BK=bk,
            LD=1 + q, LW=1 + q, TF=_PREC["dact"][q], num_warps=nw, num_stages=ns)

        DWG = torch.empty(N, device=dev, dtype=torch.float32)
        _dwsum_kernel[(_cdiv(N, 256),)](PART, DWG, N, NP, BLK=256, num_warps=4)

        DLOG = torch.zeros(T, E, device=dev, dtype=torch.float32)
        BA = _p2(A, 1)
        BTR = max(1, min(64, 8192 // BA))
        _rbwd_kernel[(_cdiv(T, BTR),)](S, sel, W, SSUM, INV, DWG, DLOG,
                                       T, E, A,
                                       BT=BTR, BA=BA, num_warps=4)

        DXS = torch.empty(N, D, device=dev, dtype=torch.float32)
        bn, bk, nw, ns = _dcfg("dxs", q, D, F2)
        _dxs_kernel[(MT, _cdiv(D, bn))](
            DGU, w1, TE, TM, EOFF, CNT, DXS, D, F2,
            BM=BM, BN=bn, BK=bk, LG=_LIMB["dxs"], LW=1 + q, TF=_PREC["dxs"][q], num_warps=nw, num_stages=ns)

        dx = torch.empty(T, D, device=dev, dtype=x.dtype)
        BND = min(128, _p2(D))
        _dx_kernel[(_cdiv(T, 64), _cdiv(D, BND))](
            DXS, DLOG, router_w, INV, dx, T, D, E, A,
            router_w.stride(0), router_w.stride(1),
            BT=64, BN=BND, BE=_p2(min(E, 64)), LL=2, LR=1 + q,
            TF=_PREC["dx"][q],
            num_warps=4, num_stages=3)

        bm, bn, bk, nw, ns = _wcfg("dw1", q, D, F2, E)
        dw1 = torch.empty_like(w1)
        _dw1_kernel[(_cdiv(D, bm), _cdiv(F2, bn), E)](
            x, DGU, RID, EOFF, CNT, dw1, D, F2, x.stride(0), x.stride(1),
            BM=bm, BN=bn, BK=bk, LX=1 + q, LG=_LIMB["dw1"], TF=_PREC["dw1"][q], num_warps=nw, num_stages=ns)

        bm, bn, bk, nw, ns = _wcfg("dw2", q, F, D, E)
        dw2 = torch.empty_like(w2)
        _dw2_kernel[(_cdiv(F, bm), _cdiv(D, bn), E)](
            WACT, dy, RID, EOFF, CNT, dw2, F, D, dy.stride(0),
            BM=bm, BN=bn, BK=bk, LA=_LIMB["dw2"], LD=1 + q, TF=_PREC["dw2"][q], num_warps=nw, num_stages=ns)

        bm, bn, bk, nw, ns = _wcfg("drw", q, D, E, 1)
        SP = max(1, min(16, 132 // max(_cdiv(D, bm) * _cdiv(E, bn), 1)))
        CH = _cdiv(_cdiv(T, SP), 32) * 32
        SP = _cdiv(T, CH)
        PRW = torch.empty(SP, D, E, device=dev, dtype=torch.float32)
        _drw_kernel[(_cdiv(D, bm), _cdiv(E, bn), SP)](
            x, DLOG, PRW, T, D, E, x.stride(0), x.stride(1), CH,
            BM=bm, BN=bn, BK=bk, LX=1 + q, LL=2, TF=_PREC["drw"][q], num_warps=nw, num_stages=ns)
        drw = torch.empty(D, E, device=dev, dtype=router_w.dtype)
        _drw_reduce_kernel[(_cdiv(D * E, 256),)](PRW, drw, D * E, SP,
                                                 BLK=256, num_warps=4)

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
