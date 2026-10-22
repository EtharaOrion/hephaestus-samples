"""Routed fused mixture-of-experts layer, forward and backward.

Structure:

  * scores, selection keys, combine weights and the routing bookkeeping are
    computed once per call; `torch.topk` on the packed integer keys and
    `torch.argsort` on the [T*A] expert ids are the only framework routing
    steps.
  * every (token, expert) pair becomes one row of an expanded [T*A, .] space
    sorted by expert id, so each expert's rows form one contiguous slice and
    the expert projections are tiled GEMMs over that slice.
  * a block table maps a flat program id to (expert, first row, valid rows),
    so a row-parallel kernel launches one program per row tile of one expert
    and never straddles an expert boundary.
  * the combine and the dx accumulation run per token over the A rows that
    token owns, so every destination is written by exactly one program;
    there are no atomics of any kind.

Precision. Every accumulator is float32. Operands reach the tensor cores as
a sequence of bfloat16 limbs whose sum reproduces the float32 value: one
limb when the value is a stored bfloat16 input (exact on its own), two when
it is a float32 quantity produced by this module, three when it is a
float32 input, which a hi/mid/lo triple represents exactly. The dot of two
such operands is expanded as the sum of the limb products of total order
<= 1 for the row-space gradients, and <= 2 wherever the result feeds a long
cancelling reduction -- the forward projections and the weight gradients.
Limbs are written once by the kernel that produces the value, so the GEMMs
that consume them are pure tensor-core loops. Per-row combine weights scale
accumulators, or the float32 value before it is split, so they never round
an operand. The router GEMMs are small and run on tf32x3 dots.
"""

import torch
import triton
import triton.language as tl

_LIMB = tl.bfloat16     # limb type of the split operand representation


# ------------------------------------------------------------------ dots ---

@triton.jit
def _split2(v):
    """Two bfloat16 limbs of a float32 value."""
    h = v.to(_LIMB)
    return h, (v - h.to(tl.float32)).to(_LIMB)


@triton.jit
def _split3(v):
    """Three bfloat16 limbs, which represent a float32 value exactly."""
    h = v.to(_LIMB)
    r = v - h.to(tl.float32)
    m = r.to(_LIMB)
    return h, m, (r - m.to(tl.float32)).to(_LIMB)


@triton.jit
def _ldot(a0, a1, a2, b0, b1, b2, acc, cor, NA: tl.constexpr,
          NB: tl.constexpr, ORDER: tl.constexpr, SPLIT: tl.constexpr):
    """Sum of the limb products a_i b_j with i + j <= ORDER. NA and NB are
    how many limbs each side really has, so a one-limb operand contributes
    exactly one product.

    With SPLIT the correction products ride their own float32 accumulator
    and are added once at the end. Corrections are orders of magnitude
    smaller than the leading product, so folding them in term by term
    rounds most of each away before it can cancel against the next; the
    separate accumulator keeps them at their own exponent. It costs a
    second accumulator's worth of registers, so it is enabled only where
    the leading product alone is too coarse."""
    acc = tl.dot(a0, b0, acc)
    if ORDER >= 1 and NB > 1:
        if SPLIT:
            cor = tl.dot(a0, b1, cor)
        else:
            acc = tl.dot(a0, b1, acc)
    if ORDER >= 1 and NA > 1:
        if SPLIT:
            cor = tl.dot(a1, b0, cor)
        else:
            acc = tl.dot(a1, b0, acc)
    if ORDER >= 2 and NA > 1 and NB > 1:
        if SPLIT:
            cor = tl.dot(a1, b1, cor)
        else:
            acc = tl.dot(a1, b1, acc)
    if ORDER >= 2 and NB > 2:
        if SPLIT:
            cor = tl.dot(a0, b2, cor)
        else:
            acc = tl.dot(a0, b2, acc)
    if ORDER >= 2 and NA > 2:
        if SPLIT:
            cor = tl.dot(a2, b0, cor)
        else:
            acc = tl.dot(a2, b0, acc)
    return acc, cor


# ---------------------------------------------------------------- routing --

@triton.jit
def _router_fwd_kernel(X, RW, RB, S, KEYS, T, D, E, sxt, srwd,
                       BT: tl.constexpr, BE: tl.constexpr, BK: tl.constexpr):
    """s = sigmoid(x @ router_w) in float32, plus the order-preserving
    integer key packing (s + router_b, inverted expert index)."""
    rt = tl.program_id(0) * BT + tl.arange(0, BT)
    re = tl.program_id(1) * BE + tl.arange(0, BE)
    mt = rt < T
    me = re < E
    xb = X + rt.to(tl.int64) * sxt
    acc = tl.zeros((BT, BE), dtype=tl.float32)
    for k0 in range(0, D, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < D
        a = tl.load(xb[:, None] + rk[None, :], mask=mt[:, None] & mk[None, :],
                    other=0.0).to(tl.float32)
        b = tl.load(RW + rk[:, None] * srwd + re[None, :],
                    mask=mk[:, None] & me[None, :], other=0.0).to(tl.float32)
        acc = tl.dot(a, b, acc, input_precision="tf32x3")
    s = tl.sigmoid(acc)
    tl.store(S + rt[:, None] * E + re[None, :], s,
             mask=mt[:, None] & me[None, :])
    rb = tl.load(RB + re, mask=me, other=0.0)
    biased = s + rb[None, :]
    biased = tl.where(biased == 0.0, 0.0, biased)
    ib = biased.to(tl.int32, bitcast=True).to(tl.int64)
    u = tl.where(ib >= 0, ib ^ 0x80000000, (~ib) & 0xFFFFFFFF)
    keys = (u << 10) | (E - 1 - re[None, :]).to(tl.int64)
    tl.store(KEYS + rt[:, None] * E + re[None, :], keys,
             mask=mt[:, None] & me[None, :])


@triton.jit
def _cw_kernel(S, SEL, CW, SELI, T, E, A, BT: tl.constexpr, BA: tl.constexpr):
    """Combine weights: raw scores of the selected experts, normalized."""
    rt = tl.program_id(0) * BT + tl.arange(0, BT)
    ra = tl.arange(0, BA)
    m = (rt < T)[:, None] & (ra < A)[None, :]
    off = rt[:, None].to(tl.int64) * A + ra[None, :]
    sel = tl.load(SEL + off, mask=m, other=0).to(tl.int32)
    s = tl.load(S + rt[:, None].to(tl.int64) * E + sel, mask=m, other=0.0)
    tot = tl.sum(s, axis=1)
    tl.store(CW + off, s / tot[:, None], mask=m)
    tl.store(SELI + off, sel, mask=m)


@triton.jit
def _row_meta_kernel(ORDER, SELI, CWF, ROW_TOK, ROW_E, ROW_CW, INV, N, A,
                     BLOCK: tl.constexpr):
    """Per dispatched row: token, expert, combine weight, and the inverse
    permutation that lets a token walk the A rows it owns."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < N
    o = tl.load(ORDER + i, mask=m, other=0)
    tl.store(ROW_TOK + i, (o // A).to(tl.int32), mask=m)
    tl.store(ROW_E + i, tl.load(SELI + o, mask=m, other=0), mask=m)
    tl.store(ROW_CW + i, tl.load(CWF + o, mask=m, other=0.0), mask=m)
    tl.store(INV + o, i.to(tl.int32), mask=m)


@triton.jit
def _starts_kernel(ROW_E, START, N):
    """START[e] = first dispatched row whose expert id is >= e."""
    e = tl.program_id(0)
    lo = 0
    hi = N
    while lo < hi:
        mid = (lo + hi) // 2
        if tl.load(ROW_E + mid) < e:
            lo = mid + 1
        else:
            hi = mid
    tl.store(START + e, lo)


@triton.jit
def _padcum_kernel(START, PADCUM, E, BM: tl.constexpr, BE: tl.constexpr):
    """Exclusive prefix sum of the per-expert row-block counts."""
    re = tl.arange(0, BE)
    me = re < E
    s0 = tl.load(START + re, mask=me, other=0)
    s1 = tl.load(START + re + 1, mask=me, other=0)
    nb = tl.where(me, (s1 - s0 + BM - 1) // BM, 0)
    tl.store(PADCUM + re, tl.cumsum(nb, axis=0) - nb, mask=me)
    tl.store(PADCUM + E, tl.sum(nb, axis=0))


@triton.jit
def _blktab_kernel(PADCUM, START, BLK_E, BLK_R, BLK_N, E, BM: tl.constexpr):
    """Flat program id -> (expert, first row, valid rows)."""
    p = tl.program_id(0)
    if p >= tl.load(PADCUM + E):
        tl.store(BLK_E + p, -1)
        tl.store(BLK_R + p, 0)
        tl.store(BLK_N + p, 0)
    else:
        lo = 0
        hi = E - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if tl.load(PADCUM + mid) <= p:
                lo = mid
            else:
                hi = mid - 1
        r0 = tl.load(START + lo) + (p - tl.load(PADCUM + lo)) * BM
        tl.store(BLK_E + p, lo)
        tl.store(BLK_R + p, r0)
        tl.store(BLK_N + p, tl.minimum(tl.load(START + lo + 1) - r0, BM))


@triton.jit
def _split_kernel(SRC, L0, L1, L2, n, BLOCK: tl.constexpr):
    """Write the three-limb representation of a contiguous float32 buffer."""
    base = tl.program_id(0).to(tl.int64) * BLOCK
    i = base + tl.arange(0, BLOCK)
    m = i < n
    v = tl.load(SRC + i, mask=m, other=0.0)
    h, mi, lo = _split3(v)
    tl.store(L0 + i, h, mask=m)
    tl.store(L1 + i, mi, mask=m)
    tl.store(L2 + i, lo, mask=m)


# ------------------------------------------------------------ expert fwd ---

@triton.jit
def _gemm1_kernel(X0, X1, X2, W0, W1_, W2_, ROW_TOK, BLK_E, BLK_R, BLK_N, GU,
                  ACT0, ACT1, ACT2, D, F, sw1e, sw1d, NL: tl.constexpr,
                  NC: tl.constexpr, SPLIT: tl.constexpr, BM: tl.constexpr,
                  BN: tl.constexpr, BK: tl.constexpr):
    """gu = xs @ w1[e]; the gate and up halves share the loaded token tile,
    and the activation leaves split into its own limbs."""
    p = tl.program_id(0)
    e = tl.load(BLK_E + p)
    if e >= 0:
        r0 = tl.load(BLK_R + p)
        nr = tl.load(BLK_N + p)
        rm = tl.arange(0, BM)
        rows = r0 + rm
        mrow = rm < nr
        tok = tl.load(ROW_TOK + rows, mask=mrow, other=0)
        xb = tok.to(tl.int64) * D
        rn = tl.program_id(1) * BN + tl.arange(0, BN)
        mn = rn < F
        wb = e.to(tl.int64) * sw1e
        accg = tl.zeros((BM, BN), dtype=tl.float32)
        accu = tl.zeros((BM, BN), dtype=tl.float32)
        corg = tl.zeros((BM, BN), dtype=tl.float32)
        coru = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, D, BK):
            rk = k0 + tl.arange(0, BK)
            mk = rk < D
            xo = xb[:, None] + rk[None, :]
            xm = mrow[:, None] & mk[None, :]
            wo = wb + rk[:, None] * sw1d + rn[None, :]
            wm = mk[:, None] & mn[None, :]
            a0 = tl.load(X0 + xo, mask=xm, other=0.0)
            g0 = tl.load(W0 + wo, mask=wm, other=0.0)
            u0 = tl.load(W0 + wo + F, mask=wm, other=0.0)
            a1 = a0
            a2 = a0
            g1 = g0
            g2 = g0
            u1 = u0
            u2 = u0
            if NL > 1:
                a1 = tl.load(X1 + xo, mask=xm, other=0.0)
                g1 = tl.load(W1_ + wo, mask=wm, other=0.0)
                u1 = tl.load(W1_ + wo + F, mask=wm, other=0.0)
            if NL > 2:
                a2 = tl.load(X2 + xo, mask=xm, other=0.0)
                g2 = tl.load(W2_ + wo, mask=wm, other=0.0)
                u2 = tl.load(W2_ + wo + F, mask=wm, other=0.0)
            accg, corg = _ldot(a0, a1, a2, g0, g1, g2, accg, corg, NL, NL,
                               2, SPLIT)
            accu, coru = _ldot(a0, a1, a2, u0, u1, u2, accu, coru, NL, NL,
                               2, SPLIT)
        accg += corg
        accu += coru
        act = accg * tl.sigmoid(accg) * accu
        ah, am_, al = _split3(act)
        ob = rows.to(tl.int64) * (2 * F)
        om = mrow[:, None] & mn[None, :]
        tl.store(GU + ob[:, None] + rn[None, :], accg, mask=om)
        tl.store(GU + ob[:, None] + rn[None, :] + F, accu, mask=om)
        ao = rows.to(tl.int64)[:, None] * F + rn[None, :]
        tl.store(ACT0 + ao, ah, mask=om)
        tl.store(ACT1 + ao, am_, mask=om)
        if NC > 2:
            tl.store(ACT2 + ao, al, mask=om)


@triton.jit
def _gemm2_kernel(ACT0, ACT1, ACT2, W0, W1_, W2_, BLK_E, BLK_R, BLK_N, YP,
                  F, D, sw2e, sw2f, NL: tl.constexpr, NC: tl.constexpr,
                  SPLIT: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                  BK: tl.constexpr):
    """yp = act @ w2[e], unweighted; the combine applies cw."""
    p = tl.program_id(0)
    e = tl.load(BLK_E + p)
    if e >= 0:
        r0 = tl.load(BLK_R + p)
        nr = tl.load(BLK_N + p)
        rm = tl.arange(0, BM)
        rows = r0 + rm
        mrow = rm < nr
        ab = rows.to(tl.int64) * F
        rn = tl.program_id(1) * BN + tl.arange(0, BN)
        mn = rn < D
        wb = e.to(tl.int64) * sw2e
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        cor = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, F, BK):
            rk = k0 + tl.arange(0, BK)
            mk = rk < F
            ao = ab[:, None] + rk[None, :]
            am = mrow[:, None] & mk[None, :]
            wo = wb + rk[:, None] * sw2f + rn[None, :]
            wm = mk[:, None] & mn[None, :]
            a0 = tl.load(ACT0 + ao, mask=am, other=0.0)
            a1 = tl.load(ACT1 + ao, mask=am, other=0.0)
            b0 = tl.load(W0 + wo, mask=wm, other=0.0)
            a2 = a0
            b1 = b0
            b2 = b0
            if NC > 2:
                a2 = tl.load(ACT2 + ao, mask=am, other=0.0)
            if NL > 1:
                b1 = tl.load(W1_ + wo, mask=wm, other=0.0)
            if NL > 2:
                b2 = tl.load(W2_ + wo, mask=wm, other=0.0)
            acc, cor = _ldot(a0, a1, a2, b0, b1, b2, acc, cor, NC, NL, 2,
                             SPLIT)
        tl.store(YP + rows.to(tl.int64)[:, None] * D + rn[None, :],
                 acc + cor, mask=mrow[:, None] & mn[None, :])


@triton.jit
def _combine_kernel(YP, CW, INV, Y, T, D, A, BT: tl.constexpr,
                    BD: tl.constexpr):
    """y[t] = sum_a cw[t, a] * yp[row(t, a)]."""
    rt = tl.program_id(0) * BT + tl.arange(0, BT)
    rd = tl.program_id(1) * BD + tl.arange(0, BD)
    mt = rt < T
    m = mt[:, None] & (rd < D)[None, :]
    acc = tl.zeros((BT, BD), dtype=tl.float32)
    base = rt.to(tl.int64) * A
    for a in range(0, A):
        row = tl.load(INV + base + a, mask=mt, other=0)
        w = tl.load(CW + base + a, mask=mt, other=0.0)
        acc += w[:, None] * tl.load(
            YP + row.to(tl.int64)[:, None] * D + rd[None, :], mask=m,
            other=0.0)
    tl.store(Y + rt[:, None].to(tl.int64) * D + rd[None, :],
             acc.to(Y.dtype.element_ty), mask=m)


# ------------------------------------------------------------ expert bwd ---

@triton.jit
def _dact_kernel(DY0, DY1, DY2, W0, W1_, W2_, GU, ROW_TOK, ROW_CW, BLK_E,
                 BLK_R, BLK_N, DGU0, DGU1, DGU2, AW0, AW1, AW2, D, F, sw2e,
                 sw2f, NL: tl.constexpr, NC: tl.constexpr,
                 ORDER: tl.constexpr, SPLIT: tl.constexpr, BM: tl.constexpr,
                 BN: tl.constexpr, BK: tl.constexpr):
    """dact = cw * (dy[tok] @ w2[e]^T), then the silu*mul local backward.
    The same tile emits the cw-scaled activation limbs the dw2 GEMM
    consumes."""
    p = tl.program_id(0)
    e = tl.load(BLK_E + p)
    pid_n = tl.program_id(1)
    if e >= 0:
        r0 = tl.load(BLK_R + p)
        nr = tl.load(BLK_N + p)
        rm = tl.arange(0, BM)
        rows = r0 + rm
        mrow = rm < nr
        tok = tl.load(ROW_TOK + rows, mask=mrow, other=0)
        cw = tl.load(ROW_CW + rows, mask=mrow, other=0.0)
        db = tok.to(tl.int64) * D
        rn = pid_n * BN + tl.arange(0, BN)
        mn = rn < F
        wb = e.to(tl.int64) * sw2e + rn[None, :] * sw2f
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        cor = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, D, BK):
            rk = k0 + tl.arange(0, BK)
            mk = rk < D
            do = db[:, None] + rk[None, :]
            dm = mrow[:, None] & mk[None, :]
            wo = wb + rk[:, None]
            wm = mk[:, None] & mn[None, :]
            a0 = tl.load(DY0 + do, mask=dm, other=0.0)
            b0 = tl.load(W0 + wo, mask=wm, other=0.0)
            a1 = a0
            a2 = a0
            b1 = b0
            b2 = b0
            if NL > 1:
                a1 = tl.load(DY1 + do, mask=dm, other=0.0)
                b1 = tl.load(W1_ + wo, mask=wm, other=0.0)
            if NL > 2 and ORDER > 1:
                a2 = tl.load(DY2 + do, mask=dm, other=0.0)
                b2 = tl.load(W2_ + wo, mask=wm, other=0.0)
            acc, cor = _ldot(a0, a1, a2, b0, b1, b2, acc, cor, NL, NL,
                             ORDER, SPLIT)
        acc += cor
        ob = rows.to(tl.int64) * (2 * F)
        om = mrow[:, None] & mn[None, :]
        g = tl.load(GU + ob[:, None] + rn[None, :], mask=om, other=0.0)
        u = tl.load(GU + ob[:, None] + rn[None, :] + F, mask=om, other=0.0)
        sg = tl.sigmoid(g)
        silu = g * sg
        aw0, aw1, aw2 = _split3(silu * u * cw[:, None])
        ao = rows.to(tl.int64)[:, None] * F + rn[None, :]
        tl.store(AW0 + ao, aw0, mask=om)
        tl.store(AW1 + ao, aw1, mask=om)
        dact = acc * cw[:, None]
        dg0, dg1, dg2 = _split3(dact * u * (sg + silu * (1.0 - sg)))
        du0, du1, du2 = _split3(dact * silu)
        go = ob[:, None] + rn[None, :]
        tl.store(DGU0 + go, dg0, mask=om)
        tl.store(DGU0 + go + F, du0, mask=om)
        tl.store(DGU1 + go, dg1, mask=om)
        tl.store(DGU1 + go + F, du1, mask=om)
        if NC > 2:
            tl.store(AW2 + ao, aw2, mask=om)
            tl.store(DGU2 + go, dg2, mask=om)
            tl.store(DGU2 + go + F, du2, mask=om)


@triton.jit
def _dcw_kernel(DY, YP, ROW_TOK, ORDER, DCW, N, D, sdyt, BR: tl.constexpr,
                BD: tl.constexpr):
    """dcw[t, a] = <dy[t], yp[row(t, a)]>, over the same unweighted expert
    output the forward combined, and written back in (token, slot) order."""
    rr = tl.program_id(0) * BR + tl.arange(0, BR)
    mr = rr < N
    tok = tl.load(ROW_TOK + rr, mask=mr, other=0)
    yb = YP + rr.to(tl.int64) * D
    db = DY + tok.to(tl.int64) * sdyt
    acc = tl.zeros((BR, BD), dtype=tl.float32)
    for k0 in range(0, D, BD):
        rd = k0 + tl.arange(0, BD)
        md = mr[:, None] & (rd < D)[None, :]
        acc += (tl.load(yb[:, None] + rd[None, :], mask=md, other=0.0)
                * tl.load(db[:, None] + rd[None, :], mask=md,
                          other=0.0).to(tl.float32))
    tl.store(DCW + tl.load(ORDER + rr, mask=mr, other=0),
             tl.sum(acc, axis=1), mask=mr)


@triton.jit
def _router_bwd_kernel(S, SELI, CW, DCW, DLOG, T, E, A, BT: tl.constexpr):
    """Backprop the normalization and the sigmoid into the selected scores."""
    rt = tl.program_id(0) * BT + tl.arange(0, BT)
    mt = rt < T
    base = rt.to(tl.int64) * A
    sb = rt.to(tl.int64) * E
    tot = tl.zeros((BT,), dtype=tl.float32)
    inner = tl.zeros((BT,), dtype=tl.float32)
    for a in range(0, A):
        sel = tl.load(SELI + base + a, mask=mt, other=0)
        tot += tl.load(S + sb + sel, mask=mt, other=0.0)
        inner += (tl.load(DCW + base + a, mask=mt, other=0.0)
                  * tl.load(CW + base + a, mask=mt, other=0.0))
    for a in range(0, A):
        sel = tl.load(SELI + base + a, mask=mt, other=0)
        s = tl.load(S + sb + sel, mask=mt, other=0.0)
        ds = (tl.load(DCW + base + a, mask=mt, other=0.0) - inner) / tot
        tl.store(DLOG + sb + sel, ds * s * (1.0 - s), mask=mt)


@triton.jit
def _dxs_kernel(DGU0, DGU1, W0, W1_, BLK_E, BLK_R, BLK_N, DXS, F, D,
                sw1e, sw1d, NL: tl.constexpr, SPLIT: tl.constexpr, BM: tl.constexpr,
                BN: tl.constexpr, BK: tl.constexpr):
    """dxs = dgu @ w1[e]^T, one row per dispatched (token, expert) pair."""
    p = tl.program_id(0)
    e = tl.load(BLK_E + p)
    if e >= 0:
        r0 = tl.load(BLK_R + p)
        nr = tl.load(BLK_N + p)
        rm = tl.arange(0, BM)
        rows = r0 + rm
        mrow = rm < nr
        ab = rows.to(tl.int64) * (2 * F)
        rn = tl.program_id(1) * BN + tl.arange(0, BN)
        mn = rn < D
        wb = e.to(tl.int64) * sw1e + rn[None, :] * sw1d
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        cor = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, 2 * F, BK):
            rk = k0 + tl.arange(0, BK)
            mk = rk < 2 * F
            ao = ab[:, None] + rk[None, :]
            am = mrow[:, None] & mk[None, :]
            wo = wb + rk[:, None]
            wm = mk[:, None] & mn[None, :]
            a0 = tl.load(DGU0 + ao, mask=am, other=0.0)
            b0 = tl.load(W0 + wo, mask=wm, other=0.0)
            a1 = a0
            b1 = b0
            if NL > 1:
                a1 = tl.load(DGU1 + ao, mask=am, other=0.0)
                b1 = tl.load(W1_ + wo, mask=wm, other=0.0)
            acc, cor = _ldot(a0, a1, a1, b0, b1, b1, acc, cor, NL, NL, 1,
                             SPLIT)
        tl.store(DXS + rows.to(tl.int64)[:, None] * D + rn[None, :],
                 acc + cor, mask=mrow[:, None] & mn[None, :])


@triton.jit
def _dw1_kernel(X0, X1, X2, DGU0, DGU1, DGU2, ROW_TOK, START, DW1, D, F,
                sw1e, sw1d, NL: tl.constexpr, NC: tl.constexpr, SPLIT: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                NT: tl.constexpr):
    """dw1[e] = xs[e]^T @ dgu[e], reduced over the expert's own rows."""
    e = tl.program_id(1)
    pid = tl.program_id(0)
    rm = (pid // NT) * BM + tl.arange(0, BM)
    rn = (pid % NT) * BN + tl.arange(0, BN)
    s0 = tl.load(START + e)
    s1 = tl.load(START + e + 1)
    mm = rm < D
    mn = rn < 2 * F
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    cor = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(s0, s1, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < s1
        tok = tl.load(ROW_TOK + rk, mask=mk, other=0)
        xo = tok[:, None].to(tl.int64) * D + rm[None, :]
        xm = mk[:, None] & mm[None, :]
        go = rk[:, None].to(tl.int64) * (2 * F) + rn[None, :]
        gm = mk[:, None] & mn[None, :]
        a0 = tl.trans(tl.load(X0 + xo, mask=xm, other=0.0))
        b0 = tl.load(DGU0 + go, mask=gm, other=0.0)
        b1 = tl.load(DGU1 + go, mask=gm, other=0.0)
        a1 = a0
        a2 = a0
        b2 = b0
        if NL > 1:
            a1 = tl.trans(tl.load(X1 + xo, mask=xm, other=0.0))
        if NL > 2:
            a2 = tl.trans(tl.load(X2 + xo, mask=xm, other=0.0))
        if NC > 2:
            b2 = tl.load(DGU2 + go, mask=gm, other=0.0)
        acc, cor = _ldot(a0, a1, a2, b0, b1, b2, acc, cor, NL, NC, 2,
                         SPLIT)
    tl.store(DW1 + e.to(tl.int64) * sw1e + rm[:, None] * sw1d + rn[None, :],
             (acc + cor).to(DW1.dtype.element_ty),
             mask=mm[:, None] & mn[None, :])


@triton.jit
def _dw2_kernel(AW0, AW1, AW2, DY0, DY1, DY2, ROW_TOK, START, DW2, F, D,
                sw2e, sw2f, NL: tl.constexpr, NC: tl.constexpr, SPLIT: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                NT: tl.constexpr):
    """dw2[e] = (cw * act)[e]^T @ dy[e]; cw scaled the float32 activation
    before it was split, so the dy operand is never rescaled."""
    e = tl.program_id(1)
    pid = tl.program_id(0)
    rm = (pid // NT) * BM + tl.arange(0, BM)
    rn = (pid % NT) * BN + tl.arange(0, BN)
    s0 = tl.load(START + e)
    s1 = tl.load(START + e + 1)
    mm = rm < F
    mn = rn < D
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    cor = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(s0, s1, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < s1
        ao = rk[:, None].to(tl.int64) * F + rm[None, :]
        am = mk[:, None] & mm[None, :]
        tok = tl.load(ROW_TOK + rk, mask=mk, other=0)
        do = tok[:, None].to(tl.int64) * D + rn[None, :]
        dm = mk[:, None] & mn[None, :]
        a0 = tl.trans(tl.load(AW0 + ao, mask=am, other=0.0))
        a1 = tl.trans(tl.load(AW1 + ao, mask=am, other=0.0))
        b0 = tl.load(DY0 + do, mask=dm, other=0.0)
        a2 = a0
        b1 = b0
        b2 = b0
        if NC > 2:
            a2 = tl.trans(tl.load(AW2 + ao, mask=am, other=0.0))
        if NL > 1:
            b1 = tl.load(DY1 + do, mask=dm, other=0.0)
        if NL > 2:
            b2 = tl.load(DY2 + do, mask=dm, other=0.0)
        acc, cor = _ldot(a0, a1, a2, b0, b1, b2, acc, cor, NC, NL, 2,
                         SPLIT)
    tl.store(DW2 + e.to(tl.int64) * sw2e + rm[:, None] * sw2f + rn[None, :],
             (acc + cor).to(DW2.dtype.element_ty),
             mask=mm[:, None] & mn[None, :])


@triton.jit
def _dx_kernel(DXS, DLOG, RW, INV, DX, T, D, E, A, srwd, BT: tl.constexpr,
               BD: tl.constexpr, BE: tl.constexpr):
    """dx = sum over the token's dispatched rows of dxs, plus dlog @ rw^T."""
    rt = tl.program_id(0) * BT + tl.arange(0, BT)
    rd = tl.program_id(1) * BD + tl.arange(0, BD)
    mt = rt < T
    md = rd < D
    m = mt[:, None] & md[None, :]
    acc = tl.zeros((BT, BD), dtype=tl.float32)
    base = rt.to(tl.int64) * A
    for a in range(0, A):
        row = tl.load(INV + base + a, mask=mt, other=0)
        acc += tl.load(DXS + row.to(tl.int64)[:, None] * D + rd[None, :],
                       mask=m, other=0.0)
    for e0 in range(0, E, BE):
        re = e0 + tl.arange(0, BE)
        me = re < E
        dl = tl.load(DLOG + rt[:, None].to(tl.int64) * E + re[None, :],
                     mask=mt[:, None] & me[None, :], other=0.0)
        rwt = tl.load(RW + rd[None, :].to(tl.int64) * srwd + re[:, None],
                      mask=me[:, None] & md[None, :], other=0.0).to(tl.float32)
        acc = tl.dot(dl, rwt, acc, input_precision="tf32x3")
    tl.store(DX + rt[:, None].to(tl.int64) * D + rd[None, :],
             acc.to(DX.dtype.element_ty), mask=m)


@triton.jit
def _drw_kernel(X, DLOG, DRW, T, D, E, sxt, srwd, BM: tl.constexpr,
                BN: tl.constexpr, BK: tl.constexpr):
    """drouter_w = x^T @ dlog."""
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    mm = rm < D
    mn = rn < E
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, T, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < T
        a = tl.load(X + rk[:, None].to(tl.int64) * sxt + rm[None, :],
                    mask=mk[:, None] & mm[None, :], other=0.0).to(tl.float32)
        b = tl.load(DLOG + rk[:, None].to(tl.int64) * E + rn[None, :],
                    mask=mk[:, None] & mn[None, :], other=0.0)
        acc = tl.dot(tl.trans(a), b, acc, input_precision="tf32x3")
    tl.store(DRW + rm[:, None] * srwd + rn[None, :],
             acc.to(DRW.dtype.element_ty), mask=mm[:, None] & mn[None, :])


# ------------------------------------------------------------- launchers ---

_SPLIT_BLOCK = 1024

# (BM, BN, BK, num_warps, num_stages) per GEMM, indexed by operand dtype.
# The float32 rows carry more limb tiles per operand, so they run shallower
# tiles to stay inside the per-block shared-memory budget.
_CFG = {
    False: {"gemm1": (128, 64, 64, 4, 3), "gemm2": (128, 256, 64, 8, 3),
            "dact": (128, 64, 64, 8, 4), "dxs": (128, 256, 64, 8, 4),
            "dw1": (128, 128, 32, 4, 3), "dw2": (128, 128, 64, 4, 3)},
    True: {"gemm1": (128, 128, 32, 8, 3), "gemm2": (128, 256, 32, 8, 3),
           "dact": (128, 64, 32, 8, 3), "dxs": (128, 256, 32, 8, 4),
           "dw1": (64, 128, 32, 4, 2), "dw2": (128, 128, 32, 8, 2)},
}

# All row-parallel dispatch GEMMs share this row tile, so the block table
# that maps a program id to (expert, first row) is built once per call.
_BMROW = 128

# GEMMs whose correction limbs need their own accumulator. Only the sites
# whose result feeds a long cancelling reduction do.
_SPLITSITES = {"gemm1": True, "gemm2": True, "dact": True, "dxs": False, "dw1": True, "dw2": True}

# Limb-product order of the GEMM that produces the hidden gradient. Its
# result is the operand of the dw1 reduction, so on float32 inputs it is
# expanded to the same order as the weight gradients themselves.
_ORDER = {False: 1, True: 2}


def _cdiv(a, b):
    return (a + b - 1) // b


def _tiles(fp32, name, n, m=None):
    """Tile shape for one GEMM. n is the width of the free axis the BN tile
    walks; m, when given, is the length of the BM axis and is returned with
    the rest. Tiles wider than their axis shrink, so a short axis does not
    launch programs that are almost entirely masked out."""
    BM, BN, BK, w, s = _CFG[fp32][name]
    if n < BN:
        BN = max(16, triton.next_power_of_2(n))
    if m is None:
        return BN, BK, min(w, max(1, _BMROW * BN // 1024)), s
    if m < BM:
        BM = max(16, triton.next_power_of_2(m))
    return BM, BN, BK, min(w, max(1, BM * BN // 1024)), s


def _ble(E):
    """Expert-axis tile for the router-sized GEMMs."""
    return min(64, max(16, triton.next_power_of_2(E)))


def _operands(t, fp32):
    """Tensor-core limb operands of a stored input tensor: itself when it is
    already bfloat16, its exact three-limb split when it is float32."""
    if not fp32:
        return t, t, t
    n = t.numel()
    buf = torch.empty(3 * n, device=t.device, dtype=torch.bfloat16)
    _split_kernel[(_cdiv(n, _SPLIT_BLOCK),)](
        t, buf[:n], buf[n:2 * n], buf[2 * n:], n, BLOCK=_SPLIT_BLOCK,
        num_warps=8)
    return (buf[:n], buf[n:2 * n], buf[2 * n:])


def _limbs(n, nc, dev):
    """Storage for the nc limbs of a computed float32 value of n elements.
    The unused third slot aliases the first, so the kernels can take a
    pointer unconditionally and read it only when nc says it is live."""
    buf = torch.empty(nc * n, device=dev, dtype=torch.bfloat16)
    parts = [buf[i * n:(i + 1) * n] for i in range(nc)]
    return parts + [parts[0]] * (3 - nc)


def _routing(x, rw, rb, A):
    T, D = x.shape
    E = rw.shape[1]
    dev = x.device
    s = torch.empty(T, E, device=dev, dtype=torch.float32)
    keys = torch.empty(T, E, device=dev, dtype=torch.int64)
    BE = _ble(E)
    _router_fwd_kernel[(_cdiv(T, 64), _cdiv(E, BE))](
        x, rw, rb, s, keys, T, D, E, x.stride(0), rw.stride(0),
        BT=64, BE=BE, BK=64, num_warps=4, num_stages=3)
    sel = torch.topk(keys, A, dim=-1, largest=True, sorted=True).indices
    N = T * A
    cw = torch.empty(T, A, device=dev, dtype=torch.float32)
    seli = torch.empty(N, device=dev, dtype=torch.int32)
    _cw_kernel[(_cdiv(T, 32),)](s, sel, cw, seli, T, E, A, BT=32,
                                BA=triton.next_power_of_2(A), num_warps=4)
    order = torch.argsort(seli, stable=True)
    row_tok = torch.empty(N, device=dev, dtype=torch.int32)
    row_e = torch.empty(N, device=dev, dtype=torch.int32)
    row_cw = torch.empty(N, device=dev, dtype=torch.float32)
    inv = torch.empty(N, device=dev, dtype=torch.int32)
    _row_meta_kernel[(_cdiv(N, 256),)](order, seli, cw, row_tok, row_e, row_cw,
                                       inv, N, A, BLOCK=256, num_warps=4)
    start = torch.empty(E + 1, device=dev, dtype=torch.int32)
    padcum = torch.empty(E + 1, device=dev, dtype=torch.int32)
    _starts_kernel[(E + 1,)](row_e, start, N, num_warps=1)
    _padcum_kernel[(1,)](start, padcum, E, BM=_BMROW,
                         BE=triton.next_power_of_2(E), num_warps=4)
    maxb = _cdiv(N, _BMROW) + E
    blk = torch.empty(3, maxb, device=dev, dtype=torch.int32)
    _blktab_kernel[(maxb,)](padcum, start, blk[0], blk[1], blk[2], E,
                            BM=_BMROW, num_warps=1)
    return s, cw, seli, order, row_tok, row_cw, inv, start, blk, maxb


class _FusedMoE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, router_w, router_b, w1, w2, A):
        A = int(A)
        x = x.contiguous()
        router_w = router_w.contiguous()
        w1 = w1.contiguous()
        w2 = w2.contiguous()
        T, D = x.shape
        E = router_w.shape[1]
        F = w1.shape[2] // 2
        N = T * A
        dev, dt = x.device, x.dtype
        fp32 = dt == torch.float32
        nl = 3 if fp32 else 1
        nc = 3 if fp32 else 2

        (s, cw, seli, order, row_tok, row_cw, inv, start, blk,
         maxb) = _routing(x, router_w, router_b, A)

        xs = _operands(x, fp32)
        w1s = _operands(w1, fp32)
        w2s = _operands(w2, fp32)

        gu = torch.empty(N, 2 * F, device=dev, dtype=torch.float32)
        act = _limbs(N * F, nc, dev)
        BN, BK, nw, ns = _tiles(fp32, "gemm1", F)
        _gemm1_kernel[(maxb, _cdiv(F, BN))](
            xs[0], xs[1], xs[2], w1s[0], w1s[1], w1s[2], row_tok,
            blk[0], blk[1], blk[2], gu, act[0], act[1], act[2], D, F,
            D * 2 * F, 2 * F,
            NL=nl, NC=nc, SPLIT=fp32 and _SPLITSITES['gemm1'], BM=_BMROW, BN=BN, BK=BK,
            num_warps=nw, num_stages=ns)

        yp = torch.empty(N, D, device=dev, dtype=torch.float32)
        BN2, BK2, nw2, ns2 = _tiles(fp32, "gemm2", D)
        _gemm2_kernel[(maxb, _cdiv(D, BN2))](
            act[0], act[1], act[2], w2s[0], w2s[1], w2s[2], blk[0], blk[1],
            blk[2], yp, F, D, F * D, D,
            NL=nl, NC=nc, SPLIT=fp32 and _SPLITSITES['gemm2'], BM=_BMROW, BN=BN2, BK=BK2,
            num_warps=nw2, num_stages=ns2)

        y = torch.empty(T, D, device=dev, dtype=dt)
        BD = min(128, triton.next_power_of_2(D))
        _combine_kernel[(_cdiv(T, 32), _cdiv(D, BD))](
            yp, cw, inv, y, T, D, A, BT=32, BD=BD, num_warps=4)

        ctx.save_for_backward(x, router_w, *xs, *w1s, *w2s, s, cw, seli,
                              order, row_tok, row_cw, inv, start, blk, gu, yp,
                              w1, w2)
        ctx.maxb = maxb
        ctx.meta = (T, D, E, F, A, N, fp32)
        return y

    @staticmethod
    def backward(ctx, dy):
        (x, router_w, x0, x1, x2, w10, w11, w12, w20, w21, w22, s, cw, seli,
         order, row_tok, row_cw, inv, start, blk, gu, yp,
         w1, w2) = ctx.saved_tensors
        T, D, E, F, A, N, fp32 = ctx.meta
        maxb = ctx.maxb
        dy = dy.contiguous()
        dev = x.device
        nl = 3 if fp32 else 1
        nc = 3 if fp32 else 2
        BD = min(128, triton.next_power_of_2(D))

        dcw = torch.empty(N, device=dev, dtype=torch.float32)
        _dcw_kernel[(_cdiv(N, 32),)](dy, yp, row_tok, order, dcw, N, D,
                                     dy.stride(0), BR=32, BD=BD, num_warps=4)

        dys = _operands(dy, fp32)
        dgu = _limbs(N * 2 * F, nc, dev)
        aw = _limbs(N * F, nc, dev)
        BN, BK, nw, ns = _tiles(fp32, "dact", F)
        _dact_kernel[(maxb, _cdiv(F, BN))](
            dys[0], dys[1], dys[2], w20, w21, w22, gu, row_tok, row_cw,
            blk[0], blk[1], blk[2], dgu[0], dgu[1], dgu[2], aw[0], aw[1],
            aw[2], D, F, F * D, D,
            NL=nl, NC=nc, ORDER=_ORDER[fp32], SPLIT=fp32 and _SPLITSITES['dact'], BM=_BMROW, BN=BN,
            BK=BK, num_warps=nw, num_stages=ns)
        dlog = torch.zeros(T, E, device=dev, dtype=torch.float32)
        _router_bwd_kernel[(_cdiv(T, 64),)](s, seli, cw, dcw, dlog, T, E, A,
                                            BT=64, num_warps=4)

        dxs = torch.empty(N, D, device=dev, dtype=torch.float32)
        BN2, BK2, nw2, ns2 = _tiles(fp32, "dxs", D)
        _dxs_kernel[(maxb, _cdiv(D, BN2))](
            dgu[0], dgu[1], w10, w11, blk[0], blk[1], blk[2], dxs, F, D,
            D * 2 * F, 2 * F,
            NL=nl, SPLIT=fp32 and _SPLITSITES['dxs'], BM=_BMROW, BN=BN2, BK=BK2, num_warps=nw2,
            num_stages=ns2)

        dw1 = torch.empty_like(w1)
        wm, wn, wk, wnw, wns = _tiles(fp32, "dw1", 2 * F, D)
        nt1 = _cdiv(2 * F, wn)
        _dw1_kernel[(_cdiv(D, wm) * nt1, E)](
            x0, x1, x2, dgu[0], dgu[1], dgu[2], row_tok, start, dw1, D, F,
            w1.stride(0), w1.stride(1),
            NL=nl, NC=nc, SPLIT=fp32 and _SPLITSITES['dw1'], BM=wm, BN=wn, BK=wk, NT=nt1,
            num_warps=wnw, num_stages=wns)

        dw2 = torch.empty_like(w2)
        vm, vn, vk, vnw, vns = _tiles(fp32, "dw2", D, F)
        nt2 = _cdiv(D, vn)
        _dw2_kernel[(_cdiv(F, vm) * nt2, E)](
            aw[0], aw[1], aw[2], dys[0], dys[1], dys[2], row_tok, start, dw2,
            F, D, w2.stride(0), w2.stride(1),
            NL=nl, NC=nc, SPLIT=fp32 and _SPLITSITES['dw2'], BM=vm, BN=vn, BK=vk, NT=nt2,
            num_warps=vnw, num_stages=vns)

        dx = torch.empty(T, D, device=dev, dtype=x.dtype)
        BE = _ble(E)
        _dx_kernel[(_cdiv(T, 32), _cdiv(D, BD))](
            dxs, dlog, router_w, inv, dx, T, D, E, A, router_w.stride(0),
            BT=32, BD=BD, BE=BE, num_warps=4)

        drw = torch.empty_like(router_w)
        _drw_kernel[(_cdiv(D, 64), _cdiv(E, BE))](
            x, dlog, drw, T, D, E, x.stride(0), router_w.stride(0),
            BM=64, BN=BE, BK=32, num_warps=4, num_stages=3)

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
    return _FusedMoE.apply(x, router_w, router_b, w1, w2, A)
