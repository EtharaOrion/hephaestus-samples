"""Routed fused mixture-of-experts layer, forward and backward.

Routing (sigmoid scores, biased top-A with the lower-index tie-break, raw-score
combine weights) and the SwiGLU expert stack both live inside one custom
autograd Function.  Tokens are expanded into T*A (token, expert) pairs, sorted
by expert, and every matmul runs as a tiled Triton kernel over that sorted
layout.

Precision policy, uniform over all inputs of a given dtype:

  * scores accumulate in float32 and the biased score is packed with the
    inverted expert index into a unique int64 key, so descending key order is
    (score descending, expert index ascending);
  * float32 inputs use input_precision="tf32x3" for every dot;
  * bfloat16 inputs use bfloat16 dots, except that operands holding float32
    values which feed dw1, dw2, drouter_w or dx are decomposed into an exact
    bfloat16 high/low pair and contracted with two dots.

Every reduction runs in a fixed index order inside a single program; there are
no atomics on any output or gradient path.
"""

import torch
import triton
import triton.language as tl

_I32MAX = tl.constexpr(0x7FFFFFFF)


# --------------------------------------------------------------------------
# routing
# --------------------------------------------------------------------------

@triton.jit
def _div(a, b):
    """Correctly rounded float32 division.

    Triton's `/` and tl.fdiv land within about two ulp of the true quotient,
    which is enough to make s/s differ from one; the combine weights and the
    normalization gradient are both differences taken around that quotient.
    """
    return tl.extra.cuda.libdevice.div_rn(a, b)


@triton.jit
def _rcp(b):
    """Correctly rounded float32 reciprocal."""
    return tl.extra.cuda.libdevice.div_rn(1.0, b)


@triton.jit
def _limbs(a, L: tl.constexpr):
    """a as L bfloat16 pieces whose sum reproduces it to 8*L mantissa bits."""
    h = a.to(tl.bfloat16)
    r = a - h.to(tl.float32)
    m = r.to(tl.bfloat16)
    if L == 2:
        return h, m, m
    return h, m, (r - m.to(tl.float32)).to(tl.bfloat16)


@triton.jit
def _dotx(a, b, L: tl.constexpr = 2):
    """One float32-effective tensor-core product.

    Operands that arrive as float32 are decomposed into L exact bfloat16
    pieces and every cross term of total order below L is accumulated in
    float32, so 8*L mantissa bits of each float32 operand reach the tensor
    cores. Operands that arrive as bfloat16 are already exact at that width
    and contribute a single term.
    """
    af: tl.constexpr = a.dtype == tl.float32
    bf: tl.constexpr = b.dtype == tl.float32
    if af and bf:
        a0, a1, a2 = _limbs(a, L)
        b0, b1, b2 = _limbs(b, L)
        acc = tl.dot(a0, b0) + (tl.dot(a1, b0) + tl.dot(a0, b1))
        if L >= 3:
            acc += tl.dot(a2, b0) + tl.dot(a0, b2) + tl.dot(a1, b1)
        return acc
    elif af:
        a0, a1, a2 = _limbs(a, L)
        acc = tl.dot(a0, b) + tl.dot(a1, b)
        if L >= 3:
            acc += tl.dot(a2, b)
        return acc
    elif bf:
        b0, b1, b2 = _limbs(b, L)
        acc = tl.dot(a, b0) + tl.dot(a, b1)
        if L >= 3:
            acc += tl.dot(a, b2)
        return acc
    else:
        return tl.dot(a, b)


@triton.jit
def _router_kernel(X, RW, S, KEY, RB, T, D, E,
                   sx0, sx1, sr0, sr1,
                   BT: tl.constexpr, BE: tl.constexpr, BK: tl.constexpr,
                   FP32: tl.constexpr):
    """s = sigmoid(x @ router_w) in float32, plus the packed selection key.

    The logits run at three bfloat16 limbs rather than through a single
    float32 dot: on a router whose columns nearly coincide the summation
    cancels, and the limb route tracks the reference's logits an order of
    magnitude more closely there.
    """
    pid = tl.program_id(0)
    rt = pid * BT + tl.arange(0, BT)
    re = tl.arange(0, BE)
    mt = rt < T
    me = re < E
    acc = tl.zeros((BT, BE), dtype=tl.float32)
    for k0 in range(0, D, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < D
        a = tl.load(X + rt[:, None] * sx0 + rk[None, :] * sx1,
                    mask=mt[:, None] & mk[None, :], other=0.0)
        b = tl.load(RW + rk[:, None] * sr0 + re[None, :] * sr1,
                    mask=mk[:, None] & me[None, :], other=0.0)
        acc += _dotx(a, b, 3)
    # The score-gradient path evaluates s*(1 - s), and 1 - s cancels hard
    # once s approaches one: an s that is a few ulp away from the value the
    # reference computes moves that factor by orders of magnitude more.
    # libdevice's exp keeps the sigmoid within one ulp.
    s = _rcp(1.0 + tl.extra.cuda.libdevice.exp(-acc))
    m = mt[:, None] & me[None, :]
    tl.store(S + rt[:, None] * E + re[None, :], s, mask=m)
    rb = tl.load(RB + re, mask=me, other=0.0)
    b = s + rb[None, :]
    b = tl.where(b == 0.0, 0.0, b)
    ib = b.to(tl.int32, bitcast=True).to(tl.int64)
    u = tl.where(ib >= 0, ib ^ 0x80000000, (~ib) & 0xFFFFFFFF)
    key = (u << 10) | (E - 1 - re).to(tl.int64)[None, :]
    tl.store(KEY + rt[:, None] * E + re[None, :], key, mask=m)


@triton.jit
def _wcomb_kernel(S, SEL, WT, SS, T, E, A,
                  BT: tl.constexpr, BA: tl.constexpr):
    """Combine weights: raw selected scores normalized by their own sum."""
    rt = tl.program_id(0) * BT + tl.arange(0, BT)
    ra = tl.arange(0, BA)
    mt = rt < T
    ma = ra < A
    m = mt[:, None] & ma[None, :]
    sel = tl.load(SEL + rt[:, None] * A + ra[None, :], mask=m, other=0).to(tl.int32)
    s = tl.load(S + rt[:, None] * E + sel, mask=m, other=0.0)
    denom = tl.sum(tl.where(m, s, 0.0), axis=1)
    tl.store(WT + rt[:, None] * A + ra[None, :], _div(s, denom[:, None]),
             mask=m)
    tl.store(SS + rt[:, None] * A + ra[None, :], s, mask=m)


@triton.jit
def _smeta_kernel(ORDER, WT, SROW, SWT, INV, N, A, BLK: tl.constexpr):
    """Sorted-pair metadata: source token, combine weight, inverse position."""
    i = tl.program_id(0) * BLK + tl.arange(0, BLK)
    m = i < N
    o = tl.load(ORDER + i, mask=m, other=0).to(tl.int32)
    tl.store(SROW + i, o // A, mask=m)
    tl.store(SWT + i, tl.load(WT + o, mask=m, other=0.0), mask=m)
    tl.store(INV + o, i.to(tl.int32), mask=m)


@triton.jit
def _off_kernel(SE, OFF, TCUM, N, E, BM,
                BE: tl.constexpr, NBITS: tl.constexpr):
    """Expert row ranges by branchless partition search, plus tile prefix sums."""
    re = tl.arange(0, BE)
    lo = tl.zeros((BE,), dtype=tl.int32)
    cnt = tl.full((BE,), N, dtype=tl.int32)
    for _ in range(NBITS):
        half = cnt // 2
        mid = lo + half
        v = tl.load(SE + mid, mask=mid < N, other=_I32MAX)
        take = v < re
        lo = tl.where(take, mid + 1, lo)
        cnt = tl.where(take, cnt - half - 1, half)
    off = tl.where(re <= E, lo, N)
    tl.store(OFF + re, off, mask=re <= E)
    nxt = tl.zeros((BE,), dtype=tl.int32)
    cnt2 = tl.full((BE,), N, dtype=tl.int32)
    for _ in range(NBITS):
        half = cnt2 // 2
        mid = nxt + half
        v = tl.load(SE + mid, mask=mid < N, other=_I32MAX)
        take = v < re + 1
        nxt = tl.where(take, mid + 1, nxt)
        cnt2 = tl.where(take, cnt2 - half - 1, half)
    ntile = tl.where(re < E, (nxt - off + BM - 1) // BM, 0)
    cum = tl.cumsum(ntile, axis=0) - ntile
    tl.store(TCUM + re, cum, mask=re <= E)


@triton.jit
def _tiles_kernel(OFF, TCUM, TE, TR, TN, NTMAX, E, BM,
                  BP: tl.constexpr, NEBITS: tl.constexpr):
    """Row-tile descriptors: owning expert, first sorted row, row count."""
    p = tl.program_id(0) * BP + tl.arange(0, BP)
    mp = p < NTMAX
    lo = tl.zeros((BP,), dtype=tl.int32)
    cnt = tl.full((BP,), E + 1, dtype=tl.int32)
    for _ in range(NEBITS):
        half = cnt // 2
        mid = lo + half
        v = tl.load(TCUM + mid, mask=mid <= E, other=_I32MAX)
        take = v <= p
        lo = tl.where(take, mid + 1, lo)
        cnt = tl.where(take, cnt - half - 1, half)
    e = tl.minimum(tl.maximum(lo - 1, 0), E - 1)
    o0 = tl.load(OFF + e, mask=mp, other=0)
    o1 = tl.load(OFF + e + 1, mask=mp, other=0)
    c0 = tl.load(TCUM + e, mask=mp, other=0)
    r0 = o0 + (p - c0) * BM
    n = tl.minimum(BM, o1 - r0)
    n = tl.where((lo >= 1) & (lo <= E) & (n > 0), n, 0)
    tl.store(TE + p, e, mask=mp)
    tl.store(TR + p, r0, mask=mp)
    tl.store(TN + p, n, mask=mp)


# --------------------------------------------------------------------------
# forward expert compute
# --------------------------------------------------------------------------

@triton.jit
def _gemm1_kernel(X, W1, ACT, ACL, AWH, AWL, GU, TE, TR, TN, SROW, SWT,
                  D, F, sx0, sx1, sw10, sw11, sw12,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                  FP32: tl.constexpr, SPLIT: tl.constexpr, LY: tl.constexpr):
    """act = silu(xg @ w1_gate) * (xg @ w1_up); both halves of h are kept.

    With SPLIT the activation is also stored pre-multiplied by the combine
    weight as an exact bfloat16 high/low pair, which is the operand dw2
    contracts; ACT then only has to carry the precision gemm2 consumes.
    """
    p = tl.program_id(0)
    n = tl.load(TN + p)
    if n == 0:
        return
    e = tl.load(TE + p)
    r0 = tl.load(TR + p)
    rm = tl.arange(0, BM)
    mrow = rm < n
    rows = tl.load(SROW + r0 + rm, mask=mrow, other=0)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    mn = rn < F
    wg = W1 + e.to(tl.int64) * sw10 + rn[None, :] * sw12
    wu = wg + F * sw12
    accg = tl.zeros((BM, BN), dtype=tl.float32)
    accu = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, D, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < D
        a = tl.load(X + rows[:, None] * sx0 + rk[None, :] * sx1,
                    mask=mrow[:, None] & mk[None, :], other=0.0)
        bg = tl.load(wg + rk[:, None] * sw11,
                     mask=mk[:, None] & mn[None, :], other=0.0)
        bu = tl.load(wu + rk[:, None] * sw11,
                     mask=mk[:, None] & mn[None, :], other=0.0)
        accg += _dotx(a, bg, LY)
        accu += _dotx(a, bu, LY)
    act = accg * tl.sigmoid(accg) * accu
    ri = (r0 + rm).to(tl.int64)
    m = mrow[:, None] & mn[None, :]
    o = ri[:, None] * F + rn[None, :]
    ah = act.to(ACT.dtype.element_ty)
    tl.store(ACT + o, ah, mask=m)
    if SPLIT:
        tl.store(ACL + o, (act - ah.to(tl.float32)).to(tl.bfloat16), mask=m)
    aw = act * tl.load(SWT + r0 + rm, mask=mrow, other=0.0)[:, None]
    h = aw.to(AWH.dtype.element_ty)
    tl.store(AWH + o, h, mask=m)
    if SPLIT:
        tl.store(AWL + o, (aw - h.to(tl.float32)).to(tl.bfloat16), mask=m)
    gu = GU + ri[:, None] * (2 * F) + rn[None, :]
    tl.store(gu, accg, mask=m)
    tl.store(gu + F, accu, mask=m)


@triton.jit
def _gemm2_kernel(ACT, ACL, W2, P, SWT, TE, TR, TN,
                  F, D, sw20, sw21, sw22,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                  SPLIT: tl.constexpr, LY: tl.constexpr):
    """p = combine_weight * (act @ w2)."""
    p = tl.program_id(0)
    n = tl.load(TN + p)
    if n == 0:
        return
    e = tl.load(TE + p)
    r0 = tl.load(TR + p)
    rm = tl.arange(0, BM)
    mrow = rm < n
    ri = (r0 + rm).to(tl.int64)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    mn = rn < D
    wp = W2 + e.to(tl.int64) * sw20 + rn[None, :] * sw22
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, F, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < F
        oa = ri[:, None] * F + rk[None, :]
        ma = mrow[:, None] & mk[None, :]
        b = tl.load(wp + rk[:, None] * sw21,
                    mask=mk[:, None] & mn[None, :], other=0.0)
        acc += _dotx(tl.load(ACT + oa, mask=ma, other=0.0), b, LY)
        if SPLIT:
            acc += tl.dot(tl.load(ACL + oa, mask=ma, other=0.0), b)
    wt = tl.load(SWT + r0 + rm, mask=mrow, other=0.0)
    tl.store(P + ri[:, None] * D + rn[None, :], acc * wt[:, None],
             mask=mrow[:, None] & mn[None, :])


@triton.jit
def _combine_kernel(P, INV, Y, T, D, A,
                    BT: tl.constexpr, BD: tl.constexpr):
    """y[t] = sum over the A weighted expert outputs of token t, in slot order."""
    rt = tl.program_id(0) * BT + tl.arange(0, BT)
    rd = tl.program_id(1) * BD + tl.arange(0, BD)
    mt = rt < T
    md = rd < D
    acc = tl.zeros((BT, BD), dtype=tl.float32)
    for a in range(A):
        i = tl.load(INV + rt * A + a, mask=mt, other=0).to(tl.int64)
        acc += tl.load(P + i[:, None] * D + rd[None, :],
                       mask=mt[:, None] & md[None, :], other=0.0)
    tl.store(Y + rt[:, None] * D + rd[None, :],
             acc.to(Y.dtype.element_ty), mask=mt[:, None] & md[None, :])


# --------------------------------------------------------------------------
# backward
# --------------------------------------------------------------------------

@triton.jit
def _dact_kernel(DY, W2, ACT, ACL, GU, DGU, DGL, PART, SWT, SROW, TE, TR, TN,
                 F, D, N, sd0, sd1, sw20, sw21, sw22,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                 FP32: tl.constexpr, SPLIT: tl.constexpr, LP: tl.constexpr):
    """dh from dy through w2 and the SwiGLU, plus partial <dy, expert out>.

    The partial is contracted against the float32 activation rather than
    against the stored expert output, so it does not inherit the rounding
    gemm2's operand carries.
    """
    p = tl.program_id(0)
    n = tl.load(TN + p)
    if n == 0:
        return
    e = tl.load(TE + p)
    r0 = tl.load(TR + p)
    rm = tl.arange(0, BM)
    mrow = rm < n
    ri = (r0 + rm).to(tl.int64)
    rows = tl.load(SROW + r0 + rm, mask=mrow, other=0)
    pn = tl.program_id(1)
    rn = pn * BN + tl.arange(0, BN)
    mn = rn < F
    wp = W2 + e.to(tl.int64) * sw20 + rn[None, :] * sw21
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, D, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < D
        a = tl.load(DY + rows[:, None] * sd0 + rk[None, :] * sd1,
                    mask=mrow[:, None] & mk[None, :], other=0.0)
        b = tl.load(wp + rk[:, None] * sw22,
                    mask=mk[:, None] & mn[None, :], other=0.0)
        acc += _dotx(a, b, LP)
    m = mrow[:, None] & mn[None, :]
    oa = ri[:, None] * F + rn[None, :]
    a_v = tl.load(ACT + oa, mask=m, other=0.0).to(tl.float32)
    if SPLIT:
        a_v += tl.load(ACL + oa, mask=m, other=0.0).to(tl.float32)
    tl.store(PART + pn.to(tl.int64) * N + r0 + rm,
             tl.sum(tl.where(m, acc * a_v, 0.0), axis=1), mask=mrow)
    wt = tl.load(SWT + r0 + rm, mask=mrow, other=0.0)
    da = acc * wt[:, None]
    gu = GU + ri[:, None] * (2 * F) + rn[None, :]
    g = tl.load(gu, mask=m, other=0.0)
    u = tl.load(gu + F, mask=m, other=0.0)
    sg = tl.sigmoid(g)
    silu = g * sg
    dg = da * u * (sg + silu * (1.0 - sg))
    du = da * silu
    o = DGU + ri[:, None] * (2 * F) + rn[None, :]
    if SPLIT:
        gh = dg.to(tl.bfloat16)
        uh = du.to(tl.bfloat16)
        tl.store(o, gh, mask=m)
        tl.store(o + F, uh, mask=m)
        ol = DGL + ri[:, None] * (2 * F) + rn[None, :]
        tl.store(ol, (dg - gh.to(tl.float32)).to(tl.bfloat16), mask=m)
        tl.store(ol + F, (du - uh.to(tl.float32)).to(tl.bfloat16), mask=m)
    else:
        tl.store(o, dg, mask=m)
        tl.store(o + F, du, mask=m)


@triton.jit
def _dscore_kernel(PART, INV, WT, SS, SEL, DLOG, T, E, A, N, NFB,
                   BT: tl.constexpr, BA: tl.constexpr):
    """Combine-weight cotangents through the normalization and the sigmoid."""
    rt = tl.program_id(0) * BT + tl.arange(0, BT)
    ra = tl.arange(0, BA)
    mt = rt < T
    ma = ra < A
    m = mt[:, None] & ma[None, :]
    i = tl.load(INV + rt[:, None] * A + ra[None, :], mask=m, other=0).to(tl.int64)
    dwa = tl.zeros((BT, BA), dtype=tl.float32)
    for fb in range(NFB):
        dwa += tl.load(PART + fb.to(tl.int64) * N + i, mask=m, other=0.0)
    w = tl.load(WT + rt[:, None] * A + ra[None, :], mask=m, other=0.0)
    s = tl.load(SS + rt[:, None] * A + ra[None, :], mask=m, other=0.0)
    denom = tl.sum(tl.where(m, s, 0.0), axis=1)
    dot = tl.sum(tl.where(m, dwa * w, 0.0), axis=1)
    ds = _div(dwa - dot[:, None], denom[:, None])
    sel = tl.load(SEL + rt[:, None] * A + ra[None, :], mask=m, other=0).to(tl.int32)
    tl.store(DLOG + rt[:, None] * E + sel, ds * s * (1.0 - s), mask=m)


@triton.jit
def _dw2_kernel(AWH, AWL, DY, SROW, OFF, DW2,
                F, D, NTD, sd0, sd1, sg0, sg1, sg2,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                FP32: tl.constexpr, LW: tl.constexpr):
    """dw2[e] = sum over the expert's sorted rows of (w * act)^T dy."""
    e = tl.program_id(1)
    pid = tl.program_id(0)
    rf = (pid // NTD) * BM + tl.arange(0, BM)
    rd = (pid % NTD) * BN + tl.arange(0, BN)
    mf = rf < F
    md = rd < D
    lo = tl.load(OFF + e)
    hi = tl.load(OFF + e + 1)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(lo, hi, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < hi
        rows = tl.load(SROW + rk, mask=mk, other=0)
        oa = rk[:, None].to(tl.int64) * F + rf[None, :]
        ma = mk[:, None] & mf[None, :]
        b = tl.load(DY + rows[:, None] * sd0 + rd[None, :] * sd1,
                    mask=mk[:, None] & md[None, :], other=0.0)
        acc += _dotx(tl.trans(tl.load(AWH + oa, mask=ma, other=0.0)), b, LW)
        if not FP32:
            acc += tl.dot(tl.trans(tl.load(AWL + oa, mask=ma, other=0.0)), b)
    tl.store(DW2 + e.to(tl.int64) * sg0 + rf[:, None] * sg1 + rd[None, :] * sg2,
             acc.to(DW2.dtype.element_ty), mask=mf[:, None] & md[None, :])


@triton.jit
def _dw1_kernel(X, DGU, DGL, SROW, OFF, DW1,
                D, F2, NTN, sx0, sx1, sg0, sg1, sg2,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                FP32: tl.constexpr, LW: tl.constexpr):
    """dw1[e] = sum over the expert's sorted rows of xg^T dh."""
    e = tl.program_id(1)
    pid = tl.program_id(0)
    rd = (pid // NTN) * BM + tl.arange(0, BM)
    rn = (pid % NTN) * BN + tl.arange(0, BN)
    md = rd < D
    mn = rn < F2
    lo = tl.load(OFF + e)
    hi = tl.load(OFF + e + 1)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(lo, hi, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < hi
        rows = tl.load(SROW + rk, mask=mk, other=0)
        a = tl.trans(tl.load(X + rows[:, None] * sx0 + rd[None, :] * sx1,
                             mask=mk[:, None] & md[None, :], other=0.0))
        ob = rk[:, None].to(tl.int64) * F2 + rn[None, :]
        mb = mk[:, None] & mn[None, :]
        acc += _dotx(a, tl.load(DGU + ob, mask=mb, other=0.0), LW)
        if not FP32:
            acc += tl.dot(a, tl.load(DGL + ob, mask=mb, other=0.0))
    tl.store(DW1 + e.to(tl.int64) * sg0 + rd[:, None] * sg1 + rn[None, :] * sg2,
             acc.to(DW1.dtype.element_ty), mask=md[:, None] & mn[None, :])


@triton.jit
def _dxs_kernel(DGU, W1, DXG, TE, TR, TN,
                F2, D, sw10, sw11, sw12,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                FP32: tl.constexpr):
    """dxg = dh @ w1[e]^T for each dispatched pair."""
    p = tl.program_id(0)
    n = tl.load(TN + p)
    if n == 0:
        return
    e = tl.load(TE + p)
    r0 = tl.load(TR + p)
    rm = tl.arange(0, BM)
    mrow = rm < n
    ri = (r0 + rm).to(tl.int64)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    mn = rn < D
    wp = W1 + e.to(tl.int64) * sw10 + rn[None, :] * sw11
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, F2, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < F2
        a = tl.load(DGU + ri[:, None] * F2 + rk[None, :],
                    mask=mrow[:, None] & mk[None, :], other=0.0)
        b = tl.load(wp + rk[:, None] * sw12,
                    mask=mk[:, None] & mn[None, :], other=0.0)
        acc += _dotx(a, b)
    tl.store(DXG + ri[:, None] * D + rn[None, :], acc,
             mask=mrow[:, None] & mn[None, :])


@triton.jit
def _dxcomb_kernel(DXG, DLOG, RW, DX, INV, T, D, E, A,
                   sr0, sr1, sq0, sq1,
                   BT: tl.constexpr, BD: tl.constexpr, BE: tl.constexpr,
                   FP32: tl.constexpr):
    """dx = sum of the A expert paths plus dlogits @ router_w^T."""
    rt = tl.program_id(0) * BT + tl.arange(0, BT)
    rd = tl.program_id(1) * BD + tl.arange(0, BD)
    mt = rt < T
    md = rd < D
    acc = tl.zeros((BT, BD), dtype=tl.float32)
    for a in range(A):
        i = tl.load(INV + rt * A + a, mask=mt, other=0).to(tl.int64)
        acc += tl.load(DXG + i[:, None] * D + rd[None, :],
                       mask=mt[:, None] & md[None, :], other=0.0)
    re = tl.arange(0, BE)
    me = re < E
    dl = tl.load(DLOG + rt[:, None] * E + re[None, :],
                 mask=mt[:, None] & me[None, :], other=0.0)
    rwt = tl.load(RW + re[:, None] * sr1 + rd[None, :] * sr0,
                  mask=me[:, None] & md[None, :], other=0.0)
    acc += _dotx(dl, rwt)
    tl.store(DX + rt[:, None] * sq0 + rd[None, :] * sq1,
             acc.to(DX.dtype.element_ty), mask=mt[:, None] & md[None, :])


@triton.jit
def _drw_kernel(X, DLOG, DRW, T, D, E, sx0, sx1, sq0, sq1,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                FP32: tl.constexpr, LR: tl.constexpr):
    """drouter_w = x^T @ dlogits."""
    rd = tl.program_id(0) * BM + tl.arange(0, BM)
    re = tl.program_id(1) * BN + tl.arange(0, BN)
    md = rd < D
    me = re < E
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, T, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < T
        a = tl.trans(tl.load(X + rk[:, None] * sx0 + rd[None, :] * sx1,
                             mask=mk[:, None] & md[None, :], other=0.0))
        b = tl.load(DLOG + rk[:, None] * E + re[None, :],
                    mask=mk[:, None] & me[None, :], other=0.0)
        acc += _dotx(a, b, LR)
    tl.store(DRW + rd[:, None] * sq0 + re[None, :] * sq1,
             acc.to(DRW.dtype.element_ty), mask=md[:, None] & me[None, :])


# --------------------------------------------------------------------------
# host side
# --------------------------------------------------------------------------

def _p2(v):
    return 1 << max(0, (int(v) - 1).bit_length())


def _bits(v):
    return int(max(1, v)).bit_length() + 1


def _cap(v, hi, lo=16):
    return min(hi, max(lo, _p2(v)))


# (BM, BN, BK, num_warps, num_stages) per GEMM, indexed by the dtype of x.
# The float32 rows carry three-pass tf32 accumulation, which costs registers
# and shared memory, so they take narrower tiles and fewer stages.
_CFG = {
    "gemm1": {False: (128, 128, 32, 8, 4), True: (128, 64, 64, 8, 3)},
    "gemm2": {False: (128, 256, 64, 8, 4), True: (128, 128, 32, 8, 3)},
    "dact": {False: (128, 64, 64, 8, 4), True: (128, 128, 64, 8, 3)},
    "dxs": {False: (128, 256, 64, 8, 4), True: (128, 128, 64, 8, 3)},
    "dw1": {False: (128, 128, 64, 8, 4), True: (128, 128, 64, 8, 3)},
    "dw2": {False: (128, 128, 64, 8, 4), True: (128, 128, 64, 8, 3)},
    "drw": {False: (64, 64, 64, 4, 3), True: (64, 64, 32, 4, 3)},
}


_SMEM = 216 * 1024   # per-block shared memory the H100 offers, less a margin


def _cfg(name, fp32, m=None, n=None, na=1, nb=1):
    """Tile for one GEMM, shrunk until its operand pipeline fits in shared
    memory. na and nb count the tiles the loop loads on each side -- a limb
    pair loads two where a plain operand loads one -- and the pipeliner
    stages every one of them num_stages deep."""
    bm, bn, bk, w, st = _CFG[name][bool(fp32)]
    if m is not None:
        bm = _cap(m, bm)
    if n is not None:
        bn = _cap(n, bn)
    width = 4 if fp32 else 2

    def need():
        return (na * bm + nb * bn) * bk * width * st

    while st > 2 and need() > _SMEM:
        st -= 1
    while bk > 16 and need() > _SMEM:
        bk //= 2
    while bn > 16 and need() > _SMEM:
        bn //= 2
    if bm * bn <= 4096:
        w = min(w, 4)
    return bm, bn, bk, w, st


class _FusedMoE(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, router_w, router_b, w1, w2, A):
        A = int(A)
        T, D = x.shape
        E = router_w.shape[1]
        F2 = w1.shape[2]
        F = F2 // 2
        dev = x.device
        fp32 = x.dtype == torch.float32
        N = T * A

        s = torch.empty((T, E), device=dev, dtype=torch.float32)
        key = torch.empty((T, E), device=dev, dtype=torch.int64)
        be = max(16, _p2(E))
        bt = 64 if be <= 64 else (32 if be <= 128 else 16)
        _router_kernel[(triton.cdiv(T, bt),)](
            x, router_w, s, key, router_b, T, D, E,
            x.stride(0), x.stride(1), router_w.stride(0), router_w.stride(1),
            BT=bt, BE=be, BK=32, FP32=fp32, num_warps=4, num_stages=3)

        sel = torch.topk(key, A, dim=-1, largest=True, sorted=True).indices
        selc = sel.to(torch.int32).contiguous()

        wt = torch.empty((T, A), device=dev, dtype=torch.float32)
        ss = torch.empty((T, A), device=dev, dtype=torch.float32)
        _wcomb_kernel[(triton.cdiv(T, 64),)](
            s, selc, wt, ss, T, E, A, BT=64, BA=max(1, _p2(A)), num_warps=4)

        sorted_e, order = torch.sort(selc.view(-1), stable=True)
        srow = torch.empty(N, device=dev, dtype=torch.int32)
        swt = torch.empty(N, device=dev, dtype=torch.float32)
        inv = torch.empty(N, device=dev, dtype=torch.int32)
        _smeta_kernel[(triton.cdiv(N, 256),)](
            order, wt, srow, swt, inv, N, A, BLK=256, num_warps=4)

        rpe = N // max(1, E)
        BM = 128 if rpe >= 96 else (64 if rpe >= 24 else 32)
        ntmax = triton.cdiv(N, BM) + E
        off = torch.empty(E + 1, device=dev, dtype=torch.int32)
        tcum = torch.empty(E + 1, device=dev, dtype=torch.int32)
        _off_kernel[(1,)](sorted_e, off, tcum, N, E, BM,
                          BE=max(16, _p2(E + 1)), NBITS=_bits(N), num_warps=4)
        te = torch.empty(ntmax, device=dev, dtype=torch.int32)
        tr = torch.empty(ntmax, device=dev, dtype=torch.int32)
        tn = torch.empty(ntmax, device=dev, dtype=torch.int32)
        _tiles_kernel[(triton.cdiv(ntmax, 128),)](
            off, tcum, te, tr, tn, ntmax, E, BM,
            BP=128, NEBITS=_bits(E + 1), num_warps=4)

        # act stays float32: gemm2 rounds it to its own operand precision,
        # and the score-gradient partial contracts against it directly. The
        # pre-weighted copy is what dw2 contracts, so on the bfloat16 rows it
        # is stored as an exact bfloat16 high/low pair instead.
        split = not fp32
        # Limb count for the projections whose result is differenced against
        # something close to itself downstream: y against the reference's y,
        # and dh against the activation inside the score-gradient partial.
        # The float32 rows need three there; on the bfloat16 rows one limb is
        # the width of the inputs themselves.
        ly = 3 if fp32 else 1
        idt = torch.bfloat16 if split else torch.float32
        act = torch.empty((N, F), device=dev, dtype=idt)
        acl = torch.empty((N, F) if split else (0, 0), device=dev, dtype=idt)
        awh = torch.empty((N, F), device=dev, dtype=idt)
        awl = torch.empty((N, F) if split else (0, 0), device=dev, dtype=idt)
        gu = torch.empty((N, F2), device=dev, dtype=torch.float32)
        _, bn1, bk1, w1w, s1 = _cfg("gemm1", fp32, n=F, nb=2)
        _gemm1_kernel[(ntmax, triton.cdiv(F, bn1))](
            x, w1, act, acl, awh, awl, gu, te, tr, tn, srow, swt, D, F,
            x.stride(0), x.stride(1), w1.stride(0), w1.stride(1), w1.stride(2),
            BM=BM, BN=bn1, BK=bk1, FP32=fp32, SPLIT=split, LY=ly,
            num_warps=w1w, num_stages=s1)

        pbuf = torch.empty((N, D), device=dev, dtype=torch.float32)
        _, bn2, bk2, w2w, s2 = _cfg("gemm2", fp32, n=D, na=2 if split else 1)
        _gemm2_kernel[(ntmax, triton.cdiv(D, bn2))](
            act, acl, w2, pbuf, swt, te, tr, tn, F, D,
            w2.stride(0), w2.stride(1), w2.stride(2),
            BM=BM, BN=bn2, BK=bk2, SPLIT=split, LY=ly,
            num_warps=w2w, num_stages=s2)

        y = torch.empty((T, D), device=dev, dtype=x.dtype)
        bd = _cap(D, 128)
        _combine_kernel[(triton.cdiv(T, 32), triton.cdiv(D, bd))](
            pbuf, inv, y, T, D, A, BT=32, BD=bd, num_warps=4)

        ctx.save_for_backward(x, router_w, w1, w2, act, acl, awh, awl, gu,
                              srow, swt, inv, selc, ss, wt, off, te, tr, tn)
        ctx.meta = (T, D, F, F2, E, A, N, BM, ntmax, fp32, ly)
        return y

    @staticmethod
    def backward(ctx, dy):
        (x, router_w, w1, w2, act, acl, awh, awl, gu, srow, swt, inv,
         selc, ss, wt, off, te, tr, tn) = ctx.saved_tensors
        T, D, F, F2, E, A, N, BM, ntmax, fp32, ly = ctx.meta
        dev = x.device
        dy = dy.contiguous()

        split = not fp32
        idt = torch.bfloat16 if split else torch.float32
        _, bnd, bkd, wd, sd = _cfg("dact", fp32, n=F)
        nfb = triton.cdiv(F, bnd)
        dgu = torch.empty((N, F2), device=dev, dtype=idt)
        dgl = torch.empty((N, F2) if split else (0, 0), device=dev, dtype=idt)
        part = torch.empty((nfb, N), device=dev, dtype=torch.float32)
        _dact_kernel[(ntmax, nfb)](
            dy, w2, act, acl, gu, dgu, dgl, part, swt, srow, te, tr, tn,
            F, D, N,
            dy.stride(0), dy.stride(1),
            w2.stride(0), w2.stride(1), w2.stride(2),
            BM=BM, BN=bnd, BK=bkd, FP32=fp32, SPLIT=split, LP=ly,
            num_warps=wd, num_stages=sd)

        dlog = torch.zeros((T, E), device=dev, dtype=torch.float32)
        _dscore_kernel[(triton.cdiv(T, 64),)](
            part, inv, wt, ss, selc, dlog, T, E, A, N, nfb,
            BT=64, BA=max(1, _p2(A)), num_warps=4)

        # The weight gradients are long per-expert row sums, so on the
        # float32 rows their operands carry the full float32 width.
        lw = 3 if fp32 else 2
        dw1 = torch.empty_like(w1)
        bm1, bn1, bk1, ww1, st1 = _cfg("dw1", fp32, m=D, n=F2,
                                       nb=2 if split else 1)
        ntn = triton.cdiv(F2, bn1)
        _dw1_kernel[(triton.cdiv(D, bm1) * ntn, E)](
            x, dgu, dgl, srow, off, dw1, D, F2, ntn,
            x.stride(0), x.stride(1),
            dw1.stride(0), dw1.stride(1), dw1.stride(2),
            BM=bm1, BN=bn1, BK=bk1, FP32=fp32, LW=lw,
            num_warps=ww1, num_stages=st1)

        dw2 = torch.empty_like(w2)
        bm2, bn2, bk2, ww2, st2 = _cfg("dw2", fp32, m=F, n=D,
                                       na=2 if split else 1)
        ntd = triton.cdiv(D, bn2)
        _dw2_kernel[(triton.cdiv(F, bm2) * ntd, E)](
            awh, awl, dy, srow, off, dw2, F, D, ntd,
            dy.stride(0), dy.stride(1),
            dw2.stride(0), dw2.stride(1), dw2.stride(2),
            BM=bm2, BN=bn2, BK=bk2, FP32=fp32, LW=lw,
            num_warps=ww2, num_stages=st2)

        dxg = torch.empty((N, D), device=dev, dtype=torch.float32)
        _, bnx, bkx, wx, sx = _cfg("dxs", fp32, n=D)
        _dxs_kernel[(ntmax, triton.cdiv(D, bnx))](
            dgu, w1, dxg, te, tr, tn, F2, D,
            w1.stride(0), w1.stride(1), w1.stride(2),
            BM=BM, BN=bnx, BK=bkx, FP32=fp32, num_warps=wx, num_stages=sx)

        dx = torch.empty((T, D), device=dev, dtype=x.dtype)
        bdc = _cap(D, 128)
        _dxcomb_kernel[(triton.cdiv(T, 32), triton.cdiv(D, bdc))](
            dxg, dlog, router_w, dx, inv, T, D, E, A,
            router_w.stride(0), router_w.stride(1), dx.stride(0), dx.stride(1),
            BT=32, BD=bdc, BE=max(16, _p2(E)), FP32=fp32, num_warps=4)

        drw = torch.empty((D, E), device=dev, dtype=router_w.dtype)
        bmr, bnr, bkr, wr, sr = _cfg("drw", fp32, m=D, n=E)
        _drw_kernel[(triton.cdiv(D, bmr), triton.cdiv(E, bnr))](
            x, dlog, drw, T, D, E,
            x.stride(0), x.stride(1), drw.stride(0), drw.stride(1),
            BM=bmr, BN=bnr, BK=bkr, FP32=fp32, LR=3,
            num_warps=wr, num_stages=sr)

        return (dx, drw, None, dw1, dw2, None)


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
