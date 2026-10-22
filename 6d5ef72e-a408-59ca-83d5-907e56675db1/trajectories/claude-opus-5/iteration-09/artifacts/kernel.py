"""Routed fused mixture-of-experts layer, forward and backward.

Pipeline
--------
Routing turns every token into A (token, expert) pairs.  The P = T*A pairs are
sorted by expert -- stably, so the order inside an expert is the flat (token,
a) order -- and each expert's run is padded up to a multiple of the GEMM row
tile, so a row tile never straddles two experts and every expert projection is
one kernel launch over a grid of (padded row tile, column tile).

    T, D, F, E, A     tokens, model dim, expert hidden dim, experts, top-A
    P = T*A           dispatched slots
    s[t, e]           sigmoid router score
    w[t, a]           combine weight: raw selected scores normalised over a
    gu[p]             gate|up pre-activations x @ w1[e],  [P, 2F]
    actw[p]           w_p * (silu(gate) * up),            [P, F]
    yp[p]             actw[p] @ w2[e],                    [P, D]
    y[t]              sum of yp over the A slots of token t

Folding the combine weight into the hidden state before the second projection
makes the forward combine a plain sum and removes the division by the score
sum from the backward.  Writing  R_p = <dy[token(p)], yp[p]>, which the second
projection's backward already forms, the score gradient is

    dlogit[t, a] = (1 - s_a) * (R_a - w_a * sum_b R_b)

because the 1/S from d(s_a / S) cancels against the w_a folded inside R_a.
That bracket is a cancelling difference, so R is accumulated in float32
registers from an unrounded elementwise product rather than read back from a
rounded matmul result.

Precision
---------
Scores, every accumulator, the sigmoid, the silu, the normalisation and the
combine are float32.  Every dot carries float32-effective operands.  On
bfloat16 inputs the weights, x and dy are bfloat16-exact, and the
float32-valued intermediates (actw, dgu, dlogit) that feed the weight
gradients are split into an exact bfloat16 high/low limb pair consumed as two
tensor-core dots.  On float32 inputs both operands are expanded into bfloat16
limbs in registers and the product is reassembled from the cross terms; see
_dotf.

Accumulation order
------------------
Every output element of every kernel is written by exactly one program, and
sums that span programs are loops over partial buffers taken in index order,
so the arithmetic does not depend on how the blocks are scheduled.
"""

import torch
import triton
import triton.language as tl

_GN = 8            # row tiles per L2 swizzle group
_DACT_BM = 64      # row tile of the hidden-gradient kernel
_DACT_BN = 64      # its column tile


@triton.jit
def _swizzle(pid, NM, NN, GN: tl.constexpr):
    """order tiles so a group of row tiles sweeps the columns together."""
    ng = GN * NN
    first = (pid // ng) * GN
    gsz = min(NM - first, GN)
    r = pid % ng
    return first + r % gsz, r // gsz


@triton.jit
def _limbs(v):
    """exact three-limb bfloat16 expansion of a float32 tile."""
    h = v.to(tl.bfloat16)
    r = v - h.to(tl.float32)
    m = r.to(tl.bfloat16)
    return h, m, (r - m.to(tl.float32)).to(tl.bfloat16)


@triton.jit
def _dotf(a, b, acc, cor):
    """float32 a @ b on bfloat16 tensor cores, float32-effective.

    bfloat16 keeps 8 mantissa bits, so three limbs span a float32 significand
    exactly.  Of the nine cross products the six of relative weight 2^-16 and
    above are kept and the rest, starting at 2^-24, are dropped.

    The five correction terms go into their own accumulator rather than onto
    the leading one.  Added to the leading accumulator each correction is
    below its last bit and rounds straight back off, which leaves the
    reduction no more accurate than a single bfloat16 pass however many limbs
    are formed; the caller adds the two accumulators once at the end.  The
    limbs live in registers, so shared memory still holds only the float32
    operand tiles a single-pass dot would stage.
    """
    a0, a1, a2 = _limbs(a)
    b0, b1, b2 = _limbs(b)
    cor = tl.dot(a2, b0, cor)
    cor = tl.dot(a0, b2, cor)
    cor = tl.dot(a1, b1, cor)
    cor = tl.dot(a1, b0, cor)
    cor = tl.dot(a0, b1, cor)
    return tl.dot(a0, b0, acc), cor


@triton.jit
def _dot(a, b, acc, cor, FP32: tl.constexpr):
    if FP32:
        return _dotf(a, b, acc, cor)
    return tl.dot(a, b, acc), cor


# ---------------------------------------------------------------------------
# routing
# ---------------------------------------------------------------------------


@triton.jit
def _score_kernel(X, RW, RB, S, KEY, T, D, E, sx0, sr0,
                  FP32: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                  BK: tl.constexpr):
    """s = sigmoid(x @ router_w), and the total-order selection key.

    KEY[t, e] packs the order-preserving integer bits of the biased score with
    the inverted expert index, so "largest key" is "largest biased score,
    smallest expert index": every key is unique and no tie is left open.
    -0.0 is folded onto +0.0 first so both zeros form one tie group.
    """
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    mm = rm < T
    mn = rn < E
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    cor = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, D, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < D
        a = tl.load(X + rm[:, None].to(tl.int64) * sx0 + rk[None, :],
                    mask=mm[:, None] & mk[None, :], other=0.0)
        b = tl.load(RW + rk[:, None] * sr0 + rn[None, :],
                    mask=mk[:, None] & mn[None, :], other=0.0)
        acc, cor = _dot(a, b, acc, cor, FP32)
    s = tl.sigmoid(acc + cor)
    m = mm[:, None] & mn[None, :]
    o = rm[:, None].to(tl.int64) * E + rn[None, :]
    tl.store(S + o, s, mask=m)
    b = s + tl.load(RB + rn, mask=mn, other=0.0)[None, :]
    b = tl.where(b == 0.0, 0.0, b)
    ib = b.to(tl.int32, bitcast=True).to(tl.int64)
    u = tl.where(ib >= 0, ib ^ 0x80000000, (~ib) & 0xFFFFFFFF)
    tl.store(KEY + o, (u << 11) | (E - 1 - rn).to(tl.int64)[None, :], mask=m)


@triton.jit
def _weight_kernel(S, SEL, W, T, E, A, BT: tl.constexpr, BA: tl.constexpr):
    """combine weights: raw selected scores, normalised over the A slots."""
    rt = tl.program_id(0) * BT + tl.arange(0, BT)
    ra = tl.arange(0, BA)
    m = (rt[:, None] < T) & (ra[None, :] < A)
    i = rt[:, None].to(tl.int64) * A + ra[None, :]
    e = tl.load(SEL + i, mask=m, other=0)
    s = tl.load(S + rt[:, None].to(tl.int64) * E + e, mask=m, other=0.0)
    tl.store(W + i, s / tl.sum(tl.where(m, s, 0.0), axis=1)[:, None], mask=m)


@triton.jit
def _offset_kernel(SE, OFF, P, E, NSTEP, BE: tl.constexpr):
    """expert run boundaries: OFF[e] is the first slot of expert e.

    A lower bound of e in the expert-sorted key list, so OFF[E] = P and the
    sentinel OFF[E+1] = P makes an out-of-range tile an empty range.
    """
    re = tl.arange(0, BE)
    m = re <= E + 1
    lo = tl.zeros((BE,), dtype=tl.int32)
    hi = tl.full((BE,), P, dtype=tl.int32)
    for _ in range(NSTEP):
        mid = (lo + hi) // 2
        c = tl.load(SE + mid, mask=(mid < P) & m, other=0x7FFFFFFF) < re
        lo = tl.where(c, mid + 1, lo)
        hi = tl.where(c, hi, mid)
    tl.store(OFF + re, lo, mask=m)


@triton.jit
def _toff_kernel(OFF, TOFF, E, BM_G, BE: tl.constexpr):
    """first row tile of each expert once its run is padded up to BM_G."""
    re = tl.arange(0, BE)
    m = re < E
    cnt = (tl.load(OFF + re + 1, mask=m, other=0)
           - tl.load(OFF + re, mask=m, other=0))
    nt = tl.where(m, (cnt + BM_G - 1) // BM_G, 0)
    tl.store(TOFF + re, tl.cumsum(nt, axis=0) - nt, mask=m)
    tl.store(TOFF + E, tl.sum(nt, axis=0))


@triton.jit
def _dispatch_kernel(ORDER, W, PTOK, PW, SLOT, P, A, BLK: tl.constexpr):
    """materialise the dispatch: source token, weight, and slot of (t, a)."""
    r = tl.program_id(0) * BLK + tl.arange(0, BLK)
    m = r < P
    o = tl.load(ORDER + r, mask=m, other=0)
    tl.store(PTOK + r, (o // A).to(tl.int32), mask=m)
    tl.store(PW + r, tl.load(W + o, mask=m, other=0.0), mask=m)
    tl.store(SLOT + o, r.to(tl.int32), mask=m)


@triton.jit
def _tilemap_kernel(TOFF, TM_E, TM_T, NT, E, NSTEP, BLK: tl.constexpr):
    """for each padded row tile: its expert, and that expert's first tile.

    Tiles past the last real one map to expert E, whose empty [P, P) range
    makes the GEMM programs return immediately.
    """
    rt = tl.program_id(0) * BLK + tl.arange(0, BLK)
    mt = rt < NT
    lo = tl.zeros((BLK,), dtype=tl.int32)
    hi = tl.full((BLK,), E, dtype=tl.int32)
    for _ in range(NSTEP):
        mid = (lo + hi) // 2
        c = tl.load(TOFF + mid, mask=(mid < E) & mt, other=0x7FFFFFFF) <= rt
        lo = tl.where(c, mid + 1, lo)
        hi = tl.where(c, hi, mid)
    e = lo - 1
    live = (e >= 0) & (rt < tl.load(TOFF + E))
    tl.store(TM_E + rt, tl.where(live, e, E), mask=mt)
    tl.store(TM_T + rt, tl.load(TOFF + e, mask=live & mt, other=0), mask=mt)


# ---------------------------------------------------------------------------
# forward expert path
# ---------------------------------------------------------------------------


@triton.jit
def _gemm1_kernel(X, W1, GU, ACTH, ACTL, PTOK, PW, TM_E, TM_T, OFF,
                  NM, NN, D, F, sx0, sw0, sw1, sg0, sa0,
                  SPLIT_OUT: tl.constexpr, FP32: tl.constexpr,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                  GN: tl.constexpr):
    """gate|up projection, silu-multiply, and the combine weight folded in.

    One program covers the gate and the up half of the same column tile, so
    the row panel of x is read once for both halves.
    """
    pid_m, pid_n = _swizzle(tl.program_id(0), NM, NN, GN)
    e = tl.load(TM_E + pid_m).to(tl.int64)
    lo = tl.load(OFF + e)
    hi = tl.load(OFF + e + 1)
    if lo >= hi:
        return
    rp = (pid_m - tl.load(TM_T + pid_m)) * BM + lo + tl.arange(0, BM)
    ok = rp < hi
    tok = tl.load(PTOK + rp, mask=ok, other=0).to(tl.int64)
    rn = pid_n * BN + tl.arange(0, BN)
    mn = rn < F
    wg = W1 + e * sw0 + rn[None, :]
    xp = X + tok[:, None] * sx0
    ag = tl.zeros((BM, BN), dtype=tl.float32)
    au = tl.zeros((BM, BN), dtype=tl.float32)
    cg = tl.zeros((BM, BN), dtype=tl.float32)
    cu = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, D, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < D
        bmask = mk[:, None] & mn[None, :]
        a = tl.load(xp + rk[None, :], mask=ok[:, None] & mk[None, :],
                    other=0.0)
        bg = tl.load(wg + rk[:, None] * sw1, mask=bmask, other=0.0)
        bu = tl.load(wg + F + rk[:, None] * sw1, mask=bmask, other=0.0)
        ag, cg = _dot(a, bg, ag, cg, FP32)
        au, cu = _dot(a, bu, au, cu, FP32)
    ag += cg
    au += cu
    m = ok[:, None] & mn[None, :]
    g = GU + rp[:, None].to(tl.int64) * sg0 + rn[None, :]
    tl.store(g, ag, mask=m)
    tl.store(g + F, au, mask=m)
    wt = tl.load(PW + rp, mask=ok, other=0.0)
    act = (ag * tl.sigmoid(ag)) * au * wt[:, None]
    o = rp[:, None].to(tl.int64) * sa0 + rn[None, :]
    if SPLIT_OUT:
        h = act.to(tl.bfloat16)
        tl.store(ACTH + o, h, mask=m)
        tl.store(ACTL + o, (act - h.to(tl.float32)).to(tl.bfloat16), mask=m)
    else:
        tl.store(ACTH + o, act, mask=m)


@triton.jit
def _gemm2_kernel(ACTH, ACTL, W2, YP, TM_E, TM_T, OFF, NM, NN, F, D,
                  sa0, sw0, sw1, sy0, SPLIT: tl.constexpr,
                  FP32: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                  BK: tl.constexpr, GN: tl.constexpr):
    """second projection of the already-weighted hidden state."""
    pid_m, pid_n = _swizzle(tl.program_id(0), NM, NN, GN)
    e = tl.load(TM_E + pid_m).to(tl.int64)
    lo = tl.load(OFF + e)
    hi = tl.load(OFF + e + 1)
    if lo >= hi:
        return
    rp = (pid_m - tl.load(TM_T + pid_m)) * BM + lo + tl.arange(0, BM)
    ok = rp < hi
    rn = pid_n * BN + tl.arange(0, BN)
    mn = rn < D
    wp = W2 + e * sw0 + rn[None, :]
    ab = rp[:, None].to(tl.int64) * sa0
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    cor = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, F, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < F
        am = ok[:, None] & mk[None, :]
        a = tl.load(ACTH + ab + rk[None, :], mask=am, other=0.0)
        b = tl.load(wp + rk[:, None] * sw1, mask=mk[:, None] & mn[None, :],
                    other=0.0)
        acc, cor = _dot(a, b, acc, cor, FP32)
        if SPLIT:
            acc = tl.dot(tl.load(ACTL + ab + rk[None, :], mask=am,
                                 other=0.0), b, acc)
    tl.store(YP + rp[:, None].to(tl.int64) * sy0 + rn[None, :], acc + cor,
             mask=ok[:, None] & mn[None, :])


@triton.jit
def _combine_kernel(YP, SLOT, Y, T, D, A, sy0, so0, OUT_BF: tl.constexpr,
                    BT: tl.constexpr, BD: tl.constexpr):
    """y[t] = sum of the token's A slot outputs, in slot order."""
    rt = tl.program_id(0) * BT + tl.arange(0, BT)
    rd = tl.program_id(1) * BD + tl.arange(0, BD)
    mt = rt < T
    md = rd < D
    m = mt[:, None] & md[None, :]
    acc = tl.zeros((BT, BD), dtype=tl.float32)
    for a in range(A):
        p = tl.load(SLOT + rt * A + a, mask=mt, other=0).to(tl.int64)
        acc += tl.load(YP + p[:, None] * sy0 + rd[None, :], mask=m, other=0.0)
    o = Y + rt[:, None].to(tl.int64) * so0 + rd[None, :]
    if OUT_BF:
        tl.store(o, acc.to(tl.bfloat16), mask=m)
    else:
        tl.store(o, acc, mask=m)


# ---------------------------------------------------------------------------
# backward
# ---------------------------------------------------------------------------


@triton.jit
def _dact_kernel(DY, W2, ACTH, ACTL, GU, PTOK, PW, PR, DGH, DGL,
                 TM_E, TM_T, OFF, NM, NN, P, F, D, sd0, sw0, sw1, sa0, sg0,
                 SPLIT_OUT: tl.constexpr, FP32: tl.constexpr,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                 GN: tl.constexpr):
    """hidden gradient, silu-multiply backward, and the score-gradient term.

    R_p = <dy[token], yp[p]> is formed here as <dy @ w2^T, actw> summed in
    float32 registers, so the cancelling quantity the router gradient is built
    from never passes through a rounded operand.
    """
    pid_m, pid_n = _swizzle(tl.program_id(0), NM, NN, GN)
    e = tl.load(TM_E + pid_m).to(tl.int64)
    lo = tl.load(OFF + e)
    hi = tl.load(OFF + e + 1)
    if lo >= hi:
        return
    rp = (pid_m - tl.load(TM_T + pid_m)) * BM + lo + tl.arange(0, BM)
    ok = rp < hi
    tok = tl.load(PTOK + rp, mask=ok, other=0).to(tl.int64)
    rn = pid_n * BN + tl.arange(0, BN)
    mn = rn < F
    wp = W2 + e * sw0 + rn[None, :] * sw1
    dyp = DY + tok[:, None] * sd0
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    cor = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, D, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < D
        a = tl.load(dyp + rk[None, :], mask=ok[:, None] & mk[None, :],
                    other=0.0)
        b = tl.load(wp + rk[:, None], mask=mk[:, None] & mn[None, :],
                    other=0.0)
        acc, cor = _dot(a, b, acc, cor, FP32)
    acc += cor
    m = ok[:, None] & mn[None, :]
    ao = rp[:, None].to(tl.int64) * sa0 + rn[None, :]
    if SPLIT_OUT:
        actw = (tl.load(ACTH + ao, mask=m, other=0.0).to(tl.float32)
                + tl.load(ACTL + ao, mask=m, other=0.0).to(tl.float32))
    else:
        actw = tl.load(ACTH + ao, mask=m, other=0.0)
    tl.store(PR + pid_n.to(tl.int64) * P + rp, tl.sum(acc * actw, axis=1),
             mask=ok)

    d = acc * tl.load(PW + rp, mask=ok, other=0.0)[:, None]
    go = GU + rp[:, None].to(tl.int64) * sg0 + rn[None, :]
    g = tl.load(go, mask=m, other=0.0)
    sig = tl.sigmoid(g)
    sl = g * sig
    dg = d * tl.load(go + F, mask=m, other=0.0) * (sig + sl * (1.0 - sig))
    du = d * sl
    oh = DGH + rp[:, None].to(tl.int64) * sg0 + rn[None, :]
    if SPLIT_OUT:
        hg = dg.to(tl.bfloat16)
        hu = du.to(tl.bfloat16)
        tl.store(oh, hg, mask=m)
        tl.store(oh + F, hu, mask=m)
        ol = DGL + rp[:, None].to(tl.int64) * sg0 + rn[None, :]
        tl.store(ol, (dg - hg.to(tl.float32)).to(tl.bfloat16), mask=m)
        tl.store(ol + F, (du - hu.to(tl.float32)).to(tl.bfloat16), mask=m)
    else:
        tl.store(oh, dg, mask=m)
        tl.store(oh + F, du, mask=m)


@triton.jit
def _dscore_kernel(PR, NPART, P, SLOT, SEL, S, DLA, T, E, A,
                   BT: tl.constexpr, BA: tl.constexpr):
    """dlogit of the selected slots: (1 - s) * (R_a - w_a * sum_b R_b).

    The per-column-tile partials of R are summed here in fixed index order.
    """
    rt = tl.program_id(0) * BT + tl.arange(0, BT)
    ra = tl.arange(0, BA)
    m = (rt[:, None] < T) & (ra[None, :] < A)
    i = rt[:, None].to(tl.int64) * A + ra[None, :]
    p = tl.load(SLOT + i, mask=m, other=0).to(tl.int64)
    e = tl.load(SEL + i, mask=m, other=0)
    r = tl.zeros((BT, BA), dtype=tl.float32)
    for k in range(NPART):
        r += tl.load(PR + k.to(tl.int64) * P + p, mask=m, other=0.0)
    s = tl.load(S + rt[:, None].to(tl.int64) * E + e, mask=m, other=0.0)
    w = s / tl.sum(tl.where(m, s, 0.0), axis=1)[:, None]
    tl.store(DLA + i,
             (1.0 - s) * (r - w * tl.sum(tl.where(m, r, 0.0), axis=1)[:, None]),
             mask=m)


@triton.jit
def _dlspread_kernel(DLA, SEL, DLH, DLL, T, E, A, SPLIT_OUT: tl.constexpr,
                     BT: tl.constexpr, BE: tl.constexpr):
    """spread the A selected score gradients over a dense [T, E] row.

    The row is assembled in registers by summing the A one-hot contributions
    and stored once, so a zero fill and a scatter never contend for the same
    address.
    """
    rt = tl.program_id(0) * BT + tl.arange(0, BT)
    re = tl.program_id(1) * BE + tl.arange(0, BE)
    mt = rt < T
    m = mt[:, None] & (re[None, :] < E)
    acc = tl.zeros((BT, BE), dtype=tl.float32)
    for a in range(A):
        i = rt.to(tl.int64) * A + a
        e = tl.load(SEL + i, mask=mt, other=-1)
        v = tl.load(DLA + i, mask=mt, other=0.0)
        acc += tl.where(e[:, None] == re[None, :], v[:, None], 0.0)
    o = rt[:, None].to(tl.int64) * E + re[None, :]
    if SPLIT_OUT:
        h = acc.to(tl.bfloat16)
        tl.store(DLH + o, h, mask=m)
        tl.store(DLL + o, (acc - h.to(tl.float32)).to(tl.bfloat16), mask=m)
    else:
        tl.store(DLH + o, acc, mask=m)


@triton.jit
def _dw1_kernel(X, GH, GL, DW1, PTOK, OFF, D, F2, sx0, sg0, sw0, sw1,
                SPLIT: tl.constexpr, FP32: tl.constexpr, OUT_BF: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                SK: tl.constexpr):
    """dw1[e] = x[rows of e]^T @ dgu[e]; empty experts store exact zeros.

    With SK > 1 each program covers one fixed slice of the expert's rows and
    writes its own partial buffer, which a reduce kernel then sums in slice
    order.
    """
    pid_e = tl.program_id(2)
    e = (pid_e // SK).to(tl.int64)
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    mm = rm < D
    mn = rn < F2
    lo = tl.load(OFF + e)
    hi = tl.load(OFF + e + 1)
    if SK > 1:
        pid_k = pid_e % SK
        chunk = tl.cdiv(tl.cdiv(hi - lo, BK), SK) * BK
        lo += pid_k * chunk
        hi = min(lo + chunk, hi)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    cor = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(lo, hi, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < hi
        tok = tl.load(PTOK + rk, mask=mk, other=0).to(tl.int64)
        a = tl.load(X + tok[None, :] * sx0 + rm[:, None],
                    mask=mk[None, :] & mm[:, None], other=0.0)
        bm = mk[:, None] & mn[None, :]
        go = rk[:, None].to(tl.int64) * sg0 + rn[None, :]
        b = tl.load(GH + go, mask=bm, other=0.0)
        acc, cor = _dot(a, b, acc, cor, FP32)
        if SPLIT:
            acc = tl.dot(a, tl.load(GL + go, mask=bm, other=0.0), acc)
    acc += cor
    o = DW1 + pid_e.to(tl.int64) * sw0 + rm[:, None] * sw1 + rn[None, :]
    m = mm[:, None] & mn[None, :]
    if OUT_BF:
        tl.store(o, acc.to(tl.bfloat16), mask=m)
    else:
        tl.store(o, acc, mask=m)


@triton.jit
def _dw2_kernel(AH, AL, DY, DW2, PTOK, OFF, F, D, sa0, sd0, sw0, sw1,
                SPLIT: tl.constexpr, FP32: tl.constexpr, OUT_BF: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                SK: tl.constexpr):
    """dw2[e] = actw[rows of e]^T @ dy; empty experts store exact zeros.

    SK slices the expert's rows exactly as in the first weight gradient.
    """
    pid_e = tl.program_id(2)
    e = (pid_e // SK).to(tl.int64)
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    mm = rm < F
    mn = rn < D
    lo = tl.load(OFF + e)
    hi = tl.load(OFF + e + 1)
    if SK > 1:
        pid_k = pid_e % SK
        chunk = tl.cdiv(tl.cdiv(hi - lo, BK), SK) * BK
        lo += pid_k * chunk
        hi = min(lo + chunk, hi)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    cor = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(lo, hi, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < hi
        tok = tl.load(PTOK + rk, mask=mk, other=0).to(tl.int64)
        am = mk[None, :] & mm[:, None]
        ao = rk[None, :].to(tl.int64) * sa0 + rm[:, None]
        a = tl.load(AH + ao, mask=am, other=0.0)
        b = tl.load(DY + tok[:, None] * sd0 + rn[None, :],
                    mask=mk[:, None] & mn[None, :], other=0.0)
        acc, cor = _dot(a, b, acc, cor, FP32)
        if SPLIT:
            acc = tl.dot(tl.load(AL + ao, mask=am, other=0.0), b, acc)
    acc += cor
    o = DW2 + pid_e.to(tl.int64) * sw0 + rm[:, None] * sw1 + rn[None, :]
    m = mm[:, None] & mn[None, :]
    if OUT_BF:
        tl.store(o, acc.to(tl.bfloat16), mask=m)
    else:
        tl.store(o, acc, mask=m)


@triton.jit
def _dxp_kernel(GH, GL, W1, DXP, TM_E, TM_T, OFF, NM, NN, F2, D, sg0, sw0,
                sw1, sx0, SPLIT: tl.constexpr, FP32: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                GN: tl.constexpr):
    """per-slot input gradient of the expert path: dxp = dgu @ w1[e]^T."""
    pid_m, pid_n = _swizzle(tl.program_id(0), NM, NN, GN)
    e = tl.load(TM_E + pid_m).to(tl.int64)
    lo = tl.load(OFF + e)
    hi = tl.load(OFF + e + 1)
    if lo >= hi:
        return
    rp = (pid_m - tl.load(TM_T + pid_m)) * BM + lo + tl.arange(0, BM)
    ok = rp < hi
    rn = pid_n * BN + tl.arange(0, BN)
    mn = rn < D
    wp = W1 + e * sw0 + rn[None, :] * sw1
    gp = rp[:, None].to(tl.int64) * sg0
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    cor = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, F2, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < F2
        am = ok[:, None] & mk[None, :]
        a = tl.load(GH + gp + rk[None, :], mask=am, other=0.0)
        b = tl.load(wp + rk[:, None], mask=mk[:, None] & mn[None, :],
                    other=0.0)
        acc, cor = _dot(a, b, acc, cor, FP32)
        if SPLIT:
            acc = tl.dot(tl.load(GL + gp + rk[None, :], mask=am,
                                 other=0.0), b, acc)
    tl.store(DXP + rp[:, None].to(tl.int64) * sx0 + rn[None, :], acc + cor,
             mask=ok[:, None] & mn[None, :])


@triton.jit
def _dx_kernel(DXP, DLH, DLL, RW, SLOT, DX, T, D, E, A, sx0, sr0, so0,
               OUT_BF: tl.constexpr, FP32: tl.constexpr, SPLIT: tl.constexpr,
               BT: tl.constexpr, BD: tl.constexpr, BE: tl.constexpr):
    """dx = sum of the token's slot gradients, plus dlogit @ router_w^T."""
    rt = tl.program_id(0) * BT + tl.arange(0, BT)
    rd = tl.program_id(1) * BD + tl.arange(0, BD)
    mt = rt < T
    md = rd < D
    m = mt[:, None] & md[None, :]
    acc = tl.zeros((BT, BD), dtype=tl.float32)
    cor = tl.zeros((BT, BD), dtype=tl.float32)
    for a in range(A):
        p = tl.load(SLOT + rt * A + a, mask=mt, other=0).to(tl.int64)
        acc += tl.load(DXP + p[:, None] * sx0 + rd[None, :], mask=m,
                       other=0.0)
    for e0 in range(0, E, BE):
        re = e0 + tl.arange(0, BE)
        dm = mt[:, None] & (re[None, :] < E)
        do = rt[:, None].to(tl.int64) * E + re[None, :]
        b = tl.load(RW + rd[None, :] * sr0 + re[:, None],
                    mask=(re[:, None] < E) & md[None, :], other=0.0)
        acc, cor = _dot(tl.load(DLH + do, mask=dm, other=0.0), b, acc,
                        cor, FP32)
        if SPLIT:
            acc = tl.dot(tl.load(DLL + do, mask=dm, other=0.0), b, acc)
    acc += cor
    o = DX + rt[:, None].to(tl.int64) * so0 + rd[None, :]
    if OUT_BF:
        tl.store(o, acc.to(tl.bfloat16), mask=m)
    else:
        tl.store(o, acc, mask=m)


@triton.jit
def _drw_kernel(X, DLH, DLL, DRW, T, D, E, sx0, SPLIT: tl.constexpr,
                FP32: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                BK: tl.constexpr, SK: tl.constexpr):
    """drouter_w = x^T @ dlogit, split over tokens into per-split buffers."""
    pid_k = tl.program_id(1)
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(2) * BN + tl.arange(0, BN)
    mm = rm < D
    mn = rn < E
    chunk = tl.cdiv(tl.cdiv(T, BK), SK) * BK
    k_hi = min(pid_k * chunk + chunk, T)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    cor = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(pid_k * chunk, k_hi, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < k_hi
        a = tl.load(X + rk[None, :].to(tl.int64) * sx0 + rm[:, None],
                    mask=mk[None, :] & mm[:, None], other=0.0)
        bm = mk[:, None] & mn[None, :]
        do = rk[:, None].to(tl.int64) * E + rn[None, :]
        b = tl.load(DLH + do, mask=bm, other=0.0)
        acc, cor = _dot(a, b, acc, cor, FP32)
        if SPLIT:
            acc = tl.dot(a, tl.load(DLL + do, mask=bm, other=0.0), acc)
    tl.store(DRW + pid_k.to(tl.int64) * D * E + rm[:, None] * E + rn[None, :],
             acc + cor, mask=mm[:, None] & mn[None, :])


@triton.jit
def _reduce_kernel(SRC, DST, N, NPART, OUT_BF: tl.constexpr,
                   BLK: tl.constexpr):
    """fixed-order sum of NPART identically shaped partial buffers."""
    r = tl.program_id(0) * BLK + tl.arange(0, BLK)
    m = r < N
    acc = tl.zeros((BLK,), dtype=tl.float32)
    for i in range(NPART):
        acc += tl.load(SRC + i.to(tl.int64) * N + r, mask=m, other=0.0)
    if OUT_BF:
        tl.store(DST + r, acc.to(tl.bfloat16), mask=m)
    else:
        tl.store(DST + r, acc, mask=m)


@triton.jit
def _wreduce_kernel(SRC, DST, N, SK: tl.constexpr, OUT_BF: tl.constexpr,
                    BLK: tl.constexpr):
    """sum an expert's SK row-slice partials, in slice order."""
    e = tl.program_id(1).to(tl.int64)
    r = tl.program_id(0) * BLK + tl.arange(0, BLK)
    m = r < N
    acc = tl.zeros((BLK,), dtype=tl.float32)
    for i in range(SK):
        acc += tl.load(SRC + (e * SK + i) * N + r, mask=m, other=0.0)
    if OUT_BF:
        tl.store(DST + e * N + r, acc.to(tl.bfloat16), mask=m)
    else:
        tl.store(DST + e * N + r, acc, mask=m)


# ---------------------------------------------------------------------------
# launch
# ---------------------------------------------------------------------------


_SMEM = 200 * 1024      # per-block operand budget, under the 227 KB limit


def _p2(n, lo=16, hi=None):
    """smallest power of two at least n, clamped into [lo, hi] when hi is set.

    Callers that pass hi tile the axis and so may clamp; callers that leave it
    open need a block that spans the axis in one program.
    """
    v = lo
    while v < n:
        v *= 2
    return v if hi is None else min(v, hi)


def _acc_budget(fp32):
    """accumulator tile area a program can hold in registers.

    The float32 path runs a second accumulator for the limb corrections, so
    its tiles have to be half the area of the bfloat16 path's.
    """
    return 16384 if fp32 else 32768


def _fit(bm, bn, bk, esize, stages, nb=1, na=1):
    """shrink (stages, bk) until the operand tiles fit the smem budget."""
    def need(k, st):
        return (na * bm * k + nb * k * bn) * esize * st
    while stages > 2 and need(bk, stages) > _SMEM:
        stages -= 1
    while bk > 16 and need(bk, stages) > _SMEM:
        bk //= 2
    while stages > 1 and need(bk, stages) > _SMEM:
        stages -= 1
    return bk, stages


def _row_tile(rows):
    """row tile of every slot-major GEMM, and the granule each run is padded to.

    `rows` is the mean number of dispatched rows an expert holds, so a layer
    whose experts are individually small does not pay a large row tile of
    mostly-masked lanes.  All four slot GEMMs share it, so one tile map serves
    them all.
    """
    return 128 if rows >= 128 else (64 if rows >= 48 else
                                    (32 if rows >= 20 else 16))


def _slot_cfg(rows, n, k, fp32, nb=1, na=1, wide=True, ncap=1024,
              warps=None, stages=None):
    """row/column/reduction tile, warps and stages for a slot-major GEMM.

    `ncap`, `warps` and `stages` let a kernel that carries more per-element
    state than a plain GEMM ask for a narrower, deeper-pipelined tile.
    """
    es = 4 if fp32 else 2
    bm = _row_tile(rows)
    bn = 256 if (wide and n >= 256 and bm >= 64) else (128 if n >= 128 else
                                                       (64 if n >= 64 else 32))
    bn = min(bn, ncap)
    while bm * bn > _acc_budget(fp32):
        bn //= 2
    bk = 64 if k >= 64 else 32
    nw = warps or (8 if bm * bn >= 16384 else 4)
    ns = stages or (4 if bm * bn <= 32768 else 3)
    bk, ns = _fit(bm, bn, bk, es, ns, nb, na)
    return bm, bn, bk, nw, ns


def _cfgw(m, n, rows, fp32):
    """tiles, warps, stages for the per-expert weight-gradient GEMMs.

    Both operands of these GEMMs stream from global memory once per tile with
    no reuse across the grid, so a short reduction tile with a deep pipeline
    beats a long one.
    """
    es = 4 if fp32 else 2
    bm = 128 if m >= 128 else (64 if m >= 64 else 32)
    bn = 128 if n >= 128 else (64 if n >= 64 else 32)
    while bm * bn > _acc_budget(fp32):
        bn //= 2
    bk = 32 if rows >= 32 else 16
    nw = 4
    bk, ns = _fit(bm, bn, bk, es, 5, 1 if fp32 else 2)
    return bm, bn, bk, nw, ns


_SMS = 132          # H100 streaming multiprocessors


def _splitk(tiles_per_expert, E, rows, bk):
    """row-slices per expert for a weight gradient, from the launch geometry.

    Few output tiles leave most of the device idle while each program walks a
    long row reduction; slicing that reduction fills the machine at the cost
    of one extra pass over the partials.
    """
    sk = 1
    while (sk < 8 and tiles_per_expert * E * sk < _SMS * 2
           and rows >= 2 * sk * bk):
        sk *= 2
    return sk


def _wbuf(shape, sk, odt, dev):
    """output and accumulation buffers for a split weight gradient."""
    out = torch.empty(shape, device=dev, dtype=odt)
    if sk == 1:
        return out, out
    return out, torch.empty((shape[0] * sk,) + shape[1:], device=dev,
                            dtype=torch.float32)


def _wsum(part, out, n, sk, E, out_bf, dev):
    if sk == 1:
        return
    _wreduce_kernel[(triton.cdiv(n, 1024), E)](
        part, out, n, SK=sk, OUT_BF=out_bf, BLK=1024, num_warps=4)


def _tilemap(off, P, E, bm, dev):
    """tile -> (expert, expert's first tile) for a given row tile size."""
    toff = torch.empty(E + 1, device=dev, dtype=torch.int32)
    _toff_kernel[(1,)](off, toff, E, bm, BE=_p2(E + 1), num_warps=4)
    nt = P // bm + E
    tm_e = torch.empty(nt, device=dev, dtype=torch.int32)
    tm_t = torch.empty(nt, device=dev, dtype=torch.int32)
    _tilemap_kernel[(triton.cdiv(nt, 128),)](
        toff, tm_e, tm_t, nt, E, max(E.bit_length(), 1), BLK=128, num_warps=4)
    return nt, tm_e, tm_t


class _MoE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, router_w, router_b, w1, w2, A):
        A = int(A)
        # the kernels index the last axis with unit stride and carry the
        # others explicitly
        x = x.contiguous()
        router_w = router_w.contiguous()
        router_b = router_b.contiguous()
        w1 = w1.contiguous()
        w2 = w2.contiguous()
        T, D = x.shape
        E = router_w.shape[1]
        F2 = w1.shape[2]
        F = F2 // 2
        P = T * A
        dev = x.device
        fp32 = x.dtype == torch.float32
        split = not fp32
        f32 = torch.float32
        i32 = torch.int32
        bf = torch.bfloat16
        rows = P / max(E, 1)

        s = torch.empty(T, E, device=dev, dtype=f32)
        key = torch.empty(T, E, device=dev, dtype=torch.int64)
        bt = 64 if T >= 64 else 16
        be = _p2(E, 16, 64)
        bk, ns = _fit(bt, be, 64, 4 if fp32 else 2, 3)
        _score_kernel[(triton.cdiv(T, bt), triton.cdiv(E, be))](
            x, router_w, router_b, s, key, T, D, E, x.stride(0),
            router_w.stride(0), FP32=fp32, BM=bt, BN=be, BK=bk,
            num_warps=4, num_stages=ns)

        sel = torch.topk(key, A, dim=-1, largest=True,
                         sorted=True).indices.to(i32)

        ba = _p2(A, 8)
        bta = max(4096 // ba, 1)
        w = torch.empty(T, A, device=dev, dtype=f32)
        _weight_kernel[(triton.cdiv(T, bta),)](
            s, sel, w, T, E, A, BT=bta, BA=ba, num_warps=4)

        # permitted routing call on the [T*A] metadata: a stable sort by
        # expert, so the slot order inside an expert is the flat (token, a)
        # order and never depends on scheduling.
        flat = sel.view(-1)
        order = torch.argsort(flat, stable=True)
        se = flat.index_select(0, order)

        off = torch.empty(E + 2, device=dev, dtype=i32)
        _offset_kernel[(1,)](se, off, P, E, max(P.bit_length(), 1),
                             BE=_p2(E + 2), num_warps=4)

        ptok = torch.empty(P, device=dev, dtype=i32)
        pw = torch.empty(P, device=dev, dtype=f32)
        slot = torch.empty(P, device=dev, dtype=i32)
        _dispatch_kernel[(triton.cdiv(P, 512),)](
            order, w, ptok, pw, slot, P, A, BLK=512, num_warps=4)

        gu = torch.empty(P, F2, device=dev, dtype=f32)
        acth = torch.empty(P, F, device=dev, dtype=bf if split else f32)
        actl = torch.empty(P, F, device=dev, dtype=bf) if split else acth

        bm = _row_tile(rows)
        nt, tm_e, tm_t = _tilemap(off, P, E, bm, dev)
        # the hidden-gradient kernel also carries the silu backward and the
        # score term, so it runs a narrower tile off its own map
        bma = min(bm, _DACT_BM)
        if bma == bm:
            nta, tma_e, tma_t = nt, tm_e, tm_t
        else:
            nta, tma_e, tma_t = _tilemap(off, P, E, bma, dev)

        bn1, bk1, nw1, ns1 = _slot_cfg(rows, F, D, fp32, nb=2, wide=False)[1:]
        nn1 = triton.cdiv(F, bn1)
        _gemm1_kernel[(nt * nn1,)](
            x, w1, gu, acth, actl, ptok, pw, tm_e, tm_t, off, nt, nn1, D, F,
            x.stride(0), w1.stride(0), w1.stride(1), F2, F,
            SPLIT_OUT=split, FP32=fp32, BM=bm, BN=bn1, BK=bk1, GN=_GN,
            num_warps=nw1, num_stages=ns1)

        yp = torch.empty(P, D, device=dev, dtype=f32)
        bn2, bk2, nw2, ns2 = _slot_cfg(rows, D, F, fp32,
                                       na=2 if split else 1)[1:]
        nn2 = triton.cdiv(D, bn2)
        _gemm2_kernel[(nt * nn2,)](
            acth, actl, w2, yp, tm_e, tm_t, off, nt, nn2, F, D, F,
            w2.stride(0), w2.stride(1), D, SPLIT=split, FP32=fp32, BM=bm,
            BN=bn2, BK=bk2, GN=_GN, num_warps=nw2, num_stages=ns2)

        y = torch.empty(T, D, device=dev, dtype=x.dtype)
        bd = _p2(D, 16, 256)
        _combine_kernel[(triton.cdiv(T, 16), triton.cdiv(D, bd))](
            yp, slot, y, T, D, A, D, y.stride(0), OUT_BF=not fp32, BT=16,
            BD=bd, num_warps=4)

        ctx.save_for_backward(x, router_w, w1, w2, s, sel, slot, ptok, pw,
                              off, acth, actl, gu, tm_e, tm_t, tma_e, tma_t)
        ctx.meta = (T, D, F, F2, E, A, P, nt, bm, nta, bma, rows, fp32,
                    x.dtype)
        return y

    @staticmethod
    def backward(ctx, dy):
        (x, router_w, w1, w2, s, sel, slot, ptok, pw, off, acth, actl, gu,
         tm_e, tm_t, tma_e, tma_t) = ctx.saved_tensors
        T, D, F, F2, E, A, P, nt, bm, nta, bma, rows, fp32, odt = ctx.meta
        dev = x.device
        split = not fp32
        f32 = torch.float32
        bf = torch.bfloat16
        dy = dy.contiguous()
        if dy.dtype != odt:
            dy = dy.to(odt)

        dgh = torch.empty(P, F2, device=dev, dtype=bf if split else f32)
        dgl = torch.empty(P, F2, device=dev, dtype=bf) if split else dgh
        bnA, bkA, nwA, nsA = _slot_cfg(rows, F, D, fp32, wide=False,
                                       ncap=_DACT_BN, warps=4, stages=5)[1:]
        nnA = triton.cdiv(F, bnA)
        pr = torch.empty(nnA, P, device=dev, dtype=f32)
        _dact_kernel[(nta * nnA,)](
            dy, w2, acth, actl, gu, ptok, pw, pr, dgh, dgl, tma_e, tma_t,
            off, nta, nnA, P, F, D, dy.stride(0), w2.stride(0), w2.stride(1),
            F, F2, SPLIT_OUT=split, FP32=fp32, BM=bma, BN=bnA, BK=bkA, GN=1,
            num_warps=nwA, num_stages=nsA)

        ba = _p2(A, 8)
        bta = max(4096 // ba, 1)
        dla = torch.empty(T, A, device=dev, dtype=f32)
        _dscore_kernel[(triton.cdiv(T, bta),)](
            pr, nnA, P, slot, sel, s, dla, T, E, A, BT=bta, BA=ba,
            num_warps=4)
        dlh = torch.empty(T, E, device=dev, dtype=bf if split else f32)
        dll = torch.empty(T, E, device=dev, dtype=bf) if split else dlh
        be = _p2(E, 16, 128)
        _dlspread_kernel[(triton.cdiv(T, 32), triton.cdiv(E, be))](
            dla, sel, dlh, dll, T, E, A, SPLIT_OUT=split, BT=32, BE=be,
            num_warps=4)

        m1, n1, k1, nw1, ns1 = _cfgw(D, F2, rows, fp32)
        g1 = triton.cdiv(D, m1) * triton.cdiv(F2, n1)
        sk1 = _splitk(g1, E, rows, k1)
        dw1, p1 = _wbuf(w1.shape, sk1, odt, dev)
        _dw1_kernel[(triton.cdiv(D, m1), triton.cdiv(F2, n1), E * sk1)](
            x, dgh, dgl, p1, ptok, off, D, F2, x.stride(0), F2,
            D * F2, F2, SPLIT=split, FP32=fp32, OUT_BF=split and sk1 == 1,
            BM=m1, BN=n1, BK=k1, SK=sk1, num_warps=nw1, num_stages=ns1)
        _wsum(p1, dw1, D * F2, sk1, E, split, dev)

        m2, n2, k2, nw2, ns2 = _cfgw(F, D, rows, fp32)
        g2 = triton.cdiv(F, m2) * triton.cdiv(D, n2)
        sk2 = _splitk(g2, E, rows, k2)
        dw2, p2 = _wbuf(w2.shape, sk2, odt, dev)
        _dw2_kernel[(triton.cdiv(F, m2), triton.cdiv(D, n2), E * sk2)](
            acth, actl, dy, p2, ptok, off, F, D, F, dy.stride(0),
            F * D, D, SPLIT=split, FP32=fp32, OUT_BF=split and sk2 == 1,
            BM=m2, BN=n2, BK=k2, SK=sk2, num_warps=nw2, num_stages=ns2)
        _wsum(p2, dw2, F * D, sk2, E, split, dev)

        dxp = torch.empty(P, D, device=dev, dtype=f32)
        bnX, bkX, nwX, nsX = _slot_cfg(rows, D, F2, fp32)[1:]
        nnX = triton.cdiv(D, bnX)
        _dxp_kernel[(nt * nnX,)](
            dgh, dgl, w1, dxp, tm_e, tm_t, off, nt, nnX, F2, D, F2,
            w1.stride(0), w1.stride(1), D, SPLIT=False, FP32=fp32, BM=bm,
            BN=bnX, BK=bkX, GN=_GN, num_warps=nwX, num_stages=nsX)

        dx = torch.empty(T, D, device=dev, dtype=odt)
        bd = _p2(D, 16, 128)
        _dx_kernel[(triton.cdiv(T, 32), triton.cdiv(D, bd))](
            dxp, dlh, dll, router_w, slot, dx, T, D, E, A, D,
            router_w.stride(0), dx.stride(0), OUT_BF=not fp32, FP32=fp32,
            SPLIT=split, BT=32, BD=bd, BE=_p2(E, 16, 64), num_warps=4)

        sk = 8 if T >= 2048 else (4 if T >= 512 else 1)
        be = _p2(E, 16, 64)
        drw_p = torch.empty(sk, D, E, device=dev, dtype=f32)
        bk, ns = _fit(64, be, 64, 4 if fp32 else 2, 3)
        _drw_kernel[(triton.cdiv(D, 64), sk, triton.cdiv(E, be))](
            x, dlh, dll, drw_p, T, D, E, x.stride(0), SPLIT=split, FP32=fp32,
            BM=64, BN=be, BK=bk, SK=sk, num_warps=4, num_stages=ns)
        drw = torch.empty(D, E, device=dev, dtype=odt)
        _reduce_kernel[(triton.cdiv(D * E, 512),)](
            drw_p, drw, D * E, sk, OUT_BF=not fp32, BLK=512, num_warps=4)
        return dx, drw, None, dw1, dw2, None


def fused_moe(
    x: torch.Tensor,         # [T, D] float32 or bfloat16
    router_w: torch.Tensor,  # [D, E] same dtype as x
    router_b: torch.Tensor,  # [E] float32, selection-only bias, no gradient
    w1: torch.Tensor,        # [E, D, 2F]  [:, :, :F] gate, [:, :, F:] up
    w2: torch.Tensor,        # [E, F, D]
    A: int,                  # experts per token
) -> torch.Tensor:           # [T, D] same dtype as x
    """Entry point: sigmoid-routed top-A SwiGLU mixture of experts."""
    return _MoE.apply(x, router_w, router_b, w1, w2, A)
