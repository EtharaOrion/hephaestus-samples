"""Routed fused mixture-of-experts layer, forward and backward.

Dispatch shape
--------------
The T tokens are expanded into N = T*A (token, expert) rows which are sorted
by expert, so every expert's rows form one contiguous span of the sorted row
space.  A tile map built from the span boundaries names, for each M-tile of
every row-space GEMM, the expert that tile belongs to and the row it starts
at, so one launch covers all experts and no tile straddles a boundary.
Row-space results are written back at the pre-sort slot index t*A + a, which
makes the per-token combine a contiguous read of A rows.

Weight placement
----------------
The combine weight w of a row is folded into the stored activation
(act = w * silu(gate) * up).  The second forward projection then needs no
epilogue scale, the cotangent enters both backward GEMMs unscaled in its
native dtype, and the gradient of the combine weight comes out of the
backward as q = w * dL/dw.  The score gradient is written in terms of q,

    dlogit_a = q_a (1 - s_a) - (sum_b q_b) s_a (1 - s_a) / sum_b s_b,

which is the exact chain rule with the 1/w and 1/s_a factors cancelled
algebraically, so a zero raw score never divides.

Precision
---------
Router scores are float32: tf32x3 on float32 inputs, exact bfloat16 products
on bfloat16 inputs, float32 accumulation in both cases.  Every GEMM
accumulates in float32.  One rule covers all operands everywhere: a value
that is float32 reaches the tensor cores as an exact bfloat16 hi/lo pair
(the lo*lo cross term lies below the float32 significand and is not formed),
and a value that is already bfloat16 is exact in a single limb.

The two float32 intermediates that later GEMMs re-read many times -- the
weighted activation and the hidden gradient -- are stored already split
into that hi/lo pair rather than as float32.  The bytes and the arithmetic
are the same either way; splitting at the single write site keeps the
conversion out of every consumer's inner loop and halves the shared memory
each staged operand tile needs.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# split-precision dots
# ---------------------------------------------------------------------------


@triton.jit
def _split(x):
    """Exact bfloat16 hi/lo pair of a float32 value: x == hi + lo + eps.

    The residual is below the float32 significand relative to hi, so a dot
    over (hi, lo) carries float32-effective operand precision.
    """
    v = x.to(tl.float32)
    hi = v.to(tl.bfloat16)
    return hi, (v - hi.to(tl.float32)).to(tl.bfloat16)


@triton.jit
def _dotp(ah, al, b, acc, SB: tl.constexpr):
    """acc += (ah + al)[M,K] @ b[K,N] for an operand already split."""
    if SB:
        bh, bl = _split(b)
        acc = tl.dot(ah, bh, acc)
        acc = tl.dot(al, bh, acc)
        acc = tl.dot(ah, bl, acc)
    else:
        bh = b.to(tl.bfloat16)
        acc = tl.dot(ah, bh, acc)
        acc = tl.dot(al, bh, acc)
    return acc


@triton.jit
def _dotp_bt(ah, al, b, acc, SB: tl.constexpr):
    """acc += (ah + al)[M,K] @ b[N,K]^T for an operand already split."""
    if SB:
        bh, bl = _split(b)
        bt = tl.trans(bh)
        acc = tl.dot(ah, bt, acc)
        acc = tl.dot(al, bt, acc)
        acc = tl.dot(ah, tl.trans(bl), acc)
    else:
        bt = tl.trans(b.to(tl.bfloat16))
        acc = tl.dot(ah, bt, acc)
        acc = tl.dot(al, bt, acc)
    return acc


@triton.jit
def _dotp_at(ah, al, b, acc, SB: tl.constexpr):
    """acc += (ah + al)[K,M]^T @ b[K,N] for an operand already split."""
    aht = tl.trans(ah)
    alt = tl.trans(al)
    if SB:
        bh, bl = _split(b)
        acc = tl.dot(aht, bh, acc)
        acc = tl.dot(alt, bh, acc)
        acc = tl.dot(aht, bl, acc)
    else:
        bh = b.to(tl.bfloat16)
        acc = tl.dot(aht, bh, acc)
        acc = tl.dot(alt, bh, acc)
    return acc


@triton.jit
def _dot2(a, b, acc, SA: tl.constexpr, SB: tl.constexpr):
    """acc += a[M,K] @ b[K,N], splitting whichever sides SA/SB mark."""
    if SA:
        ah, al = _split(a)
        acc = _dotp(ah, al, b, acc, SB)
    elif SB:
        bh, bl = _split(b)
        ah = a.to(tl.bfloat16)
        acc = tl.dot(ah, bh, acc)
        acc = tl.dot(ah, bl, acc)
    else:
        acc = tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16), acc)
    return acc


@triton.jit
def _dot2_bt(a, b, acc, SA: tl.constexpr, SB: tl.constexpr):
    """acc += a[M,K] @ b[N,K]^T, splitting whichever sides SA/SB mark."""
    if SA:
        ah, al = _split(a)
        acc = _dotp_bt(ah, al, b, acc, SB)
    elif SB:
        bh, bl = _split(b)
        ah = a.to(tl.bfloat16)
        acc = tl.dot(ah, tl.trans(bh), acc)
        acc = tl.dot(ah, tl.trans(bl), acc)
    else:
        acc = tl.dot(a.to(tl.bfloat16), tl.trans(b.to(tl.bfloat16)), acc)
    return acc


@triton.jit
def _dot2_at(a, b, acc, SA: tl.constexpr, SB: tl.constexpr):
    """acc += a[K,M]^T @ b[K,N], splitting whichever sides SA/SB mark."""
    if SA:
        ah, al = _split(a)
        acc = _dotp_at(ah, al, b, acc, SB)
    elif SB:
        bh, bl = _split(b)
        aht = tl.trans(a.to(tl.bfloat16))
        acc = tl.dot(aht, bh, acc)
        acc = tl.dot(aht, bl, acc)
    else:
        acc = tl.dot(tl.trans(a.to(tl.bfloat16)), b.to(tl.bfloat16), acc)
    return acc


# ---------------------------------------------------------------------------
# routing
# ---------------------------------------------------------------------------


@triton.jit
def _router_kernel(X, RW, RB, SRAW, KEY, T, D, E,
                   sxm, sxk, srk, sre,
                   BT: tl.constexpr, BE: tl.constexpr, BK: tl.constexpr,
                   SHIFT: tl.constexpr, F32: tl.constexpr):
    """s = sigmoid(x @ router_w) in float32, and the packed selection key.

    KEY[t, e] = (order-preserving bits of s[t,e] + router_b[e]) << SHIFT
                | (E - 1 - e)
    with SHIFT wide enough to hold E-1, so the score field and the index
    field never overlap and every key is distinct.  Descending key order is
    therefore exactly (biased score descending, expert index ascending).
    -0.0 is folded onto +0.0 before the bit pattern is taken.
    """
    pid_t = tl.program_id(0).to(tl.int64)
    pid_e = tl.program_id(1).to(tl.int64)
    rt = pid_t * BT + tl.arange(0, BT).to(tl.int64)
    re = pid_e * BE + tl.arange(0, BE).to(tl.int64)
    mt = rt < T
    me = re < E
    acc = tl.zeros((BT, BE), dtype=tl.float32)
    for k0 in range(0, D, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < D
        a = tl.load(X + rt[:, None] * sxm + rk[None, :] * sxk,
                    mask=mt[:, None] & mk[None, :], other=0.0)
        b = tl.load(RW + rk[:, None] * srk + re[None, :] * sre,
                    mask=mk[:, None] & me[None, :], other=0.0)
        if F32:
            acc = tl.dot(a, b, acc, input_precision="tf32x3")
        else:
            acc = tl.dot(a, b, acc)
    s = tl.sigmoid(acc)
    tl.store(SRAW + rt[:, None] * E + re[None, :], s,
             mask=mt[:, None] & me[None, :])
    rb = tl.load(RB + re, mask=me, other=0.0)
    bs = s + rb[None, :]
    bs = tl.where(bs == 0.0, 0.0, bs)
    ib = bs.to(tl.int32, bitcast=True).to(tl.int64)
    u = tl.where(ib >= 0, ib ^ 0x80000000, (~ib) & 0xFFFFFFFF)
    key = (u << SHIFT) | (E - 1 - re)[None, :]
    tl.store(KEY + rt[:, None] * E + re[None, :], key,
             mask=mt[:, None] & me[None, :])


@triton.jit
def _weights_kernel(SRAW, SEL, W, T, E, A, BT: tl.constexpr,
                    BA: tl.constexpr):
    """w[t, a] = s[t, sel[t, a]] / sum_a s[t, sel[t, a]], raw scores only."""
    pid = tl.program_id(0).to(tl.int64)
    rt = pid * BT + tl.arange(0, BT).to(tl.int64)
    ra = tl.arange(0, BA)
    mt = rt < T
    ma = ra < A
    m = mt[:, None] & ma[None, :]
    sel = tl.load(SEL + rt[:, None] * A + ra[None, :], mask=m, other=0)
    s = tl.load(SRAW + rt[:, None] * E + sel, mask=m, other=0.0)
    w = s / tl.sum(tl.where(m, s, 0.0), axis=1)[:, None]
    tl.store(W + rt[:, None] * A + ra[None, :], w, mask=m)


@triton.jit
def _offs_kernel(SE, OFFS, N, E):
    """OFFS[e] = first sorted row whose expert id is >= e; OFFS[E] = N."""
    e = tl.program_id(0)
    lo = 0
    hi = N
    while lo < hi:
        mid = (lo + hi) // 2
        v = tl.load(SE + mid)
        if v < e:
            lo = mid + 1
        else:
            hi = mid
    tl.store(OFFS + e, lo)


@triton.jit
def _meta_kernel(ORDER, WFLAT, POS, TOK, WSORT, N, A, BLOCK: tl.constexpr):
    """Per sorted row: its pre-sort slot, its token and its combine weight."""
    o = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    m = o < N
    p = tl.load(ORDER + o, mask=m, other=0)
    tl.store(POS + o, p.to(tl.int32), mask=m)
    tl.store(TOK + o, (p // A).to(tl.int32), mask=m)
    tl.store(WSORT + o, tl.load(WFLAT + p, mask=m, other=0.0), mask=m)


@triton.jit
def _tilemap_kernel(OFFS, TILE_E, TILE_R, E, NTMAX,
                    BM: tl.constexpr, BE: tl.constexpr):
    """Flatten the per-expert row spans into a list of (expert, first row).

    Program e places its own ceil(count_e / BM) tiles at the exclusive
    prefix of the tile counts of the experts before it, so the map is a
    pure function of the span boundaries.  Slots past the last real tile
    get expert -1 and the GEMMs launched over the upper bound exit on it.
    """
    e = tl.program_id(0)
    ri = tl.arange(0, BE)
    prev = tl.load(OFFS + ri, mask=ri < e, other=0)
    nxt = tl.load(OFFS + ri + 1, mask=ri < e, other=0)
    base = tl.sum(tl.cdiv(nxt - prev, BM))
    lo = tl.load(OFFS + e)
    hi = tl.load(OFFS + e + 1)
    n = tl.cdiv(hi - lo, BM)
    for i0 in range(0, n, 64):
        i = i0 + tl.arange(0, 64)
        mi = i < n
        tl.store(TILE_E + base + i, e, mask=mi)
        tl.store(TILE_R + base + i, lo + i * BM, mask=mi)
    if e == E - 1:
        tot = base + n
        for j0 in range(tot, NTMAX, 64):
            j = j0 + tl.arange(0, 64)
            tl.store(TILE_E + j, -1, mask=j < NTMAX)


# ---------------------------------------------------------------------------
# forward
# ---------------------------------------------------------------------------


@triton.jit
def _gemm1_kernel(X, W1, TOK, WSORT, TILE_E, TILE_R, OFFS, AH, AL, PG, PU,
                  D, F, sxm, sxk, sw1e, sw1k, sw1n,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                  SX: tl.constexpr, SW: tl.constexpr):
    """h = x[tok] @ w1[e], with the whole SwiGLU epilogue folded in.

    One program owns a gate column block and the matching up column block,
    so silu(gate) * up is formed while both are still in registers.  The
    weighted activation leaves as its exact bfloat16 hi/lo pair (AH, AL) --
    the same decomposition its consumers would build, done once here at the
    single write site instead of on each of their many re-reads.  PG and PU
    are the two SwiGLU partials, so the backward reloads neither gate nor
    up and evaluates no sigmoid.
    """
    t = tl.program_id(0)
    e = tl.load(TILE_E + t)
    if e < 0:
        return
    pid_n = tl.program_id(1)
    r0 = tl.load(TILE_R + t)
    hi = tl.load(OFFS + e + 1)
    rm = r0 + tl.arange(0, BM)
    mm = rm < hi
    tok = tl.load(TOK + rm, mask=mm, other=0).to(tl.int64)
    rn = pid_n * BN + tl.arange(0, BN)
    mn = rn < F
    accg = tl.zeros((BM, BN), dtype=tl.float32)
    accu = tl.zeros((BM, BN), dtype=tl.float32)
    xp = X + tok[:, None] * sxm
    wg = W1 + e.to(tl.int64) * sw1e + rn[None, :] * sw1n
    wu = wg + F * sw1n
    for k0 in range(0, D, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < D
        mb = mk[:, None] & mn[None, :]
        a = tl.load(xp + rk[None, :] * sxk, mask=mm[:, None] & mk[None, :],
                    other=0.0)
        bg = tl.load(wg + rk[:, None] * sw1k, mask=mb, other=0.0)
        bu = tl.load(wu + rk[:, None] * sw1k, mask=mb, other=0.0)
        if SX:
            ah, al = _split(a.to(tl.float32))
            accg = _dotp(ah, al, bg, accg, SW)
            accu = _dotp(ah, al, bu, accu, SW)
        else:
            accg = _dot2(a, bg, accg, False, SW)
            accu = _dot2(a, bu, accu, False, SW)
    m = mm[:, None] & mn[None, :]
    rm64 = rm[:, None].to(tl.int64)
    off = rm64 * F + rn[None, :]
    w = tl.load(WSORT + rm, mask=mm, other=0.0)
    sg = tl.sigmoid(accg)
    silu = accg * sg
    act = w[:, None] * silu * accu
    hi_, lo_ = _split(act)
    tl.store(AH + off, hi_, mask=m)
    tl.store(AL + off, lo_, mask=m)
    tl.store(PG + off, w[:, None] * accu * (sg + silu * (1.0 - sg)), mask=m)
    tl.store(PU + off, w[:, None] * silu, mask=m)


@triton.jit
def _gemm2_kernel(AH, AL, W2, TILE_E, TILE_R, OFFS, POS, YP,
                  F, D, sw2e, sw2k, sw2n,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                  SW: tl.constexpr):
    """yp[pos[r], :] = actw[r, :] @ w2[e(r)], already carrying the weight."""
    t = tl.program_id(0)
    e = tl.load(TILE_E + t)
    if e < 0:
        return
    pid_n = tl.program_id(1)
    r0 = tl.load(TILE_R + t)
    hi = tl.load(OFFS + e + 1)
    rm = r0 + tl.arange(0, BM)
    mm = rm < hi
    rn = pid_n * BN + tl.arange(0, BN)
    mn = rn < D
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    ap = rm[:, None].to(tl.int64) * F
    wp = W2 + e.to(tl.int64) * sw2e + rn[None, :] * sw2n
    for k0 in range(0, F, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < F
        ma = mm[:, None] & mk[None, :]
        ah = tl.load(AH + ap + rk[None, :], mask=ma, other=0.0)
        al = tl.load(AL + ap + rk[None, :], mask=ma, other=0.0)
        b = tl.load(wp + rk[:, None] * sw2k, mask=mk[:, None] & mn[None, :],
                    other=0.0)
        acc = _dotp(ah, al, b, acc, SW)
    p = tl.load(POS + rm, mask=mm, other=0).to(tl.int64)
    tl.store(YP + p[:, None] * D + rn[None, :], acc,
             mask=mm[:, None] & mn[None, :])


@triton.jit
def _combine_kernel(YP, Y, T, D, A, OD: tl.constexpr, BT: tl.constexpr,
                    BD: tl.constexpr):
    """y[t] = sum over the A slots of token t, added in slot order."""
    rt = tl.program_id(0).to(tl.int64) * BT + tl.arange(0, BT).to(tl.int64)
    rd = tl.program_id(1) * BD + tl.arange(0, BD)
    mt = rt < T
    md = rd < D
    m = mt[:, None] & md[None, :]
    acc = tl.zeros((BT, BD), dtype=tl.float32)
    base = YP + rt[:, None] * A * D + rd[None, :]
    for a in range(A):
        acc += tl.load(base + a * D, mask=m, other=0.0)
    if OD == 0:
        tl.store(Y + rt[:, None] * D + rd[None, :], acc, mask=m)
    else:
        tl.store(Y + rt[:, None] * D + rd[None, :], acc.to(tl.bfloat16),
                 mask=m)


# ---------------------------------------------------------------------------
# backward
# ---------------------------------------------------------------------------


@triton.jit
def _dact_kernel(DY, W2, AH, AL, PG, PU, TOK, POS, TILE_E, TILE_R, OFFS,
                 GH, GL, QP, N, F, D, F2, sdym, sdyn, sw2e, sw2k, sw2n,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                 SD: tl.constexpr, SW: tl.constexpr):
    """dact = dy[tok] @ w2[e]^T, the SwiGLU backward, and the score dot.

    The cotangent enters unscaled in its native dtype; the combine weight
    already sits in PG and PU.  The same tile contracts the accumulator
    against the weighted activation, which yields the per-row partial of
    w * dL/dw without ever forming the expert output.  The hidden gradient
    leaves as its exact bfloat16 hi/lo pair for the two GEMMs that read it.
    """
    t = tl.program_id(0)
    e = tl.load(TILE_E + t)
    if e < 0:
        return
    pid_n = tl.program_id(1)
    r0 = tl.load(TILE_R + t)
    hi = tl.load(OFFS + e + 1)
    rm = r0 + tl.arange(0, BM)
    mm = rm < hi
    rn = pid_n * BN + tl.arange(0, BN)
    mn = rn < F
    tok = tl.load(TOK + rm, mask=mm, other=0).to(tl.int64)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    gp = DY + tok[:, None] * sdym
    wp = W2 + e.to(tl.int64) * sw2e + rn[:, None] * sw2k
    for k0 in range(0, D, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < D
        g = tl.load(gp + rk[None, :] * sdyn, mask=mm[:, None] & mk[None, :],
                    other=0.0)
        b = tl.load(wp + rk[None, :] * sw2n, mask=mn[:, None] & mk[None, :],
                    other=0.0)
        acc = _dot2_bt(g, b, acc, SD, SW)
    rm64 = rm[:, None].to(tl.int64)
    m = mm[:, None] & mn[None, :]
    off = rm64 * F + rn[None, :]
    a = tl.load(AH + off, mask=m, other=0.0).to(tl.float32) + \
        tl.load(AL + off, mask=m, other=0.0).to(tl.float32)
    q = tl.sum(a * acc, axis=1)
    p = tl.load(POS + rm, mask=mm, other=0).to(tl.int64)
    tl.store(QP + pid_n.to(tl.int64) * N + p, q, mask=mm)
    dg = acc * tl.load(PG + off, mask=m, other=0.0)
    du = acc * tl.load(PU + off, mask=m, other=0.0)
    gh, gl = _split(dg)
    uh, ul = _split(du)
    hp = rm64 * F2 + rn[None, :]
    tl.store(GH + hp, gh, mask=m)
    tl.store(GL + hp, gl, mask=m)
    tl.store(GH + hp + F, uh, mask=m)
    tl.store(GL + hp + F, ul, mask=m)


@triton.jit
def _dscore_kernel(QP, SRAW, SEL, DS, N, T, E, A, NP,
                   BA: tl.constexpr):
    """Reduce the score-gradient partials and scatter one dense [T, E] row.

    q_a is w_a * dL/dw_a, so the combine-weight normalisation collapses to
    dlogit_a = q_a (1 - s_a) - (sum q) s_a (1 - s_a) / sum s with no
    division by a combine weight or by an individual score.
    """
    t = tl.program_id(0).to(tl.int64)
    ra = tl.arange(0, BA)
    ma = ra < A
    p = t * A + ra
    acc = tl.zeros((BA,), dtype=tl.float32)
    for i in range(NP):
        acc += tl.load(QP + i * N + p, mask=ma, other=0.0)
    sel = tl.load(SEL + p, mask=ma, other=0).to(tl.int64)
    s = tl.load(SRAW + t * E + sel, mask=ma, other=0.0)
    ssum = tl.sum(tl.where(ma, s, 0.0))
    qsum = tl.sum(tl.where(ma, acc, 0.0))
    ds = acc * (1.0 - s) - qsum * (s * (1.0 - s)) / ssum
    tl.store(DS + t * E + sel, ds, mask=ma)


@triton.jit
def _zero_kernel(P, n, BLOCK: tl.constexpr):
    o = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    tl.store(P + o, 0.0, mask=o < n)


@triton.jit
def _dxs_kernel(GH, GL, W1, TILE_E, TILE_R, OFFS, POS, DXP,
                D, F2, sw1e, sw1k, sw1n,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                SW: tl.constexpr):
    """dxp[pos[r], :] = dh[r, :] @ w1[e(r)]^T."""
    t = tl.program_id(0)
    e = tl.load(TILE_E + t)
    if e < 0:
        return
    pid_n = tl.program_id(1)
    r0 = tl.load(TILE_R + t)
    hi = tl.load(OFFS + e + 1)
    rm = r0 + tl.arange(0, BM)
    mm = rm < hi
    rn = pid_n * BN + tl.arange(0, BN)
    mn = rn < D
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    dp = rm[:, None].to(tl.int64) * F2
    wp = W1 + e.to(tl.int64) * sw1e + rn[:, None] * sw1k
    for k0 in range(0, F2, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < F2
        mg = mm[:, None] & mk[None, :]
        gh = tl.load(GH + dp + rk[None, :], mask=mg, other=0.0)
        gl = tl.load(GL + dp + rk[None, :], mask=mg, other=0.0)
        b = tl.load(wp + rk[None, :] * sw1n, mask=mn[:, None] & mk[None, :],
                    other=0.0)
        acc = _dotp_bt(gh, gl, b, acc, SW)
    p = tl.load(POS + rm, mask=mm, other=0).to(tl.int64)
    tl.store(DXP + p[:, None] * D + rn[None, :], acc,
             mask=mm[:, None] & mn[None, :])


@triton.jit
def _dxcomb_kernel(DXP, DS, RW, DX, T, D, E, A, srk, sre,
                   BT: tl.constexpr, BD: tl.constexpr, BE: tl.constexpr,
                   OD: tl.constexpr, SW: tl.constexpr):
    """dx[t] = sum_a dxp[t*A + a] + ds[t] @ router_w^T."""
    pid_t = tl.program_id(0)
    rt = pid_t * BT + tl.arange(0, BT)
    rd = tl.program_id(1) * BD + tl.arange(0, BD)
    mt = rt < T
    md = rd < D
    m = mt[:, None] & md[None, :]
    acc = tl.zeros((BT, BD), dtype=tl.float32)
    base = DXP + rt[:, None].to(tl.int64) * A * D + rd[None, :]
    for a in range(A):
        acc += tl.load(base + a * D, mask=m, other=0.0)
    for k0 in range(0, E, BE):
        rk = k0 + tl.arange(0, BE)
        mk = rk < E
        g = tl.load(DS + rt[:, None] * E + rk[None, :],
                    mask=mt[:, None] & mk[None, :], other=0.0)
        b = tl.load(RW + rd[:, None] * srk + rk[None, :] * sre,
                    mask=md[:, None] & mk[None, :], other=0.0)
        acc = _dot2_bt(g, b, acc, True, SW)
    if OD == 0:
        tl.store(DX + rt[:, None] * D + rd[None, :], acc, mask=m)
    else:
        tl.store(DX + rt[:, None] * D + rd[None, :], acc.to(tl.bfloat16),
                 mask=m)


@triton.jit
def _dw1_kernel(X, GH, GL, TOK, OFFS, DW1, D, F2, sxm, sxk,
                sw1e, sw1k, sw1n,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                OD: tl.constexpr, SX: tl.constexpr):
    """dw1[e] = x[tok]^T @ dh over expert e's rows; one program per tile.

    The whole row span is reduced by the one program that owns the tile, so
    the sum runs in row order with no split-K partials to merge.
    """
    pid = tl.program_id(0)
    e = tl.program_id(1).to(tl.int64)
    nm = tl.cdiv(D, BM)
    rm = (pid % nm) * BM + tl.arange(0, BM)
    rn = (pid // nm) * BN + tl.arange(0, BN)
    mm = rm < D
    mn = rn < F2
    lo = tl.load(OFFS + e)
    hi = tl.load(OFFS + e + 1)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(lo, hi, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < hi
        tok = tl.load(TOK + rk, mask=mk, other=0).to(tl.int64)
        a = tl.load(X + tok[:, None] * sxm + rm[None, :] * sxk,
                    mask=mk[:, None] & mm[None, :], other=0.0)
        gp = rk[:, None].to(tl.int64) * F2 + rn[None, :]
        mg = mk[:, None] & mn[None, :]
        gh = tl.load(GH + gp, mask=mg, other=0.0)
        gl = tl.load(GL + gp, mask=mg, other=0.0)
        if SX:
            ah, al = _split(a)
            aht = tl.trans(ah)
            acc = tl.dot(aht, gh, acc)
            acc = tl.dot(tl.trans(al), gh, acc)
            acc = tl.dot(aht, gl, acc)
        else:
            at = tl.trans(a.to(tl.bfloat16))
            acc = tl.dot(at, gh, acc)
            acc = tl.dot(at, gl, acc)
    p = DW1 + e * sw1e + rm[:, None] * sw1k + rn[None, :] * sw1n
    m = mm[:, None] & mn[None, :]
    if OD == 0:
        tl.store(p, acc, mask=m)
    else:
        tl.store(p, acc.to(tl.bfloat16), mask=m)


@triton.jit
def _dw2_kernel(AH, AL, DY, TOK, OFFS, DW2, F, D, sdym, sdyn,
                sw2e, sw2k, sw2n,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                OD: tl.constexpr, SD: tl.constexpr):
    """dw2[e] = actw^T @ dy[tok] over expert e's rows; one program per tile."""
    pid = tl.program_id(0)
    e = tl.program_id(1).to(tl.int64)
    nm = tl.cdiv(F, BM)
    rm = (pid % nm) * BM + tl.arange(0, BM)
    rn = (pid // nm) * BN + tl.arange(0, BN)
    mm = rm < F
    mn = rn < D
    lo = tl.load(OFFS + e)
    hi = tl.load(OFFS + e + 1)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(lo, hi, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < hi
        tok = tl.load(TOK + rk, mask=mk, other=0).to(tl.int64)
        ap = rk[:, None].to(tl.int64) * F + rm[None, :]
        ma = mk[:, None] & mm[None, :]
        ah = tl.load(AH + ap, mask=ma, other=0.0)
        al = tl.load(AL + ap, mask=ma, other=0.0)
        g = tl.load(DY + tok[:, None] * sdym + rn[None, :] * sdyn,
                    mask=mk[:, None] & mn[None, :], other=0.0)
        acc = _dotp_at(ah, al, g, acc, SD)
    p = DW2 + e * sw2e + rm[:, None] * sw2k + rn[None, :] * sw2n
    m = mm[:, None] & mn[None, :]
    if OD == 0:
        tl.store(p, acc, mask=m)
    else:
        tl.store(p, acc.to(tl.bfloat16), mask=m)


@triton.jit
def _drw_kernel(X, DS, DRW, T, D, E, sxm, sxk, srk, sre,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                OD: tl.constexpr, SX: tl.constexpr):
    """drw = x^T @ ds, reduced over all tokens by one program per tile."""
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    mm = rm < D
    mn = rn < E
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, T, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < T
        a = tl.load(X + rk[:, None] * sxm + rm[None, :] * sxk,
                    mask=mk[:, None] & mm[None, :], other=0.0)
        g = tl.load(DS + rk[:, None] * E + rn[None, :],
                    mask=mk[:, None] & mn[None, :], other=0.0)
        acc = _dot2_at(a, g, acc, SX, True)
    p = DRW + rm[:, None] * srk + rn[None, :] * sre
    m = mm[:, None] & mn[None, :]
    if OD == 0:
        tl.store(p, acc, mask=m)
    else:
        tl.store(p, acc.to(tl.bfloat16), mask=m)


# ---------------------------------------------------------------------------
# host side
# ---------------------------------------------------------------------------


def _p2(n):
    return 1 << max(0, (n - 1).bit_length())


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _tile(dim, want, floor=16):
    return _clamp(min(want, _p2(dim)), floor, want)


# Per-role tile shapes as (BM, BN, BK, num_warps, num_stages), bfloat16 row
# first and float32 row second.
_TILES = {
    "gemm1":       ((64, 128, 64, 8, 5), (64, 128, 32, 4, 3)),
    "gemm2":       ((128, 256, 64, 8, 3), (128, 128, 32, 4, 3)),
    "dact":        ((64, 128, 64, 4, 4), (64, 128, 32, 4, 5)),
    "dxs":         ((128, 256, 64, 8, 3), (128, 256, 32, 8, 4)),
    "dw1":         ((128, 128, 32, 4, 5), (64, 256, 32, 4, 5)),
    "dw2":         ((128, 128, 32, 4, 5), (128, 128, 32, 8, 5)),
    "drw":         ((128, 64, 64, 8, 3), (64, 64, 128, 4, 3)),
}

# H100 per-block shared-memory limit, less a margin for the epilogue's
# reduction scratch.
_SMEM = 227 * 1024


def _cfg(role, m, n, k, f32, planes):
    """Clamp a role's tile shape to this GEMM's extents and to shared memory.

    `planes` lists the loaded operand tiles as (on_m, on_n, itemsize), where
    on_m/on_n say which of BM/BN the tile spans besides BK.  A tile that had
    to shrink gives warps back rather than leaving each warp a sliver of the
    accumulator, and the pipeline then loses stages, and finally K, until
    the operand buffers fit: an fp32 operand split into two bfloat16 limb
    planes costs the same bytes, but the pipeliner stages each plane
    separately, which a single fp32 plane does not.
    """
    bm0, bn0, bk, w, s = _TILES[role][1 if f32 else 0]
    bm = _tile(m, bm0)
    bn = _tile(n, bn0)
    bk = _tile(k, bk)
    while w > 1 and bm * bn * 4 <= bm0 * bn0:
        w //= 2
        bm0 //= 2

    def per_k():
        return sum((bm if pm else 1) * (bn if pn else 1) * sz
                   for pm, pn, sz in planes)

    while s > 2 and per_k() * bk * s > _SMEM:
        s -= 1
    while bk > 16 and per_k() * bk * s > _SMEM:
        bk //= 2
    return bm, bn, bk, w, s


def _empty(*shape, dtype=torch.float32, device=None):
    return torch.empty(shape, dtype=dtype, device=device)


class _TileMaps:
    """Tile maps for the row-space GEMMs, one per distinct BM in a call.

    Built fresh from this call's span boundaries; the dict only keeps the
    two or three BM values the launches in flight ask for.
    """

    def __init__(self, offs, E, N, dev):
        self.offs, self.E, self.N, self.dev = offs, E, N, dev
        self.maps = {}

    def get(self, bm):
        m = self.maps.get(bm)
        if m is None:
            ntmax = triton.cdiv(self.N, bm) + self.E
            te = _empty(ntmax, dtype=torch.int32, device=self.dev)
            tr = _empty(ntmax, dtype=torch.int32, device=self.dev)
            _tilemap_kernel[(self.E,)](self.offs, te, tr, self.E, ntmax,
                                       BM=bm, BE=_p2(self.E), num_warps=4)
            m = (te, tr, ntmax)
            self.maps[bm] = m
        return m


class _MoE(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, router_w, router_b, w1, w2, A):
        T, D = x.shape
        E = router_w.shape[1]
        F2 = w1.shape[2]
        F = F2 // 2
        A = int(A)
        dev = x.device
        sx = x.dtype == torch.float32
        sw1 = w1.dtype == torch.float32
        sw2 = w2.dtype == torch.float32

        # ---- router scores and selection keys -----------------------------
        sraw = _empty(T, E, device=dev)
        keys = _empty(T, E, dtype=torch.int64, device=dev)
        be = _tile(E, 64, floor=16)
        bt = _tile(T, 64, floor=16)
        bk = _tile(D, 64, floor=16)
        shift = max((E - 1).bit_length(), 1)
        _router_kernel[(triton.cdiv(T, bt), triton.cdiv(E, be))](
            x, router_w, router_b, sraw, keys, T, D, E,
            x.stride(0), x.stride(1), router_w.stride(0), router_w.stride(1),
            BT=bt, BE=be, BK=bk, SHIFT=shift, F32=sx,
            num_warps=4, num_stages=3)

        sel = torch.topk(keys, A, dim=-1, largest=True, sorted=True).indices
        sorted_e, order = torch.sort(sel.reshape(-1), stable=True)

        N = T * A
        offs = _empty(E + 1, dtype=torch.int32, device=dev)
        _offs_kernel[(E + 1,)](sorted_e, offs, N, E, num_warps=1)

        w = _empty(T, A, device=dev)
        ba = _p2(A)
        bt = _clamp(4096 // ba, 1, 64)
        _weights_kernel[(triton.cdiv(T, bt),)](
            sraw, sel, w, T, E, A, BT=bt, BA=ba, num_warps=4)

        pos = _empty(N, dtype=torch.int32, device=dev)
        tok = _empty(N, dtype=torch.int32, device=dev)
        wsort = _empty(N, device=dev)
        _meta_kernel[(triton.cdiv(N, 256),)](
            order, w, pos, tok, wsort, N, A, BLOCK=256, num_warps=4)

        # ---- expert compute ------------------------------------------------
        tm = _TileMaps(offs, E, N, dev)
        bf = torch.bfloat16

        ah = _empty(N, F, dtype=bf, device=dev)
        al = _empty(N, F, dtype=bf, device=dev)
        pg = _empty(N, F, device=dev)
        pu = _empty(N, F, device=dev)
        # x tile (split when float32), gate and up tiles of w1
        bm1, bn1, bk1, w1w, w1s = _cfg(
            "gemm1", N, F, D, sx or sw1,
            [(1, 0, 4 if sx else 2)] + [(0, 1, 4 if sw1 else 2)] * 2)
        te1, tr1, nt1 = tm.get(bm1)
        _gemm1_kernel[(nt1, triton.cdiv(F, bn1))](
            x, w1, tok, wsort, te1, tr1, offs, ah, al, pg, pu, D, F,
            x.stride(0), x.stride(1), w1.stride(0), w1.stride(1), w1.stride(2),
            BM=bm1, BN=bn1, BK=bk1, SX=sx, SW=sw1,
            num_warps=w1w, num_stages=w1s)

        yp = _empty(N, D, device=dev)
        # two bfloat16 activation limb planes, one w2 tile
        bm2, bn2, bk2, w2w, w2s = _cfg(
            "gemm2", N, D, F, sw2,
            [(1, 0, 2), (1, 0, 2), (0, 1, 4 if sw2 else 2)])
        te2, tr2, nt2 = tm.get(bm2)
        _gemm2_kernel[(nt2, triton.cdiv(D, bn2))](
            ah, al, w2, te2, tr2, offs, pos, yp, F, D,
            w2.stride(0), w2.stride(1), w2.stride(2),
            BM=bm2, BN=bn2, BK=bk2, SW=sw2,
            num_warps=w2w, num_stages=w2s)

        y = _empty(T, D, dtype=x.dtype, device=dev)
        bd = _tile(D, 128, floor=16)
        btc = _clamp(2048 // bd, 1, 32)
        _combine_kernel[(triton.cdiv(T, btc), triton.cdiv(D, bd))](
            yp, y, T, D, A, OD=0 if sx else 1, BT=btc, BD=bd, num_warps=4)

        ctx.save_for_backward(x, router_w, w1, w2, sraw, sel, pos, tok,
                              offs, ah, al, pg, pu)
        ctx.meta = (T, D, E, F, F2, A, N, sx, sw1, sw2)
        return y

    @staticmethod
    def backward(ctx, dy):
        (x, router_w, w1, w2, sraw, sel, pos, tok, offs,
         ah, al, pg, pu) = ctx.saved_tensors
        T, D, E, F, F2, A, N, sx, sw1, sw2 = ctx.meta
        dev = x.device
        dy = dy.contiguous()
        sd = dy.dtype == torch.float32

        tm = _TileMaps(offs, E, N, dev)
        bf = torch.bfloat16

        gh = _empty(N, F2, dtype=bf, device=dev)
        gl = _empty(N, F2, dtype=bf, device=dev)
        # cotangent tile, w2 tile
        bma, bna, bka, waw, was = _cfg(
            "dact", N, F, D, sd or sw2,
            [(1, 0, 4 if sd else 2), (0, 1, 4 if sw2 else 2)])
        tea, tra, nta = tm.get(bma)
        npn = triton.cdiv(F, bna)
        qp = _empty(npn, N, device=dev)
        _dact_kernel[(nta, npn)](
            dy, w2, ah, al, pg, pu, tok, pos, tea, tra, offs, gh, gl, qp,
            N, F, D, F2, dy.stride(0), dy.stride(1),
            w2.stride(0), w2.stride(1), w2.stride(2),
            BM=bma, BN=bna, BK=bka, SD=sd, SW=sw2,
            num_warps=waw, num_stages=was)

        ds = _empty(T, E, device=dev)
        _zero_kernel[(triton.cdiv(T * E, 1024),)](ds, T * E, BLOCK=1024,
                                                  num_warps=4)
        _dscore_kernel[(T,)](qp, sraw, sel, ds, N, T, E, A, npn,
                             BA=_p2(A), num_warps=4)

        dxp = _empty(N, D, device=dev)
        # two bfloat16 hidden-gradient limb planes, one w1 tile
        bmx, bnx, bkx, wxw, wxs = _cfg(
            "dxs", N, D, F2, sw1,
            [(1, 0, 2), (1, 0, 2), (0, 1, 4 if sw1 else 2)])
        tex, trx, ntx = tm.get(bmx)
        _dxs_kernel[(ntx, triton.cdiv(D, bnx))](
            gh, gl, w1, tex, trx, offs, pos, dxp, D, F2,
            w1.stride(0), w1.stride(1), w1.stride(2),
            BM=bmx, BN=bnx, BK=bkx, SW=sw1,
            num_warps=wxw, num_stages=wxs)

        dx = _empty(T, D, dtype=x.dtype, device=dev)
        bd = _tile(D, 128, floor=16)
        btc = _clamp(2048 // bd, 1, 32)
        _dxcomb_kernel[(triton.cdiv(T, btc), triton.cdiv(D, bd))](
            dxp, ds, router_w, dx, T, D, E, A,
            router_w.stride(0), router_w.stride(1),
            BT=btc, BD=bd, BE=_tile(E, 64), OD=0 if sx else 1,
            SW=(router_w.dtype == torch.float32), num_warps=4)

        dw1 = _empty(E, D, F2, dtype=w1.dtype, device=dev)
        # x tile (split when float32), two bfloat16 hidden-gradient planes
        b1m, b1n, b1k, u1w, u1s = _cfg(
            "dw1", D, F2, N, sx,
            [(1, 0, 4 if sx else 2), (0, 1, 2), (0, 1, 2)])
        _dw1_kernel[(triton.cdiv(D, b1m) * triton.cdiv(F2, b1n), E)](
            x, gh, gl, tok, offs, dw1, D, F2,
            x.stride(0), x.stride(1), dw1.stride(0), dw1.stride(1),
            dw1.stride(2), BM=b1m, BN=b1n, BK=b1k,
            OD=0 if sw1 else 1, SX=sx, num_warps=u1w, num_stages=u1s)

        dw2 = _empty(E, F, D, dtype=w2.dtype, device=dev)
        # two bfloat16 activation limb planes, cotangent tile
        b2m, b2n, b2k, u2w, u2s = _cfg(
            "dw2", F, D, N, sd,
            [(1, 0, 2), (1, 0, 2), (0, 1, 4 if sd else 2)])
        _dw2_kernel[(triton.cdiv(F, b2m) * triton.cdiv(D, b2n), E)](
            ah, al, dy, tok, offs, dw2, F, D, dy.stride(0), dy.stride(1),
            dw2.stride(0), dw2.stride(1), dw2.stride(2),
            BM=b2m, BN=b2n, BK=b2k, OD=0 if sw2 else 1, SD=sd,
            num_warps=u2w, num_stages=u2s)

        drw = _empty(D, E, dtype=router_w.dtype, device=dev)
        # x tile (split when float32), float32 score-gradient tile
        b3m, b3n, b3k, u3w, u3s = _cfg(
            "drw", D, E, T, sx, [(1, 0, 4 if sx else 2), (0, 1, 4)])
        _drw_kernel[(triton.cdiv(D, b3m), triton.cdiv(E, b3n))](
            x, ds, drw, T, D, E, x.stride(0), x.stride(1),
            drw.stride(0), drw.stride(1), BM=b3m, BN=b3n, BK=b3k,
            OD=0 if router_w.dtype == torch.float32 else 1, SX=sx,
            num_warps=u3w, num_stages=u3s)

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
    return _MoE.apply(x, router_w, router_b, w1, w2, A)
