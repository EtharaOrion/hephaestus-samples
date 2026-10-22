"""Routed fused mixture-of-experts layer, forward and backward.

Layout of the pipeline below: the router scores every token and packs a
selection key; the top-A picks become a list of (token, expert) pairs sorted
by expert; a tile plan cuts each expert's contiguous block of pairs into row
tiles; and the expert projections, the combine and the backward run as tiled
GEMMs over those tiles, so one weight tile serves many rows.

float32 operands ride the bfloat16 tensor cores as hi/lo limb pairs. Limbs of
the two stored intermediates are written once by their producer, and each
consumer reads back the number of limbs its own output precision needs.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# device helpers
# ---------------------------------------------------------------------------

@triton.jit
def _mdot(a, b, acc, ASPLIT: tl.constexpr, BSPLIT: tl.constexpr):
    """acc + a @ b through bfloat16 tensor cores.

    An operand flagged as split is a float32 value decomposed here into a
    bfloat16 high part plus its bfloat16 remainder; the cross terms land in
    the same float32 accumulator. An unflagged operand is already bfloat16.
    """
    if ASPLIT:
        ah = a.to(tl.bfloat16)
        al = (a - ah.to(tl.float32)).to(tl.bfloat16)
    else:
        ah = a
    if BSPLIT:
        bh = b.to(tl.bfloat16)
        bl = (b - bh.to(tl.float32)).to(tl.bfloat16)
    else:
        bh = b
    acc = tl.dot(ah, bh, acc)
    if ASPLIT:
        acc = tl.dot(al, bh, acc)
    if BSPLIT:
        acc = tl.dot(ah, bl, acc)
    return acc


@triton.jit
def _limbdot(ah, al, b, acc, BSPLIT: tl.constexpr, ALIMBS: tl.constexpr):
    """acc + a @ b where a already arrives as bfloat16 limbs.

    ALIMBS says how many limbs of a participate: the high limb alone
    reproduces a bfloat16-rounded operand, both limbs reproduce the float32
    value the limbs were split from.
    """
    if BSPLIT:
        bh = b.to(tl.bfloat16)
        bl = (b - bh.to(tl.float32)).to(tl.bfloat16)
    else:
        bh = b
    acc = tl.dot(ah, bh, acc)
    if ALIMBS > 1:
        acc = tl.dot(al, bh, acc)
    if BSPLIT:
        acc = tl.dot(ah, bl, acc)
    return acc


@triton.jit
def _sat(acc, LIM: tl.constexpr):
    """Clamp to +-LIM, passing NaN through unchanged.

    A row sum whose exact value sits just below the destination maximum can
    round to either side of it depending on the order the partial products
    are summed.
    """
    c = tl.minimum(tl.maximum(acc, -LIM), LIM)
    return tl.where(acc == acc, c, acc)


@triton.jit
def _swizzle(pid, num_m, num_n, GROUP_M: tl.constexpr):
    wide = GROUP_M * num_n
    gid = pid // wide
    first = gid * GROUP_M
    gsz = min(num_m - first, GROUP_M)
    pid_m = first + ((pid % wide) % gsz)
    pid_n = (pid % wide) // gsz
    return pid_m, pid_n


# ---------------------------------------------------------------------------
# routing
# ---------------------------------------------------------------------------

@triton.jit
def _router_kernel(X, RW, RB, S, KEY, T, D, E,
                   SHIFT: tl.constexpr, F32: tl.constexpr,
                   BT: tl.constexpr, BE: tl.constexpr, BK: tl.constexpr):
    """s = sigmoid(x @ router_w) in float32, plus the packed selection key.

    The key is the order-preserving integer image of the biased score shifted
    above a field wide enough for every expert index, OR'd with the inverted
    index, so descending key order is (score descending, index ascending).
    """
    pid_t = tl.program_id(0)
    pid_e = tl.program_id(1)
    rt = pid_t * BT + tl.arange(0, BT)
    re = pid_e * BE + tl.arange(0, BE)
    tm = rt < T
    em = re < E
    acc = tl.zeros((BT, BE), tl.float32)
    xp = X + rt[:, None].to(tl.int64) * D
    wp = RW + re[None, :].to(tl.int64)
    for k0 in range(0, D, BK):
        rk = k0 + tl.arange(0, BK)
        km = rk < D
        a = tl.load(xp + rk[None, :], mask=tm[:, None] & km[None, :], other=0.0)
        b = tl.load(wp + rk[:, None].to(tl.int64) * E,
                    mask=km[:, None] & em[None, :], other=0.0)
        if F32:
            acc = tl.dot(a, b, acc, input_precision="ieee")
        else:
            acc = tl.dot(a, b, acc)
    s = tl.sigmoid(acc)
    off = rt[:, None].to(tl.int64) * E + re[None, :]
    msk = tm[:, None] & em[None, :]
    tl.store(S + off, s, mask=msk)
    rb = tl.load(RB + re, mask=em, other=0.0)
    biased = s + rb[None, :]
    biased = tl.where(biased == 0.0, 0.0, biased)
    ib = biased.to(tl.int32, bitcast=True).to(tl.int64)
    u = tl.where(ib >= 0, ib + 2147483648, -ib - 1)
    key = (u << SHIFT) | (E - 1 - re[None, :]).to(tl.int64)
    tl.store(KEY + off, key, mask=msk)


@triton.jit
def _pairkey_kernel(SEL, PK, N, A, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < N
    e = tl.load(SEL + i, mask=m, other=0)
    tl.store(PK + i, e * N + i.to(tl.int64), mask=m)


@triton.jit
def _bounds_kernel(PKS, OFFS, N, E):
    """First position in the expert-sorted pair list belonging to expert e."""
    e = tl.program_id(0)
    lo = 0
    hi = N
    while lo < hi:
        mid = (lo + hi) // 2
        v = tl.load(PKS + mid) // N
        if v < e:
            lo = mid + 1
        else:
            hi = mid
    tl.store(OFFS + e, lo)


@triton.jit
def _eorder_kernel(OFFS, EORD, E, BR: tl.constexpr, BC: tl.constexpr):
    """Expert indices ordered by descending row count, index ascending on ties.

    Each expert's rank is the count of experts that outrank it, accumulated a
    column block at a time so the comparison tile stays bounded as E grows.
    The ranks form a permutation; the weight-gradient grids walk it so the
    experts holding the most rows are issued first.
    """
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    rlive = r < E
    cnt = tl.load(OFFS + r + 1, mask=rlive, other=0) - tl.load(
        OFFS + r, mask=rlive, other=0)
    rank = tl.zeros((BR,), tl.int32)
    for c0 in range(0, E, BC):
        c = c0 + tl.arange(0, BC)
        clive = c < E
        ccnt = tl.load(OFFS + c + 1, mask=clive, other=0) - tl.load(
            OFFS + c, mask=clive, other=0)
        gt = ccnt[None, :] > cnt[:, None]
        eq = (ccnt[None, :] == cnt[:, None]) & (c[None, :] < r[:, None])
        rank += tl.sum(tl.where(clive[None, :] & (gt | eq), 1, 0), 1)
    tl.store(EORD + rank, r, mask=rlive)


@triton.jit
def _plan_kernel(OFFS, TOFF, E, BM: tl.constexpr, BE: tl.constexpr):
    """Exclusive prefix sum of each expert's row-tile count.

    One program walks the experts a block at a time, carrying the running
    total across blocks, so E is not bounded by a single tile.
    """
    carry = 0
    for e0 in range(0, E + 1, BE):
        e = e0 + tl.arange(0, BE)
        live = e < E
        beg = tl.load(OFFS + e, mask=e <= E, other=0)
        end = tl.load(OFFS + e + 1, mask=live, other=0)
        tiles = tl.where(live, (end - beg + BM - 1) // BM, 0)
        incl = tl.cumsum(tiles, 0)
        tl.store(TOFF + e, carry + incl - tiles, mask=e <= E)
        carry += tl.sum(tiles, 0)


@triton.jit
def _tileid_kernel(TOFF, OFFS, TE, TR, TN, E, NT,
                   BM: tl.constexpr, BLOCK: tl.constexpr,
                   STEPS: tl.constexpr):
    """Flat row-tile index -> (expert, first row, row count).

    The owning expert is the last one whose tile offset does not exceed the
    flat index, found by a fixed-depth binary search over TOFF.
    """
    j = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    jm = j < NT
    total = tl.load(TOFF + E)
    lo = tl.zeros((BLOCK,), tl.int32)
    hi = tl.full((BLOCK,), E - 1, tl.int32)
    for _ in tl.static_range(STEPS):
        mid = (lo + hi + 1) // 2
        take = tl.load(TOFF + mid, mask=jm, other=0) <= j
        lo = tl.where(take, mid, lo)
        hi = tl.where(take, hi, mid - 1)
    e = lo
    beg = tl.load(OFFS + e, mask=jm, other=0)
    end = tl.load(OFFS + e + 1, mask=jm, other=0)
    te = tl.load(TOFF + e, mask=jm, other=0)
    row0 = beg + (j - te) * BM
    live = jm & (j < total)
    tl.store(TE + j, tl.where(live, e, -1), mask=jm)
    tl.store(TR + j, row0, mask=jm)
    tl.store(TN + j, tl.minimum(end - row0, BM), mask=jm)


@triton.jit
def _wnorm_kernel(S, SEL, WF, T, E, A, BT: tl.constexpr, BA: tl.constexpr):
    """Combine weights: raw sigmoid scores of the selected experts, normalized."""
    rt = tl.program_id(0) * BT + tl.arange(0, BT)
    ra = tl.arange(0, BA)
    tm = rt < T
    am = ra < A
    msk = tm[:, None] & am[None, :]
    sel = tl.load(SEL + rt[:, None].to(tl.int64) * A + ra[None, :],
                  mask=msk, other=0)
    s = tl.load(S + rt[:, None].to(tl.int64) * E + sel, mask=msk, other=0.0)
    tot = tl.sum(tl.where(msk, s, 0.0), 1)
    tl.store(WF + rt[:, None].to(tl.int64) * A + ra[None, :],
             s / tot[:, None], mask=msk)


@triton.jit
def _meta_kernel(ORDER, WF, TOK, POS, CW, N, A, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < N
    o = tl.load(ORDER + i, mask=m, other=0)
    tl.store(TOK + i, (o // A).to(tl.int32), mask=m)
    tl.store(POS + o, i, mask=m)
    tl.store(CW + i, tl.load(WF + o, mask=m, other=0.0), mask=m)


# ---------------------------------------------------------------------------
# forward expert pipeline
# ---------------------------------------------------------------------------

@triton.jit
def _gemm1_kernel(X, W1, H, AH, AL, TOK, TE, TR, TN, D, F, NM, NN,
                  XS: tl.constexpr, WS: tl.constexpr,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                  GROUP_M: tl.constexpr):
    """h = x[tok] @ w1[e] for one row tile, then silu(gate) * up as limbs."""
    pid_m, pid_n = _swizzle(tl.program_id(0), NM, NN, GROUP_M)
    e = tl.load(TE + pid_m)
    if e < 0:
        return
    row0 = tl.load(TR + pid_m)
    nrow = tl.load(TN + pid_m)
    rm = tl.arange(0, BM)
    mm = rm < nrow
    tok = tl.load(TOK + row0 + rm, mask=mm, other=0).to(tl.int64)
    rn = pid_n * BN + tl.arange(0, BN)
    nm = rn < F
    accg = tl.zeros((BM, BN), tl.float32)
    accu = tl.zeros((BM, BN), tl.float32)
    xp = X + tok[:, None] * D
    wbase = W1 + e.to(tl.int64) * (D * 2 * F)
    for k0 in range(0, D, BK):
        rk = k0 + tl.arange(0, BK)
        km = rk < D
        a = tl.load(xp + rk[None, :], mask=mm[:, None] & km[None, :], other=0.0)
        wp = wbase + rk[:, None].to(tl.int64) * (2 * F)
        bg = tl.load(wp + rn[None, :], mask=km[:, None] & nm[None, :], other=0.0)
        bu = tl.load(wp + rn[None, :] + F, mask=km[:, None] & nm[None, :],
                     other=0.0)
        accg = _mdot(a, bg, accg, XS, WS)
        accu = _mdot(a, bu, accu, XS, WS)
    rows = (row0 + rm).to(tl.int64)
    hp = H + rows[:, None] * (2 * F) + rn[None, :]
    st = mm[:, None] & nm[None, :]
    tl.store(hp, accg, mask=st)
    tl.store(hp + F, accu, mask=st)
    act = accg * tl.sigmoid(accg) * accu
    ah = act.to(tl.bfloat16)
    off = rows[:, None] * F + rn[None, :]
    tl.store(AH + off, ah, mask=st)
    tl.store(AL + off, (act - ah.to(tl.float32)).to(tl.bfloat16), mask=st)


@triton.jit
def _gemm2_kernel(AH, AL, W2, YP, TE, TR, TN, F, D, NM, NN,
                  WS: tl.constexpr, ALIMBS: tl.constexpr,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                  GROUP_M: tl.constexpr):
    """yp = act @ w2[e] for one row tile."""
    pid_m, pid_n = _swizzle(tl.program_id(0), NM, NN, GROUP_M)
    e = tl.load(TE + pid_m)
    if e < 0:
        return
    row0 = tl.load(TR + pid_m)
    nrow = tl.load(TN + pid_m)
    rm = tl.arange(0, BM)
    mm = rm < nrow
    rows = (row0 + rm).to(tl.int64)
    rn = pid_n * BN + tl.arange(0, BN)
    nm = rn < D
    acc = tl.zeros((BM, BN), tl.float32)
    abase = rows[:, None] * F
    wbase = W2 + e.to(tl.int64) * (F * D)
    for k0 in range(0, F, BK):
        rk = k0 + tl.arange(0, BK)
        km = rk < F
        am = mm[:, None] & km[None, :]
        ah = tl.load(AH + abase + rk[None, :], mask=am, other=0.0)
        if ALIMBS > 1:
            al = tl.load(AL + abase + rk[None, :], mask=am, other=0.0)
        else:
            al = ah
        b = tl.load(wbase + rk[:, None].to(tl.int64) * D + rn[None, :],
                    mask=km[:, None] & nm[None, :], other=0.0)
        acc = _limbdot(ah, al, b, acc, WS, ALIMBS)
    tl.store(YP + rows[:, None] * D + rn[None, :], acc,
             mask=mm[:, None] & nm[None, :])


@triton.jit
def _combine_kernel(YP, POS, WF, Y, T, D, A, LIM: tl.constexpr,
                    BT: tl.constexpr, BD: tl.constexpr):
    """y[t] = sum over the A slots of w * yp[slot row], slots walked in order."""
    rt = tl.program_id(0) * BT + tl.arange(0, BT)
    rd = tl.program_id(1) * BD + tl.arange(0, BD)
    tm = rt < T
    dm = rd < D
    st = tm[:, None] & dm[None, :]
    acc = tl.zeros((BT, BD), tl.float32)
    base = rt.to(tl.int64) * A
    for a in range(A):
        p = tl.load(POS + base + a, mask=tm, other=0).to(tl.int64)
        w = tl.load(WF + base + a, mask=tm, other=0.0)
        v = tl.load(YP + p[:, None] * D + rd[None, :], mask=st, other=0.0)
        acc += w[:, None] * v
    tl.store(Y + rt[:, None].to(tl.int64) * D + rd[None, :],
             _sat(acc, LIM).to(Y.dtype.element_ty), mask=st)


# ---------------------------------------------------------------------------
# backward
# ---------------------------------------------------------------------------

@triton.jit
def _dcw_kernel(DY, YP, TOK, DCW, N, D, BR: tl.constexpr, BD: tl.constexpr):
    """Gradient of each pair's combine weight: <dy[token], yp[pair row]>."""
    ri = tl.program_id(0) * BR + tl.arange(0, BR)
    rm = ri < N
    tok = tl.load(TOK + ri, mask=rm, other=0).to(tl.int64)
    acc = tl.zeros((BR, BD), tl.float32)
    for d0 in range(0, D, BD):
        rd = d0 + tl.arange(0, BD)
        st = rm[:, None] & (rd < D)[None, :]
        g = tl.load(DY + tok[:, None] * D + rd[None, :], mask=st,
                    other=0.0).to(tl.float32)
        v = tl.load(YP + ri[:, None].to(tl.int64) * D + rd[None, :], mask=st,
                    other=0.0)
        acc += g * v
    tl.store(DCW + ri, tl.sum(acc, 1), mask=rm)


@triton.jit
def _dact_kernel(DY, W2, H, CW, GH, GL, TOK, TE, TR, TN, F, D, NM, NN,
                 YS: tl.constexpr, WS: tl.constexpr,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                 GROUP_M: tl.constexpr):
    """dact = cw * (dy[tok] @ w2[e]^T), then back through silu(gate) * up."""
    pid_m, pid_n = _swizzle(tl.program_id(0), NM, NN, GROUP_M)
    e = tl.load(TE + pid_m)
    if e < 0:
        return
    row0 = tl.load(TR + pid_m)
    nrow = tl.load(TN + pid_m)
    rm = tl.arange(0, BM)
    mm = rm < nrow
    rows = (row0 + rm).to(tl.int64)
    tok = tl.load(TOK + row0 + rm, mask=mm, other=0).to(tl.int64)
    rn = pid_n * BN + tl.arange(0, BN)
    nm = rn < F
    acc = tl.zeros((BM, BN), tl.float32)
    dp = DY + tok[:, None] * D
    wbase = W2 + e.to(tl.int64) * (F * D)
    for k0 in range(0, D, BK):
        rk = k0 + tl.arange(0, BK)
        km = rk < D
        a = tl.load(dp + rk[None, :], mask=mm[:, None] & km[None, :], other=0.0)
        bt = tl.load(wbase + rn[:, None].to(tl.int64) * D + rk[None, :],
                     mask=nm[:, None] & km[None, :], other=0.0)
        acc = _mdot(a, tl.trans(bt), acc, YS, WS)
    cw = tl.load(CW + row0 + rm, mask=mm, other=0.0)
    acc = acc * cw[:, None]
    hp = H + rows[:, None] * (2 * F) + rn[None, :]
    st = mm[:, None] & nm[None, :]
    g = tl.load(hp, mask=st, other=0.0)
    u = tl.load(hp + F, mask=st, other=0.0)
    sg = tl.sigmoid(g)
    silu = g * sg
    dg = acc * u * (sg + silu * (1.0 - sg))
    du = acc * silu
    off = rows[:, None] * (2 * F) + rn[None, :]
    dgh = dg.to(tl.bfloat16)
    duh = du.to(tl.bfloat16)
    tl.store(GH + off, dgh, mask=st)
    tl.store(GH + off + F, duh, mask=st)
    tl.store(GL + off, (dg - dgh.to(tl.float32)).to(tl.bfloat16), mask=st)
    tl.store(GL + off + F, (du - duh.to(tl.float32)).to(tl.bfloat16), mask=st)


@triton.jit
def _dxs_kernel(GH, GL, W1, DXS, TE, TR, TN, D, F, NM, NN,
                WS: tl.constexpr, ALIMBS: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                GROUP_M: tl.constexpr):
    """Per-pair input gradient dgu @ w1[e]^T."""
    pid_m, pid_n = _swizzle(tl.program_id(0), NM, NN, GROUP_M)
    e = tl.load(TE + pid_m)
    if e < 0:
        return
    row0 = tl.load(TR + pid_m)
    nrow = tl.load(TN + pid_m)
    rm = tl.arange(0, BM)
    mm = rm < nrow
    rows = (row0 + rm).to(tl.int64)
    rn = pid_n * BN + tl.arange(0, BN)
    nm = rn < D
    acc = tl.zeros((BM, BN), tl.float32)
    K = 2 * F
    gbase = rows[:, None] * K
    wbase = W1 + e.to(tl.int64) * (D * K)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        km = rk < K
        am = mm[:, None] & km[None, :]
        ah = tl.load(GH + gbase + rk[None, :], mask=am, other=0.0)
        if ALIMBS > 1:
            al = tl.load(GL + gbase + rk[None, :], mask=am, other=0.0)
        else:
            al = ah
        bt = tl.load(wbase + rn[:, None].to(tl.int64) * K + rk[None, :],
                     mask=nm[:, None] & km[None, :], other=0.0)
        acc = _limbdot(ah, al, tl.trans(bt), acc, WS, ALIMBS)
    tl.store(DXS + rows[:, None] * D + rn[None, :], acc,
             mask=mm[:, None] & nm[None, :])


@triton.jit
def _dw1_kernel(X, GH, GL, DW1, TOK, OFFS, EORD, D, F, NM, NN,
                XS: tl.constexpr, LIM: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """dw1[e] = x[tok]^T @ dgu over expert e's own row range."""
    pid = tl.program_id(0)
    e = tl.load(EORD + pid // (NM * NN))
    r = pid % (NM * NN)
    pid_m = r // NN
    pid_n = r % NN
    beg = tl.load(OFFS + e)
    end = tl.load(OFFS + e + 1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    K2 = 2 * F
    mm = rm < D
    nm = rn < K2
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(beg, end, BK):
        rk = k0 + tl.arange(0, BK)
        km = rk < end
        tok = tl.load(TOK + rk, mask=km, other=0).to(tl.int64)
        a = tl.load(X + tok[:, None] * D + rm[None, :],
                    mask=km[:, None] & mm[None, :], other=0.0)
        off = rk[:, None].to(tl.int64) * K2 + rn[None, :]
        bm = km[:, None] & nm[None, :]
        bh = tl.load(GH + off, mask=bm, other=0.0)
        bl = tl.load(GL + off, mask=bm, other=0.0)
        at = tl.trans(a)
        if XS:
            ah = at.to(tl.bfloat16)
            alo = (at - ah.to(tl.float32)).to(tl.bfloat16)
            acc = tl.dot(alo, bh, acc)
        else:
            ah = at
        acc = tl.dot(ah, bh, acc)
        acc = tl.dot(ah, bl, acc)
    tl.store(DW1 + e.to(tl.int64) * (D * K2) + rm[:, None].to(tl.int64) * K2
             + rn[None, :], _sat(acc, LIM).to(DW1.dtype.element_ty),
             mask=mm[:, None] & nm[None, :])


@triton.jit
def _dw2_kernel(AH, AL, DY, CW, DW2, TOK, OFFS, EORD, F, D, NM, NN,
                YS: tl.constexpr, LIM: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """dw2[e] = (cw * act)^T @ dy[tok] over expert e's own row range."""
    pid = tl.program_id(0)
    e = tl.load(EORD + pid // (NM * NN))
    r = pid % (NM * NN)
    pid_m = r // NN
    pid_n = r % NN
    beg = tl.load(OFFS + e)
    end = tl.load(OFFS + e + 1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    mm = rm < F
    nm = rn < D
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(beg, end, BK):
        rk = k0 + tl.arange(0, BK)
        km = rk < end
        tok = tl.load(TOK + rk, mask=km, other=0).to(tl.int64)
        cw = tl.load(CW + rk, mask=km, other=0.0)
        off = rk[:, None].to(tl.int64) * F + rm[None, :]
        am = km[:, None] & mm[None, :]
        av = (tl.load(AH + off, mask=am, other=0.0).to(tl.float32)
              + tl.load(AL + off, mask=am, other=0.0).to(tl.float32))
        a = tl.trans(av * cw[:, None])
        b = tl.load(DY + tok[:, None] * D + rn[None, :],
                    mask=km[:, None] & nm[None, :], other=0.0)
        acc = _mdot(a, b, acc, True, YS)
    tl.store(DW2 + e.to(tl.int64) * (F * D) + rm[:, None].to(tl.int64) * D
             + rn[None, :], _sat(acc, LIM).to(DW2.dtype.element_ty),
             mask=mm[:, None] & nm[None, :])


@triton.jit
def _rbwd_kernel(S, SEL, WF, DCW, POS, DLOG, T, E, A,
                 BT: tl.constexpr, BA: tl.constexpr):
    """Through the weight normalization and the sigmoid, into the logits."""
    rt = tl.program_id(0) * BT + tl.arange(0, BT)
    ra = tl.arange(0, BA)
    tm = rt < T
    am = ra < A
    msk = tm[:, None] & am[None, :]
    base = rt[:, None].to(tl.int64) * A + ra[None, :]
    sel = tl.load(SEL + base, mask=msk, other=0)
    s = tl.load(S + rt[:, None].to(tl.int64) * E + sel, mask=msk, other=0.0)
    w = tl.load(WF + base, mask=msk, other=0.0)
    pos = tl.load(POS + base, mask=msk, other=0).to(tl.int64)
    dcw = tl.load(DCW + pos, mask=msk, other=0.0)
    tot = tl.sum(tl.where(msk, s, 0.0), 1)
    inner = tl.sum(tl.where(msk, dcw * w, 0.0), 1)
    ds = (dcw - inner[:, None]) / tot[:, None]
    tl.store(DLOG + rt[:, None].to(tl.int64) * E + sel, ds * s * (1.0 - s),
             mask=msk)


@triton.jit
def _drw_kernel(X, DLOG, DRW, T, D, E, SPLITK,
                XS: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """Partial x^T @ dlog for one slice of tokens."""
    g = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    mm = rm < D
    nm = rn < E
    acc = tl.zeros((BM, BN), tl.float32)
    span = tl.cdiv(tl.cdiv(T, BK), SPLITK) * BK
    lo = g * span
    hi = min(lo + span, T)
    for k0 in range(lo, hi, BK):
        rk = k0 + tl.arange(0, BK)
        km = rk < hi
        a = tl.load(X + rk[:, None].to(tl.int64) * D + rm[None, :],
                    mask=km[:, None] & mm[None, :], other=0.0)
        b = tl.load(DLOG + rk[:, None].to(tl.int64) * E + rn[None, :],
                    mask=km[:, None] & nm[None, :], other=0.0)
        acc = _mdot(tl.trans(a), b, acc, XS, True)
    tl.store(DRW + g.to(tl.int64) * (D * E) + rm[:, None].to(tl.int64) * E
             + rn[None, :], acc, mask=mm[:, None] & nm[None, :])


@triton.jit
def _drwsum_kernel(PART, DRW, N, SPLITK, LIM: tl.constexpr,
                   BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < N
    acc = tl.zeros((BLOCK,), tl.float32)
    for g in range(SPLITK):
        acc += tl.load(PART + g.to(tl.int64) * N + i, mask=m, other=0.0)
    tl.store(DRW + i, _sat(acc, LIM).to(DRW.dtype.element_ty), mask=m)


@triton.jit
def _dxcomb_kernel(DXS, POS, DLOG, RW, DX, T, D, E, A,
                   RS: tl.constexpr, LIM: tl.constexpr,
                   BT: tl.constexpr, BD: tl.constexpr, BK: tl.constexpr):
    """dx = sum of the token's A per-pair gradients plus dlog @ router_w^T."""
    rt = tl.program_id(0) * BT + tl.arange(0, BT)
    rd = tl.program_id(1) * BD + tl.arange(0, BD)
    tm = rt < T
    dm = rd < D
    st = tm[:, None] & dm[None, :]
    acc = tl.zeros((BT, BD), tl.float32)
    base = rt.to(tl.int64) * A
    for a in range(A):
        p = tl.load(POS + base + a, mask=tm, other=0).to(tl.int64)
        acc += tl.load(DXS + p[:, None] * D + rd[None, :], mask=st, other=0.0)
    racc = tl.zeros((BT, BD), tl.float32)
    for k0 in range(0, E, BK):
        rk = k0 + tl.arange(0, BK)
        km = rk < E
        g = tl.load(DLOG + rt[:, None].to(tl.int64) * E + rk[None, :],
                    mask=tm[:, None] & km[None, :], other=0.0)
        bt = tl.load(RW + rd[:, None].to(tl.int64) * E + rk[None, :],
                     mask=dm[:, None] & km[None, :], other=0.0)
        racc = _mdot(g, tl.trans(bt), racc, True, RS)
    tl.store(DX + rt[:, None].to(tl.int64) * D + rd[None, :],
             _sat(acc + racc, LIM).to(DX.dtype.element_ty), mask=st)


# ---------------------------------------------------------------------------
# launch configuration
# ---------------------------------------------------------------------------

def _finite_max(dtype):
    return float(torch.finfo(dtype).max)


def _p2(n):
    return 1 << max(0, (n - 1).bit_length())


def _cap(b, n):
    return max(16, min(b, _p2(n)))


# Limbs of a stored float32 intermediate a consumer reads back. act reaches
# router_w's gradient through yp and the combine-weight gradient, and every
# weight-gradient reduction reads its operand in full, so those consumers take
# both limbs. dgu reaches only dx, which is returned in the input dtype.
_ACT_LIMBS = 2
_DXS_LIMBS = {False: 1, True: 2}

# Tile geometry per kernel and per input dtype, from a sweep over the row-tile
# and weight-gradient GEMMs. BM of the four row-tile kernels selects which
# precomputed tile plan the launch reads.
_CFG = {
    False: {
        "gemm1": dict(BM=128, BN=128, BK=32, num_warps=8, num_stages=4),
        "gemm2": dict(BM=128, BN=256, BK=64, num_warps=8, num_stages=3),
        "dact": dict(BM=64, BN=128, BK=64, num_warps=8, num_stages=4),
        "dxs": dict(BM=128, BN=256, BK=64, num_warps=8, num_stages=4),
        "dw1": dict(BM=128, BN=128, BK=64, num_warps=8, num_stages=4),
        "dw2": dict(BM=64, BN=256, BK=64, num_warps=4, num_stages=4),
    },
    True: {
        "gemm1": dict(BM=128, BN=64, BK=32, num_warps=4, num_stages=4),
        "gemm2": dict(BM=128, BN=128, BK=32, num_warps=4, num_stages=3),
        "dact": dict(BM=128, BN=128, BK=64, num_warps=8, num_stages=3),
        "dxs": dict(BM=128, BN=256, BK=32, num_warps=8, num_stages=3),
        "dw1": dict(BM=64, BN=256, BK=32, num_warps=4, num_stages=3),
        "dw2": dict(BM=128, BN=256, BK=32, num_warps=8, num_stages=3),
    },
}
_GROUP_M = 8

# Row-tile heights any row GEMM may ask for; a plan is built for each.
_PLAN_BM = (64, 128)


def _route(x, router_w, router_b, A):
    T, D = x.shape
    E = router_w.shape[1]
    N = T * A
    dev = x.device
    f32 = x.dtype == torch.float32

    s = torch.empty(T, E, device=dev, dtype=torch.float32)
    key = torch.empty(T, E, device=dev, dtype=torch.int64)
    shift = max(10, (E - 1).bit_length())
    be = _cap(64, E)
    bt = max(16, min(128, _p2(T)))
    _router_kernel[(triton.cdiv(T, bt), triton.cdiv(E, be))](
        x, router_w, router_b, s, key, T, D, E,
        SHIFT=shift, F32=f32, BT=bt, BE=be, BK=_cap(32, D),
        num_warps=4, num_stages=3)

    sel = torch.topk(key, A, dim=-1, largest=True, sorted=True).indices

    pk = torch.empty(N, device=dev, dtype=torch.int64)
    _pairkey_kernel[(triton.cdiv(N, 256),)](sel, pk, N, A, BLOCK=256,
                                            num_warps=4)
    pks, order = torch.sort(pk)

    offs = torch.empty(E + 1, device=dev, dtype=torch.int32)
    _bounds_kernel[(E + 1,)](pks, offs, N, E, num_warps=1)

    bec = min(_p2(E + 1), 1024)
    eord = torch.empty(E, device=dev, dtype=torch.int32)
    _eorder_kernel[(triton.cdiv(E, 128),)](offs, eord, E, BR=128, BC=bec,
                                           num_warps=4)

    steps = max(1, (E - 1).bit_length())
    plans = {}
    for bm in _PLAN_BM:
        toff = torch.empty(E + 1, device=dev, dtype=torch.int32)
        _plan_kernel[(1,)](offs, toff, E, BM=bm, BE=bec, num_warps=4)
        nt = triton.cdiv(N, bm) + E
        te = torch.empty(nt, device=dev, dtype=torch.int32)
        tr = torch.empty(nt, device=dev, dtype=torch.int32)
        tn = torch.empty(nt, device=dev, dtype=torch.int32)
        _tileid_kernel[(triton.cdiv(nt, 128),)](
            toff, offs, te, tr, tn, E, nt, BM=bm, BLOCK=128, STEPS=steps,
            num_warps=4)
        plans[bm] = (te, tr, tn, nt)

    ba = _p2(max(A, 1))
    btw = max(1, min(64, 8192 // ba))
    wf = torch.empty(T, A, device=dev, dtype=torch.float32)
    _wnorm_kernel[(triton.cdiv(T, btw),)](s, sel, wf, T, E, A, BT=btw, BA=ba,
                                          num_warps=4)

    tok = torch.empty(N, device=dev, dtype=torch.int32)
    pos = torch.empty(N, device=dev, dtype=torch.int32)
    cw = torch.empty(N, device=dev, dtype=torch.float32)
    _meta_kernel[(triton.cdiv(N, 256),)](order, wf, tok, pos, cw, N, A,
                                         BLOCK=256, num_warps=4)
    return s, sel, wf, tok, pos, cw, offs, eord, plans


def _fwd(x, router_w, router_b, w1, w2, A):
    T, D = x.shape
    E = router_w.shape[1]
    F = w1.shape[2] // 2
    N = T * A
    dev = x.device
    f32 = x.dtype == torch.float32

    (s, sel, wf, tok, pos, cw, offs, eord,
     plans) = _route(x, router_w, router_b, A)
    cfg = _CFG[f32]

    h = torch.empty(N, 2 * F, device=dev, dtype=torch.float32)
    ah = torch.empty(N, F, device=dev, dtype=torch.bfloat16)
    al = torch.empty(N, F, device=dev, dtype=torch.bfloat16)
    c = cfg["gemm1"]
    te, tr, tn, nt = plans[c["BM"]]
    bn1 = _cap(c["BN"], F)
    nn1 = triton.cdiv(F, bn1)
    _gemm1_kernel[(nt * nn1,)](
        x, w1, h, ah, al, tok, te, tr, tn, D, F, nt, nn1,
        XS=f32, WS=f32, BM=c["BM"], BN=bn1, BK=_cap(c["BK"], D),
        GROUP_M=_GROUP_M, num_warps=c["num_warps"],
        num_stages=c["num_stages"])

    yp = torch.empty(N, D, device=dev, dtype=torch.float32)
    c = cfg["gemm2"]
    te, tr, tn, nt = plans[c["BM"]]
    bn2 = _cap(c["BN"], D)
    nn2 = triton.cdiv(D, bn2)
    _gemm2_kernel[(nt * nn2,)](
        ah, al, w2, yp, te, tr, tn, F, D, nt, nn2,
        WS=f32, ALIMBS=_ACT_LIMBS, BM=c["BM"], BN=bn2, BK=_cap(c["BK"], F),
        GROUP_M=_GROUP_M, num_warps=c["num_warps"],
        num_stages=c["num_stages"])

    y = torch.empty(T, D, device=dev, dtype=x.dtype)
    bdc = _cap(128, D)
    btc = max(16, min(64, _p2(T)))
    _combine_kernel[(triton.cdiv(T, btc), triton.cdiv(D, bdc))](
        yp, pos, wf, y, T, D, A, LIM=_finite_max(x.dtype), BT=btc, BD=bdc,
        num_warps=4, num_stages=2)

    saved = (s, sel, wf, tok, pos, cw, offs, eord, plans, h, ah, al, yp)
    return y, saved, (T, D, E, F, N)


def _bwd(dy, x, router_w, w1, w2, A, saved, dims):
    s, sel, wf, tok, pos, cw, offs, eord, plans, h, ah, al, yp = saved
    T, D, E, F, N = dims
    dev = x.device
    f32 = x.dtype == torch.float32
    ys = dy.dtype == torch.float32
    cfg = _CFG[f32]
    lim = _finite_max(x.dtype)

    dcw = torch.empty(N, device=dev, dtype=torch.float32)
    _dcw_kernel[(triton.cdiv(N, 64),)](dy, yp, tok, dcw, N, D, BR=64,
                                       BD=_cap(128, D), num_warps=4,
                                       num_stages=2)

    gh = torch.empty(N, 2 * F, device=dev, dtype=torch.bfloat16)
    gl = torch.empty(N, 2 * F, device=dev, dtype=torch.bfloat16)
    c = cfg["dact"]
    te, tr, tn, nt = plans[c["BM"]]
    bna = _cap(c["BN"], F)
    nna = triton.cdiv(F, bna)
    _dact_kernel[(nt * nna,)](
        dy, w2, h, cw, gh, gl, tok, te, tr, tn, F, D, nt, nna,
        YS=ys, WS=f32, BM=c["BM"], BN=bna, BK=_cap(c["BK"], D),
        GROUP_M=_GROUP_M, num_warps=c["num_warps"],
        num_stages=c["num_stages"])

    dxs = torch.empty(N, D, device=dev, dtype=torch.float32)
    c = cfg["dxs"]
    te, tr, tn, nt = plans[c["BM"]]
    bnx = _cap(c["BN"], D)
    nnx = triton.cdiv(D, bnx)
    _dxs_kernel[(nt * nnx,)](
        gh, gl, w1, dxs, te, tr, tn, D, F, nt, nnx,
        WS=f32, ALIMBS=_DXS_LIMBS[f32], BM=c["BM"], BN=bnx,
        BK=_cap(c["BK"], 2 * F), GROUP_M=_GROUP_M,
        num_warps=c["num_warps"], num_stages=c["num_stages"])

    dw1 = torch.empty_like(w1)
    c = cfg["dw1"]
    bm1, bn1 = _cap(c["BM"], D), _cap(c["BN"], 2 * F)
    m1, n1 = triton.cdiv(D, bm1), triton.cdiv(2 * F, bn1)
    _dw1_kernel[(E * m1 * n1,)](
        x, gh, gl, dw1, tok, offs, eord, D, F, m1, n1,
        XS=f32, LIM=lim, BM=bm1, BN=bn1, BK=c["BK"],
        num_warps=c["num_warps"], num_stages=c["num_stages"])

    dw2 = torch.empty_like(w2)
    c = cfg["dw2"]
    bm2, bn2 = _cap(c["BM"], F), _cap(c["BN"], D)
    m2, n2 = triton.cdiv(F, bm2), triton.cdiv(D, bn2)
    _dw2_kernel[(E * m2 * n2,)](
        ah, al, dy, cw, dw2, tok, offs, eord, F, D, m2, n2,
        YS=ys, LIM=lim, BM=bm2, BN=bn2, BK=c["BK"],
        num_warps=c["num_warps"], num_stages=c["num_stages"])

    dlog = torch.zeros(T, E, device=dev, dtype=torch.float32)
    ba = _p2(max(A, 1))
    btw = max(1, min(64, 8192 // ba))
    _rbwd_kernel[(triton.cdiv(T, btw),)](
        s, sel, wf, dcw, pos, dlog, T, E, A, BT=btw, BA=ba, num_warps=4)

    bmr, bnr = _cap(64, D), _cap(64, E)
    bkr = _cap(64, T)
    gm, gn = triton.cdiv(D, bmr), triton.cdiv(E, bnr)
    splitk = max(1, min(triton.cdiv(T, bkr), 256 // max(gm * gn, 1)))
    part = torch.empty(splitk, D, E, device=dev, dtype=torch.float32)
    _drw_kernel[(splitk, gm, gn)](
        x, dlog, part, T, D, E, splitk,
        XS=f32, BM=bmr, BN=bnr, BK=bkr, num_warps=4, num_stages=3)
    drw = torch.empty_like(router_w)
    _drwsum_kernel[(triton.cdiv(D * E, 256),)](
        part, drw, D * E, splitk, LIM=lim, BLOCK=256, num_warps=4)

    dx = torch.empty_like(x)
    bdx = _cap(128, D)
    btx = max(16, min(64, _p2(T)))
    _dxcomb_kernel[(triton.cdiv(T, btx), triton.cdiv(D, bdx))](
        dxs, pos, dlog, router_w, dx, T, D, E, A,
        RS=f32, LIM=lim, BT=btx, BD=bdx, BK=_cap(64, E), num_warps=4,
        num_stages=2)

    return dx, drw, dw1, dw2


class _FusedMoe(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, router_w, router_b, w1, w2, A):
        A = int(A)
        x = x.contiguous()
        router_w = router_w.contiguous()
        w1 = w1.contiguous()
        w2 = w2.contiguous()
        y, saved, dims = _fwd(x, router_w, router_b.contiguous(), w1, w2, A)
        s, sel, wf, tok, pos, cw, offs, eord, plans, h, ah, al, yp = saved
        plan_t = [t for bm in _PLAN_BM for t in plans[bm][:3]]
        ctx.save_for_backward(x, router_w, w1, w2, s, sel, wf, tok, pos, cw,
                              offs, eord, h, ah, al, yp, *plan_t)
        ctx.nts = tuple(plans[bm][3] for bm in _PLAN_BM)
        ctx.dims = dims
        ctx.A = A
        return y

    @staticmethod
    def backward(ctx, dy):
        (x, router_w, w1, w2, s, sel, wf, tok, pos, cw, offs, eord, h, ah,
         al, yp) = ctx.saved_tensors[:16]
        plan_t = ctx.saved_tensors[16:]
        plans = {bm: (*plan_t[3 * i:3 * i + 3], ctx.nts[i])
                 for i, bm in enumerate(_PLAN_BM)}
        saved = (s, sel, wf, tok, pos, cw, offs, eord, plans, h, ah, al, yp)
        dx, drw, dw1, dw2 = _bwd(dy.contiguous(), x, router_w, w1, w2, ctx.A,
                                 saved, ctx.dims)
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
