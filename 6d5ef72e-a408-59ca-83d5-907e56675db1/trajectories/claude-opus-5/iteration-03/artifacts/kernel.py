"""Routed fused mixture-of-experts layer, forward and backward.

Tokens are dispatched to experts by sorting the T*A (token, expert) pairs by
expert id, which turns the per-expert projections into contiguous row ranges
of one big matrix. Every (expert, row-tile) pair is one program of a grouped
GEMM; the tile-to-expert map is built on device from the routing metadata, so
a tile never straddles two experts and no expert is ever skipped.

Operand precision is expressed as a limb count. A float32 value is carried as
a sum of bfloat16 limbs (each limb is the round-to-nearest residual of the
previous one), and a product of two such expansions is accumulated in float32
by one tensor-core dot per retained term. A bfloat16 input is one exact limb;
a float32 input takes three, and a float32 intermediate feeding a weight
gradient takes two. The count per operand is fixed per kernel and per input
dtype.

Two rearrangements keep intermediates out of the reductions that feed the
weight gradients: the combine weight multiplies the finished dact rows rather
than the dy tile, and in dw2 it rides on the activation, which is a float32
temporary already. Both leave dy an unmodified input operand.

Every accumulation lands in exactly one program: the dispatch GEMMs partition
rows, the weight gradients partition each expert's own row range, and the two
combines walk the A slots of a token in slot order. There are no atomics, so
repeated runs agree bit for bit.
"""

import torch
import triton
import triton.language as tl

# Operand-precision codes for _ldot. The number of tensor-core dots issued is
# 1, 2, 2 and 6 respectively.
P_HH = tl.constexpr(0)    # both operands one bfloat16 limb
P_AL = tl.constexpr(1)    # left operand two limbs, right one
P_BL = tl.constexpr(2)    # left operand one limb, right two
P_X6 = tl.constexpr(3)    # both operands three limbs


@triton.jit
def _ldot(a, b, acc, P: tl.constexpr):
    """acc + a @ b, with each operand expanded to its declared limb count.

    Operands already stored as bfloat16 pass straight through as their own
    single limb, so the tile they were loaded into keeps its native width.
    A two-limb operand adds the cross term with the other operand's leading
    limb, which is the term that carries the residual back into the sum.
    """
    if P == P_X6:
        return tl.dot(a.to(tl.float32), b.to(tl.float32), acc,
                      input_precision="bf16x6")
    ah = a.to(tl.bfloat16)
    bh = b.to(tl.bfloat16)
    acc = tl.dot(ah, bh, acc)
    if P == P_AL:
        al = (a.to(tl.float32) - ah.to(tl.float32)).to(tl.bfloat16)
        acc = tl.dot(al, bh, acc)
    if P == P_BL:
        bl = (b.to(tl.float32) - bh.to(tl.float32)).to(tl.bfloat16)
        acc = tl.dot(ah, bl, acc)
    return acc


# --------------------------------------------------------------------------
# routing
# --------------------------------------------------------------------------

@triton.jit
def _router_kernel(X, RW, RB, S, KEY, T, D, E,
                   sx0, sx1, sr0, sr1,
                   BT: tl.constexpr, BE: tl.constexpr, BD: tl.constexpr,
                   P: tl.constexpr):
    """s = sigmoid(x @ router_w) and the packed selection key of s + router_b.

    The key is the order-preserving integer image of the biased float32 score
    shifted left and OR'd with the inverted expert index, so descending key
    order is (biased score descending, expert index ascending) and no two
    keys of one token are ever equal. -0.0 is folded onto +0.0 first.

    Two experts whose router columns are bitwise equal feed identical operands
    through an identical reduction order here, so their scores come out
    bitwise equal and their tie is decided by the index field of the key.
    """
    pid = tl.program_id(0)
    rt = pid * BT + tl.arange(0, BT)
    re = tl.arange(0, BE)
    mt = rt < T
    me = re < E
    xp = X + rt[:, None].to(tl.int64) * sx0
    acc = tl.zeros((BT, BE), dtype=tl.float32)
    for k0 in range(0, D, BD):
        rk = k0 + tl.arange(0, BD)
        mk = rk < D
        a = tl.load(xp + rk[None, :] * sx1,
                    mask=mt[:, None] & mk[None, :], other=0.0)
        b = tl.load(RW + rk[:, None] * sr0 + re[None, :] * sr1,
                    mask=mk[:, None] & me[None, :], other=0.0)
        acc = _ldot(a, b, acc, P)
    s = tl.sigmoid(acc)
    tl.store(S + rt[:, None].to(tl.int64) * E + re[None, :], s,
             mask=mt[:, None] & me[None, :])

    rb = tl.load(RB + re, mask=me, other=0.0)
    b = s + rb[None, :]
    b = tl.where(b == 0.0, 0.0, b)
    ib = b.to(tl.int32, bitcast=True).to(tl.int64)
    u = tl.where(ib >= 0, ib ^ 0x80000000, (~ib) & 0xFFFFFFFF)
    key = (u << 10) | (E - 1 - re[None, :]).to(tl.int64)
    tl.store(KEY + rt[:, None].to(tl.int64) * E + re[None, :], key,
             mask=mt[:, None] & me[None, :])


@triton.jit
def _weights_kernel(S, SEL, W, SSEL, SSUM, T, E, A,
                    BT: tl.constexpr, BA: tl.constexpr):
    """Combine weights: raw selected scores normalized by their own sum."""
    rt = tl.program_id(0) * BT + tl.arange(0, BT)
    ra = tl.arange(0, BA)
    mt = rt < T
    ma = ra < A
    m = mt[:, None] & ma[None, :]
    off = rt[:, None].to(tl.int64) * A + ra[None, :]
    sel = tl.load(SEL + off, mask=m, other=0).to(tl.int64)
    ssel = tl.load(S + rt[:, None].to(tl.int64) * E + sel, mask=m, other=0.0)
    ssum = tl.sum(tl.where(m, ssel, 0.0), axis=1)
    tl.store(W + off, ssel / ssum[:, None], mask=m)
    tl.store(SSEL + off, ssel, mask=m)
    tl.store(SSUM + rt, ssum, mask=mt)


@triton.jit
def _pairmeta_kernel(ORDER, W, TOKID, POS, WS, N, A, BLK: tl.constexpr):
    """Per-pair metadata in dispatch order: source token, inverse permutation
    and the gathered combine weight."""
    p = tl.program_id(0) * BLK + tl.arange(0, BLK)
    m = p < N
    o = tl.load(ORDER + p, mask=m, other=0).to(tl.int64)
    tl.store(TOKID + p, (o // A).to(tl.int32), mask=m)
    tl.store(POS + o, p.to(tl.int32), mask=m)
    tl.store(WS + p, tl.load(W + o, mask=m, other=0.0), mask=m)


@triton.jit
def _lower_bound(SFLAT, want, N, BE: tl.constexpr, LOGN: tl.constexpr):
    """First index into the sorted expert list whose value is >= want."""
    lo = tl.zeros((BE,), dtype=tl.int32)
    hi = tl.full((BE,), N, dtype=tl.int32)
    for _ in tl.static_range(LOGN):
        mid = (lo + hi) // 2
        v = tl.load(SFLAT + mid, mask=(mid < N) & (lo < hi), other=0x7FFFFFFF)
        take = v < want
        lo = tl.where((lo < hi) & take, mid + 1, lo)
        hi = tl.where((lo < hi) & (take == 0), mid, hi)
    return lo


@triton.jit
def _meta_kernel(SFLAT, ESTART, CUM, TOTAL, N, E, BM_G,
                 BE: tl.constexpr, LOGN: tl.constexpr):
    """Expert row ranges and the exclusive prefix sum of per-expert row-tile
    counts, from the expert-sorted pair list."""
    re = tl.arange(0, BE)
    lo = _lower_bound(SFLAT, re, N, BE, LOGN)
    hi = _lower_bound(SFLAT, re + 1, N, BE, LOGN)
    tl.store(ESTART + re, lo, mask=re <= E)
    ntile = tl.where(re < E, (hi - lo + BM_G - 1) // BM_G, 0)
    tl.store(CUM + re, tl.cumsum(ntile, axis=0) - ntile, mask=re < E)
    tl.store(TOTAL, tl.sum(ntile, axis=0))


@triton.jit
def _tilemap_kernel(CUM, TOTAL, TILE_E, TILE_M, NT, E,
                    BLK: tl.constexpr, LOGE: tl.constexpr):
    """Row-tile index -> (expert, tile within expert)."""
    t = tl.program_id(0) * BLK + tl.arange(0, BLK)
    m = t < NT
    total = tl.load(TOTAL)
    lo = tl.zeros((BLK,), dtype=tl.int32)
    hi = tl.full((BLK,), E, dtype=tl.int32)
    for _ in tl.static_range(LOGE):
        mid = (lo + hi) // 2
        v = tl.load(CUM + mid, mask=(mid < E) & (lo < hi), other=0)
        take = v <= t
        lo = tl.where((lo < hi) & take, mid + 1, lo)
        hi = tl.where((lo < hi) & (take == 0), mid, hi)
    e = lo - 1
    base = tl.load(CUM + tl.maximum(e, 0), mask=(e >= 0) & (e < E), other=0)
    tl.store(TILE_E + t, tl.where(t < total, e, -1), mask=m)
    tl.store(TILE_M + t, t - base, mask=m)


# --------------------------------------------------------------------------
# expert compute -- forward
# --------------------------------------------------------------------------

@triton.jit
def _gemm1_kernel(X, W1, GU, ACT, TOKID, TILE_E, TILE_M, ESTART,
                  D, F, sx0, sx1, sw0, sw1, sw2,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                  P: tl.constexpr):
    """gate|up = x[rows] @ w1[e]; both halves share one loaded x tile.

    The tile holds gate and up together, so silu(gate) * up is finished here
    and stored alongside them; the consumers that need only the activation
    then read F columns instead of 2F.
    """
    pid = tl.program_id(0)
    e = tl.load(TILE_E + pid)
    if e >= 0:
        r0 = tl.load(ESTART + e)
        r1 = tl.load(ESTART + e + 1)
        rows = r0 + tl.load(TILE_M + pid) * BM + tl.arange(0, BM)
        mr = rows < r1
        tok = tl.load(TOKID + rows, mask=mr, other=0).to(tl.int64)
        rn = tl.program_id(1) * BN + tl.arange(0, BN)
        mn = rn < F
        xp = X + tok[:, None] * sx0
        wg = W1 + e.to(tl.int64) * sw0 + rn[None, :] * sw2
        wu = wg + F * sw2
        accg = tl.zeros((BM, BN), dtype=tl.float32)
        accu = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, D, BK):
            rk = k0 + tl.arange(0, BK)
            mk = rk < D
            a = tl.load(xp + rk[None, :] * sx1,
                        mask=mr[:, None] & mk[None, :], other=0.0)
            bg = tl.load(wg + rk[:, None] * sw1,
                         mask=mk[:, None] & mn[None, :], other=0.0)
            bu = tl.load(wu + rk[:, None] * sw1,
                         mask=mk[:, None] & mn[None, :], other=0.0)
            accg = _ldot(a, bg, accg, P)
            accu = _ldot(a, bu, accu, P)
        m = mr[:, None] & mn[None, :]
        gp = GU + rows[:, None].to(tl.int64) * (2 * F) + rn[None, :]
        tl.store(gp, accg, mask=m)
        tl.store(gp + F, accu, mask=m)
        tl.store(ACT + rows[:, None].to(tl.int64) * F + rn[None, :],
                 (accg * tl.sigmoid(accg)) * accu, mask=m)


@triton.jit
def _gemm2_kernel(ACT, W2, YP, TILE_E, TILE_M, ESTART,
                  D, F, sw0, sw1, sw2,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                  P: tl.constexpr):
    """yp = act[rows] @ w2[e]."""
    pid = tl.program_id(0)
    e = tl.load(TILE_E + pid)
    if e >= 0:
        r0 = tl.load(ESTART + e)
        r1 = tl.load(ESTART + e + 1)
        rows = r0 + tl.load(TILE_M + pid) * BM + tl.arange(0, BM)
        mr = rows < r1
        rn = tl.program_id(1) * BN + tl.arange(0, BN)
        mn = rn < D
        ap = ACT + rows[:, None].to(tl.int64) * F
        wp = W2 + e.to(tl.int64) * sw0 + rn[None, :] * sw2
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, F, BK):
            rk = k0 + tl.arange(0, BK)
            mk = rk < F
            a = tl.load(ap + rk[None, :],
                        mask=mr[:, None] & mk[None, :], other=0.0)
            b = tl.load(wp + rk[:, None] * sw1,
                        mask=mk[:, None] & mn[None, :], other=0.0)
            acc = _ldot(a, b, acc, P)
        tl.store(YP + rows[:, None].to(tl.int64) * D + rn[None, :], acc,
                 mask=mr[:, None] & mn[None, :])


@triton.jit
def _combine_kernel(YP, W, POS, Y, T, D, A, sy0,
                    BT: tl.constexpr, BN: tl.constexpr):
    """y[t] = sum_a w[t, a] * yp[pair(t, a)], summed in slot order."""
    rt = tl.program_id(0) * BT + tl.arange(0, BT)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    mt = rt < T
    mn = rn < D
    acc = tl.zeros((BT, BN), dtype=tl.float32)
    for a in range(0, A):
        off = rt.to(tl.int64) * A + a
        p = tl.load(POS + off, mask=mt, other=0).to(tl.int64)
        wv = tl.load(W + off, mask=mt, other=0.0)
        v = tl.load(YP + p[:, None] * D + rn[None, :],
                    mask=mt[:, None] & mn[None, :], other=0.0)
        acc += wv[:, None] * v
    tl.store(Y + rt[:, None].to(tl.int64) * sy0 + rn[None, :],
             acc.to(Y.dtype.element_ty), mask=mt[:, None] & mn[None, :])


# --------------------------------------------------------------------------
# expert compute -- backward
# --------------------------------------------------------------------------

@triton.jit
def _dwv_kernel(DY, YP, TOKID, DWV, N, D, sd0,
                BM: tl.constexpr, BK: tl.constexpr):
    """dwv[p] = <dy[token(p)], yp[p]>, the cotangent of the combine weight."""
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    mr = rows < N
    tok = tl.load(TOKID + rows, mask=mr, other=0).to(tl.int64)
    acc = tl.zeros((BM, BK), dtype=tl.float32)
    for k0 in range(0, D, BK):
        rk = k0 + tl.arange(0, BK)
        m = mr[:, None] & (rk[None, :] < D)
        dyv = tl.load(DY + tok[:, None] * sd0 + rk[None, :],
                      mask=m, other=0.0).to(tl.float32)
        ypv = tl.load(YP + rows[:, None].to(tl.int64) * D + rk[None, :],
                      mask=m, other=0.0)
        acc += dyv * ypv
    tl.store(DWV + rows, tl.sum(acc, axis=1), mask=mr)


@triton.jit
def _dact_kernel(DY, W2, GU, DGU, TOKID, WS, TILE_E, TILE_M, ESTART,
                 D, F, sd0, sw0, sw1, sw2,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                 P: tl.constexpr):
    """dgate|dup from dact = w * (dy[token] @ w2[e]^T) and the SwiGLU rule.

    The combine weight scales the finished row rather than the dy tile, which
    keeps dy an unmodified input operand inside the reduction.
    """
    pid = tl.program_id(0)
    e = tl.load(TILE_E + pid)
    if e >= 0:
        r0 = tl.load(ESTART + e)
        r1 = tl.load(ESTART + e + 1)
        rows = r0 + tl.load(TILE_M + pid) * BM + tl.arange(0, BM)
        mr = rows < r1
        tok = tl.load(TOKID + rows, mask=mr, other=0).to(tl.int64)
        wsv = tl.load(WS + rows, mask=mr, other=0.0)
        rn = tl.program_id(1) * BN + tl.arange(0, BN)
        mn = rn < F
        dyp = DY + tok[:, None] * sd0
        wp = W2 + e.to(tl.int64) * sw0 + rn[None, :] * sw1
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, D, BK):
            rk = k0 + tl.arange(0, BK)
            mk = rk < D
            a = tl.load(dyp + rk[None, :],
                        mask=mr[:, None] & mk[None, :], other=0.0)
            b = tl.load(wp + rk[:, None] * sw2,
                        mask=mk[:, None] & mn[None, :], other=0.0)
            acc = _ldot(a, b, acc, P)
        acc = acc * wsv[:, None]
        gp = GU + rows[:, None].to(tl.int64) * (2 * F) + rn[None, :]
        m = mr[:, None] & mn[None, :]
        g = tl.load(gp, mask=m, other=0.0)
        u = tl.load(gp + F, mask=m, other=0.0)
        sg = tl.sigmoid(g)
        silu = g * sg
        dp = DGU + rows[:, None].to(tl.int64) * (2 * F) + rn[None, :]
        tl.store(dp, acc * u * (sg + silu * (1.0 - sg)), mask=m)
        tl.store(dp + F, acc * silu, mask=m)


@triton.jit
def _dw2_kernel(ACT, DY, TOKID, WS, ESTART, DW2, D, F, sd0, sw0, sw1, sw2,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                P: tl.constexpr):
    """dw2[e] = (w * act[rows])^T @ dy[token(rows)] over the expert's rows.

    The combine weight rides on the activation, which is a float32 temporary
    already, so dy stays an unmodified input operand.
    """
    e = tl.program_id(2)
    r0 = tl.load(ESTART + e)
    r1 = tl.load(ESTART + e + 1)
    rm = tl.program_id(1) * BM + tl.arange(0, BM)
    rn = tl.program_id(0) * BN + tl.arange(0, BN)
    mm = rm < F
    mn = rn < D
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(r0, r1, BK):
        rows = k0 + tl.arange(0, BK)
        mk = rows < r1
        wsv = tl.load(WS + rows, mask=mk, other=0.0)
        a = tl.load(ACT + rows[:, None].to(tl.int64) * F + rm[None, :],
                    mask=mk[:, None] & mm[None, :], other=0.0) * wsv[:, None]
        tok = tl.load(TOKID + rows, mask=mk, other=0).to(tl.int64)
        b = tl.load(DY + tok[:, None] * sd0 + rn[None, :],
                    mask=mk[:, None] & mn[None, :], other=0.0)
        acc = _ldot(tl.trans(a), b, acc, P)
    tl.store(DW2 + e.to(tl.int64) * sw0 + rm[:, None] * sw1 + rn[None, :] * sw2,
             acc.to(DW2.dtype.element_ty), mask=mm[:, None] & mn[None, :])


@triton.jit
def _dxg_kernel(DGU, W1, DXP, TILE_E, TILE_M, ESTART,
                D, F, sw0, sw1, sw2,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                P: tl.constexpr):
    """dx contribution of one pair: dgu[p] @ w1[e]^T."""
    pid = tl.program_id(0)
    e = tl.load(TILE_E + pid)
    if e >= 0:
        r0 = tl.load(ESTART + e)
        r1 = tl.load(ESTART + e + 1)
        rows = r0 + tl.load(TILE_M + pid) * BM + tl.arange(0, BM)
        mr = rows < r1
        rn = tl.program_id(1) * BN + tl.arange(0, BN)
        mn = rn < D
        dp = DGU + rows[:, None].to(tl.int64) * (2 * F)
        wp = W1 + e.to(tl.int64) * sw0 + rn[None, :] * sw1
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, 2 * F, BK):
            rk = k0 + tl.arange(0, BK)
            mk = rk < 2 * F
            a = tl.load(dp + rk[None, :],
                        mask=mr[:, None] & mk[None, :], other=0.0)
            b = tl.load(wp + rk[:, None] * sw2,
                        mask=mk[:, None] & mn[None, :], other=0.0)
            acc = _ldot(a, b, acc, P)
        tl.store(DXP + rows[:, None].to(tl.int64) * D + rn[None, :], acc,
                 mask=mr[:, None] & mn[None, :])


@triton.jit
def _dw1_kernel(X, DGU, TOKID, ESTART, DW1, D, F, sx0, sx1, sw0, sw1, sw2,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                P: tl.constexpr):
    """dw1[e] = x[token(rows)]^T @ dgu[rows] over the expert's rows."""
    e = tl.program_id(2)
    r0 = tl.load(ESTART + e)
    r1 = tl.load(ESTART + e + 1)
    rm = tl.program_id(1) * BM + tl.arange(0, BM)
    rn = tl.program_id(0) * BN + tl.arange(0, BN)
    mm = rm < D
    mn = rn < 2 * F
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(r0, r1, BK):
        rows = k0 + tl.arange(0, BK)
        mk = rows < r1
        tok = tl.load(TOKID + rows, mask=mk, other=0).to(tl.int64)
        a = tl.load(X + tok[:, None] * sx0 + rm[None, :] * sx1,
                    mask=mk[:, None] & mm[None, :], other=0.0)
        b = tl.load(DGU + rows[:, None].to(tl.int64) * (2 * F) + rn[None, :],
                    mask=mk[:, None] & mn[None, :], other=0.0)
        acc = _ldot(tl.trans(a), b, acc, P)
    tl.store(DW1 + e.to(tl.int64) * sw0 + rm[:, None] * sw1 + rn[None, :] * sw2,
             acc.to(DW1.dtype.element_ty), mask=mm[:, None] & mn[None, :])


@triton.jit
def _dlogit_kernel(DWV, POS, W, SSEL, SSUM, SEL, DLOG, T, E, A,
                   BT: tl.constexpr, BA: tl.constexpr):
    """Router-logit cotangent: through the weight normalization and the
    sigmoid, scattered to the selected experts only."""
    rt = tl.program_id(0) * BT + tl.arange(0, BT)
    ra = tl.arange(0, BA)
    mt = rt < T
    ma = ra < A
    m = mt[:, None] & ma[None, :]
    off = rt[:, None].to(tl.int64) * A + ra[None, :]
    p = tl.load(POS + off, mask=m, other=0).to(tl.int64)
    dwv = tl.load(DWV + p, mask=m, other=0.0)
    wv = tl.load(W + off, mask=m, other=0.0)
    inner = tl.sum(tl.where(m, dwv * wv, 0.0), axis=1)
    ssum = tl.load(SSUM + rt, mask=mt, other=1.0)
    ssel = tl.load(SSEL + off, mask=m, other=0.0)
    dl = (dwv - inner[:, None]) / ssum[:, None] * ssel * (1.0 - ssel)
    sel = tl.load(SEL + off, mask=m, other=0).to(tl.int64)
    tl.store(DLOG + rt[:, None].to(tl.int64) * E + sel, dl, mask=m)


@triton.jit
def _drw_kernel(X, DLOG, DRW, T, D, E, sx0, sx1,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                P: tl.constexpr):
    """d router_w = x^T @ dlogit. dlogit is a difference of comparable terms,
    so this reduction loses leading digits and carries the tightest operand
    precision of any GEMM here."""
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.arange(0, BN)
    mm = rm < D
    mn = rn < E
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, T, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < T
        a = tl.load(X + rk[:, None].to(tl.int64) * sx0 + rm[None, :] * sx1,
                    mask=mk[:, None] & mm[None, :], other=0.0)
        b = tl.load(DLOG + rk[:, None].to(tl.int64) * E + rn[None, :],
                    mask=mk[:, None] & mn[None, :], other=0.0)
        acc = _ldot(tl.trans(a), b, acc, P)
    tl.store(DRW + rm[:, None] * E + rn[None, :], acc,
             mask=mm[:, None] & mn[None, :])


@triton.jit
def _dxcombine_kernel(DXP, POS, DLOG, RW, DX, T, D, E, A, sr0, sr1, sd0,
                      BT: tl.constexpr, BN: tl.constexpr, BE: tl.constexpr,
                      P: tl.constexpr):
    """dx = sum_a dxp[pair(t, a)] + dlogit @ router_w^T, the per-pair terms
    summed in slot order."""
    rt = tl.program_id(0) * BT + tl.arange(0, BT)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    mt = rt < T
    mn = rn < D
    acc = tl.zeros((BT, BN), dtype=tl.float32)
    for a in range(0, A):
        off = rt.to(tl.int64) * A + a
        p = tl.load(POS + off, mask=mt, other=0).to(tl.int64)
        acc += tl.load(DXP + p[:, None] * D + rn[None, :],
                       mask=mt[:, None] & mn[None, :], other=0.0)
    for k0 in range(0, E, BE):
        rk = k0 + tl.arange(0, BE)
        mk = rk < E
        dl = tl.load(DLOG + rt[:, None].to(tl.int64) * E + rk[None, :],
                     mask=mt[:, None] & mk[None, :], other=0.0)
        rw = tl.load(RW + rn[:, None] * sr0 + rk[None, :] * sr1,
                     mask=mn[:, None] & mk[None, :], other=0.0)
        acc = _ldot(dl, tl.trans(rw), acc, P)
    tl.store(DX + rt[:, None].to(tl.int64) * sd0 + rn[None, :],
             acc.to(DX.dtype.element_ty), mask=mt[:, None] & mn[None, :])


# --------------------------------------------------------------------------
# host side
# --------------------------------------------------------------------------

def _p2(n):
    return 1 << max(0, int(n) - 1).bit_length()


# Tile shapes per GEMM, as (BM, BN, BK, num_warps, num_stages), indexed by
# input dtype. The float32 rows carry three bfloat16 limbs per operand, so
# their operand tiles and pipeline depth are sized against the same 227 KB
# shared-memory budget as the single-limb bfloat16 rows.
_CFG = {
    "gemm1": {True: (128, 64, 64, 8, 3), False: (128, 128, 32, 8, 4)},
    "gemm2": {True: (128, 128, 64, 8, 3), False: (128, 256, 32, 8, 4)},
    "dact":  {True: (128, 128, 64, 8, 3), False: (128, 64, 64, 4, 4)},
    "dxg":   {True: (128, 128, 64, 8, 3), False: (128, 256, 64, 8, 4)},
    "dw1":   {True: (64, 128, 64, 4, 4), False: (256, 64, 64, 8, 2)},
    "dw2":   {True: (128, 128, 64, 8, 3), False: (64, 256, 32, 4, 4)},
}


def _tiles(name, f32, *caps):
    """Tile config for one GEMM, shrunk so a tile never dwarfs its axis."""
    bm, bn, bk, nw, ns = _CFG[name][f32]
    bm = min(bm, max(16, _p2(caps[0])))
    bn = min(bn, max(16, _p2(caps[1])))
    return bm, bn, bk, nw, ns


class _FusedMoe(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, router_w, router_b, w1, w2, A):
        A = int(A)
        T, D = x.shape
        E = router_w.shape[1]
        F = w1.shape[2] // 2
        N = T * A
        dev = x.device
        f32 = x.dtype == torch.float32
        # A float32 operand needs three bfloat16 limbs to be float32-effective;
        # a bfloat16 input is one exact limb, and the float32 intermediates
        # feeding the weight gradients still need two.
        p_ii = P_X6 if f32 else P_HH    # input x input
        p_ti = P_X6 if f32 else P_AL    # intermediate x input
        p_it = P_X6 if f32 else P_BL    # input x intermediate

        s = torch.empty(T, E, device=dev, dtype=torch.float32)
        key = torch.empty(T, E, device=dev, dtype=torch.int64)
        BE = max(16, _p2(E))
        BT = 64 if T >= 64 else max(16, _p2(T))
        _router_kernel[(triton.cdiv(T, BT),)](
            x, router_w, router_b, s, key, T, D, E,
            x.stride(0), x.stride(1), router_w.stride(0), router_w.stride(1),
            BT=BT, BE=BE, BD=32, P=P_X6, num_warps=4, num_stages=2)

        sel = torch.topk(key, A, dim=-1, largest=True, sorted=True).indices

        w = torch.empty(T, A, device=dev, dtype=torch.float32)
        ssel = torch.empty(T, A, device=dev, dtype=torch.float32)
        ssum = torch.empty(T, device=dev, dtype=torch.float32)
        _weights_kernel[(triton.cdiv(T, BT),)](
            s, sel, w, ssel, ssum, T, E, A,
            BT=BT, BA=max(1, _p2(A)), num_warps=4)

        sflat, order = torch.sort(sel.reshape(-1).to(torch.int32), stable=True)

        tokid = torch.empty(N, device=dev, dtype=torch.int32)
        pos = torch.empty(N, device=dev, dtype=torch.int32)
        ws = torch.empty(N, device=dev, dtype=torch.float32)
        _pairmeta_kernel[(triton.cdiv(N, 256),)](
            order, w, tokid, pos, ws, N, A, BLK=256, num_warps=4)

        # One row-tile height serves every dispatch-ordered GEMM so the tile
        # map is built once.
        BMG = _tiles("gemm1", f32, N, F)[0]
        estart = torch.empty(E + 1, device=dev, dtype=torch.int32)
        cum = torch.empty(E, device=dev, dtype=torch.int32)
        total = torch.empty(1, device=dev, dtype=torch.int32)
        _meta_kernel[(1,)](sflat, estart, cum, total, N, E, BMG,
                           BE=max(16, _p2(E + 1)),
                           LOGN=int(N).bit_length() + 1, num_warps=4)

        NT = triton.cdiv(N, BMG) + E
        tile_e = torch.empty(NT, device=dev, dtype=torch.int32)
        tile_m = torch.empty(NT, device=dev, dtype=torch.int32)
        _tilemap_kernel[(triton.cdiv(NT, 128),)](
            cum, total, tile_e, tile_m, NT, E, BLK=128,
            LOGE=int(E).bit_length() + 1, num_warps=4)

        gu = torch.empty(N, 2 * F, device=dev, dtype=torch.float32)
        act = torch.empty(N, F, device=dev, dtype=torch.float32)
        _, bn, bk, nw, ns = _tiles("gemm1", f32, N, F)
        _gemm1_kernel[(NT, triton.cdiv(F, bn))](
            x, w1, gu, act, tokid, tile_e, tile_m, estart, D, F,
            x.stride(0), x.stride(1), w1.stride(0), w1.stride(1), w1.stride(2),
            BM=BMG, BN=bn, BK=bk, P=p_ii, num_warps=nw, num_stages=ns)

        yp = torch.empty(N, D, device=dev, dtype=torch.float32)
        _, bn, bk, nw, ns = _tiles("gemm2", f32, N, D)
        _gemm2_kernel[(NT, triton.cdiv(D, bn))](
            act, w2, yp, tile_e, tile_m, estart, D, F,
            w2.stride(0), w2.stride(1), w2.stride(2),
            BM=BMG, BN=bn, BK=bk, P=p_ti, num_warps=nw, num_stages=ns)

        y = torch.empty(T, D, device=dev, dtype=x.dtype)
        BNC = 64 if D >= 64 else max(16, _p2(D))
        _combine_kernel[(triton.cdiv(T, BT), triton.cdiv(D, BNC))](
            yp, w, pos, y, T, D, A, y.stride(0), BT=BT, BN=BNC, num_warps=4)

        ctx.save_for_backward(x, router_w, w1, w2, gu, act, yp, w, ssel, ssum,
                              sel, tokid, pos, ws, estart, tile_e, tile_m)
        ctx.meta = (T, D, F, E, A, N, NT, BMG, f32, p_ii, p_ti, p_it)
        return y

    @staticmethod
    def backward(ctx, dy):
        (x, router_w, w1, w2, gu, act, yp, w, ssel, ssum, sel,
         tokid, pos, ws, estart, tile_e, tile_m) = ctx.saved_tensors
        T, D, F, E, A, N, NT, BMG, f32, p_ii, p_ti, p_it = ctx.meta
        dev = x.device
        dy = dy.contiguous()

        dwv = torch.empty(N, device=dev, dtype=torch.float32)
        _dwv_kernel[(triton.cdiv(N, 128),)](
            dy, yp, tokid, dwv, N, D, dy.stride(0), BM=128, BK=64, num_warps=4)

        dgu = torch.empty(N, 2 * F, device=dev, dtype=torch.float32)
        _, bn, bk, nw, ns = _tiles("dact", f32, N, F)
        _dact_kernel[(NT, triton.cdiv(F, bn))](
            dy, w2, gu, dgu, tokid, ws, tile_e, tile_m, estart, D, F,
            dy.stride(0), w2.stride(0), w2.stride(1), w2.stride(2),
            BM=BMG, BN=bn, BK=bk, P=p_ii, num_warps=nw, num_stages=ns)

        dw2 = torch.empty_like(w2)
        bm, bn, bk, nw, ns = _tiles("dw2", f32, F, D)
        _dw2_kernel[(triton.cdiv(D, bn), triton.cdiv(F, bm), E)](
            act, dy, tokid, ws, estart, dw2, D, F, dy.stride(0),
            dw2.stride(0), dw2.stride(1), dw2.stride(2),
            BM=bm, BN=bn, BK=bk, P=p_ti, num_warps=nw, num_stages=ns)

        dxp = torch.empty(N, D, device=dev, dtype=torch.float32)
        _, bn, bk, nw, ns = _tiles("dxg", f32, N, D)
        _dxg_kernel[(NT, triton.cdiv(D, bn))](
            dgu, w1, dxp, tile_e, tile_m, estart, D, F,
            w1.stride(0), w1.stride(1), w1.stride(2),
            BM=BMG, BN=bn, BK=bk, P=p_ti, num_warps=nw, num_stages=ns)

        dw1 = torch.empty_like(w1)
        bm, bn, bk, nw, ns = _tiles("dw1", f32, D, 2 * F)
        _dw1_kernel[(triton.cdiv(2 * F, bn), triton.cdiv(D, bm), E)](
            x, dgu, tokid, estart, dw1, D, F, x.stride(0), x.stride(1),
            dw1.stride(0), dw1.stride(1), dw1.stride(2),
            BM=bm, BN=bn, BK=bk, P=p_it, num_warps=nw, num_stages=ns)

        dlog = torch.zeros(T, E, device=dev, dtype=torch.float32)
        BT = 64 if T >= 64 else max(16, _p2(T))
        _dlogit_kernel[(triton.cdiv(T, BT),)](
            dwv, pos, w, ssel, ssum, sel, dlog, T, E, A,
            BT=BT, BA=max(1, _p2(A)), num_warps=4)

        drw = torch.empty(D, E, device=dev, dtype=torch.float32)
        _drw_kernel[(triton.cdiv(D, 64),)](
            x, dlog, drw, T, D, E, x.stride(0), x.stride(1),
            BM=64, BN=max(16, _p2(E)), BK=64, P=P_X6, num_warps=4, num_stages=2)

        dx = torch.empty(T, D, device=dev, dtype=x.dtype)
        BNC = 64 if D >= 64 else max(16, _p2(D))
        _dxcombine_kernel[(triton.cdiv(T, BT), triton.cdiv(D, BNC))](
            dxp, pos, dlog, router_w, dx, T, D, E, A,
            router_w.stride(0), router_w.stride(1), dx.stride(0),
            BT=BT, BN=BNC, BE=max(16, _p2(E)), P=P_X6,
            num_warps=4, num_stages=2)

        return (dx, drw.to(router_w.dtype), None, dw1, dw2, None)


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
