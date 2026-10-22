"""Routed fused mixture-of-experts layer, forward and backward.

Forward pipeline per call:

  router GEMM -> sigmoid scores + order-preserving int64 selection keys
  top-A on the keys -> per-token expert set
  normalize the raw selected scores -> combine weights
  stable sort of the [T*A] pair list by expert -> contiguous per-expert runs
  tiled expert GEMMs over those runs (SwiGLU), gathering rows on the fly
  gather-combine of the A partial outputs per token

The backward mirrors it: activation-gradient GEMMs over the same runs,
per-expert weight-gradient GEMMs whose reduction axis is the run, and a dense
[T, E] logit gradient scattered from the A selected slots per token that then
feeds two small router GEMMs.

The float32-valued intermediates that a matmul re-reads -- the SwiGLU
activation and the hidden gradient -- are stored as an exact bfloat16
high/low limb pair. A dot against such a pair takes the high limb alone in
the two activation-gradient GEMMs, both limbs in the two weight-gradient
GEMMs, and three limb products when the other operand is float32 as well.
"""

import torch
import triton
import triton.language as tl


def _npow2(n: int) -> int:
    n = int(n)
    return 1 << (n - 1).bit_length() if n > 1 else 1


@triton.jit
def _split(v):
    """Exact bfloat16 high/low decomposition of a float32 tile."""
    hi = v.to(tl.bfloat16)
    lo = (v - hi.to(tl.float32)).to(tl.bfloat16)
    return hi, lo


@triton.jit
def _swizzle(pid, nm, nn, GROUP: tl.constexpr):
    """Flat program id -> (row tile, column tile) in row-groups of GROUP.

    Walking a group of GROUP row tiles across all column tiles before moving
    on keeps both operands of the tile resident in L2 for the whole group.
    """
    per = GROUP * nn
    gid = pid // per
    first = gid * GROUP
    gsz = min(nm - first, GROUP)
    return first + ((pid % per) % gsz), (pid % per) // gsz


@triton.jit
def _dot_raw(a, b, acc, FP32: tl.constexpr):
    """Dot of two operands held in the input dtype."""
    if FP32:
        ah, al = _split(a)
        bh, bl = _split(b)
        acc = tl.dot(ah, bh, acc)
        acc = tl.dot(al, bh, acc)
        acc = tl.dot(ah, bl, acc)
    else:
        acc = tl.dot(a, b, acc)
    return acc


@triton.jit
def _dot_lim(ah, al, b, acc, FP32: tl.constexpr, TWO: tl.constexpr):
    """Dot of a stored limb pair against an input-dtype operand."""
    if FP32:
        bh, bl = _split(b)
        acc = tl.dot(ah, bh, acc)
        acc = tl.dot(al, bh, acc)
        acc = tl.dot(ah, bl, acc)
    else:
        acc = tl.dot(ah, b, acc)
        if TWO:
            acc = tl.dot(al, b, acc)
    return acc


# ---------------------------------------------------------------- routing ---

@triton.jit
def _router_kernel(X, RW, RB, S, KEY, T, D, E,
                   BT: tl.constexpr, BE: tl.constexpr, BK: tl.constexpr,
                   FP32: tl.constexpr, SHIFT: tl.constexpr):
    """scores = sigmoid(x @ router_w), plus the packed selection keys.

    The key is the biased score's order-preserving integer image shifted left
    by SHIFT (at least the bit width of E-1) OR'd with the inverted expert
    index, so descending key order is (score descending, index ascending) and
    every key in a row is distinct.
    """
    pid_t = tl.program_id(0)
    pid_e = tl.program_id(1)
    rt = pid_t * BT + tl.arange(0, BT)
    re = pid_e * BE + tl.arange(0, BE)
    mt = rt < T
    me = re < E
    acc = tl.zeros((BT, BE), tl.float32)
    xp = X + rt[:, None].to(tl.int64) * D
    wp = RW + re[None, :]
    for k0 in range(0, D, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < D
        a = tl.load(xp + rk[None, :], mask=mt[:, None] & mk[None, :], other=0.0)
        b = tl.load(wp + rk[:, None].to(tl.int64) * E,
                    mask=mk[:, None] & me[None, :], other=0.0)
        if FP32:
            acc = tl.dot(a, b, acc, input_precision="tf32x3")
        else:
            acc = tl.dot(a, b, acc)
    s = tl.sigmoid(acc)
    m = mt[:, None] & me[None, :]
    tl.store(S + rt[:, None].to(tl.int64) * E + re[None, :], s, mask=m)

    b = s + tl.load(RB + re, mask=me, other=0.0)[None, :]
    b = tl.where(b == 0.0, 0.0, b)
    ib = b.to(tl.int32, bitcast=True).to(tl.int64)
    u = tl.where(ib >= 0, ib ^ 0x80000000, ~ib)
    key = (u << SHIFT) | (E - 1 - re)[None, :].to(tl.int64)
    tl.store(KEY + rt[:, None].to(tl.int64) * E + re[None, :], key, mask=m)


@triton.jit
def _wgt_kernel(S, SEL, WGT, SSUM, T, E, A,
                BT: tl.constexpr, BA: tl.constexpr):
    """Combine weights: raw selected scores divided by their sum."""
    pid = tl.program_id(0)
    rt = pid * BT + tl.arange(0, BT)
    ra = tl.arange(0, BA)
    mt = rt < T
    m = mt[:, None] & (ra < A)[None, :]
    off = rt[:, None].to(tl.int64) * A + ra[None, :]
    sel = tl.load(SEL + off, mask=m, other=0)
    s = tl.load(S + rt[:, None].to(tl.int64) * E + sel, mask=m, other=0.0)
    ssum = tl.sum(s, axis=1)
    tl.store(WGT + off, s / ssum[:, None], mask=m)
    tl.store(SSUM + rt, ssum, mask=mt)


@triton.jit
def _dispatch_kernel(ORDER, WGT, SROW, SW, POS, N, A, BLK: tl.constexpr):
    """Sorted-slot metadata: source row, combine weight, inverse permutation."""
    pid = tl.program_id(0)
    i = pid * BLK + tl.arange(0, BLK)
    m = i < N
    o = tl.load(ORDER + i, mask=m, other=0)
    tl.store(SROW + i, (o // A).to(tl.int32), mask=m)
    tl.store(SW + i, tl.load(WGT + o, mask=m, other=0.0), mask=m)
    tl.store(POS + o, i.to(tl.int32), mask=m)


@triton.jit
def _bounds_kernel(VALS, OFFS, N, E, BLK: tl.constexpr):
    """OFFS[e] = first position of expert e in the expert-sorted slot list."""
    pid = tl.program_id(0)
    e = pid * BLK + tl.arange(0, BLK)
    m = e <= E
    lo = tl.zeros((BLK,), tl.int32)
    hi = tl.full((BLK,), N, tl.int32)
    for _ in range(32):
        act = lo < hi
        mid = tl.where(act, (lo + hi) // 2, lo)
        v = tl.load(VALS + mid, mask=m & act, other=0).to(tl.int32)
        less = v < e
        lo = tl.where(act & less, mid + 1, lo)
        hi = tl.where(act & (less == 0), mid, hi)
    tl.store(OFFS + e, lo, mask=m)


@triton.jit
def _tilemap_kernel(OFFS, TILE, NTMAX, E, BM: tl.constexpr, BE: tl.constexpr,
                    BLK: tl.constexpr):
    """Per-tile (expert, first slot, row count); expert -1 past the last tile.

    Each program re-derives the exclusive tile-count prefix sum over experts
    with one cumsum, then inverts it for its own block of tiles by counting
    how many prefixes each tile index passes.
    """
    e = tl.arange(0, BE)
    me = e < E
    o0 = tl.load(OFFS + e, mask=me, other=0)
    o1 = tl.load(OFFS + e + 1, mask=me, other=0)
    nt = tl.where(me, (o1 - o0 + BM - 1) // BM, 0)
    cum = tl.cumsum(nt, axis=0) - nt
    total = tl.sum(nt, axis=0)
    i = tl.program_id(0) * BLK + tl.arange(0, BLK)
    mi = i < NTMAX
    live = mi & (i < total)
    pick = tl.sum(tl.where((cum[:, None] <= i[None, :]) & me[:, None], 1, 0),
                  axis=0) - 1
    pick = tl.where(live, pick, 0)
    hit = e[:, None] == pick[None, :]
    base = tl.sum(tl.where(hit, cum[:, None], 0), axis=0)
    s0 = tl.sum(tl.where(hit, o0[:, None], 0), axis=0)
    s1 = tl.sum(tl.where(hit, o1[:, None], 0), axis=0)
    start = s0 + (i - base) * BM
    tl.store(TILE + i * 4, tl.where(live, pick, -1), mask=mi)
    tl.store(TILE + i * 4 + 1, start, mask=mi)
    tl.store(TILE + i * 4 + 2, tl.minimum(s1 - start, BM), mask=mi)


# ------------------------------------------------------------- expert fwd ---

@triton.jit
def _gemm1_kernel(X, W1, SROW, SW, TILE, H, ACTH, ACTL, D, F, F2, NT,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                  FP32: tl.constexpr, GROUP: tl.constexpr):
    """gate|up projections of the gathered rows, and w * silu(gate) * up.

    The combine weight is folded into the stored activation limbs, so both
    readers of the activation (the second projection and its weight gradient)
    consume the weighted value with no further scaling.
    """
    pid_m, pid_n = _swizzle(tl.program_id(0), NT, tl.cdiv(F, BN), GROUP)
    e = tl.load(TILE + pid_m * 4)
    if e >= 0:
        s0 = tl.load(TILE + pid_m * 4 + 1)
        nrow = tl.load(TILE + pid_m * 4 + 2)
        rm = tl.arange(0, BM)
        mrow = rm < nrow
        rows = tl.load(SROW + s0 + rm, mask=mrow, other=0).to(tl.int64)
        rn = pid_n * BN + tl.arange(0, BN)
        mn = rn < F
        wp = W1 + e.to(tl.int64) * (D * F2) + rn[None, :]
        xp = X + rows[:, None] * D
        accg = tl.zeros((BM, BN), tl.float32)
        accu = tl.zeros((BM, BN), tl.float32)
        for k0 in range(0, D, BK):
            rk = k0 + tl.arange(0, BK)
            mk = rk < D
            a = tl.load(xp + rk[None, :], mask=mrow[:, None] & mk[None, :],
                        other=0.0)
            bp = wp + rk[:, None].to(tl.int64) * F2
            mb = mk[:, None] & mn[None, :]
            accg = _dot_raw(a, tl.load(bp, mask=mb, other=0.0), accg, FP32)
            accu = _dot_raw(a, tl.load(bp + F, mask=mb, other=0.0), accu, FP32)
        slot = (s0 + rm)[:, None].to(tl.int64)
        ms = mrow[:, None] & mn[None, :]
        hp = H + slot * F2 + rn[None, :]
        tl.store(hp, accg, mask=ms)
        tl.store(hp + F, accu, mask=ms)
        sw = tl.load(SW + s0 + rm, mask=mrow, other=0.0)
        ah, al = _split((accg * tl.sigmoid(accg)) * accu * sw[:, None])
        tl.store(ACTH + slot * F + rn[None, :], ah, mask=ms)
        tl.store(ACTL + slot * F + rn[None, :], al, mask=ms)


@triton.jit
def _gemm2_kernel(ACTH, ACTL, W2, TILE, FP, F, D, NT,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                  FP32: tl.constexpr, GROUP: tl.constexpr):
    """Expert output projection of the already weighted activation."""
    pid_m, pid_n = _swizzle(tl.program_id(0), NT, tl.cdiv(D, BN), GROUP)
    e = tl.load(TILE + pid_m * 4)
    if e >= 0:
        s0 = tl.load(TILE + pid_m * 4 + 1)
        nrow = tl.load(TILE + pid_m * 4 + 2)
        rm = tl.arange(0, BM)
        mrow = rm < nrow
        slot = (s0 + rm).to(tl.int64)
        rn = pid_n * BN + tl.arange(0, BN)
        mn = rn < D
        ap = slot[:, None] * F
        wp = W2 + e.to(tl.int64) * (F * D) + rn[None, :]
        acc = tl.zeros((BM, BN), tl.float32)
        for k0 in range(0, F, BK):
            rk = k0 + tl.arange(0, BK)
            mk = rk < F
            ma = mrow[:, None] & mk[None, :]
            ah = tl.load(ACTH + ap + rk[None, :], mask=ma, other=0.0)
            al = tl.load(ACTL + ap + rk[None, :], mask=ma, other=0.0)
            b = tl.load(wp + rk[:, None].to(tl.int64) * D,
                        mask=mk[:, None] & mn[None, :], other=0.0)
            acc = _dot_lim(ah, al, b, acc, FP32, False)
        tl.store(FP + slot[:, None] * D + rn[None, :],
                 acc.to(FP.dtype.element_ty),
                 mask=mrow[:, None] & mn[None, :])


@triton.jit
def _combine_kernel(FP, POS, Y, T, D, A, BT: tl.constexpr, BN: tl.constexpr):
    """y[t] = sum over the token's A slots of its weighted expert output."""
    pid_t = tl.program_id(0)
    pid_n = tl.program_id(1)
    rt = pid_t * BT + tl.arange(0, BT)
    rn = pid_n * BN + tl.arange(0, BN)
    mt = rt < T
    mn = rn < D
    m = mt[:, None] & mn[None, :]
    acc = tl.zeros((BT, BN), tl.float32)
    for a in range(A):
        p = tl.load(POS + rt * A + a, mask=mt, other=0).to(tl.int64)
        acc += tl.load(FP + p[:, None] * D + rn[None, :], mask=m, other=0.0)
    tl.store(Y + rt[:, None].to(tl.int64) * D + rn[None, :],
             acc.to(Y.dtype.element_ty), mask=m)


# ------------------------------------------------------------- expert bwd ---

@triton.jit
def _dhalf(acc, ACTH, ACTL, H, DHH, DHL, sw, slot, rn, mrow, mn, F, F2,
           DWP, N, blk):
    """SwiGLU backward for one column block of q, plus its <act, q> partial."""
    ms = mrow[:, None] & mn[None, :]
    ap = slot[:, None] * F + rn[None, :]
    av = (tl.load(ACTH + ap, mask=ms, other=0.0).to(tl.float32)
          + tl.load(ACTL + ap, mask=ms, other=0.0).to(tl.float32))
    # av carries the combine weight, so this row sum is w * d(loss)/d(w)
    tl.store(DWP + blk * N + slot, tl.sum(acc * av, axis=1), mask=mrow)
    da = acc * sw[:, None]
    hp = H + slot[:, None] * F2 + rn[None, :]
    g = tl.load(hp, mask=ms, other=0.0)
    u = tl.load(hp + F, mask=ms, other=0.0)
    sg = tl.sigmoid(g)
    silu = g * sg
    gh, gl = _split(da * u * (sg + silu * (1.0 - sg)))
    dp = slot[:, None] * F2 + rn[None, :]
    tl.store(DHH + dp, gh, mask=ms)
    tl.store(DHL + dp, gl, mask=ms)
    uh, ul = _split(da * silu)
    tl.store(DHH + dp + F, uh, mask=ms)
    tl.store(DHL + dp + F, ul, mask=ms)


@triton.jit
def _dact_kernel(DY, W2, ACTH, ACTL, H, DHH, DHL, SW, SROW, TILE, DWP,
                 D, F, F2, N, NT,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                 FP32: tl.constexpr, GROUP: tl.constexpr):
    """q = dy @ w2^T; SwiGLU backward into the hidden-gradient limbs; <act, q>.

    Two adjacent column blocks of q share one loaded tile of dy per step, and
    their epilogues run one after the other so only one block's worth of
    activation state is live at a time.
    """
    pid_m, pid_n = _swizzle(tl.program_id(0), NT, tl.cdiv(F, 2 * BN), GROUP)
    e = tl.load(TILE + pid_m * 4)
    if e >= 0:
        s0 = tl.load(TILE + pid_m * 4 + 1)
        nrow = tl.load(TILE + pid_m * 4 + 2)
        rm = tl.arange(0, BM)
        mrow = rm < nrow
        slot = (s0 + rm).to(tl.int64)
        rows = tl.load(SROW + s0 + rm, mask=mrow, other=0).to(tl.int64)
        rn = pid_n * 2 * BN + tl.arange(0, BN)
        rn2 = rn + BN
        mn = rn < F
        mn2 = rn2 < F
        dyp = DY + rows[:, None] * D
        wp = W2 + e.to(tl.int64) * (F * D) + rn[:, None].to(tl.int64) * D
        acc = tl.zeros((BM, BN), tl.float32)
        acc2 = tl.zeros((BM, BN), tl.float32)
        for k0 in range(0, D, BK):
            rk = k0 + tl.arange(0, BK)
            mk = rk < D
            a = tl.load(dyp + rk[None, :], mask=mrow[:, None] & mk[None, :],
                        other=0.0)
            b = tl.load(wp + rk[None, :], mask=mn[:, None] & mk[None, :],
                        other=0.0)
            b2 = tl.load(wp + BN * D + rk[None, :],
                         mask=mn2[:, None] & mk[None, :], other=0.0)
            acc = _dot_raw(a, tl.trans(b), acc, FP32)
            acc2 = _dot_raw(a, tl.trans(b2), acc2, FP32)
        sw = tl.load(SW + slot, mask=mrow, other=0.0)
        _dhalf(acc, ACTH, ACTL, H, DHH, DHL, sw, slot, rn, mrow, mn, F, F2,
               DWP, N, 2 * pid_n)
        _dhalf(acc2, ACTH, ACTL, H, DHH, DHL, sw, slot, rn2, mrow, mn2, F, F2,
               DWP, N, 2 * pid_n + 1)


@triton.jit
def _dwred_kernel(DWP, DWS, N, NB, BLK: tl.constexpr):
    """Ordered reduction of the per-column-block <act, q> partials."""
    pid = tl.program_id(0)
    i = pid * BLK + tl.arange(0, BLK)
    m = i < N
    acc = tl.zeros((BLK,), tl.float32)
    for b in range(NB):
        acc += tl.load(DWP + b * N + i, mask=m, other=0.0)
    tl.store(DWS + i, acc, mask=m)


@triton.jit
def _dlogit_kernel(S, SEL, SSUM, POS, DWS, DL, T, E, A,
                   BT: tl.constexpr, BA: tl.constexpr, BE: tl.constexpr):
    """Scatter the selected-score gradients into the dense [T, E] logit grad.

    DWS holds p_a = w_a * d(loss)/d(w_a). With w_a = s_a / ssum, the chain
    through the normalization and the sigmoid is
        dlogit_a = (1 - s_a) * (p_a * ssum - s_a * sum_b p_b) / ssum.

    One program owns all E columns of its token block, so it clears them and
    then writes the A selected entries with no ordering between programs.
    """
    pid = tl.program_id(0)
    rt = pid * BT + tl.arange(0, BT)
    ra = tl.arange(0, BA)
    mt = rt < T
    zero = tl.zeros((BT, BE), tl.float32)
    for e0 in range(0, E, BE):
        re = e0 + tl.arange(0, BE)
        tl.store(DL + rt[:, None].to(tl.int64) * E + re[None, :], zero,
                 mask=mt[:, None] & (re < E)[None, :])
    m = mt[:, None] & (ra < A)[None, :]
    off = rt[:, None].to(tl.int64) * A + ra[None, :]
    sel = tl.load(SEL + off, mask=m, other=0)
    pos = tl.load(POS + off, mask=m, other=0).to(tl.int64)
    p = tl.load(DWS + pos, mask=m, other=0.0)
    ssum = tl.load(SSUM + rt, mask=mt, other=1.0)[:, None]
    s = tl.load(S + rt[:, None].to(tl.int64) * E + sel, mask=m, other=0.0)
    tot = tl.sum(p, axis=1)[:, None]
    tl.store(DL + rt[:, None].to(tl.int64) * E + sel,
             (1.0 - s) * (p * ssum - s * tot) / ssum, mask=m)


@triton.jit
def _dxs_kernel(DHH, DHL, W1, TILE, DXP, D, F2, NT,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                FP32: tl.constexpr, GROUP: tl.constexpr):
    """Per-slot input gradient of the expert body: dh @ w1^T."""
    pid_m, pid_n = _swizzle(tl.program_id(0), NT, tl.cdiv(D, BN), GROUP)
    e = tl.load(TILE + pid_m * 4)
    if e >= 0:
        s0 = tl.load(TILE + pid_m * 4 + 1)
        nrow = tl.load(TILE + pid_m * 4 + 2)
        rm = tl.arange(0, BM)
        mrow = rm < nrow
        slot = (s0 + rm).to(tl.int64)
        rn = pid_n * BN + tl.arange(0, BN)
        mn = rn < D
        hp = slot[:, None] * F2
        wp = W1 + e.to(tl.int64) * (D * F2) + rn[:, None].to(tl.int64) * F2
        acc = tl.zeros((BM, BN), tl.float32)
        for k0 in range(0, F2, BK):
            rk = k0 + tl.arange(0, BK)
            mk = rk < F2
            ma = mrow[:, None] & mk[None, :]
            ah = tl.load(DHH + hp + rk[None, :], mask=ma, other=0.0)
            al = tl.load(DHL + hp + rk[None, :], mask=ma, other=0.0)
            b = tl.load(wp + rk[None, :], mask=mn[:, None] & mk[None, :],
                        other=0.0)
            acc = _dot_lim(ah, al, tl.trans(b), acc, FP32, False)
        tl.store(DXP + slot[:, None] * D + rn[None, :],
                 acc.to(DXP.dtype.element_ty),
                 mask=mrow[:, None] & mn[None, :])


@triton.jit
def _dxcomb_kernel(DXP, POS, DL, RW, DX, T, D, E, A,
                   BT: tl.constexpr, BN: tl.constexpr, BE: tl.constexpr,
                   FP32: tl.constexpr):
    """dx = gathered expert-body gradients + dlogit @ router_w^T."""
    pid_t = tl.program_id(0)
    pid_n = tl.program_id(1)
    rt = pid_t * BT + tl.arange(0, BT)
    rn = pid_n * BN + tl.arange(0, BN)
    mt = rt < T
    mn = rn < D
    m = mt[:, None] & mn[None, :]
    acc = tl.zeros((BT, BN), tl.float32)
    for a in range(A):
        p = tl.load(POS + rt * A + a, mask=mt, other=0).to(tl.int64)
        acc += tl.load(DXP + p[:, None] * D + rn[None, :], mask=m, other=0.0)
    for k0 in range(0, E, BE):
        rk = k0 + tl.arange(0, BE)
        mk = rk < E
        dl = tl.load(DL + rt[:, None].to(tl.int64) * E + rk[None, :],
                     mask=mt[:, None] & mk[None, :], other=0.0)
        w = tl.load(RW + rn[:, None].to(tl.int64) * E + rk[None, :],
                    mask=mn[:, None] & mk[None, :], other=0.0)
        hi, lo = _split(dl)
        acc = _dot_lim(hi, lo, tl.trans(w), acc, FP32, True)
    tl.store(DX + rt[:, None].to(tl.int64) * D + rn[None, :],
             acc.to(DX.dtype.element_ty), mask=m)


@triton.jit
def _dw1_kernel(X, DHH, DHL, SROW, OFFS, DW1, D, F2,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                FP32: tl.constexpr, GROUP: tl.constexpr):
    """dw1[e] = x_e^T @ dh_e over the expert's run of slots."""
    e = tl.program_id(1)
    o0 = tl.load(OFFS + e)
    o1 = tl.load(OFFS + e + 1)
    pid_d, pid_f = _swizzle(tl.program_id(0), tl.cdiv(D, BM),
                            tl.cdiv(F2, BN), GROUP)
    rd = pid_d * BM + tl.arange(0, BM)
    rf = pid_f * BN + tl.arange(0, BN)
    md = rd < D
    mf = rf < F2
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(o0, o1, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < o1
        rows = tl.load(SROW + rk, mask=mk, other=0).to(tl.int64)
        a = tl.load(X + rows[:, None] * D + rd[None, :],
                    mask=mk[:, None] & md[None, :], other=0.0)
        hp = rk[:, None].to(tl.int64) * F2 + rf[None, :]
        mh = mk[:, None] & mf[None, :]
        hh = tl.load(DHH + hp, mask=mh, other=0.0)
        hl = tl.load(DHL + hp, mask=mh, other=0.0)
        at = tl.trans(a)
        if FP32:
            ah, al = _split(at)
            acc = tl.dot(ah, hh, acc)
            acc = tl.dot(al, hh, acc)
            acc = tl.dot(ah, hl, acc)
        else:
            acc = tl.dot(at, hh, acc)
            acc = tl.dot(at, hl, acc)
    tl.store(DW1 + e.to(tl.int64) * (D * F2) + rd[:, None].to(tl.int64) * F2
             + rf[None, :], acc.to(DW1.dtype.element_ty),
             mask=md[:, None] & mf[None, :])


@triton.jit
def _dw2_kernel(ACTH, ACTL, DY, SROW, OFFS, DW2, F, D,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                FP32: tl.constexpr, GROUP: tl.constexpr):
    """dw2[e] = (w * act_e)^T @ dy_e over the expert's run of slots."""
    e = tl.program_id(1)
    o0 = tl.load(OFFS + e)
    o1 = tl.load(OFFS + e + 1)
    pid_f, pid_d = _swizzle(tl.program_id(0), tl.cdiv(F, BM),
                            tl.cdiv(D, BN), GROUP)
    rf = pid_f * BM + tl.arange(0, BM)
    rd = pid_d * BN + tl.arange(0, BN)
    mf = rf < F
    md = rd < D
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(o0, o1, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < o1
        rows = tl.load(SROW + rk, mask=mk, other=0).to(tl.int64)
        ap = rk[:, None].to(tl.int64) * F + rf[None, :]
        ma = mk[:, None] & mf[None, :]
        aht = tl.trans(tl.load(ACTH + ap, mask=ma, other=0.0))
        alt = tl.trans(tl.load(ACTL + ap, mask=ma, other=0.0))
        dy = tl.load(DY + rows[:, None] * D + rd[None, :],
                     mask=mk[:, None] & md[None, :], other=0.0)
        if FP32:
            dh, dl = _split(dy)
            acc = tl.dot(aht, dh, acc)
            acc = tl.dot(alt, dh, acc)
            acc = tl.dot(aht, dl, acc)
        else:
            acc = tl.dot(aht, dy, acc)
            acc = tl.dot(alt, dy, acc)
    tl.store(DW2 + e.to(tl.int64) * (F * D) + rf[:, None].to(tl.int64) * D
             + rd[None, :], acc.to(DW2.dtype.element_ty),
             mask=mf[:, None] & md[None, :])


@triton.jit
def _drw_kernel(X, DL, DRW, T, D, E,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                FP32: tl.constexpr):
    """drouter_w = x^T @ dlogit."""
    rd = tl.program_id(0) * BM + tl.arange(0, BM)
    re = tl.program_id(1) * BN + tl.arange(0, BN)
    md = rd < D
    me = re < E
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, T, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < T
        a = tl.load(X + rk[:, None].to(tl.int64) * D + rd[None, :],
                    mask=mk[:, None] & md[None, :], other=0.0)
        dl = tl.load(DL + rk[:, None].to(tl.int64) * E + re[None, :],
                     mask=mk[:, None] & me[None, :], other=0.0)
        hi, lo = _split(dl)
        at = tl.trans(a)
        if FP32:
            ah, al = _split(at)
            acc = tl.dot(ah, hi, acc)
            acc = tl.dot(al, hi, acc)
            acc = tl.dot(ah, lo, acc)
        else:
            acc = tl.dot(at, hi, acc)
            acc = tl.dot(at, lo, acc)
    tl.store(DRW + rd[:, None].to(tl.int64) * E + re[None, :],
             acc.to(DRW.dtype.element_ty), mask=md[:, None] & me[None, :])


# ------------------------------------------------------------ host driver ---

GROUP = 8


def _cap(b, n):
    """Largest power-of-two tile <= b that is not wider than the axis."""
    while b > 16 and b > n:
        b //= 2
    return b


def _plan(N, E, D, F, fp32):
    """Tile shapes per kernel: (slot/row tile, col tile, k tile, warps, stages).

    Chosen from the tensor shapes and the input dtype. The slot tile follows
    the mean run length N/E, since a run shorter than the tile leaves the rows
    past its end idle. Float32 operand tiles are four bytes wide and carry
    three limb products per step, so the float32 tiles and stage counts are
    smaller: a block must fit the 227 KB of shared memory an H100 SM offers.
    """
    F2 = 2 * F
    avg = N / max(E, 1)
    slot = 128 if avg >= 192.0 else (64 if avg >= 48.0 else 32)
    half = min(slot, 64)
    if fp32:
        return {
            "g1": (slot, _cap(64, F), _cap(32, D), 8, 3),
            "g2": (slot, _cap(128, D), _cap(32, F), 8, 3),
            "da": (half, _cap(64, F), _cap(32, D), 4, 3),
            "dx": (slot, _cap(128, D), _cap(32, F2), 8, 3),
            "w1": (_cap(128, D), _cap(64, F2), 32, 4, 3),
            "w2": (_cap(64, F), _cap(128, D), 32, 4, 3),
        }
    return {
        "g1": (slot, _cap(128, F), _cap(64, D), 8, 4),
        "g2": (slot, _cap(256, D), _cap(64, F), 8, 4),
        "da": (half, _cap(64, F), _cap(64, D), 4, 4),
        "dx": (slot, _cap(256, D), _cap(64, F2), 8, 4),
        "w1": (_cap(128, D), _cap(128, F2), 32, 4, 5),
        "w2": (_cap(64, F), _cap(256, D), 64, 4, 4),
    }


def _route(x, router_w, router_b, A):
    T, D = x.shape
    E = router_w.shape[1]
    fp32 = x.dtype == torch.float32
    dev = x.device
    s = torch.empty(T, E, device=dev, dtype=torch.float32)
    key = torch.empty(T, E, device=dev, dtype=torch.int64)
    be = max(16, min(_npow2(E), 64))
    shift = max(10, (E - 1).bit_length())
    _router_kernel[(triton.cdiv(T, 64), triton.cdiv(E, be))](
        x, router_w, router_b, s, key, T, D, E,
        BT=64, BE=be, BK=64, FP32=fp32, SHIFT=shift,
        num_warps=4, num_stages=3)

    sel = torch.topk(key, A, dim=-1, largest=True, sorted=True).indices
    wgt = torch.empty(T, A, device=dev, dtype=torch.float32)
    ssum = torch.empty(T, device=dev, dtype=torch.float32)
    _wgt_kernel[(triton.cdiv(T, 64),)](s, sel, wgt, ssum, T, E, A,
                                       BT=64, BA=_npow2(A), num_warps=4)
    return s, sel, wgt, ssum


def _dispatch(sel, wgt, T, E, A, bms):
    N = T * A
    dev = sel.device
    svals, order = torch.sort(sel.reshape(-1), stable=True)
    srow = torch.empty(N, device=dev, dtype=torch.int32)
    sw = torch.empty(N, device=dev, dtype=torch.float32)
    pos = torch.empty(N, device=dev, dtype=torch.int32)
    _dispatch_kernel[(triton.cdiv(N, 256),)](order, wgt, srow, sw, pos, N, A,
                                             BLK=256, num_warps=4)
    offs = torch.empty(E + 1, device=dev, dtype=torch.int32)
    _bounds_kernel[(triton.cdiv(E + 1, 128),)](svals, offs, N, E, BLK=128,
                                               num_warps=4)
    be = max(_npow2(E), 64)
    maps = {}
    for bm in sorted(set(bms)):
        ntmax = triton.cdiv(N, bm) + E
        tile = torch.empty(ntmax * 4, device=dev, dtype=torch.int32)
        _tilemap_kernel[(triton.cdiv(ntmax, 256),)](
            offs, tile, ntmax, E, BM=bm, BE=be, BLK=256, num_warps=4)
        maps[bm] = (tile, ntmax)
    return srow, sw, pos, offs, maps


class _FusedMoe(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, router_w, router_b, w1, w2, A):
        A = int(A)
        T, D = x.shape
        E = router_w.shape[1]
        F2 = w1.shape[2]
        F = F2 // 2
        N = T * A
        fp32 = x.dtype == torch.float32
        dev = x.device

        x = x.contiguous()
        router_w = router_w.contiguous()
        w1 = w1.contiguous()
        w2 = w2.contiguous()

        s, sel, wgt, ssum = _route(x, router_w, router_b, A)
        cfg = _plan(N, E, D, F, fp32)
        srow, sw, pos, offs, maps = _dispatch(
            sel, wgt, T, E, A, [cfg[k][0] for k in ("g1", "g2", "da", "dx")])

        h = torch.empty(N, F2, device=dev, dtype=torch.float32)
        acth = torch.empty(N, F, device=dev, dtype=torch.bfloat16)
        actl = torch.empty(N, F, device=dev, dtype=torch.bfloat16)
        bm, bn, bk, w, st = cfg["g1"]
        tile, ntmax = maps[bm]
        _gemm1_kernel[(ntmax * triton.cdiv(F, bn),)](
            x, w1, srow, sw, tile, h, acth, actl, D, F, F2, ntmax,
            BM=bm, BN=bn, BK=bk, FP32=fp32, GROUP=GROUP,
            num_warps=w, num_stages=st)

        fp = torch.empty(N, D, device=dev, dtype=x.dtype)
        bm, bn, bk, w, st = cfg["g2"]
        tile, ntmax = maps[bm]
        _gemm2_kernel[(ntmax * triton.cdiv(D, bn),)](
            acth, actl, w2, tile, fp, F, D, ntmax,
            BM=bm, BN=bn, BK=bk, FP32=fp32, GROUP=GROUP,
            num_warps=w, num_stages=st)

        y = torch.empty(T, D, device=dev, dtype=x.dtype)
        _combine_kernel[(triton.cdiv(T, 32), triton.cdiv(D, _cap(128, D)))](
            fp, pos, y, T, D, A, BT=32, BN=_cap(128, D), num_warps=4)

        ctx.save_for_backward(x, router_w, w1, w2, s, sel, ssum,
                              srow, sw, pos, offs, h, acth, actl)
        ctx.meta = (T, D, E, F, F2, A, N, fp32, cfg, maps)
        return y

    @staticmethod
    def backward(ctx, dy):
        (x, router_w, w1, w2, s, sel, ssum,
         srow, sw, pos, offs, h, acth, actl) = ctx.saved_tensors
        T, D, E, F, F2, A, N, fp32, cfg, maps = ctx.meta
        dev = x.device
        dy = dy.contiguous()

        dhh = torch.empty(N, F2, device=dev, dtype=torch.bfloat16)
        dhl = torch.empty(N, F2, device=dev, dtype=torch.bfloat16)
        bm, bn, bk, w, st = cfg["da"]
        tile, ntmax = maps[bm]
        npair = triton.cdiv(F, 2 * bn)
        dwp = torch.empty(2 * npair, N, device=dev, dtype=torch.float32)
        _dact_kernel[(ntmax * npair,)](
            dy, w2, acth, actl, h, dhh, dhl, sw, srow, tile, dwp, D, F, F2, N,
            ntmax, BM=bm, BN=bn, BK=bk, FP32=fp32, GROUP=GROUP,
            num_warps=w, num_stages=st)
        nbf = 2 * npair

        dws = torch.empty(N, device=dev, dtype=torch.float32)
        _dwred_kernel[(triton.cdiv(N, 256),)](dwp, dws, N, nbf, BLK=256,
                                              num_warps=4)

        dl = torch.empty(T, E, device=dev, dtype=torch.float32)
        _dlogit_kernel[(triton.cdiv(T, 64),)](
            s, sel, ssum, pos, dws, dl, T, E, A,
            BT=64, BA=_npow2(A), BE=min(256, _npow2(E)), num_warps=4)

        dxp = torch.empty(N, D, device=dev, dtype=x.dtype)
        bm, bn, bk, w, st = cfg["dx"]
        tile, ntmax = maps[bm]
        _dxs_kernel[(ntmax * triton.cdiv(D, bn),)](
            dhh, dhl, w1, tile, dxp, D, F2, ntmax,
            BM=bm, BN=bn, BK=bk, FP32=fp32, GROUP=GROUP,
            num_warps=w, num_stages=st)

        dx = torch.empty(T, D, device=dev, dtype=x.dtype)
        bnc = _cap(128, D)
        _dxcomb_kernel[(triton.cdiv(T, 32), triton.cdiv(D, bnc))](
            dxp, pos, dl, router_w, dx, T, D, E, A,
            BT=32, BN=bnc, BE=min(64, max(16, _npow2(E))), FP32=fp32,
            num_warps=4, num_stages=2)

        dw1 = torch.empty_like(w1)
        bm, bn, bk, w, st = cfg["w1"]
        _dw1_kernel[(triton.cdiv(D, bm) * triton.cdiv(F2, bn), E)](
            x, dhh, dhl, srow, offs, dw1, D, F2,
            BM=bm, BN=bn, BK=bk, FP32=fp32, GROUP=GROUP,
            num_warps=w, num_stages=st)

        dw2 = torch.empty_like(w2)
        bm, bn, bk, w, st = cfg["w2"]
        _dw2_kernel[(triton.cdiv(F, bm) * triton.cdiv(D, bn), E)](
            acth, actl, dy, srow, offs, dw2, F, D,
            BM=bm, BN=bn, BK=bk, FP32=fp32, GROUP=GROUP,
            num_warps=w, num_stages=st)

        drw = torch.empty_like(router_w)
        bne = min(64, max(16, _npow2(E)))
        _drw_kernel[(triton.cdiv(D, 64), triton.cdiv(E, bne))](
            x, dl, drw, T, D, E,
            BM=64, BN=bne, BK=64, FP32=fp32, num_warps=4, num_stages=3)

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
