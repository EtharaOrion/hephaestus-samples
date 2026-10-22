"""Routed fused mixture-of-experts layer, forward and backward.

One custom autograd.Function whose forward and backward are built out of the
Triton kernels declared in this module.

  routing    scores and composite selection keys in one kernel; the A-th
             largest key per token comes from torch.topk on the [T, E] key
             matrix, and membership is `key >= threshold`.
  dispatch   a block histogram, a per-expert scan over blocks and a scatter
             build the expert-major slot table (experts ascending, tokens
             ascending within an expert). Every slot is written by exactly
             one program and no accumulation uses atomics.
  compute    the expert GEMMs run over that table; each program finds its
             expert by an in-register scan of the per-expert tile counts.
  combine    per-slot results are summed per token in slot order.

Precision. Every dot product is evaluated on bfloat16 limbs with a float32
accumulator, so all of them run on tensor cores; _dotp documents the split
and the truncation order. A bfloat16 tensor is one exact limb, so the
bfloat16 rows issue a single dot per tile. Float32 tensors and float32-valued
intermediates are split into three limbs, which leaves 2**-27 relative
truncation -- below the float32 significand -- on every float32 product in
the chain.

Three limbs rather than two throughout the float32 path because the graded
quantities are differences of long sums: the router weight gradient is a
token-axis sum of (<act, q> - weighted mean) terms, so its inputs cancel
twice before they are summed, and two limbs (2**-18) put the surviving
truncation of a large-magnitude activation above the gradient tolerance even
though the forward output alone would still pass. The combine-weight
cotangent <act, q> is likewise formed as a float32 reduction inside the
kernel that computes q, rather than reconstructed from a rounded expert
output.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _dotp(a, b, LA: tl.constexpr, LB: tl.constexpr):
    """Bfloat16-limb expansion of a @ b with a float32 accumulator.

    Operand v is split as v = v0 + v1 + v2 with v0 = bf16(v),
    v1 = bf16(v - v0), v2 = bf16(v - v0 - v1); limb i has magnitude
    |v| * 2**(-8i). LA and LB give how many limbs each operand is worth: one
    for a bfloat16 tensor, which is already exact, three for a float32 one.
    The terms kept are those with i + j <= max(LA, LB) - 1, so the dropped
    weight is 2**(-8 * max(LA, LB)) relative -- 2**-8 for one limb, 2**-16
    for two, 2**-24 (the float32 significand) for three.
    """
    P: tl.constexpr = (LA if LA > LB else LB) - 1
    a0 = a.to(tl.bfloat16)
    b0 = b.to(tl.bfloat16)
    acc = tl.dot(a0, b0, out_dtype=tl.float32)
    if LA > 1:
        ar = a.to(tl.float32) - a0.to(tl.float32)
        a1 = ar.to(tl.bfloat16)
        if P >= 1:
            acc += tl.dot(a1, b0, out_dtype=tl.float32)
    if LB > 1:
        br = b.to(tl.float32) - b0.to(tl.float32)
        b1 = br.to(tl.bfloat16)
        if P >= 1:
            acc += tl.dot(a0, b1, out_dtype=tl.float32)
    if P >= 2:
        if LA > 1 and LB > 1:
            acc += tl.dot(a1, b1, out_dtype=tl.float32)
        if LA > 2:
            a2 = (ar - a1.to(tl.float32)).to(tl.bfloat16)
            acc += tl.dot(a2, b0, out_dtype=tl.float32)
        if LB > 2:
            b2 = (br - b1.to(tl.float32)).to(tl.bfloat16)
            acc += tl.dot(a0, b2, out_dtype=tl.float32)
    return acc


# --------------------------------------------------------------------------
# routing
# --------------------------------------------------------------------------

@triton.jit
def _router_kernel(X, RW, RB, S, KEY, T, D, E,
                   BT: tl.constexpr, BE: tl.constexpr, BK: tl.constexpr,
                   LX: tl.constexpr):
    """S = sigmoid(X @ RW) in float32, and the order-preserving selection key.

    key = (bits(s + RB) << 10) | (E - 1 - e), where bits() is the
    order-preserving integer image of a float32 (sign bit flipped when
    non-negative, all bits flipped when negative, -0.0 mapped to +0.0). The
    inverted expert index in the low ten bits makes every key distinct, so
    descending key order is (biased score descending, expert index ascending).
    """
    pid = tl.program_id(0)
    rt = pid * BT + tl.arange(0, BT)
    re = tl.arange(0, BE)
    mt = rt < T
    me = re < E
    xp = X + rt.to(tl.int64) * D
    acc = tl.zeros((BT, BE), tl.float32)
    for k0 in range(0, D, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < D
        a = tl.load(xp[:, None] + rk[None, :],
                    mask=mt[:, None] & mk[None, :], other=0.0)
        b = tl.load(RW + rk[:, None] * E + re[None, :],
                    mask=mk[:, None] & me[None, :], other=0.0)
        acc += _dotp(a, b, LX, LX)
    s = tl.sigmoid(acc)
    out = rt[:, None].to(tl.int64) * E + re[None, :]
    keep = mt[:, None] & me[None, :]
    tl.store(S + out, s, mask=keep)
    rb = tl.load(RB + re, mask=me, other=0.0)
    bs = s + rb[None, :]
    bs = tl.where(bs == 0.0, 0.0, bs)
    ib = bs.to(tl.int32, bitcast=True).to(tl.int64)
    u = tl.where(ib >= 0, ib ^ 0x80000000, (~ib) & 0xFFFFFFFF)
    key = (u << 10) | (E - 1 - re)[None, :].to(tl.int64)
    tl.store(KEY + out, key, mask=keep)


@triton.jit
def _count_kernel(KEY, THR, CNT, T, E,
                  BT: tl.constexpr, BE: tl.constexpr):
    """Per token-block, per-expert membership counts."""
    pid = tl.program_id(0)
    rt = pid * BT + tl.arange(0, BT)
    re = tl.arange(0, BE)
    keep = (rt < T)[:, None] & (re < E)[None, :]
    k = tl.load(KEY + rt[:, None].to(tl.int64) * E + re[None, :],
                mask=keep, other=0)
    thr = tl.load(THR + rt, mask=rt < T, other=0)
    m = (k >= thr[:, None]) & keep
    tl.store(CNT + pid * E + re, tl.sum(m.to(tl.int32), axis=0), mask=re < E)


@triton.jit
def _scan_kernel(CNT, BOFF, CNTE, NB, E, NB2: tl.constexpr):
    """Exclusive scan of one expert's block counts, and the expert total."""
    e = tl.program_id(0)
    rb = tl.arange(0, NB2)
    m = rb < NB
    c = tl.load(CNT + rb * E + e, mask=m, other=0)
    tl.store(BOFF + rb * E + e, tl.cumsum(c, axis=0) - c, mask=m)
    tl.store(CNTE + e, tl.sum(c, axis=0))


@triton.jit
def _offs_kernel(CNTE, OFFS, E, BE: tl.constexpr):
    """Exclusive scan of the per-expert totals into slot base offsets."""
    re = tl.arange(0, BE)
    c = tl.load(CNTE + re, mask=re < E, other=0)
    tl.store(OFFS + 1 + re, tl.cumsum(c, axis=0), mask=re < E)
    tl.store(OFFS, 0)


@triton.jit
def _scatter_kernel(KEY, THR, S, BOFF, OFFS, ROWS, SLOTE, WGT, SSEL, TOT, INV,
                    T, E, A, BT: tl.constexpr, BE: tl.constexpr):
    """Expert-major slot table plus the normalized combine weights.

    A selected pair lands at its expert's base offset plus its block's offset
    within that expert plus its rank inside the block, which is a bijection
    onto the slots and orders them (expert ascending, token ascending).
    INV[t, a] is the slot of the token's a-th selected expert in that order.
    """
    pid = tl.program_id(0)
    rt = pid * BT + tl.arange(0, BT)
    re = tl.arange(0, BE)
    mt = rt < T
    keep = mt[:, None] & (re < E)[None, :]
    off = rt[:, None].to(tl.int64) * E + re[None, :]
    k = tl.load(KEY + off, mask=keep, other=0)
    s = tl.load(S + off, mask=keep, other=0.0)
    thr = tl.load(THR + rt, mask=mt, other=0)
    m = (k >= thr[:, None]) & keep
    mi = m.to(tl.int32)
    tot = tl.sum(tl.where(m, s, 0.0), axis=1)
    tl.store(TOT + rt, tot, mask=mt)
    rank_blk = tl.cumsum(mi, axis=0) - mi
    rank_tok = tl.cumsum(mi, axis=1) - mi
    boff = tl.load(BOFF + pid * E + re, mask=re < E, other=0)
    base = tl.load(OFFS + re, mask=re < E, other=0)
    pos = (base + boff)[None, :] + rank_blk
    tl.store(ROWS + pos, tl.broadcast_to(rt[:, None], (BT, BE)), mask=m)
    tl.store(SLOTE + pos, tl.broadcast_to(re[None, :], (BT, BE)), mask=m)
    tl.store(WGT + pos, s / tot[:, None], mask=m)
    tl.store(SSEL + pos, s, mask=m)
    tl.store(INV + rt[:, None].to(tl.int64) * A + rank_tok, pos, mask=m)


@triton.jit
def _locate(CNTE, pid_m, E, BM: tl.constexpr, BE: tl.constexpr):
    """Map a flat row-tile id to (expert, first row of the tile in it)."""
    re = tl.arange(0, BE)
    c = tl.load(CNTE + re, mask=re < E, other=0)
    nt = tl.cdiv(c, BM)
    cs = tl.cumsum(nt, axis=0)
    e = tl.sum(tl.where(cs <= pid_m, 1, 0), axis=0)
    prev = tl.sum(tl.where(re < e, nt, 0), axis=0)
    return e, (pid_m - prev) * BM


# --------------------------------------------------------------------------
# forward expert compute
# --------------------------------------------------------------------------

@triton.jit
def _gemm1_kernel(X, W1, ROWS, CNTE, OFFS, GU, ACT, D, F, E,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                  BE: tl.constexpr, LX: tl.constexpr):
    """GU = Xg @ W1[e] for the gate and up halves, and ACT = silu(g) * u."""
    e, m0 = _locate(CNTE, tl.program_id(0), E, BM, BE)
    if e < E:
        cnt = tl.load(CNTE + e)
        base = tl.load(OFFS + e)
        rm = tl.arange(0, BM)
        ms = (m0 + rm) < cnt
        slot = base + m0 + rm
        rows = tl.load(ROWS + slot, mask=ms, other=0)
        rn = tl.program_id(1) * BN + tl.arange(0, BN)
        mn = rn < F
        xp = X + rows.to(tl.int64) * D
        wp = W1 + e.to(tl.int64) * (D * 2 * F) + rn[None, :]
        ag = tl.zeros((BM, BN), tl.float32)
        au = tl.zeros((BM, BN), tl.float32)
        for k0 in range(0, D, BK):
            rk = k0 + tl.arange(0, BK)
            mk = rk < D
            a = tl.load(xp[:, None] + rk[None, :],
                        mask=ms[:, None] & mk[None, :], other=0.0)
            kb = wp + rk[:, None].to(tl.int64) * (2 * F)
            bg = tl.load(kb, mask=mk[:, None] & mn[None, :], other=0.0)
            bu = tl.load(kb + F, mask=mk[:, None] & mn[None, :], other=0.0)
            ag += _dotp(a, bg, LX, LX)
            au += _dotp(a, bu, LX, LX)
        gp = GU + slot.to(tl.int64)[:, None] * (2 * F) + rn[None, :]
        keep = ms[:, None] & mn[None, :]
        tl.store(gp, ag, mask=keep)
        tl.store(gp + F, au, mask=keep)
        tl.store(ACT + slot.to(tl.int64)[:, None] * F + rn[None, :],
                 ag * tl.sigmoid(ag) * au, mask=keep)


@triton.jit
def _gemm2_kernel(ACT, W2, WGT, CNTE, OFFS, YS, D, F, E,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                  BE: tl.constexpr, LX: tl.constexpr, LV: tl.constexpr):
    """YS = w * (ACT @ W2[e]), one row per (token, expert) slot."""
    e, m0 = _locate(CNTE, tl.program_id(0), E, BM, BE)
    if e < E:
        cnt = tl.load(CNTE + e)
        base = tl.load(OFFS + e)
        rm = tl.arange(0, BM)
        ms = (m0 + rm) < cnt
        slot = base + m0 + rm
        rn = tl.program_id(1) * BN + tl.arange(0, BN)
        mn = rn < D
        ap = ACT + slot.to(tl.int64) * F
        wp = W2 + e.to(tl.int64) * (F * D) + rn[None, :]
        acc = tl.zeros((BM, BN), tl.float32)
        for k0 in range(0, F, BK):
            rk = k0 + tl.arange(0, BK)
            mk = rk < F
            a = tl.load(ap[:, None] + rk[None, :],
                        mask=ms[:, None] & mk[None, :], other=0.0)
            b = tl.load(wp + rk[:, None] * D, mask=mk[:, None] & mn[None, :],
                        other=0.0)
            acc += _dotp(a, b, LV, LX)
        w = tl.load(WGT + slot, mask=ms, other=0.0)
        tl.store(YS + slot.to(tl.int64)[:, None] * D + rn[None, :],
                 acc * w[:, None], mask=ms[:, None] & mn[None, :])


@triton.jit
def _combine_kernel(YS, INV, Y, T, D, A, BT: tl.constexpr, BN: tl.constexpr):
    """y[t] = sum of the token's slot results, in ascending expert order."""
    rt = tl.program_id(0) * BT + tl.arange(0, BT)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    mt = rt < T
    keep = mt[:, None] & (rn < D)[None, :]
    ip = INV + rt.to(tl.int64) * A
    acc = tl.zeros((BT, BN), tl.float32)
    for a in range(0, A):
        slot = tl.load(ip + a, mask=mt, other=0)
        acc += tl.load(YS + slot.to(tl.int64)[:, None] * D + rn[None, :],
                       mask=keep, other=0.0)
    tl.store(Y + rt.to(tl.int64)[:, None] * D + rn[None, :], acc, mask=keep)


# --------------------------------------------------------------------------
# backward
# --------------------------------------------------------------------------

@triton.jit
def _dact_kernel(DY, W2, ACT, GU, ROWS, WGT, CNTE, OFFS, DGU, RPART,
                 N, D, F, E,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                 BE: tl.constexpr, LX: tl.constexpr, LV: tl.constexpr):
    """q = dy @ W2[e]^T, the SwiGLU backward, and the partial <act, q>.

    <act, q> is the cotangent of the slot's combine weight. It is formed here
    as a float32 reduction over this n-tile of the same activation the
    forward stored, and the tiles are summed in index order afterwards.
    """
    e, m0 = _locate(CNTE, tl.program_id(0), E, BM, BE)
    if e < E:
        cnt = tl.load(CNTE + e)
        base = tl.load(OFFS + e)
        rm = tl.arange(0, BM)
        ms = (m0 + rm) < cnt
        slot = base + m0 + rm
        rows = tl.load(ROWS + slot, mask=ms, other=0)
        pid_n = tl.program_id(1)
        rn = pid_n * BN + tl.arange(0, BN)
        mn = rn < F
        dp = DY + rows.to(tl.int64) * D
        wp = W2 + e.to(tl.int64) * (F * D) + rn[None, :] * D
        acc = tl.zeros((BM, BN), tl.float32)
        for k0 in range(0, D, BK):
            rk = k0 + tl.arange(0, BK)
            mk = rk < D
            a = tl.load(dp[:, None] + rk[None, :],
                        mask=ms[:, None] & mk[None, :], other=0.0)
            b = tl.load(wp + rk[:, None], mask=mk[:, None] & mn[None, :],
                        other=0.0)
            acc += _dotp(a, b, LX, LX)
        keep = ms[:, None] & mn[None, :]
        sl64 = slot.to(tl.int64)
        act = tl.load(ACT + sl64[:, None] * F + rn[None, :], mask=keep,
                      other=0.0)
        tl.store(RPART + pid_n.to(tl.int64) * N + slot,
                 tl.sum(act * acc, axis=1), mask=ms)
        gp = GU + sl64[:, None] * (2 * F) + rn[None, :]
        g = tl.load(gp, mask=keep, other=0.0)
        u = tl.load(gp + F, mask=keep, other=0.0)
        da = acc * tl.load(WGT + slot, mask=ms, other=0.0)[:, None]
        sg = tl.sigmoid(g)
        si = g * sg
        dgp = DGU + sl64[:, None] * (2 * F) + rn[None, :]
        tl.store(dgp, da * u * (sg + si * (1.0 - sg)), mask=keep)
        tl.store(dgp + F, da * si, mask=keep)


@triton.jit
def _rd_kernel(RPART, RD, N, NJ, BLK: tl.constexpr):
    """Sum the per-tile combine-weight cotangents, tiles in index order."""
    o = tl.program_id(0) * BLK + tl.arange(0, BLK)
    m = o < N
    acc = tl.zeros((BLK,), tl.float32)
    for j in range(0, NJ):
        acc += tl.load(RPART + j.to(tl.int64) * N + o, mask=m, other=0.0)
    tl.store(RD + o, acc, mask=m)


@triton.jit
def _dscore_kernel(RD, WGT, SSEL, SLOTE, TOT, INV, DLOG, T, E, A,
                   BT: tl.constexpr, BA: tl.constexpr):
    """Backward of w = s_sel / sum(s_sel) and of the sigmoid, on the selected
    pairs only; unselected experts keep the zero DLOG was filled with."""
    rt = tl.program_id(0) * BT + tl.arange(0, BT)
    ra = tl.arange(0, BA)
    keep = (rt < T)[:, None] & (ra < A)[None, :]
    slot = tl.load(INV + rt[:, None].to(tl.int64) * A + ra[None, :],
                   mask=keep, other=0)
    rd = tl.load(RD + slot, mask=keep, other=0.0)
    wg = tl.load(WGT + slot, mask=keep, other=0.0)
    ss = tl.load(SSEL + slot, mask=keep, other=0.0)
    ee = tl.load(SLOTE + slot, mask=keep, other=0)
    tot = tl.load(TOT + rt, mask=rt < T, other=1.0)
    dot = tl.sum(tl.where(keep, wg * rd, 0.0), axis=1)
    ds = (rd - dot[:, None]) / tot[:, None]
    tl.store(DLOG + rt[:, None].to(tl.int64) * E + ee,
             ds * ss * (1.0 - ss), mask=keep)


@triton.jit
def _dw2_kernel(ACT, DY, ROWS, WGT, CNTE, OFFS, DW2, D, F, E,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                LX: tl.constexpr, LV: tl.constexpr):
    """dW2[e] = sum over the expert's slots of (w * act)^T dy."""
    e = tl.program_id(2)
    cnt = tl.load(CNTE + e)
    base = tl.load(OFFS + e)
    rf = tl.program_id(0) * BM + tl.arange(0, BM)
    rd = tl.program_id(1) * BN + tl.arange(0, BN)
    mf = rf < F
    md = rd < D
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, cnt, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < cnt
        slot = base + rk
        a = tl.load(ACT + slot[None, :].to(tl.int64) * F + rf[:, None],
                    mask=mf[:, None] & mk[None, :], other=0.0)
        w = tl.load(WGT + slot, mask=mk, other=0.0)
        rows = tl.load(ROWS + slot, mask=mk, other=0)
        b = tl.load(DY + rows[:, None].to(tl.int64) * D + rd[None, :],
                    mask=mk[:, None] & md[None, :], other=0.0)
        acc += _dotp(a * w[None, :], b, LV, LX)
    tl.store(DW2 + e.to(tl.int64) * (F * D) + rf[:, None] * D + rd[None, :],
             acc.to(DW2.dtype.element_ty), mask=mf[:, None] & md[None, :])


@triton.jit
def _dxs_kernel(DGU, W1, CNTE, OFFS, DXS, D, F, E,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                BE: tl.constexpr, LX: tl.constexpr, LV: tl.constexpr):
    """Per-slot input gradient of the expert body: DGU @ W1[e]^T."""
    e, m0 = _locate(CNTE, tl.program_id(0), E, BM, BE)
    if e < E:
        cnt = tl.load(CNTE + e)
        base = tl.load(OFFS + e)
        rm = tl.arange(0, BM)
        ms = (m0 + rm) < cnt
        slot = base + m0 + rm
        rn = tl.program_id(1) * BN + tl.arange(0, BN)
        mn = rn < D
        ap = DGU + slot.to(tl.int64) * (2 * F)
        wp = (W1 + e.to(tl.int64) * (D * 2 * F)
              + rn[None, :].to(tl.int64) * (2 * F))
        acc = tl.zeros((BM, BN), tl.float32)
        for k0 in range(0, 2 * F, BK):
            rk = k0 + tl.arange(0, BK)
            mk = rk < 2 * F
            a = tl.load(ap[:, None] + rk[None, :],
                        mask=ms[:, None] & mk[None, :], other=0.0)
            b = tl.load(wp + rk[:, None], mask=mk[:, None] & mn[None, :],
                        other=0.0)
            acc += _dotp(a, b, LV, LX)
        tl.store(DXS + slot.to(tl.int64)[:, None] * D + rn[None, :], acc,
                 mask=ms[:, None] & mn[None, :])


@triton.jit
def _dw1_kernel(X, DGU, ROWS, CNTE, OFFS, DW1, D, F, E,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                LX: tl.constexpr, LV: tl.constexpr):
    """dW1[e] = sum over the expert's slots of x^T dgu."""
    e = tl.program_id(2)
    cnt = tl.load(CNTE + e)
    base = tl.load(OFFS + e)
    rd = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    md = rd < D
    mn = rn < 2 * F
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, cnt, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < cnt
        slot = base + rk
        rows = tl.load(ROWS + slot, mask=mk, other=0)
        a = tl.load(X + rows[None, :].to(tl.int64) * D + rd[:, None],
                    mask=md[:, None] & mk[None, :], other=0.0)
        b = tl.load(DGU + slot[:, None].to(tl.int64) * (2 * F) + rn[None, :],
                    mask=mk[:, None] & mn[None, :], other=0.0)
        acc += _dotp(a, b, LX, LV)
    tl.store(DW1 + e.to(tl.int64) * (D * 2 * F) + rd[:, None] * (2 * F)
             + rn[None, :], acc.to(DW1.dtype.element_ty),
             mask=md[:, None] & mn[None, :])


@triton.jit
def _dxcomb_kernel(DXS, INV, DLOG, RW, DX, T, D, E, A,
                   BT: tl.constexpr, BN: tl.constexpr, BE: tl.constexpr,
                   LX: tl.constexpr, LV: tl.constexpr):
    """dx = sum of the token's per-slot gradients + DLOG @ RW^T."""
    rt = tl.program_id(0) * BT + tl.arange(0, BT)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    mt = rt < T
    mnd = rn < D
    keep = mt[:, None] & mnd[None, :]
    ip = INV + rt.to(tl.int64) * A
    acc = tl.zeros((BT, BN), tl.float32)
    for a in range(0, A):
        slot = tl.load(ip + a, mask=mt, other=0)
        acc += tl.load(DXS + slot.to(tl.int64)[:, None] * D + rn[None, :],
                       mask=keep, other=0.0)
    re = tl.arange(0, BE)
    me = re < E
    dl = tl.load(DLOG + rt[:, None].to(tl.int64) * E + re[None, :],
                 mask=mt[:, None] & me[None, :], other=0.0)
    rwt = tl.load(RW + rn[None, :].to(tl.int64) * E + re[:, None],
                  mask=me[:, None] & mnd[None, :], other=0.0)
    acc += _dotp(dl, rwt, LV, LX)
    tl.store(DX + rt.to(tl.int64)[:, None] * D + rn[None, :],
             acc.to(DX.dtype.element_ty), mask=keep)


@triton.jit
def _drw_kernel(X, DLOG, DRW, T, D, E, CHUNK,
                BM: tl.constexpr, BE: tl.constexpr, BK: tl.constexpr,
                LX: tl.constexpr, LV: tl.constexpr):
    """dRW = X^T @ DLOG, the token axis cut into fixed-size chunks."""
    rd = tl.program_id(0) * BM + tl.arange(0, BM)
    j = tl.program_id(1)
    re = tl.arange(0, BE)
    md = rd < D
    me = re < E
    t0 = j * CHUNK
    tend = tl.minimum(t0 + CHUNK, T)
    acc = tl.zeros((BM, BE), tl.float32)
    for k0 in range(t0, tend, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < tend
        a = tl.load(X + rk[None, :].to(tl.int64) * D + rd[:, None],
                    mask=md[:, None] & mk[None, :], other=0.0)
        b = tl.load(DLOG + rk[:, None].to(tl.int64) * E + re[None, :],
                    mask=mk[:, None] & me[None, :], other=0.0)
        acc += _dotp(a, b, LX, LV)
    tl.store(DRW + j.to(tl.int64) * (D * E) + rd[:, None] * E + re[None, :],
             acc, mask=md[:, None] & me[None, :])


@triton.jit
def _reduce_kernel(PART, OUT, n, NJ, BLK: tl.constexpr):
    """Sum chunk partials into one buffer, chunks in index order."""
    o = tl.program_id(0) * BLK + tl.arange(0, BLK)
    m = o < n
    acc = tl.zeros((BLK,), tl.float32)
    for j in range(0, NJ):
        acc += tl.load(PART + j.to(tl.int64) * n + o, mask=m, other=0.0)
    tl.store(OUT + o, acc.to(OUT.dtype.element_ty), mask=m)


# --------------------------------------------------------------------------
# launch configuration
# --------------------------------------------------------------------------

def _p2(n):
    return 1 << max(0, (int(n) - 1).bit_length())


def _mtiles(n, BM, E):
    """Upper bound on row tiles: the slots split over at most E groups."""
    return triton.cdiv(n, BM) + E


# Per-role tile shapes, indexed by dtype. Each entry is (BM, BN, BK, warps,
# stages) for the shape-independent part; the launch clamps the tile to the
# actual extents. The float32 rows carry three limbs per operand, so their
# tiles are sized for the extra operand registers and shared memory.
_TILES = {
    #                  bfloat16                    float32
    "gemm1": ((128, 64, 64, 8, 3), (128, 64, 64, 8, 3)),
    "gemm2": ((64, 128, 64, 4, 4), (128, 128, 32, 8, 3)),
    "dact":  ((64, 128, 64, 4, 4), (128, 64, 32, 4, 3)),
    "dxs":   ((64, 128, 64, 4, 4), (128, 128, 64, 8, 3)),
    "dw1":   ((128, 64, 64, 4, 2), (128, 128, 64, 8, 3)),
    "dw2":   ((64, 128, 128, 4, 4), (128, 128, 64, 8, 3)),
}


def _cfg(role, m, n, k, f32):
    """Tile shape for one GEMM role, clamped to the extents it will see.

    m is the rows the tile walks (slots per expert, or an output extent), n
    and k the output and contraction extents. Only shape metadata enters.
    """
    BM, BN, BK, nw, ns = _TILES[role][1 if f32 else 0]
    BM = min(BM, max(16, _p2(m)))
    BN = min(BN, max(16, _p2(n)))
    BK = min(BK, max(16, _p2(k)))
    while BM * BN > 16384:
        if BN > BM:
            BN //= 2
        else:
            BM //= 2
    if BM * BN <= 2048:
        nw = min(nw, 4)
    return BM, BN, BK, nw, ns


def _route(x, router_w, router_b, A, E):
    """Scores, keys and the expert-major slot table."""
    T, D = x.shape
    dev = x.device
    f32 = x.dtype == torch.float32
    BE = max(16, _p2(E))
    s = torch.empty(T, E, device=dev, dtype=torch.float32)
    key = torch.empty(T, E, device=dev, dtype=torch.int64)
    BT = 64
    _router_kernel[(triton.cdiv(T, BT),)](
        x, router_w, router_b, s, key, T, D, E,
        BT=BT, BE=BE, BK=32 if f32 else 64, LX=3 if f32 else 1,
        num_warps=8, num_stages=3)
    thr = torch.topk(key, A, dim=-1, largest=True,
                     sorted=True).values[:, A - 1].contiguous()

    BTM = 64
    while triton.cdiv(T, BTM) > 2048:
        BTM *= 2
    nb = triton.cdiv(T, BTM)
    cnt = torch.empty(nb, E, device=dev, dtype=torch.int32)
    _count_kernel[(nb,)](key, thr, cnt, T, E, BT=BTM, BE=BE, num_warps=8)
    boff = torch.empty(nb, E, device=dev, dtype=torch.int32)
    cnte = torch.empty(E, device=dev, dtype=torch.int32)
    _scan_kernel[(E,)](cnt, boff, cnte, nb, E, NB2=_p2(nb), num_warps=4)
    offs = torch.empty(E + 1, device=dev, dtype=torch.int32)
    _offs_kernel[(1,)](cnte, offs, E, BE=BE, num_warps=4)

    N = T * A
    rows = torch.empty(N, device=dev, dtype=torch.int32)
    slote = torch.empty(N, device=dev, dtype=torch.int32)
    wgt = torch.empty(N, device=dev, dtype=torch.float32)
    ssel = torch.empty(N, device=dev, dtype=torch.float32)
    tot = torch.empty(T, device=dev, dtype=torch.float32)
    inv = torch.empty(T, A, device=dev, dtype=torch.int32)
    _scatter_kernel[(nb,)](key, thr, s, boff, offs, rows, slote, wgt, ssel,
                           tot, inv, T, E, A, BT=BTM, BE=BE, num_warps=8)
    return rows, slote, wgt, ssel, tot, inv, cnte, offs


class _FusedMoe(torch.autograd.Function):
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
        dev = x.device
        f32 = x.dtype == torch.float32
        LX = 3 if f32 else 1
        LV = 3 if f32 else 2
        BE = max(16, _p2(E))
        N = T * A
        rpe = triton.cdiv(N, E)

        rows, slote, wgt, ssel, tot, inv, cnte, offs = _route(
            x, router_w, router_b.float().contiguous(), A, E)

        gu = torch.empty(N, 2 * F, device=dev, dtype=torch.float32)
        act = torch.empty(N, F, device=dev, dtype=torch.float32)
        BM, BN, BK, nw, ns = _cfg("gemm1", rpe, F, D, f32)
        _gemm1_kernel[(_mtiles(N, BM, E), triton.cdiv(F, BN))](
            x, w1, rows, cnte, offs, gu, act, D, F, E,
            BM=BM, BN=BN, BK=BK, BE=BE, LX=LX, num_warps=nw, num_stages=ns)

        ys = torch.empty(N, D, device=dev, dtype=torch.float32)
        BM, BN, BK, nw, ns = _cfg("gemm2", rpe, D, F, f32)
        _gemm2_kernel[(_mtiles(N, BM, E), triton.cdiv(D, BN))](
            act, w2, wgt, cnte, offs, ys, D, F, E,
            BM=BM, BN=BN, BK=BK, BE=BE, LX=LX, LV=LV, num_warps=nw,
            num_stages=ns)

        y = torch.empty(T, D, device=dev, dtype=x.dtype)
        BN = min(128, max(16, _p2(D)))
        BT = max(1, 4096 // BN)
        _combine_kernel[(triton.cdiv(T, BT), triton.cdiv(D, BN))](
            ys, inv, y, T, D, A, BT=BT, BN=BN, num_warps=4, num_stages=2)

        ctx.save_for_backward(x, router_w, w1, w2, rows, slote, wgt, ssel,
                              tot, inv, cnte, offs, gu, act)
        ctx.meta = (T, D, F, E, A)
        return y

    @staticmethod
    def backward(ctx, dy):
        (x, router_w, w1, w2, rows, slote, wgt, ssel, tot, inv, cnte, offs,
         gu, act) = ctx.saved_tensors
        T, D, F, E, A = ctx.meta
        dev = x.device
        f32 = x.dtype == torch.float32
        LX = 3 if f32 else 1
        LV = 3 if f32 else 2
        BE = max(16, _p2(E))
        N = T * A
        rpe = triton.cdiv(N, E)
        dy = dy.contiguous()

        dgu = torch.empty(N, 2 * F, device=dev, dtype=torch.float32)
        BM, BN, BK, nw, ns = _cfg("dact", rpe, F, D, f32)
        nj = triton.cdiv(F, BN)
        rpart = torch.empty(nj, N, device=dev, dtype=torch.float32)
        _dact_kernel[(_mtiles(N, BM, E), nj)](
            dy, w2, act, gu, rows, wgt, cnte, offs, dgu, rpart, N, D, F, E,
            BM=BM, BN=BN, BK=BK, BE=BE, LX=LX, LV=LV, num_warps=nw,
            num_stages=ns)

        rd = torch.empty(N, device=dev, dtype=torch.float32)
        _rd_kernel[(triton.cdiv(N, 512),)](rpart, rd, N, nj, BLK=512,
                                           num_warps=4)

        dlog = torch.zeros(T, E, device=dev, dtype=torch.float32)
        BA = max(1, _p2(A))
        BT = max(1, 4096 // BA)
        _dscore_kernel[(triton.cdiv(T, BT),)](
            rd, wgt, ssel, slote, tot, inv, dlog, T, E, A, BT=BT, BA=BA,
            num_warps=4)

        dw2 = torch.empty(E, F, D, device=dev, dtype=x.dtype)
        BM, BN, BK, nw, ns = _cfg("dw2", F, D, rpe, f32)
        _dw2_kernel[(triton.cdiv(F, BM), triton.cdiv(D, BN), E)](
            act, dy, rows, wgt, cnte, offs, dw2, D, F, E,
            BM=BM, BN=BN, BK=BK, LX=LX, LV=LV, num_warps=nw, num_stages=ns)

        dw1 = torch.empty(E, D, 2 * F, device=dev, dtype=x.dtype)
        BM, BN, BK, nw, ns = _cfg("dw1", D, 2 * F, rpe, f32)
        _dw1_kernel[(triton.cdiv(D, BM), triton.cdiv(2 * F, BN), E)](
            x, dgu, rows, cnte, offs, dw1, D, F, E,
            BM=BM, BN=BN, BK=BK, LX=LX, LV=LV, num_warps=nw, num_stages=ns)

        dxs = torch.empty(N, D, device=dev, dtype=torch.float32)
        BM, BN, BK, nw, ns = _cfg("dxs", rpe, D, 2 * F, f32)
        _dxs_kernel[(_mtiles(N, BM, E), triton.cdiv(D, BN))](
            dgu, w1, cnte, offs, dxs, D, F, E,
            BM=BM, BN=BN, BK=BK, BE=BE, LX=LX, LV=LV, num_warps=nw,
            num_stages=ns)

        dx = torch.empty(T, D, device=dev, dtype=x.dtype)
        BN = min(128, max(16, _p2(D)))
        BT = max(1, 2048 // BN)
        _dxcomb_kernel[(triton.cdiv(T, BT), triton.cdiv(D, BN))](
            dxs, inv, dlog, router_w, dx, T, D, E, A,
            BT=BT, BN=BN, BE=BE, LX=LX, LV=LV, num_warps=4, num_stages=2)

        BM = min(64, max(16, _p2(D)))
        nchunk = max(1, min(triton.cdiv(T, 128),
                            256 // max(1, triton.cdiv(D, BM))))
        chunk = triton.cdiv(triton.cdiv(T, nchunk), 32) * 32
        nchunk = triton.cdiv(T, chunk)
        part = torch.empty(nchunk, D, E, device=dev, dtype=torch.float32)
        _drw_kernel[(triton.cdiv(D, BM), nchunk)](
            x, dlog, part, T, D, E, chunk, BM=BM, BE=BE, BK=32,
            LX=3 if f32 else 1, LV=3, num_warps=8, num_stages=3)
        drw = torch.empty(D, E, device=dev, dtype=x.dtype)
        _reduce_kernel[(triton.cdiv(D * E, 512),)](
            part, drw, D * E, nchunk, BLK=512, num_warps=4)

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
