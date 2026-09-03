"""Private oracle -- from-scratch Triton fused MoE + SwiGLU, forward + backward.

UNVERIFIED: authored on a CPU-only box without GPU calibration. The structure
mirrors the proven moe_swiglu category oracle (which was measured to reach full
reward through every live gate), but this exact copy has NOT been run on a GPU
here. Treat kernel-launch parameters, tile sizes and the fp32/dtype policy as
a starting point to be validated during GPU calibration, not as a measured
result. Never shipped to an agent. It establishes that full reward is
reachable through every live gate (G1 symbol scan, G2 dense+grad correctness
with bitwise determinism, G3 timing, G4 adoption) with no MoE library call
anywhere.

Plan (all deterministic; no atomic adds anywhere):

  prep (torch, tiny): sort the T*A routing slots by expert (stable argsort so
  slots of one expert stay in (token, slot) order), per-expert segment
  offsets, the inverse permutation, and a tile map assigning GM-row tiles to
  expert segments.

  forward: (1) segment-tiled GEMM over gathered token rows against both
  halves of w1 with a fused silu(gate)*up epilogue; (2) segment-tiled GEMM
  against w2 producing per-slot partial outputs; (3) per-token ordered
  combine, ascending slot, accumulating in float32 -- fixed order, so the
  result is reproducible bit for bit.

  backward (custom autograd.Function): d_partial = w_slot * dY[token];
  dtopk_w by per-slot dot(dY[token], y_partial); dact via segment GEMM
  against w2^T with the exact silu backward fused in its epilogue; dx_slot
  via segment GEMM against w1^T; dx by the same ordered per-token combine;
  dW2/dW1 by per-expert GEMMs whose K loop walks the expert's segment
  sequentially -- one accumulation chain per output tile, so weight
  gradients are deterministic too. Experts no token routes to store zeros.

  Transposed operands are loaded through strides; nothing is ever copied.
  Intermediates are stored in the input dtype (the production-realistic
  choice); every tl.dot accumulates in float32.
"""

import torch
import triton
import triton.language as tl

GM, GN = 128, 128     # segment GEMM tile (rows x cols)
GKD = 32              # segment GEMM K-step
WM, WN, WK = 64, 64, 32   # per-expert weight-gradient GEMM tile
CT, CD = 32, 128      # combine tile (tokens x features)
SB, SD = 64, 128      # slot-dot tile (slots x feature-step)


@triton.jit
def _fwd_gu_kernel(X, W1, TILE_E, TILE_ROW, TILE_END, SRC_TOK, GU, ACT,
                   D, F, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """Segment GEMM 1: gu[s] = x[tok(s)] @ w1[e], epilogue act = silu(g)*u."""
    pid_t = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1).to(tl.int64)
    e = tl.load(TILE_E + pid_t).to(tl.int64)
    row0 = tl.load(TILE_ROW + pid_t)
    end = tl.load(TILE_END + pid_t)
    rows = row0 + tl.arange(0, BM).to(tl.int64)
    mrow = rows < end
    tok = tl.load(SRC_TOK + rows, mask=mrow, other=0)
    cf = pid_n * BN + tl.arange(0, BN).to(tl.int64)
    mcf = cf < F
    acc_g = tl.zeros((BM, BN), dtype=tl.float32)
    acc_u = tl.zeros((BM, BN), dtype=tl.float32)
    wbase = W1 + e * D * 2 * F
    for k0 in range(0, D, BK):
        rk = k0 + tl.arange(0, BK).to(tl.int64)
        mk = rk < D
        a = tl.load(X + tok[:, None] * D + rk[None, :],
                    mask=mrow[:, None] & mk[None, :], other=0.0)
        bg = tl.load(wbase + rk[:, None] * 2 * F + cf[None, :],
                     mask=mk[:, None] & mcf[None, :], other=0.0)
        bu = tl.load(wbase + rk[:, None] * 2 * F + F + cf[None, :],
                     mask=mk[:, None] & mcf[None, :], other=0.0)
        acc_g += tl.dot(a, bg)
        acc_u += tl.dot(a, bu)
    m2 = mrow[:, None] & mcf[None, :]
    tl.store(GU + rows[:, None] * 2 * F + cf[None, :], acc_g, mask=m2)
    tl.store(GU + rows[:, None] * 2 * F + F + cf[None, :], acc_u, mask=m2)
    s = tl.sigmoid(acc_g)
    tl.store(ACT + rows[:, None] * F + cf[None, :], acc_g * s * acc_u, mask=m2)


@triton.jit
def _fwd_out_kernel(ACT, W2, TILE_E, TILE_ROW, TILE_END, YP,
                    D, F, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """Segment GEMM 2: y_partial[s] = act[s] @ w2[e]."""
    pid_t = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1).to(tl.int64)
    e = tl.load(TILE_E + pid_t).to(tl.int64)
    row0 = tl.load(TILE_ROW + pid_t)
    end = tl.load(TILE_END + pid_t)
    rows = row0 + tl.arange(0, BM).to(tl.int64)
    mrow = rows < end
    cd = pid_n * BN + tl.arange(0, BN).to(tl.int64)
    mcd = cd < D
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    wbase = W2 + e * F * D
    for k0 in range(0, F, BK):
        rk = k0 + tl.arange(0, BK).to(tl.int64)
        mk = rk < F
        a = tl.load(ACT + rows[:, None] * F + rk[None, :],
                    mask=mrow[:, None] & mk[None, :], other=0.0)
        b = tl.load(wbase + rk[:, None] * D + cd[None, :],
                    mask=mk[:, None] & mcd[None, :], other=0.0)
        acc += tl.dot(a, b)
    tl.store(YP + rows[:, None] * D + cd[None, :], acc,
             mask=mrow[:, None] & mcd[None, :])


@triton.jit
def _combine_fwd_kernel(YP, WSLOT, INV, Y, T, D, A,
                        BT: tl.constexpr, BD: tl.constexpr):
    """y[t] = sum_a w[t,a] * y_partial[slot(t,a)], slot order ascending: one
    program owns each output tile and walks the A slots sequentially, so the
    accumulation order is fixed and there are no atomics."""
    pid_t = tl.program_id(0).to(tl.int64)
    pid_d = tl.program_id(1).to(tl.int64)
    t = pid_t * BT + tl.arange(0, BT).to(tl.int64)
    mt = t < T
    d = pid_d * BD + tl.arange(0, BD).to(tl.int64)
    md = d < D
    m2 = mt[:, None] & md[None, :]
    acc = tl.zeros((BT, BD), dtype=tl.float32)
    for a in range(A):
        s = tl.load(INV + t * A + a, mask=mt, other=0)
        w = tl.load(WSLOT + s, mask=mt, other=0.0).to(tl.float32)
        yp = tl.load(YP + s[:, None] * D + d[None, :], mask=m2, other=0.0)
        acc += w[:, None] * yp.to(tl.float32)
    tl.store(Y + t[:, None] * D + d[None, :], acc, mask=m2)


@triton.jit
def _bwd_dyp_kernel(DY, WSLOT, SRC_TOK, DYP, S, D,
                    BT: tl.constexpr, BD: tl.constexpr):
    """d_partial[s] = w_slot[s] * dY[tok(s)]."""
    pid_s = tl.program_id(0).to(tl.int64)
    pid_d = tl.program_id(1).to(tl.int64)
    s = pid_s * BT + tl.arange(0, BT).to(tl.int64)
    ms = s < S
    d = pid_d * BD + tl.arange(0, BD).to(tl.int64)
    md = d < D
    m2 = ms[:, None] & md[None, :]
    tok = tl.load(SRC_TOK + s, mask=ms, other=0)
    w = tl.load(WSLOT + s, mask=ms, other=0.0).to(tl.float32)
    dy = tl.load(DY + tok[:, None] * D + d[None, :], mask=m2, other=0.0)
    tl.store(DYP + s[:, None] * D + d[None, :], w[:, None] * dy.to(tl.float32),
             mask=m2)


@triton.jit
def _bwd_dtw_kernel(DY, YP, SRC_TOK, ORDER, DTW, S, D,
                    BS: tl.constexpr, BD: tl.constexpr):
    """dtopk_w[t,a] = dot(dY[t], y_partial[slot(t,a)]): one program per slot
    block, feature loop sequential, result scattered through the permutation
    (no duplicates, so no atomics)."""
    pid = tl.program_id(0).to(tl.int64)
    s = pid * BS + tl.arange(0, BS).to(tl.int64)
    ms = s < S
    tok = tl.load(SRC_TOK + s, mask=ms, other=0)
    pos = tl.load(ORDER + s, mask=ms, other=0)
    acc = tl.zeros((BS,), dtype=tl.float32)
    for d0 in range(0, D, BD):
        d = d0 + tl.arange(0, BD).to(tl.int64)
        md = d < D
        m2 = ms[:, None] & md[None, :]
        dy = tl.load(DY + tok[:, None] * D + d[None, :], mask=m2, other=0.0)
        yp = tl.load(YP + s[:, None] * D + d[None, :], mask=m2, other=0.0)
        acc += tl.sum(dy.to(tl.float32) * yp.to(tl.float32), axis=1)
    tl.store(DTW + pos, acc, mask=ms)


@triton.jit
def _bwd_dact_dgu_kernel(DYP, W2, GU, TILE_E, TILE_ROW, TILE_END, DGU,
                         D, F, BM: tl.constexpr, BN: tl.constexpr,
                         BK: tl.constexpr):
    """dact = d_partial @ w2^T (transpose through strides), then the exact
    silu backward in the epilogue: dg = dact*u*(s + silu*(1-s)), du = dact*silu."""
    pid_t = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1).to(tl.int64)
    e = tl.load(TILE_E + pid_t).to(tl.int64)
    row0 = tl.load(TILE_ROW + pid_t)
    end = tl.load(TILE_END + pid_t)
    rows = row0 + tl.arange(0, BM).to(tl.int64)
    mrow = rows < end
    cf = pid_n * BN + tl.arange(0, BN).to(tl.int64)
    mcf = cf < F
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    wbase = W2 + e * F * D
    for k0 in range(0, D, BK):
        rk = k0 + tl.arange(0, BK).to(tl.int64)
        mk = rk < D
        a = tl.load(DYP + rows[:, None] * D + rk[None, :],
                    mask=mrow[:, None] & mk[None, :], other=0.0)
        b = tl.load(wbase + cf[None, :] * D + rk[:, None],
                    mask=mcf[None, :] & mk[:, None], other=0.0)
        acc += tl.dot(a, b)
    m2 = mrow[:, None] & mcf[None, :]
    g = tl.load(GU + rows[:, None] * 2 * F + cf[None, :], mask=m2,
                other=0.0).to(tl.float32)
    u = tl.load(GU + rows[:, None] * 2 * F + F + cf[None, :], mask=m2,
                other=0.0).to(tl.float32)
    s = tl.sigmoid(g)
    silu = g * s
    tl.store(DGU + rows[:, None] * 2 * F + cf[None, :],
             acc * u * (s + silu * (1.0 - s)), mask=m2)
    tl.store(DGU + rows[:, None] * 2 * F + F + cf[None, :], acc * silu, mask=m2)


@triton.jit
def _bwd_dxslot_kernel(DGU, W1, TILE_E, TILE_ROW, TILE_END, DXS,
                       D, F, BM: tl.constexpr, BN: tl.constexpr,
                       BK: tl.constexpr):
    """dx_slot = dgu @ w1^T (transpose through strides), K over 2F."""
    pid_t = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1).to(tl.int64)
    e = tl.load(TILE_E + pid_t).to(tl.int64)
    row0 = tl.load(TILE_ROW + pid_t)
    end = tl.load(TILE_END + pid_t)
    rows = row0 + tl.arange(0, BM).to(tl.int64)
    mrow = rows < end
    cd = pid_n * BN + tl.arange(0, BN).to(tl.int64)
    mcd = cd < D
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    wbase = W1 + e * D * 2 * F
    for k0 in range(0, 2 * F, BK):
        rk = k0 + tl.arange(0, BK).to(tl.int64)
        mk = rk < 2 * F
        a = tl.load(DGU + rows[:, None] * 2 * F + rk[None, :],
                    mask=mrow[:, None] & mk[None, :], other=0.0)
        b = tl.load(wbase + cd[None, :] * 2 * F + rk[:, None],
                    mask=mcd[None, :] & mk[:, None], other=0.0)
        acc += tl.dot(a, b)
    tl.store(DXS + rows[:, None] * D + cd[None, :], acc,
             mask=mrow[:, None] & mcd[None, :])


@triton.jit
def _combine_bwd_kernel(DXS, INV, DX, T, D, A,
                        BT: tl.constexpr, BD: tl.constexpr):
    """dx[t] = sum_a dx_slot[slot(t,a)], ascending slot, no atomics."""
    pid_t = tl.program_id(0).to(tl.int64)
    pid_d = tl.program_id(1).to(tl.int64)
    t = pid_t * BT + tl.arange(0, BT).to(tl.int64)
    mt = t < T
    d = pid_d * BD + tl.arange(0, BD).to(tl.int64)
    md = d < D
    m2 = mt[:, None] & md[None, :]
    acc = tl.zeros((BT, BD), dtype=tl.float32)
    for a in range(A):
        s = tl.load(INV + t * A + a, mask=mt, other=0)
        v = tl.load(DXS + s[:, None] * D + d[None, :], mask=m2, other=0.0)
        acc += v.to(tl.float32)
    tl.store(DX + t[:, None] * D + d[None, :], acc, mask=m2)


@triton.jit
def _bwd_dw2_kernel(ACT, DYP, STARTS, COUNTS, DW2, D, F,
                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """dW2[e] = act_seg^T @ d_partial_seg: one program per (expert, F-tile,
    D-tile), K loop walking the segment sequentially -- a single accumulation
    chain per output tile, deterministic; empty experts store zeros."""
    e = tl.program_id(0).to(tl.int64)
    pid_f = tl.program_id(1).to(tl.int64)
    pid_d = tl.program_id(2).to(tl.int64)
    st = tl.load(STARTS + e)
    cnt = tl.load(COUNTS + e)
    rf = pid_f * BM + tl.arange(0, BM).to(tl.int64)
    mf = rf < F
    rd = pid_d * BN + tl.arange(0, BN).to(tl.int64)
    md = rd < D
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, cnt, BK):
        rows = st + k0 + tl.arange(0, BK).to(tl.int64)
        mk = (k0 + tl.arange(0, BK).to(tl.int64)) < cnt
        a = tl.load(ACT + rows[:, None] * F + rf[None, :],
                    mask=mk[:, None] & mf[None, :], other=0.0)
        b = tl.load(DYP + rows[:, None] * D + rd[None, :],
                    mask=mk[:, None] & md[None, :], other=0.0)
        acc += tl.dot(tl.trans(a), b)
    tl.store(DW2 + e * F * D + rf[:, None] * D + rd[None, :], acc,
             mask=mf[:, None] & md[None, :])


@triton.jit
def _bwd_dw1_kernel(X, DGU, SRC_TOK, STARTS, COUNTS, DW1, D, F,
                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """dW1[e] = x_seg^T @ dgu_seg, x rows gathered through src_tok; same
    deterministic per-tile accumulation chain as dW2."""
    e = tl.program_id(0).to(tl.int64)
    pid_d = tl.program_id(1).to(tl.int64)
    pid_c = tl.program_id(2).to(tl.int64)
    st = tl.load(STARTS + e)
    cnt = tl.load(COUNTS + e)
    rd = pid_d * BM + tl.arange(0, BM).to(tl.int64)
    md = rd < D
    rc = pid_c * BN + tl.arange(0, BN).to(tl.int64)
    mc = rc < 2 * F
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, cnt, BK):
        rows = st + k0 + tl.arange(0, BK).to(tl.int64)
        mk = (k0 + tl.arange(0, BK).to(tl.int64)) < cnt
        tok = tl.load(SRC_TOK + rows, mask=mk, other=0)
        a = tl.load(X + tok[:, None] * D + rd[None, :],
                    mask=mk[:, None] & md[None, :], other=0.0)
        b = tl.load(DGU + rows[:, None] * 2 * F + rc[None, :],
                    mask=mk[:, None] & mc[None, :], other=0.0)
        acc += tl.dot(tl.trans(a), b)
    tl.store(DW1 + e * D * 2 * F + rd[:, None] * 2 * F + rc[None, :], acc,
             mask=md[:, None] & mc[None, :])


def _prep(topk_idx: torch.Tensor, topk_w: torch.Tensor, E: int):
    """Routing prep: stable expert sort, segment offsets, inverse map, tile map.
    All integer torch ops (order-free or permutation writes): deterministic."""
    T, A = topk_idx.shape
    dev = topk_idx.device
    flat_idx = topk_idx.reshape(-1)
    order = torch.argsort(flat_idx, stable=True)          # [S]
    counts = torch.bincount(flat_idx, minlength=E)        # [E]
    ends = counts.cumsum(0)
    starts = ends - counts
    src_tok = order // A                                  # token of sorted slot
    inv = torch.empty_like(order)
    inv[order] = torch.arange(T * A, device=dev)
    w_slot = topk_w.reshape(-1).index_select(0, order).contiguous()
    tiles = (counts + (GM - 1)) // GM
    tile_cum = tiles.cumsum(0)
    n_tiles = int(tile_cum[-1])
    tile_expert = torch.repeat_interleave(
        torch.arange(E, device=dev, dtype=torch.int32), tiles)
    first = (tile_cum - tiles).index_select(0, tile_expert.long())
    tile_row = starts.index_select(0, tile_expert.long()) + \
        (torch.arange(n_tiles, device=dev) - first) * GM
    tile_end = ends.index_select(0, tile_expert.long())
    return (order.contiguous(), counts.contiguous(), starts.contiguous(),
            src_tok.contiguous(), inv.contiguous(), w_slot, n_tiles,
            tile_expert.contiguous(), tile_row.contiguous(),
            tile_end.contiguous())


class _WideExpertSwigluMoeOracle(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, w2, topk_idx, topk_w):
        T, D = x.shape
        E, _, F2 = w1.shape
        F = F2 // 2
        A = topk_idx.shape[1]
        S = T * A
        x = x.contiguous()
        w1 = w1.contiguous()
        w2 = w2.contiguous()
        tw = topk_w.contiguous()
        (order, counts, starts, src_tok, inv, w_slot, n_tiles,
         tile_e, tile_row, tile_end) = _prep(topk_idx, tw, E)
        dt = x.dtype
        gu = torch.empty(S, 2 * F, device=x.device, dtype=dt)
        act = torch.empty(S, F, device=x.device, dtype=dt)
        yp = torch.empty(S, D, device=x.device, dtype=dt)
        y = torch.empty(T, D, device=x.device, dtype=dt)
        _fwd_gu_kernel[(n_tiles, triton.cdiv(F, GN))](
            x, w1, tile_e, tile_row, tile_end, src_tok, gu, act, D, F,
            BM=GM, BN=GN, BK=GKD, num_warps=8)
        _fwd_out_kernel[(n_tiles, triton.cdiv(D, GN))](
            act, w2, tile_e, tile_row, tile_end, yp, D, F,
            BM=GM, BN=GN, BK=GKD, num_warps=8)
        _combine_fwd_kernel[(triton.cdiv(T, CT), triton.cdiv(D, CD))](
            yp, w_slot, inv, y, T, D, A, BT=CT, BD=CD)
        ctx.save_for_backward(x, w1, w2, tw, gu, act, yp, order, counts,
                              starts, src_tok, inv, w_slot, tile_e, tile_row,
                              tile_end)
        ctx.dims = (T, D, F, E, A, S, n_tiles)
        return y

    @staticmethod
    def backward(ctx, dy):
        (x, w1, w2, tw, gu, act, yp, order, counts, starts, src_tok, inv,
         w_slot, tile_e, tile_row, tile_end) = ctx.saved_tensors
        T, D, F, E, A, S, n_tiles = ctx.dims
        dy = dy.contiguous()
        dt = x.dtype
        dyp = torch.empty(S, D, device=x.device, dtype=dt)
        dtw = torch.empty(S, device=x.device, dtype=tw.dtype)
        dgu = torch.empty(S, 2 * F, device=x.device, dtype=dt)
        dxs = torch.empty(S, D, device=x.device, dtype=dt)
        dx = torch.empty(T, D, device=x.device, dtype=dt)
        dw1 = torch.empty(E, D, 2 * F, device=x.device, dtype=w1.dtype)
        dw2 = torch.empty(E, F, D, device=x.device, dtype=w2.dtype)
        _bwd_dyp_kernel[(triton.cdiv(S, CT), triton.cdiv(D, CD))](
            dy, w_slot, src_tok, dyp, S, D, BT=CT, BD=CD)
        _bwd_dtw_kernel[(triton.cdiv(S, SB),)](
            dy, yp, src_tok, order, dtw, S, D, BS=SB, BD=SD)
        _bwd_dact_dgu_kernel[(n_tiles, triton.cdiv(F, GN))](
            dyp, w2, gu, tile_e, tile_row, tile_end, dgu, D, F,
            BM=GM, BN=GN, BK=GKD, num_warps=8)
        _bwd_dxslot_kernel[(n_tiles, triton.cdiv(D, GN))](
            dgu, w1, tile_e, tile_row, tile_end, dxs, D, F,
            BM=GM, BN=GN, BK=GKD, num_warps=8)
        _combine_bwd_kernel[(triton.cdiv(T, CT), triton.cdiv(D, CD))](
            dxs, inv, dx, T, D, A, BT=CT, BD=CD)
        _bwd_dw2_kernel[(E, triton.cdiv(F, WM), triton.cdiv(D, WN))](
            act, dyp, starts, counts, dw2, D, F, BM=WM, BN=WN, BK=WK)
        _bwd_dw1_kernel[(E, triton.cdiv(D, WM), triton.cdiv(2 * F, WN))](
            x, dgu, src_tok, starts, counts, dw1, D, F, BM=WM, BN=WN, BK=WK)
        return dx, dw1, dw2, None, dtw.view(T, A)


def wide_expert_swiglu_moe(
    x: torch.Tensor,         # [T, D] float32 or bfloat16
    w1: torch.Tensor,        # [E, D, 2F]  [:, :, :F] gate, [:, :, F:] up
    w2: torch.Tensor,        # [E, F, D]
    topk_idx: torch.Tensor,  # [T, A] int64, values in [0, E), no gradient
    topk_w: torch.Tensor,    # [T, A] same dtype as x, normalized
) -> torch.Tensor:           # [T, D] same dtype as x
    return _WideExpertSwigluMoeOracle.apply(x, w1, w2, topk_idx, topk_w)
# FORGE-CANARY-BEGIN forge-canary/v1
# FORGE-CANARY-SLOT-0 ee63dc6e01c821ba229fe54a1b17b536b4e12f77393482fb462da30ada6c7758
# FORGE-CANARY-SLOT-1 6cc28039e2c7ac13f582035bcb25468f68dca60367d553d735b2c4eb3d6f5954
# FORGE-CANARY-SLOT-2 2f82a766307e31a9f5ea6337a11f15c1db255764d7ede76ca7b45f9b5e461f75
# FORGE-CANARY-SLOT-3 4c8cd138e01315ef2dec00edbeedc16e701d4e642b5d364680b922189959c5dc
# FORGE-CANARY-END
