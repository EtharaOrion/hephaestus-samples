"""Private oracle -- from-scratch Triton fp8-e4m3 weight-only SwiGLU MoE, fwd+bwd.

UNVERIFIED: authored on a CPU-only box without GPU calibration. Structure
mirrors the moe_swiglu category oracle's ROUTED path but adds per-output-
channel dequant into the MMA prologue: every W tile is multiplied by its
scale slice before feeding tl.dot. Weights arrive on the fp8-e4m3 grid stored
in x.dtype (a candidate wanting native e4m3 tensor cores can re-cast). Only
dx and dtopk_w are graded, so the per-expert weight-gradient kernels are
absent. No forbidden symbol anywhere: full reward is reachable through G1
(scan), G2 (dense+grad correctness with bitwise determinism), G3 (timing),
G4 (adoption) with the anchor's eager per-expert bf16 dequant pass replaced
by an in-prologue fusion.

All accumulations are deterministic: segment-tiled GEMM with a single
accumulation chain per output tile, ordered per-token slot combine, no
atomics. Tolerance floor is set by the fp8 grid noise; OUT_TOL in the
taskdef is calibrated to accept a candidate that consumes fp8 tensor cores
directly instead of the eager dequant this reference uses in its MMA prologue.
"""

import torch
import triton
import triton.language as tl

GM, GN = 128, 128
GKD = 32
CT, CD = 32, 128
SB, SD = 64, 128


@triton.jit
def _fwd_gu_kernel(
    X,
    W1Q,
    W1S,
    TILE_E,
    TILE_ROW,
    TILE_END,
    SRC_TOK,
    GU,
    ACT,
    D,
    F,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    """Segment GEMM 1 with per-out-channel fp8 dequant fused into the prologue:
    gu[s] = x[tok(s)] @ (w1_q[e] * w1_s[e][None, :]), epilogue act = silu(g)*u."""
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
    sg = tl.load(W1S + e * 2 * F + cf, mask=mcf, other=0.0).to(tl.float32)
    su = tl.load(W1S + e * 2 * F + F + cf, mask=mcf, other=0.0).to(tl.float32)
    acc_g = tl.zeros((BM, BN), dtype=tl.float32)
    acc_u = tl.zeros((BM, BN), dtype=tl.float32)
    wbase = W1Q + e * D * 2 * F
    for k0 in range(0, D, BK):
        rk = k0 + tl.arange(0, BK).to(tl.int64)
        mk = rk < D
        a = tl.load(
            X + tok[:, None] * D + rk[None, :],
            mask=mrow[:, None] & mk[None, :],
            other=0.0,
        )
        bg = tl.load(
            wbase + rk[:, None] * 2 * F + cf[None, :],
            mask=mk[:, None] & mcf[None, :],
            other=0.0,
        )
        bu = tl.load(
            wbase + rk[:, None] * 2 * F + F + cf[None, :],
            mask=mk[:, None] & mcf[None, :],
            other=0.0,
        )
        acc_g += tl.dot(a, bg)
        acc_u += tl.dot(a, bu)
    # Per-out-channel dequant scale applied ONCE in the epilogue.
    acc_g = acc_g * sg[None, :]
    acc_u = acc_u * su[None, :]
    m2 = mrow[:, None] & mcf[None, :]
    tl.store(GU + rows[:, None] * 2 * F + cf[None, :], acc_g, mask=m2)
    tl.store(GU + rows[:, None] * 2 * F + F + cf[None, :], acc_u, mask=m2)
    s = tl.sigmoid(acc_g)
    tl.store(ACT + rows[:, None] * F + cf[None, :], acc_g * s * acc_u, mask=m2)


@triton.jit
def _fwd_out_kernel(
    ACT,
    W2Q,
    W2S,
    TILE_E,
    TILE_ROW,
    TILE_END,
    YP,
    D,
    F,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    """Segment GEMM 2 with per-out-channel dequant: y_partial[s] = act[s] @
    (w2_q[e] * w2_s[e][None, :])."""
    pid_t = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1).to(tl.int64)
    e = tl.load(TILE_E + pid_t).to(tl.int64)
    row0 = tl.load(TILE_ROW + pid_t)
    end = tl.load(TILE_END + pid_t)
    rows = row0 + tl.arange(0, BM).to(tl.int64)
    mrow = rows < end
    cd = pid_n * BN + tl.arange(0, BN).to(tl.int64)
    mcd = cd < D
    sc = tl.load(W2S + e * D + cd, mask=mcd, other=0.0).to(tl.float32)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    wbase = W2Q + e * F * D
    for k0 in range(0, F, BK):
        rk = k0 + tl.arange(0, BK).to(tl.int64)
        mk = rk < F
        a = tl.load(
            ACT + rows[:, None] * F + rk[None, :],
            mask=mrow[:, None] & mk[None, :],
            other=0.0,
        )
        b = tl.load(
            wbase + rk[:, None] * D + cd[None, :],
            mask=mk[:, None] & mcd[None, :],
            other=0.0,
        )
        acc += tl.dot(a, b)
    acc = acc * sc[None, :]
    tl.store(
        YP + rows[:, None] * D + cd[None, :], acc, mask=mrow[:, None] & mcd[None, :]
    )


@triton.jit
def _combine_fwd_kernel(YP, WSLOT, INV, Y, T, D, A, BT: tl.constexpr, BD: tl.constexpr):
    """y[t] = sum_a w[t,a] * y_partial[slot(t,a)], slot order ascending."""
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
def _bwd_dyp_kernel(DY, WSLOT, SRC_TOK, DYP, S, D, BT: tl.constexpr, BD: tl.constexpr):
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
    tl.store(DYP + s[:, None] * D + d[None, :], w[:, None] * dy.to(tl.float32), mask=m2)


@triton.jit
def _bwd_dtw_kernel(
    DY, YP, SRC_TOK, ORDER, DTW, S, D, BS: tl.constexpr, BD: tl.constexpr
):
    """dtopk_w[t,a] = dot(dY[t], y_partial[slot(t,a)])."""
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
def _bwd_dact_dgu_kernel(
    DYP,
    W2Q,
    W2S,
    GU,
    TILE_E,
    TILE_ROW,
    TILE_END,
    DGU,
    D,
    F,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    """dact = d_partial @ (w2_q * w2_s)^T, transpose through strides, dequant
    scale applied per K-slice, then silu backward in the epilogue."""
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
    wbase = W2Q + e * F * D
    for k0 in range(0, D, BK):
        rk = k0 + tl.arange(0, BK).to(tl.int64)
        mk = rk < D
        # Dequant scale for the K-slice of w2 (which are D-column indices).
        sc = tl.load(W2S + e * D + rk, mask=mk, other=0.0).to(tl.float32)
        a = tl.load(
            DYP + rows[:, None] * D + rk[None, :],
            mask=mrow[:, None] & mk[None, :],
            other=0.0,
        ).to(tl.float32)
        b = tl.load(
            wbase + cf[None, :] * D + rk[:, None],
            mask=mcf[None, :] & mk[:, None],
            other=0.0,
        ).to(tl.float32)
        # Fold the K-scale into a: (a * sc) @ b == a @ (sc * b) but the
        # scale is per-column of w2 (K axis here), so multiply a by sc.
        a = a * sc[None, :]
        acc += tl.dot(a, b)
    m2 = mrow[:, None] & mcf[None, :]
    g = tl.load(GU + rows[:, None] * 2 * F + cf[None, :], mask=m2, other=0.0).to(
        tl.float32
    )
    u = tl.load(GU + rows[:, None] * 2 * F + F + cf[None, :], mask=m2, other=0.0).to(
        tl.float32
    )
    s = tl.sigmoid(g)
    silu = g * s
    tl.store(
        DGU + rows[:, None] * 2 * F + cf[None, :],
        acc * u * (s + silu * (1.0 - s)),
        mask=m2,
    )
    tl.store(DGU + rows[:, None] * 2 * F + F + cf[None, :], acc * silu, mask=m2)


@triton.jit
def _bwd_dxslot_kernel(
    DGU,
    W1Q,
    W1S,
    TILE_E,
    TILE_ROW,
    TILE_END,
    DXS,
    D,
    F,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    """dx_slot = dgu @ (w1_q * w1_s)^T, K over 2F, per-K dequant folded into dgu."""
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
    wbase = W1Q + e * D * 2 * F
    for k0 in range(0, 2 * F, BK):
        rk = k0 + tl.arange(0, BK).to(tl.int64)
        mk = rk < 2 * F
        sc = tl.load(W1S + e * 2 * F + rk, mask=mk, other=0.0).to(tl.float32)
        a = tl.load(
            DGU + rows[:, None] * 2 * F + rk[None, :],
            mask=mrow[:, None] & mk[None, :],
            other=0.0,
        ).to(tl.float32)
        b = tl.load(
            wbase + cd[None, :] * 2 * F + rk[:, None],
            mask=mcd[None, :] & mk[:, None],
            other=0.0,
        ).to(tl.float32)
        a = a * sc[None, :]
        acc += tl.dot(a, b)
    tl.store(
        DXS + rows[:, None] * D + cd[None, :], acc, mask=mrow[:, None] & mcd[None, :]
    )


@triton.jit
def _combine_bwd_kernel(DXS, INV, DX, T, D, A, BT: tl.constexpr, BD: tl.constexpr):
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


def _prep(topk_idx, topk_w, E):
    T, A = topk_idx.shape
    dev = topk_idx.device
    flat_idx = topk_idx.reshape(-1)
    order = torch.argsort(flat_idx, stable=True)
    counts = torch.bincount(flat_idx, minlength=E)
    ends = counts.cumsum(0)
    starts = ends - counts
    src_tok = order // A
    inv = torch.empty_like(order)
    inv[order] = torch.arange(T * A, device=dev)
    w_slot = topk_w.reshape(-1).index_select(0, order).contiguous()
    tiles = (counts + (GM - 1)) // GM
    tile_cum = tiles.cumsum(0)
    n_tiles = int(tile_cum[-1])
    tile_expert = torch.repeat_interleave(
        torch.arange(E, device=dev, dtype=torch.int32), tiles
    )
    first = (tile_cum - tiles).index_select(0, tile_expert.long())
    tile_row = (
        starts.index_select(0, tile_expert.long())
        + (torch.arange(n_tiles, device=dev) - first) * GM
    )
    tile_end = ends.index_select(0, tile_expert.long())
    return (
        order.contiguous(),
        counts.contiguous(),
        starts.contiguous(),
        src_tok.contiguous(),
        inv.contiguous(),
        w_slot,
        n_tiles,
        tile_expert.contiguous(),
        tile_row.contiguous(),
        tile_end.contiguous(),
    )


class _Fp8E4m3SwigluMoeOracle(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1_q, w1_s, w2_q, w2_s, topk_idx, topk_w):
        T, D = x.shape
        E, _, F2 = w1_q.shape
        F = F2 // 2
        A = topk_idx.shape[1]
        S = T * A
        x = x.contiguous()
        w1_q = w1_q.contiguous()
        w1_s = w1_s.contiguous()
        w2_q = w2_q.contiguous()
        w2_s = w2_s.contiguous()
        tw = topk_w.contiguous()
        (
            order,
            counts,
            starts,
            src_tok,
            inv,
            w_slot,
            n_tiles,
            tile_e,
            tile_row,
            tile_end,
        ) = _prep(topk_idx, tw, E)
        dt = x.dtype
        gu = torch.empty(S, 2 * F, device=x.device, dtype=dt)
        act = torch.empty(S, F, device=x.device, dtype=dt)
        yp = torch.empty(S, D, device=x.device, dtype=dt)
        y = torch.empty(T, D, device=x.device, dtype=dt)
        _fwd_gu_kernel[(n_tiles, triton.cdiv(F, GN))](
            x,
            w1_q,
            w1_s,
            tile_e,
            tile_row,
            tile_end,
            src_tok,
            gu,
            act,
            D,
            F,
            BM=GM,
            BN=GN,
            BK=GKD,
            num_warps=8,
        )
        _fwd_out_kernel[(n_tiles, triton.cdiv(D, GN))](
            act,
            w2_q,
            w2_s,
            tile_e,
            tile_row,
            tile_end,
            yp,
            D,
            F,
            BM=GM,
            BN=GN,
            BK=GKD,
            num_warps=8,
        )
        _combine_fwd_kernel[(triton.cdiv(T, CT), triton.cdiv(D, CD))](
            yp, w_slot, inv, y, T, D, A, BT=CT, BD=CD
        )
        ctx.save_for_backward(
            x,
            w1_q,
            w1_s,
            w2_q,
            w2_s,
            tw,
            gu,
            act,
            yp,
            order,
            src_tok,
            inv,
            w_slot,
            tile_e,
            tile_row,
            tile_end,
        )
        ctx.dims = (T, D, F, E, A, S, n_tiles)
        return y

    @staticmethod
    def backward(ctx, dy):
        (
            x,
            w1_q,
            w1_s,
            w2_q,
            w2_s,
            tw,
            gu,
            act,
            yp,
            order,
            src_tok,
            inv,
            w_slot,
            tile_e,
            tile_row,
            tile_end,
        ) = ctx.saved_tensors
        T, D, F, E, A, S, n_tiles = ctx.dims
        dy = dy.contiguous()
        dt = x.dtype
        dyp = torch.empty(S, D, device=x.device, dtype=dt)
        dtw = torch.empty(S, device=x.device, dtype=tw.dtype)
        dgu = torch.empty(S, 2 * F, device=x.device, dtype=dt)
        dxs = torch.empty(S, D, device=x.device, dtype=dt)
        dx = torch.empty(T, D, device=x.device, dtype=dt)
        _bwd_dyp_kernel[(triton.cdiv(S, CT), triton.cdiv(D, CD))](
            dy, w_slot, src_tok, dyp, S, D, BT=CT, BD=CD
        )
        _bwd_dtw_kernel[(triton.cdiv(S, SB),)](
            dy, yp, src_tok, order, dtw, S, D, BS=SB, BD=SD
        )
        _bwd_dact_dgu_kernel[(n_tiles, triton.cdiv(F, GN))](
            dyp,
            w2_q,
            w2_s,
            gu,
            tile_e,
            tile_row,
            tile_end,
            dgu,
            D,
            F,
            BM=GM,
            BN=GN,
            BK=GKD,
            num_warps=8,
        )
        _bwd_dxslot_kernel[(n_tiles, triton.cdiv(D, GN))](
            dgu,
            w1_q,
            w1_s,
            tile_e,
            tile_row,
            tile_end,
            dxs,
            D,
            F,
            BM=GM,
            BN=GN,
            BK=GKD,
            num_warps=8,
        )
        _combine_bwd_kernel[(triton.cdiv(T, CT), triton.cdiv(D, CD))](
            dxs, inv, dx, T, D, A, BT=CT, BD=CD
        )
        # order: x, w1_q, w1_s, w2_q, w2_s, topk_idx, topk_w
        return dx, None, None, None, None, None, dtw.view(T, A)


def fp8_e4m3_swiglu_moe(
    x: torch.Tensor,  # [T, D]
    w1_q: torch.Tensor,  # [E, D, 2F] values on the fp8-e4m3 grid
    w1_s: torch.Tensor,  # [E, 2F]    per-out-channel scale
    w2_q: torch.Tensor,  # [E, F, D]
    w2_s: torch.Tensor,  # [E, D]
    topk_idx: torch.Tensor,  # [T, A] int64
    topk_w: torch.Tensor,  # [T, A]
) -> torch.Tensor:  # [T, D]
    return _Fp8E4m3SwigluMoeOracle.apply(x, w1_q, w1_s, w2_q, w2_s, topk_idx, topk_w)
# FORGE-CANARY-BEGIN forge-canary/v1
# FORGE-CANARY-SLOT-0 ecf5206580e37be68bc54a69b3ad246bad51767f4cc933c8777403fa864ef951
# FORGE-CANARY-SLOT-1 b6e9982ed2ead637d94fb4c2a78ffdc6d891542b2ebfaeeaf60aabe932ddb3c6
# FORGE-CANARY-SLOT-2 5211f21413770f59bf72e5dbfb4cd79bc7d06b47f1ab9d1ca507ed3dc4e73d62
# FORGE-CANARY-SLOT-3 671305a7b95a9d8f290e21eefb64b67ad48046e4587d802513cc93e2d3cc4b15
# FORGE-CANARY-END
