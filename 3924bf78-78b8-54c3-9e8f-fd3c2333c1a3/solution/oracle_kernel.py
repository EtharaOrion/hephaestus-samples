"""Private oracle -- from-scratch Triton int4 W4A16 SwiGLU MoE, fwd+bwd.

UNVERIFIED: authored on a CPU-only box without GPU calibration. Reaches full
reward by staging the int4 unpack + asymmetric dequant into dedicated Triton
prologue kernels (`_unpack_w1_kernel`, `_unpack_w2_kernel`) that populate
dense fp32 scratch buffers, then running the moe_swiglu-family routed segment
GEMMs against those buffers. Every arithmetic byte -- unpack, dequant, both
expert GEMMs, silu*up, combine and backward -- lives inside kernels this
module declares, so the 0.60 adoption floor is met with margin. No forbidden
symbol anywhere. Only dx and dtopk_w are graded, so the per-expert weight-
gradient kernels are absent. All accumulations are deterministic.

Note on scratch memory: the prologue materializes fp32 weight buffers of
size [E, D, 2F] and [E, F, D]. At the largest hidden shape (E=128, D=2048,
F=1024, and the wider g3 config) that stays under 5 GB total, well inside
one H100. A candidate that fuses the unpack into each MMA prologue and
never spills the dense weights runs faster; that is exactly the headroom
this operator exposes.
"""

import torch
import triton
import triton.language as tl

GM, GN = 128, 128
GKD = 32
CT, CD = 32, 128
SB, SD = 64, 128
UM, UN = 64, 128  # unpack tile


# ===========================================================================
# int4 unpack + asymmetric dequant prologues (triton, deterministic)
# ===========================================================================


@triton.jit
def _unpack_w1_kernel(
    WPACK, WSCALE, WZP, WDQ, D, F, BM: tl.constexpr, BN: tl.constexpr
):
    """WPACK[E, D, F] uint8 -> WDQ[E, D, 2F] fp32 via asymmetric dequant:
    lo = (byte & 0xF), hi = (byte >> 4) & 0xF; column 2j = lo, 2j+1 = hi;
    then (nibble - zp[col]) * scale[col]."""
    e = tl.program_id(0).to(tl.int64)
    pid_d = tl.program_id(1).to(tl.int64)
    pid_f = tl.program_id(2).to(tl.int64)
    rd = pid_d * BM + tl.arange(0, BM).to(tl.int64)
    md = rd < D
    rf = pid_f * BN + tl.arange(0, BN).to(tl.int64)  # byte index in [0, F)
    mf = rf < F
    m2 = md[:, None] & mf[None, :]
    b = tl.load(WPACK + e * D * F + rd[:, None] * F + rf[None, :], mask=m2, other=0).to(
        tl.int32
    )
    lo = (b & 0xF).to(tl.float32)
    hi = ((b >> 4) & 0xF).to(tl.float32)
    col_lo = rf * 2  # even output cols
    col_hi = rf * 2 + 1  # odd output cols
    slo = tl.load(WSCALE + e * 2 * F + col_lo, mask=mf, other=0.0).to(tl.float32)
    shi = tl.load(WSCALE + e * 2 * F + col_hi, mask=mf, other=0.0).to(tl.float32)
    zlo = tl.load(WZP + e * 2 * F + col_lo, mask=mf, other=0).to(tl.float32)
    zhi = tl.load(WZP + e * 2 * F + col_hi, mask=mf, other=0).to(tl.float32)
    vlo = (lo - zlo[None, :]) * slo[None, :]
    vhi = (hi - zhi[None, :]) * shi[None, :]
    tl.store(WDQ + e * D * 2 * F + rd[:, None] * 2 * F + col_lo[None, :], vlo, mask=m2)
    tl.store(WDQ + e * D * 2 * F + rd[:, None] * 2 * F + col_hi[None, :], vhi, mask=m2)


@triton.jit
def _unpack_w2_kernel(
    WPACK, WSCALE, WZP, WDQ, D, F, BM: tl.constexpr, BN: tl.constexpr
):
    """WPACK[E, F, D/2] uint8 -> WDQ[E, F, D] fp32; same nibble/dequant rule
    but packed along the D axis (byte column d/2, lo = column 2j, hi = 2j+1)."""
    e = tl.program_id(0).to(tl.int64)
    pid_f = tl.program_id(1).to(tl.int64)
    pid_d = tl.program_id(2).to(tl.int64)
    rf = pid_f * BM + tl.arange(0, BM).to(tl.int64)
    mf = rf < F
    rdb = pid_d * BN + tl.arange(0, BN).to(tl.int64)  # byte index in [0, D/2)
    mdb = rdb < (D // 2)
    m2 = mf[:, None] & mdb[None, :]
    b = tl.load(
        WPACK + e * F * (D // 2) + rf[:, None] * (D // 2) + rdb[None, :],
        mask=m2,
        other=0,
    ).to(tl.int32)
    lo = (b & 0xF).to(tl.float32)
    hi = ((b >> 4) & 0xF).to(tl.float32)
    col_lo = rdb * 2
    col_hi = rdb * 2 + 1
    slo = tl.load(WSCALE + e * D + col_lo, mask=mdb, other=0.0).to(tl.float32)
    shi = tl.load(WSCALE + e * D + col_hi, mask=mdb, other=0.0).to(tl.float32)
    zlo = tl.load(WZP + e * D + col_lo, mask=mdb, other=0).to(tl.float32)
    zhi = tl.load(WZP + e * D + col_hi, mask=mdb, other=0).to(tl.float32)
    vlo = (lo - zlo[None, :]) * slo[None, :]
    vhi = (hi - zhi[None, :]) * shi[None, :]
    tl.store(WDQ + e * F * D + rf[:, None] * D + col_lo[None, :], vlo, mask=m2)
    tl.store(WDQ + e * F * D + rf[:, None] * D + col_hi[None, :], vhi, mask=m2)


# ===========================================================================
# Standard routed SwiGLU forward + dx / dtopk_w backward kernels
# (structurally identical to swiglu_moe oracle; only dw1/dw2 kernels omitted)
# ===========================================================================


@triton.jit
def _fwd_gu_kernel(
    X,
    W1,
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
    m2 = mrow[:, None] & mcf[None, :]
    tl.store(GU + rows[:, None] * 2 * F + cf[None, :], acc_g, mask=m2)
    tl.store(GU + rows[:, None] * 2 * F + F + cf[None, :], acc_u, mask=m2)
    s = tl.sigmoid(acc_g)
    tl.store(ACT + rows[:, None] * F + cf[None, :], acc_g * s * acc_u, mask=m2)


@triton.jit
def _fwd_out_kernel(
    ACT,
    W2,
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
    tl.store(
        YP + rows[:, None] * D + cd[None, :], acc, mask=mrow[:, None] & mcd[None, :]
    )


@triton.jit
def _combine_fwd_kernel(YP, WSLOT, INV, Y, T, D, A, BT: tl.constexpr, BD: tl.constexpr):
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
    W2,
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
        a = tl.load(
            DYP + rows[:, None] * D + rk[None, :],
            mask=mrow[:, None] & mk[None, :],
            other=0.0,
        )
        b = tl.load(
            wbase + cf[None, :] * D + rk[:, None],
            mask=mcf[None, :] & mk[:, None],
            other=0.0,
        )
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
    W1,
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
        a = tl.load(
            DGU + rows[:, None] * 2 * F + rk[None, :],
            mask=mrow[:, None] & mk[None, :],
            other=0.0,
        )
        b = tl.load(
            wbase + cd[None, :] * 2 * F + rk[:, None],
            mask=mcd[None, :] & mk[:, None],
            other=0.0,
        )
        acc += tl.dot(a, b)
    tl.store(
        DXS + rows[:, None] * D + cd[None, :], acc, mask=mrow[:, None] & mcd[None, :]
    )


@triton.jit
def _combine_bwd_kernel(DXS, INV, DX, T, D, A, BT: tl.constexpr, BD: tl.constexpr):
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
        src_tok.contiguous(),
        inv.contiguous(),
        w_slot,
        n_tiles,
        tile_expert.contiguous(),
        tile_row.contiguous(),
        tile_end.contiguous(),
    )


def _dequant_w1(w1_pack, w1_s, w1_zp, D, F, dtype, device):
    """Triton unpack + asymmetric dequant; produces fp32 buffer [E, D, 2F].
    Cast to `dtype` for consumption by downstream MMA kernels."""
    E = w1_pack.shape[0]
    wdq32 = torch.empty(E, D, 2 * F, device=device, dtype=torch.float32)
    _unpack_w1_kernel[(E, triton.cdiv(D, UM), triton.cdiv(F, UN))](
        w1_pack, w1_s.contiguous(), w1_zp.contiguous(), wdq32, D, F, BM=UM, BN=UN
    )
    return wdq32.to(dtype)


def _dequant_w2(w2_pack, w2_s, w2_zp, D, F, dtype, device):
    E = w2_pack.shape[0]
    wdq32 = torch.empty(E, F, D, device=device, dtype=torch.float32)
    _unpack_w2_kernel[(E, triton.cdiv(F, UM), triton.cdiv(D // 2, UN))](
        w2_pack, w2_s.contiguous(), w2_zp.contiguous(), wdq32, D, F, BM=UM, BN=UN
    )
    return wdq32.to(dtype)


class _Int4WqSwigluMoeOracle(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1_pack, w1_s, w1_zp, w2_pack, w2_s, w2_zp, topk_idx, topk_w):
        T, D = x.shape
        E = w1_pack.shape[0]
        F = w1_pack.shape[2]  # 2F/2 == F bytes per row
        A = topk_idx.shape[1]
        S = T * A
        x = x.contiguous()
        tw = topk_w.contiguous()
        dt = x.dtype
        w1 = _dequant_w1(w1_pack.contiguous(), w1_s, w1_zp, D, F, dt, x.device)
        w2 = _dequant_w2(w2_pack.contiguous(), w2_s, w2_zp, D, F, dt, x.device)
        (order, src_tok, inv, w_slot, n_tiles, tile_e, tile_row, tile_end) = _prep(
            topk_idx, tw, E
        )
        gu = torch.empty(S, 2 * F, device=x.device, dtype=dt)
        act = torch.empty(S, F, device=x.device, dtype=dt)
        yp = torch.empty(S, D, device=x.device, dtype=dt)
        y = torch.empty(T, D, device=x.device, dtype=dt)
        _fwd_gu_kernel[(n_tiles, triton.cdiv(F, GN))](
            x,
            w1,
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
            w2,
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
            w1,
            w2,
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
            w1,
            w2,
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
            w2,
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
            w1,
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
        # order: x, w1_pack, w1_s, w1_zp, w2_pack, w2_s, w2_zp, topk_idx, topk_w
        return dx, None, None, None, None, None, None, None, dtw.view(T, A)


def int4_wq_swiglu_moe(
    x: torch.Tensor,
    w1_pack: torch.Tensor,
    w1_s: torch.Tensor,
    w1_zp: torch.Tensor,
    w2_pack: torch.Tensor,
    w2_s: torch.Tensor,
    w2_zp: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_w: torch.Tensor,
) -> torch.Tensor:
    return _Int4WqSwigluMoeOracle.apply(
        x, w1_pack, w1_s, w1_zp, w2_pack, w2_s, w2_zp, topk_idx, topk_w
    )
# FORGE-CANARY-BEGIN forge-canary/v1
# FORGE-CANARY-SLOT-0 044eda056a577a65c7220299a73d3149fe07aa5f3b3fcf7a93b8490da37d53ba
# FORGE-CANARY-SLOT-1 48fe8ab47273a9b68530219458d2dabc65ac7bbd7bdb6fff6c47caec9cb76d56
# FORGE-CANARY-SLOT-2 1444827bb5ef43fcd0c4d39153007a9805bb7482d970cce6982b74b42823a7c7
# FORGE-CANARY-SLOT-3 db7c789d5a08da4d80b752723905259a7ff80f7168556ba2cda874a620216e39
# FORGE-CANARY-END
