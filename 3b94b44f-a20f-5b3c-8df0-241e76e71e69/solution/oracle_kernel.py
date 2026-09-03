"""Private oracle -- from-scratch Triton shared-expert-plus-routed SwiGLU MoE
with softmax top-K + p_sel/sum combine on the routed side, fwd + bwd.

Never shipped to an agent. Establishes that full reward is reachable through
every live gate with no grouped-mm library call anywhere, and that both the
routed grouped-GEMM path and the always-on dense shared-SwiGLU path are
carried by Triton kernels this module declares (adoption floor met by the
routed segment GEMMs plus the shared dense GEMMs together).

SIX graded gradients: dx (both paths), drouter_w (routed only), dw1, dw2
(routed), dw1_s, dw2_s (shared).

Structure:

  routing   one Triton kernel per token row: fp32 softmax over the [E]
            logits, iterative top-K on unique int64 keys, p_sel/sum combine
            weights (mixtral rule). Store p (dense fp32 for router bwd) and
            w (combine weights).
  dispatch  stable integer argsort of [T*K]; per-expert segment bounds and
            BM-row tile table.
  compute
    routed  segment SwiGLU expert GEMMs (shared _seg_mm / _seg_dw kernels,
            bf16 hi/lo split precision for weight-grad chain).
    shared  dense SwiGLU pipeline through TWO _seg_mm launches with a
            single-expert (E=1) segment covering ALL tokens (GATHER=0);
            weight grads via _seg_dw the same way. That reuses the same
            kernel family (adoption floor covered by one instruction set).
  combine   ordered per-token accumulation of K contributions + shared row
            (fused in a single kernel per D-tile; slot order deterministic).
  router bw p_sel/sum coupling + dense softmax Jacobian over the whole
            expert row, as in mixtral.
"""

import torch
import triton
import triton.language as tl

BM = 64
BD = 256


@triton.jit
def _route_topk_kernel(LOG, P, SEL, W, T, E, K: tl.constexpr, KP: tl.constexpr, BE: tl.constexpr):
    """Softmax + top-K + p_sel/sum (mixtral)."""
    t = tl.program_id(0).to(tl.int64)
    e = tl.arange(0, BE)
    m = e < E
    lo = tl.load(LOG + t * E + e.to(tl.int64), mask=m, other=-1e30)
    mx = tl.max(lo, axis=0)
    ex = tl.exp(lo - mx)
    ex = tl.where(m, ex, 0.0)
    p = ex / tl.sum(ex, axis=0)
    tl.store(P + t * E + e.to(tl.int64), p, mask=m)
    b = tl.where(p == 0.0, 0.0, p)
    ib = b.to(tl.int32, bitcast=True).to(tl.int64)
    u = tl.where(ib >= 0, ib ^ 0x80000000, (~ib) & 0xFFFFFFFF)
    ebits: tl.constexpr = 12
    key = (u << ebits) | (E - 1 - e).to(tl.int64)
    key = tl.where(m, key, -1)
    psum = 0.0
    for k in tl.static_range(K):
        mx = tl.max(key, axis=0)
        col = E - 1 - (mx & ((1 << ebits) - 1)).to(tl.int32)
        hit = e == col
        pv = tl.sum(tl.where(hit, p, 0.0), axis=0)
        tl.store(SEL + t * K + k, col)
        tl.store(W + t * K + k, pv)  # temp: raw p_sel
        psum += pv
        key = tl.where(hit, -1, key)
    # Triton needs a power-of-2 arange extent; top-k=6 (graded g2/g3) failed to COMPILE,
    # which is why both dtypes failed at one shape. Pad the k axis and mask the tail.
    kk = tl.arange(0, KP)
    km = kk < K
    ps = tl.load(W + t * K + kk, mask=km, other=0.0)
    tl.store(W + t * K + kk, ps / psum, mask=km)


# --- shared kernels (identical bones to prior oracles) ---


@triton.jit
def _prec_dot(a, b, acc, PREC: tl.constexpr):
    if PREC == 0:
        acc += tl.dot(a, b)
    elif PREC == 1:
        acc += tl.dot(a.to(tl.float32), b.to(tl.float32), input_precision="ieee")
    elif PREC == 2:
        ah = a.to(tl.bfloat16)
        al = (a - ah.to(tl.float32)).to(tl.bfloat16)
        acc += tl.dot(ah, b) + tl.dot(al, b)
    elif PREC == 3:
        ah = a.to(tl.bfloat16)
        al = (a - ah.to(tl.float32)).to(tl.bfloat16)
        bh = b.to(tl.bfloat16)
        bl = (b - bh.to(tl.float32)).to(tl.bfloat16)
        acc += tl.dot(ah, bh) + tl.dot(ah, bl) + tl.dot(al, bh)
    else:
        bh = b.to(tl.bfloat16)
        bl = (b - bh.to(tl.float32)).to(tl.bfloat16)
        acc += tl.dot(a, bh) + tl.dot(a, bl)
    return acc


@triton.jit
def _seg_mm_kernel(
    AP,
    BP,
    OP,
    TOK,
    TE,
    TR0,
    SEGEND,
    K,
    N,
    sbe,
    sbk,
    sbn,
    GATHER: tl.constexpr,
    BLK_M: tl.constexpr,
    BLK_N: tl.constexpr,
    BLK_K: tl.constexpr,
    PREC: tl.constexpr,
):
    pt = tl.program_id(0)
    pn = tl.program_id(1)
    e = tl.load(TE + pt).to(tl.int64)
    r0 = tl.load(TR0 + pt).to(tl.int64)
    end = tl.load(SEGEND + e).to(tl.int64)
    rm = r0 + tl.arange(0, BLK_M).to(tl.int64)
    rmask = rm < end
    rowid = tl.load(TOK + rm, mask=rmask, other=0).to(tl.int64) if GATHER else rm
    rn = pn * BLK_N + tl.arange(0, BLK_N)
    nmask = rn < N
    acc = tl.zeros((BLK_M, BLK_N), dtype=tl.float32)
    bbase = BP + e * sbe
    for k0 in range(0, K, BLK_K):
        rk = k0 + tl.arange(0, BLK_K)
        kmask = rk < K
        a = tl.load(
            AP + rowid[:, None] * K + rk[None, :].to(tl.int64),
            mask=rmask[:, None] & kmask[None, :],
            other=0.0,
        )
        bmat = tl.load(
            bbase + rk[:, None].to(tl.int64) * sbk + rn[None, :].to(tl.int64) * sbn,
            mask=kmask[:, None] & nmask[None, :],
            other=0.0,
        )
        acc = _prec_dot(a, bmat, acc, PREC)
    tl.store(
        OP + rm[:, None] * N + rn[None, :].to(tl.int64),
        acc,
        mask=rmask[:, None] & nmask[None, :],
    )


@triton.jit
def _seg_dw_kernel(
    AP,
    BP,
    OP,
    TOK,
    SEGSTART,
    SEGEND,
    M,
    N,
    RSA,
    GATHER: tl.constexpr,
    BLK_M: tl.constexpr,
    BLK_N: tl.constexpr,
    BLK_K: tl.constexpr,
    PREC: tl.constexpr,
):
    pe = tl.program_id(0).to(tl.int64)
    pm = tl.program_id(1)
    pn = tl.program_id(2)
    lo = tl.load(SEGSTART + pe).to(tl.int64)
    hi = tl.load(SEGEND + pe).to(tl.int64)
    rm = pm * BLK_M + tl.arange(0, BLK_M)
    rn = pn * BLK_N + tl.arange(0, BLK_N)
    mmask = rm < M
    nmask = rn < N
    acc = tl.zeros((BLK_M, BLK_N), dtype=tl.float32)
    for r0 in range(lo, hi, BLK_K):
        rr = r0 + tl.arange(0, BLK_K).to(tl.int64)
        rmask = rr < hi
        rows = tl.load(TOK + rr, mask=rmask, other=0).to(tl.int64) if GATHER else rr
        a = tl.load(
            AP + rows[:, None] * RSA + rm[None, :].to(tl.int64),
            mask=rmask[:, None] & mmask[None, :],
            other=0.0,
        )
        bmat = tl.load(
            BP + rr[:, None] * N + rn[None, :].to(tl.int64),
            mask=rmask[:, None] & nmask[None, :],
            other=0.0,
        )
        acc = _prec_dot(tl.trans(a), bmat, acc, PREC)
    tl.store(
        OP + pe * M * N + rm[:, None].to(tl.int64) * N + rn[None, :].to(tl.int64),
        acc,
        mask=mmask[:, None] & nmask[None, :],
    )


@triton.jit
def _swiglu_fwd_kernel(GU, ACT, total, F, F2, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    o = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    m = o < total
    r = o // F
    f = o % F
    g = tl.load(GU + r * F2 + f, mask=m, other=0.0)
    u = tl.load(GU + r * F2 + F + f, mask=m, other=0.0)
    tl.store(ACT + o, g * tl.sigmoid(g) * u, mask=m)


@triton.jit
def _swiglu_bwd_kernel(GU, DACT, DGU, total, F, F2, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    o = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    m = o < total
    r = o // F
    f = o % F
    g = tl.load(GU + r * F2 + f, mask=m, other=0.0)
    u = tl.load(GU + r * F2 + F + f, mask=m, other=0.0)
    da = tl.load(DACT + o, mask=m, other=0.0).to(tl.float32)
    sg = tl.sigmoid(g)
    silu = g * sg
    tl.store(DGU + r * F2 + f, da * u * (sg + silu * (1.0 - sg)), mask=m)
    tl.store(DGU + r * F2 + F + f, da * silu, mask=m)


@triton.jit
def _combine_fwd_kernel(YP_R, W, POS, YS, Y, D, K_r, BLOCK: tl.constexpr):
    """y[t, d] = sum_k w[t,k] * yp_r[POS[t,k], d] + ys[t, d]. Ordered
    accumulation over k in ascending order -- deterministic."""
    t = tl.program_id(0).to(tl.int64)
    pd = tl.program_id(1).to(tl.int64)
    d = pd * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    dm = d < D
    acc = tl.load(YS + t * D + d, mask=dm, other=0.0)
    for k in range(0, K_r):
        q = tl.load(POS + t * K_r + k).to(tl.int64)
        wv = tl.load(W + t * K_r + k)
        acc += wv * tl.load(YP_R + q * D + d, mask=dm, other=0.0)
    tl.store(Y + t * D + d, acc, mask=dm)


@triton.jit
def _dyp_kernel(DY, WS, TOKS, DYP, D, BLOCK: tl.constexpr):
    q = tl.program_id(0).to(tl.int64)
    pd = tl.program_id(1).to(tl.int64)
    d = pd * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    dm = d < D
    t = tl.load(TOKS + q).to(tl.int64)
    wv = tl.load(WS + q)
    dyv = tl.load(DY + t * D + d, mask=dm, other=0.0).to(tl.float32)
    tl.store(DYP + q * D + d, wv * dyv, mask=dm)


@triton.jit
def _dw_scalar_kernel(DY, YP, POS, DWC, D, K_r, BLOCK: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    k = tl.program_id(1).to(tl.int64)
    q = tl.load(POS + t * K_r + k).to(tl.int64)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for d0 in range(0, D, BLOCK):
        d = d0 + tl.arange(0, BLOCK).to(tl.int64)
        dm = d < D
        dyv = tl.load(DY + t * D + d, mask=dm, other=0.0).to(tl.float32)
        ypv = tl.load(YP + q * D + d, mask=dm, other=0.0)
        acc += dyv * ypv
    tl.store(DWC + t * K_r + k, tl.sum(acc, axis=0))


@triton.jit
def _dx_combine_kernel(DXS, POS, OUT, D, K_r, BLOCK: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    pd = tl.program_id(1).to(tl.int64)
    d = pd * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    dm = d < D
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in range(0, K_r):
        q = tl.load(POS + t * K_r + k).to(tl.int64)
        acc += tl.load(DXS + q * D + d, mask=dm, other=0.0)
    tl.store(OUT + t * D + d, acc, mask=dm)


def _dispatch(sel, E, K):
    dev = sel.device
    flat = sel.reshape(-1).to(torch.int64)
    order = torch.argsort(flat, stable=True)
    pos = torch.empty(flat.numel(), dtype=torch.int64, device=dev)
    pos[order] = torch.arange(flat.numel(), dtype=torch.int64, device=dev)
    toks = torch.div(order, K, rounding_mode="floor").to(torch.int32)
    counts = torch.bincount(flat, minlength=E)
    seg_end = counts.cumsum(0)
    seg_start = seg_end - counts
    tiles = (counts + BM - 1) // BM
    tile_cum = tiles.cumsum(0) - tiles
    ntiles = int(tiles.sum().item())
    texp = torch.repeat_interleave(
        torch.arange(E, dtype=torch.int64, device=dev), tiles
    )
    tid = torch.arange(ntiles, dtype=torch.int64, device=dev)
    trow = (seg_start[texp] + (tid - tile_cum[texp]) * BM).to(torch.int32)
    return (
        order,
        pos.to(torch.int32),
        toks,
        seg_start.to(torch.int32),
        seg_end.to(torch.int32),
        texp.to(torch.int32),
        trow,
        ntiles,
    )


def _mm_launch(
    ap, bp, op, toks, texp, trow, seg_end, K, N, sbe, sbk, sbn, gather, prec, ntiles
):
    if prec == 1:
        bn, bk, w, s = 64, 32, 8, 3
    else:
        bn, bk, w, s = 128, 64, 8, 3
    _seg_mm_kernel[(ntiles, triton.cdiv(N, bn))](
        ap,
        bp,
        op,
        toks,
        texp,
        trow,
        seg_end,
        K,
        N,
        sbe,
        sbk,
        sbn,
        GATHER=gather,
        BLK_M=BM,
        BLK_N=bn,
        BLK_K=bk,
        PREC=prec,
        num_warps=w,
        num_stages=s,
    )


def _dw_launch(ap, bp, op, toks, seg_start, seg_end, E, M, N, rsa, gather, prec):
    bm, bn, bk = 64, 128, 32
    w = 8 if prec == 3 else 4
    _seg_dw_kernel[(E, triton.cdiv(M, bm), triton.cdiv(N, bn))](
        ap,
        bp,
        op,
        toks,
        seg_start,
        seg_end,
        M,
        N,
        rsa,
        GATHER=gather,
        BLK_M=bm,
        BLK_N=bn,
        BLK_K=bk,
        PREC=prec,
        num_warps=w,
    )


def _dense_dispatch(T, dev):
    """Single-expert dispatch table covering all T rows in contiguous order,
    so _seg_mm/_seg_dw carry the dense shared-path GEMMs the same way."""
    ntiles_s = (T + BM - 1) // BM
    seg_start = torch.tensor([0], dtype=torch.int32, device=dev)
    seg_end = torch.tensor([T], dtype=torch.int32, device=dev)
    texp = torch.zeros(ntiles_s, dtype=torch.int32, device=dev)
    trow = torch.arange(ntiles_s, dtype=torch.int32, device=dev) * BM
    tok_id = torch.arange(T, dtype=torch.int32, device=dev)
    return seg_start, seg_end, texp, trow, tok_id, ntiles_s


class _FusedMoeOracle(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, router_w, w1, w2, w1_s, w2_s, K):
        K = int(K)
        x = x.contiguous()
        router_w = router_w.contiguous()
        w1 = w1.contiguous()
        w2 = w2.contiguous()
        w1_s = w1_s.contiguous()
        w2_s = w2_s.contiguous()
        T, D = x.shape
        E = router_w.shape[1]
        F2 = w1.shape[2]
        F = F2 // 2
        Fs2 = w1_s.shape[1]
        Fs = Fs2 // 2
        dev = x.device
        ieee = x.dtype == torch.float32
        cdt = x.dtype

        # routed routing
        logits = x.detach().float() @ router_w.detach().float()
        p_dense = torch.empty(T, E, dtype=torch.float32, device=dev)
        sel = torch.empty(T, K, dtype=torch.int32, device=dev)
        w = torch.empty(T, K, dtype=torch.float32, device=dev)
        _route_topk_kernel[(T,)](
            logits, p_dense, sel, w, T, E, K=K, KP=max(triton.next_power_of_2(K), 2), BE=max(triton.next_power_of_2(E), 2)
        )

        (order, pos, toks, seg_start, seg_end, texp, trow, ntiles) = _dispatch(
            sel, E, K
        )
        TK = T * K
        ws = w.reshape(-1).index_select(0, order).contiguous()

        prec_fwd = 1 if ieee else 0
        # routed forward
        out1_r = torch.empty(TK, F2, dtype=torch.float32, device=dev)
        _mm_launch(
            x,
            w1,
            out1_r,
            toks,
            texp,
            trow,
            seg_end,
            D,
            F2,
            D * F2,
            F2,
            1,
            1,
            prec_fwd,
            ntiles,
        )
        act_r = torch.empty(TK, F, dtype=cdt, device=dev)
        _swiglu_fwd_kernel[(triton.cdiv(TK * F, BD),)](
            out1_r, act_r, TK * F, F, F2, BLOCK=BD
        )
        yp_r = torch.empty(TK, D, dtype=torch.float32, device=dev)
        _mm_launch(
            act_r,
            w2,
            yp_r,
            toks,
            texp,
            trow,
            seg_end,
            F,
            D,
            F * D,
            D,
            1,
            0,
            prec_fwd,
            ntiles,
        )

        # shared forward through the same _seg_mm kernel (E=1 dense table)
        (ss_start, ss_end, s_texp, s_trow, s_tok, s_ntiles) = _dense_dispatch(T, dev)
        # sha w1_s viewed as [1, D, 2Fs], w2_s as [1, Fs, D]
        w1_s3 = w1_s.view(1, D, Fs2)
        w2_s3 = w2_s.view(1, Fs, D)
        out1_s = torch.empty(T, Fs2, dtype=torch.float32, device=dev)
        _mm_launch(
            x,
            w1_s3,
            out1_s,
            s_tok,
            s_texp,
            s_trow,
            ss_end,
            D,
            Fs2,
            D * Fs2,
            Fs2,
            1,
            1,
            prec_fwd,
            s_ntiles,
        )
        act_s = torch.empty(T, Fs, dtype=cdt, device=dev)
        _swiglu_fwd_kernel[(triton.cdiv(T * Fs, BD),)](
            out1_s, act_s, T * Fs, Fs, Fs2, BLOCK=BD
        )
        yp_s = torch.empty(T, D, dtype=torch.float32, device=dev)
        _mm_launch(
            act_s,
            w2_s3,
            yp_s,
            s_tok,
            s_texp,
            s_trow,
            ss_end,
            Fs,
            D,
            Fs * D,
            D,
            1,
            0,
            prec_fwd,
            s_ntiles,
        )

        # fused combine
        y = torch.empty(T, D, dtype=cdt, device=dev)
        _combine_fwd_kernel[(T, triton.cdiv(D, BD))](
            yp_r, w, pos, yp_s, y, D, K, BLOCK=BD
        )

        ctx.save_for_backward(
            x,
            router_w,
            w1,
            w2,
            w1_s,
            w2_s,
            p_dense,
            sel,
            w,
            pos,
            toks,
            seg_start,
            seg_end,
            texp,
            trow,
            ws,
            out1_r,
            out1_s,
            ss_start,
            ss_end,
            s_texp,
            s_trow,
            s_tok,
        )
        ctx.meta = (K, ntiles, s_ntiles, ieee, F, F2, Fs, Fs2)
        return y

    @staticmethod
    def backward(ctx, dy):
        (
            x,
            router_w,
            w1,
            w2,
            w1_s,
            w2_s,
            p_dense,
            sel,
            w,
            pos,
            toks,
            seg_start,
            seg_end,
            texp,
            trow,
            ws,
            out1_r,
            out1_s,
            ss_start,
            ss_end,
            s_texp,
            s_trow,
            s_tok,
        ) = ctx.saved_tensors
        K, ntiles, s_ntiles, ieee, F, F2, Fs, Fs2 = ctx.meta
        T, D = x.shape
        E = router_w.shape[1]
        TK = T * K
        dev = x.device
        cdt = x.dtype
        dy = dy.contiguous()

        p_a = 1 if ieee else 2
        p_ab = 1 if ieee else 3
        p_b = 1 if ieee else 4

        # ROUTED backward
        dyp_r = torch.empty(TK, D, dtype=torch.float32, device=dev)
        _dyp_kernel[(TK, triton.cdiv(D, BD))](dy, ws, toks, dyp_r, D, BLOCK=BD)
        act32_r = torch.empty(TK, F, dtype=torch.float32, device=dev)
        _swiglu_fwd_kernel[(triton.cdiv(TK * F, BD),)](
            out1_r, act32_r, TK * F, F, F2, BLOCK=BD
        )

        yp32_r = torch.empty(TK, D, dtype=torch.float32, device=dev)
        _mm_launch(
            act32_r,
            w2,
            yp32_r,
            toks,
            texp,
            trow,
            seg_end,
            F,
            D,
            F * D,
            D,
            1,
            0,
            p_a,
            ntiles,
        )
        dwc = torch.empty(T, K, dtype=torch.float32, device=dev)
        _dw_scalar_kernel[(T, K)](dy, yp32_r, pos, dwc, D, K, BLOCK=BD)
        del yp32_r

        dact_r = torch.empty(TK, F, dtype=torch.float32, device=dev)
        _mm_launch(
            dyp_r,
            w2,
            dact_r,
            toks,
            texp,
            trow,
            seg_end,
            D,
            F,
            F * D,
            1,
            D,
            0,
            p_a,
            ntiles,
        )
        dw2 = torch.empty(E, F, D, dtype=w2.dtype, device=dev)
        _dw_launch(act32_r, dyp_r, dw2, toks, seg_start, seg_end, E, F, D, F, 0, p_ab)
        del act32_r, dyp_r

        dgu_r = torch.empty(TK, F2, dtype=torch.float32, device=dev)
        _swiglu_bwd_kernel[(triton.cdiv(TK * F, BD),)](
            out1_r, dact_r, dgu_r, TK * F, F, F2, BLOCK=BD
        )
        del dact_r

        dw1 = torch.empty(E, D, F2, dtype=w1.dtype, device=dev)
        _dw_launch(x, dgu_r, dw1, toks, seg_start, seg_end, E, D, F2, D, 1, p_b)
        dgu_r_c = dgu_r.to(cdt)
        del dgu_r
        dxs = torch.empty(TK, D, dtype=torch.float32, device=dev)
        _mm_launch(
            dgu_r_c,
            w1,
            dxs,
            toks,
            texp,
            trow,
            seg_end,
            F2,
            D,
            D * F2,
            1,
            F2,
            0,
            1 if ieee else 0,
            ntiles,
        )
        del dgu_r_c
        dxe_r = torch.empty(T, D, dtype=torch.float32, device=dev)
        _dx_combine_kernel[(T, triton.cdiv(D, BD))](dxs, pos, dxe_r, D, K, BLOCK=BD)

        # SHARED backward (dense E=1 pipeline via same _seg_mm/_seg_dw)
        dyp_s = dy.float().contiguous()  # [T, D]
        w1_s3 = w1_s.view(1, D, Fs2)
        w2_s3 = w2_s.view(1, Fs, D)
        act32_s = torch.empty(T, Fs, dtype=torch.float32, device=dev)
        _swiglu_fwd_kernel[(triton.cdiv(T * Fs, BD),)](
            out1_s, act32_s, T * Fs, Fs, Fs2, BLOCK=BD
        )

        dact_s = torch.empty(T, Fs, dtype=torch.float32, device=dev)
        _mm_launch(
            dyp_s,
            w2_s3,
            dact_s,
            s_tok,
            s_texp,
            s_trow,
            ss_end,
            D,
            Fs,
            Fs * D,
            1,
            D,
            0,
            p_a,
            s_ntiles,
        )
        dw2s = torch.empty(1, Fs, D, dtype=w2_s.dtype, device=dev)
        _dw_launch(act32_s, dyp_s, dw2s, s_tok, ss_start, ss_end, 1, Fs, D, Fs, 0, p_ab)
        dw2_s_grad = dw2s.view(Fs, D)
        del act32_s

        dgu_s = torch.empty(T, Fs2, dtype=torch.float32, device=dev)
        _swiglu_bwd_kernel[(triton.cdiv(T * Fs, BD),)](
            out1_s, dact_s, dgu_s, T * Fs, Fs, Fs2, BLOCK=BD
        )
        del dact_s

        dw1s = torch.empty(1, D, Fs2, dtype=w1_s.dtype, device=dev)
        _dw_launch(x, dgu_s, dw1s, s_tok, ss_start, ss_end, 1, D, Fs2, D, 0, p_b)
        dw1_s_grad = dw1s.view(D, Fs2)
        dgu_s_c = dgu_s.to(cdt)
        del dgu_s
        dxe_s = torch.empty(T, D, dtype=torch.float32, device=dev)
        _mm_launch(
            dgu_s_c,
            w1_s3,
            dxe_s,
            s_tok,
            s_texp,
            s_trow,
            ss_end,
            Fs2,
            D,
            D * Fs2,
            1,
            Fs2,
            0,
            1 if ieee else 0,
            s_ntiles,
        )

        # router bwd (mixtral p_sel/sum + softmax Jacobian over whole row)
        wsum = (dwc * w).sum(dim=-1, keepdim=True)
        psum = torch.gather(p_dense, 1, sel.to(torch.int64)).sum(dim=-1, keepdim=True)
        dp_sel = (dwc - wsum) / psum
        # dp_j: at selected j -> dp_sel[t,k]; else 0
        dp = torch.zeros_like(p_dense)
        dp.scatter_(1, sel.to(torch.int64), dp_sel)
        # softmax Jacobian: dlogits = p * (dp - sum_j p_j dp_j)
        row = (p_dense * dp).sum(dim=-1, keepdim=True)
        dlogits = p_dense * (dp - row)

        xf = x.float()
        rwf = router_w.float()
        drw = (xf.t() @ dlogits).to(router_w.dtype)
        dx = (dxe_r + dxe_s + dlogits @ rwf.t()).to(x.dtype)
        return dx, drw, dw1, dw2, dw1_s_grad, dw2_s_grad, None


def fused_moe(
    x: torch.Tensor,
    router_w: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_s: torch.Tensor,
    w2_s: torch.Tensor,
    K: int,
) -> torch.Tensor:
    return _FusedMoeOracle.apply(x, router_w, w1, w2, w1_s, w2_s, K)
# FORGE-CANARY-BEGIN forge-canary/v1
# FORGE-CANARY-SLOT-0 3ba11a0060fdbdbd191746ee52a4bef4e373ae78ef266af3192272c554d43c70
# FORGE-CANARY-SLOT-1 dad2583c5cdffe95684adf9ca848bd9c00fe687400e563d18eebf627d6a77325
# FORGE-CANARY-SLOT-2 2b174defda600ed37f94172c8a2ea1a97e1059c6b89f84081fbd0587c9593690
# FORGE-CANARY-SLOT-3 799409199f4e0d6101cc6bd0efc16a073d0d4b5a9e4ac512184e3ff35e067310
# FORGE-CANARY-END
