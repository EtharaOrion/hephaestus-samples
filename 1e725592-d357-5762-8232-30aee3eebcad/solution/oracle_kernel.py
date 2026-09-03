"""Private oracle -- from-scratch Triton hierarchical group-then-expert
routed SwiGLU MoE (DeepSeek-V2 device-limited routing), fwd + bwd.

Never shipped to an agent. Establishes that full reward is reachable through
every live gate with no grouped-mm library call anywhere.

Structure (deterministic; no atomics, no value-dependent branching):

  routing   one Triton kernel per token row: fp32 softmax over the [E]
            logits (reshaped as [G, S] slabs), per-group top-2 sum for the
            group score, argmax GROUP by unique key (LOWER group index),
            intra-group top-K by unique key (LOWER local index) inside the
            chosen group. Store p (for router bwd), sel (global expert
            indices), and w (p_sel/sum combine weights).
  dispatch  stable integer argsort of [T*K]; per-expert segment bounds and
            BM-row tile table.
  compute   segment SwiGLU expert GEMMs (shared _seg_mm / _seg_dw kernels,
            bf16 hi/lo split precision for weight-grad chain).
  combine   ordered per-token accumulation of K contributions.
  router bw p_sel/sum coupling + softmax Jacobian over the whole expert row
            (dense). The two selection stages are locally constant, so no
            extra term contributes beyond the p_sel/sum + softmax chain.
"""

import torch
import triton
import triton.language as tl

BM = 64
BD = 256


@triton.jit
def _route_group_kernel(
    LOG, P, SEL, W, T, E, G, K: tl.constexpr, BE: tl.constexpr, BG: tl.constexpr
):
    """Softmax scores + hierarchical selection: top-1 group by top-2-sum,
    then top-K experts within chosen group by p; p_sel/sum combine."""
    t = tl.program_id(0).to(tl.int64)
    e = tl.arange(0, BE)
    m = e < E
    lo = tl.load(LOG + t * E + e.to(tl.int64), mask=m, other=-1e30)
    mx = tl.max(lo, axis=0)
    ex = tl.exp(lo - mx)
    ex = tl.where(m, ex, 0.0)
    p = ex / tl.sum(ex, axis=0)
    tl.store(P + t * E + e.to(tl.int64), p, mask=m)

    S: tl.constexpr = 4096  # upper bound; only S=E/G entries per group matter
    # per-group top-2 sum -> group score gs[g]
    S_actual = E // G
    g_idx = tl.arange(0, BG)
    gvalid = g_idx < G
    gs = tl.zeros((BG,), dtype=tl.float32)
    for gi in tl.static_range(BG):
        # gather this group's slab from p
        s = tl.arange(0, BE)
        sm = (s >= gi * S_actual) & (s < (gi + 1) * S_actual) & m
        slab = tl.where(sm, p, -1.0)
        m1 = tl.max(slab, axis=0)
        # Drop exactly ONE occurrence of the max. Masking every entry equal to m1
        # makes m2 the third largest when the top two are tied, but the reference
        # uses topk(2).sum(), where both tied maxima count.
        first = tl.min(tl.where(slab == m1, s, BE), axis=0)
        slab2 = tl.where(s == first, -1.0, slab)
        m2 = tl.max(slab2, axis=0)
        m2 = tl.where(S_actual > 1, m2, 0.0)
        summ = m1 + m2
        # accumulate into gs[gi]
        active = gi < G
        gs = tl.where(g_idx == gi, tl.where(active, summ, -1.0), gs)
    gs = tl.where(gvalid, gs, -1.0)

    # argmax group by unique key (LOWER group index)
    b = tl.where(gs == 0.0, 0.0, gs)
    ib = b.to(tl.int32, bitcast=True).to(tl.int64)
    u = tl.where(ib >= 0, ib ^ 0x80000000, (~ib) & 0xFFFFFFFF)
    gbits: tl.constexpr = 8  # G <= 256
    gkey = (u << gbits) | (G - 1 - g_idx).to(tl.int64)
    gkey = tl.where(gvalid, gkey, -1)
    gmax = tl.max(gkey, axis=0)
    grp = G - 1 - (gmax & ((1 << gbits) - 1)).to(tl.int32)

    # intra-group top-K
    s = tl.arange(0, BE)
    mask_grp = (s >= grp * S_actual) & (s < (grp + 1) * S_actual) & m
    p_in = tl.where(mask_grp, p, 0.0)
    b2 = tl.where(p_in == 0.0, 0.0, p_in)
    ib2 = b2.to(tl.int32, bitcast=True).to(tl.int64)
    u2 = tl.where(ib2 >= 0, ib2 ^ 0x80000000, (~ib2) & 0xFFFFFFFF)
    sbits: tl.constexpr = 12
    # local index inside the group = s - grp * S_actual
    local_idx = s - grp * S_actual
    key = (u2 << sbits) | (S_actual - 1 - local_idx).to(tl.int64)
    key = tl.where(mask_grp, key, -1)
    psum = 0.0
    for k in tl.static_range(K):
        mx = tl.max(key, axis=0)
        local = S_actual - 1 - (mx & ((1 << sbits) - 1)).to(tl.int32)
        e_pick = grp * S_actual + local
        hit = s == e_pick
        pv = tl.sum(tl.where(hit, p, 0.0), axis=0)
        tl.store(SEL + t * K + k, e_pick)
        tl.store(W + t * K + k, pv)
        psum += pv
        key = tl.where(hit, -1, key)
    kk = tl.arange(0, K)
    ps = tl.load(W + t * K + kk)
    tl.store(W + t * K + kk, ps / psum)


# --- shared kernels ---


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
def _combine_fwd_kernel(YP, W, POS, Y, D, K_r, BLOCK: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    pd = tl.program_id(1).to(tl.int64)
    d = pd * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    dm = d < D
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in range(0, K_r):
        q = tl.load(POS + t * K_r + k).to(tl.int64)
        wv = tl.load(W + t * K_r + k)
        acc += wv * tl.load(YP + q * D + d, mask=dm, other=0.0)
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


class _FusedMoeOracle(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, router_w, w1, w2, G, K):
        G = int(G)
        K = int(K)
        x = x.contiguous()
        router_w = router_w.contiguous()
        w1 = w1.contiguous()
        w2 = w2.contiguous()
        T, D = x.shape
        E = router_w.shape[1]
        F2 = w1.shape[2]
        F = F2 // 2
        dev = x.device
        ieee = x.dtype == torch.float32
        cdt = x.dtype

        logits = x.detach().float() @ router_w.detach().float()
        p_dense = torch.empty(T, E, dtype=torch.float32, device=dev)
        sel = torch.empty(T, K, dtype=torch.int32, device=dev)
        w = torch.empty(T, K, dtype=torch.float32, device=dev)
        BE = max(triton.next_power_of_2(E), 2)
        BG = max(triton.next_power_of_2(G), 2)
        _route_group_kernel[(T,)](logits, p_dense, sel, w, T, E, G, K=K, BE=BE, BG=BG)

        (order, pos, toks, seg_start, seg_end, texp, trow, ntiles) = _dispatch(
            sel, E, K
        )
        TK = T * K
        ws = w.reshape(-1).index_select(0, order).contiguous()

        prec_fwd = 1 if ieee else 0
        out1 = torch.empty(TK, F2, dtype=torch.float32, device=dev)
        _mm_launch(
            x,
            w1,
            out1,
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
        act = torch.empty(TK, F, dtype=cdt, device=dev)
        _swiglu_fwd_kernel[(triton.cdiv(TK * F, BD),)](
            out1, act, TK * F, F, F2, BLOCK=BD
        )
        yp = torch.empty(TK, D, dtype=torch.float32, device=dev)
        _mm_launch(
            act,
            w2,
            yp,
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
        y = torch.empty(T, D, dtype=cdt, device=dev)
        _combine_fwd_kernel[(T, triton.cdiv(D, BD))](yp, w, pos, y, D, K, BLOCK=BD)

        ctx.save_for_backward(
            x,
            router_w,
            w1,
            w2,
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
            out1,
        )
        ctx.meta = (K, ntiles, ieee, F, F2)
        return y

    @staticmethod
    def backward(ctx, dy):
        (
            x,
            router_w,
            w1,
            w2,
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
            out1,
        ) = ctx.saved_tensors
        K, ntiles, ieee, F, F2 = ctx.meta
        T, D = x.shape
        E = router_w.shape[1]
        TK = T * K
        dev = x.device
        cdt = x.dtype
        dy = dy.contiguous()

        p_a = 1 if ieee else 2
        p_ab = 1 if ieee else 3
        p_b = 1 if ieee else 4

        dyp = torch.empty(TK, D, dtype=torch.float32, device=dev)
        _dyp_kernel[(TK, triton.cdiv(D, BD))](dy, ws, toks, dyp, D, BLOCK=BD)
        act32 = torch.empty(TK, F, dtype=torch.float32, device=dev)
        _swiglu_fwd_kernel[(triton.cdiv(TK * F, BD),)](
            out1, act32, TK * F, F, F2, BLOCK=BD
        )

        yp32 = torch.empty(TK, D, dtype=torch.float32, device=dev)
        _mm_launch(
            act32,
            w2,
            yp32,
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
        _dw_scalar_kernel[(T, K)](dy, yp32, pos, dwc, D, K, BLOCK=BD)
        del yp32

        dact = torch.empty(TK, F, dtype=torch.float32, device=dev)
        _mm_launch(
            dyp, w2, dact, toks, texp, trow, seg_end, D, F, F * D, 1, D, 0, p_a, ntiles
        )
        dw2 = torch.empty(E, F, D, dtype=w2.dtype, device=dev)
        _dw_launch(act32, dyp, dw2, toks, seg_start, seg_end, E, F, D, F, 0, p_ab)
        del act32, dyp

        dgu = torch.empty(TK, F2, dtype=torch.float32, device=dev)
        _swiglu_bwd_kernel[(triton.cdiv(TK * F, BD),)](
            out1, dact, dgu, TK * F, F, F2, BLOCK=BD
        )
        del dact

        dw1 = torch.empty(E, D, F2, dtype=w1.dtype, device=dev)
        _dw_launch(x, dgu, dw1, toks, seg_start, seg_end, E, D, F2, D, 1, p_b)
        dgu_c = dgu.to(cdt)
        del dgu
        dxs = torch.empty(TK, D, dtype=torch.float32, device=dev)
        _mm_launch(
            dgu_c,
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
        del dgu_c
        dxe = torch.empty(T, D, dtype=torch.float32, device=dev)
        _dx_combine_kernel[(T, triton.cdiv(D, BD))](dxs, pos, dxe, D, K, BLOCK=BD)

        # router bwd (p_sel/sum + softmax Jacobian over the whole row;
        # both selection stages are locally constant)
        wsum = (dwc * w).sum(dim=-1, keepdim=True)
        psum = torch.gather(p_dense, 1, sel.to(torch.int64)).sum(dim=-1, keepdim=True)
        dp_sel = (dwc - wsum) / psum
        dp = torch.zeros_like(p_dense)
        dp.scatter_(1, sel.to(torch.int64), dp_sel)
        row = (p_dense * dp).sum(dim=-1, keepdim=True)
        dlogits = p_dense * (dp - row)
        xf = x.float()
        rwf = router_w.float()
        drw = (xf.t() @ dlogits).to(router_w.dtype)
        dx = (dxe + dlogits @ rwf.t()).to(x.dtype)
        return dx, drw, dw1, dw2, None, None


def fused_moe(
    x: torch.Tensor,
    router_w: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    G: int,
    K: int,
) -> torch.Tensor:
    return _FusedMoeOracle.apply(x, router_w, w1, w2, G, K)
# FORGE-CANARY-BEGIN forge-canary/v1
# FORGE-CANARY-SLOT-0 19386f5093ddc3f25489ad1e573171ada161012e470c3b9d8625cb5eac9c2606
# FORGE-CANARY-SLOT-1 3b085cfcafe8fff741c881194a38510b32d3e1bf8cd1dcea5c82181ef75922b8
# FORGE-CANARY-SLOT-2 b7d6aeace93d74b5bf64cb41bada31abd24dacd9928922865071163705d181cf
# FORGE-CANARY-SLOT-3 17b3553eebd0b23f27d50867357b7ddcc0cb34454ce062932fbc72fe31303d1f
# FORGE-CANARY-END
