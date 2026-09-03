"""Private oracle -- from-scratch Triton capacity-limited Switch-style MoE, fwd + bwd.

Never shipped to an agent. Establishes that full reward is reachable through
every live gate (G1 forbidden-symbol scan, G2 semantics including the exact
lower-expert-index selection tie-break and the lower-token-index capacity
tie-break, G3 timing, G4 written-kernel adoption floor) with no grouped-mm
library call anywhere.

Structure (all deterministic; no atomics, no value-dependent branching):

  routing   softmax logits via one dense matmul, then ONE Triton kernel per
            token row computes p, the argmax expert e* under the unique-key
            trick (order-preserving fp32 bits << 10 | inverted expert index)
            and the raw Switch gate p[t, e*]. Bit-equal p entries decide by
            the lower-expert-index rule structurally, never by scheduling.
  capacity  per-expert candidate top-`capacity` selection by p[:, e*] with
            LOWER-token-index tie-break, realised through one Triton kernel
            per (expert, T) that packs (order-preserving p bits << tokenbits
            | inverted token index) into unique int64 keys and picks the
            top-`capacity` in a bitonic-style reduction (no atomics). Every
            overflow candidate is written to the drop mask.
  dispatch  stable integer argsort of the [T] top-1 expert assignments on
            the KEPT-token subset, per-expert segment bounds, tile table.
  compute   segment SwiGLU expert GEMMs: one shared _seg_mm kernel invoked
            for gate/up (with fused token gather from x) and down, plus its
            dW twin for the two weight gradients. bf16 hi/lo split precision
            keeps the weight-gradient operand chain float32-effective on
            tensor cores. Dropped tokens contribute exactly zero everywhere.
  router bw the Switch gate p[t, e*] backpropagates through the softmax
            Jacobian over the WHOLE expert row (dense) into router_w and x.
            Dropped tokens receive zero softmax-row gradient because their
            gate is unused in the forward.

  Precision policy (measured against fp32 reference): forward expert GEMMs
  and the dx expert data path keep bf16 operands on bf16 inputs; every
  weight-gradient operand (dyp, recomputed act, dact, dgu) carries
  float32-effective precision via bf16 hi/lo splitting on tensor cores.
  Float32 inputs run "ieee" dots throughout; tf32 is never allowed.

  PREC modes:  0 native bf16 x bf16
               1 fp32 "ieee"
               2 split-A x native-B  (A fp32-computed, B bf16 leaf)
               3 split-A x split-B   (both fp32-computed)
               4 native-A x split-B  (A bf16 leaf, B fp32-computed)
"""

import torch
import triton
import triton.language as tl

BM = 64
BD = 256


# ---------------------------------------------------------------------------
# routing: per-token top-1 by softmax weight, unique-key tie-break
# ---------------------------------------------------------------------------


@triton.jit
def _select_top1_kernel(LOG, P, SEL, GATE, T, E, BE: tl.constexpr):
    """Per token row: fp32 softmax over E logits, top-1 by unique key,
    store p (for backward), sel (int32), gate = p[t, e*] (raw Switch)."""
    t = tl.program_id(0).to(tl.int64)
    e = tl.arange(0, BE)
    m = e < E
    lo = tl.load(LOG + t * E + e.to(tl.int64), mask=m, other=-1e30)
    mx = tl.max(lo, axis=0)
    ex = tl.exp(lo - mx)
    ex = tl.where(m, ex, 0.0)
    sm = tl.sum(ex, axis=0)
    p = ex / sm
    # store p for backward
    tl.store(P + t * E + e.to(tl.int64), p, mask=m)
    # unique-key top-1 by p desc, expert idx asc
    b = tl.where(p == 0.0, 0.0, p)
    ib = b.to(tl.int32, bitcast=True).to(tl.int64)
    u = tl.where(ib >= 0, ib ^ 0x80000000, (~ib) & 0xFFFFFFFF)
    key = (u << 10) | (E - 1 - e).to(tl.int64)
    key = tl.where(m, key, -1)
    kmax = tl.max(key, axis=0)
    e_star = E - 1 - (kmax & 1023).to(tl.int32)
    hit = e == e_star
    g_val = tl.sum(tl.where(hit, p, 0.0), axis=0)
    tl.store(SEL + t, e_star)
    tl.store(GATE + t, g_val)


@triton.jit
def _cap_keep_kernel(P, SEL, KEEP, T, E, cap, BT: tl.constexpr, TBITS: tl.constexpr):
    """Per expert e: scan T tokens, mark the top-`cap` candidates (with
    sel==e) by p[:, e] as KEEP=1. Ties broken to LOWER token index via
    the unique-key trick over (p bits << tbits) | (T-1-t)."""
    e = tl.program_id(0).to(tl.int64)
    t = tl.arange(0, BT).to(tl.int64)
    m = t < T
    p_val = tl.load(P + t * E + e, mask=m, other=0.0)
    sel_t = tl.load(SEL + t, mask=m, other=-1)
    cand = m & (sel_t == e)
    # unique key: (fp32 order bits << tbits) | (T-1-t); non-candidates -> -1
    b = tl.where(p_val == 0.0, 0.0, p_val)
    ib = b.to(tl.int32, bitcast=True).to(tl.int64)
    u = tl.where(ib >= 0, ib ^ 0x80000000, (~ib) & 0xFFFFFFFF)
    key = (u << TBITS) | (T - 1 - t)
    key = tl.where(cand, key, -1)
    # count real candidates, clamp cap
    ncand = tl.sum(tl.where(cand, 1, 0), axis=0)
    kkeep = tl.minimum(cap, ncand)
    # iterative top-cap in registers: no atomics, no scheduling dependence
    keep = tl.zeros((BT,), dtype=tl.int32)
    for _ in range(cap):
        mx = tl.max(key, axis=0)
        picked_t = T - 1 - (mx & ((1 << TBITS) - 1)).to(tl.int32)
        alive = kkeep > 0
        hit = (t == picked_t) & alive
        keep = tl.where(hit, 1, keep)
        key = tl.where(hit, -1, key)
        kkeep = kkeep - tl.where(alive, 1, 0)
    tl.store(KEEP + e * T + t, keep, mask=m)


# ---------------------------------------------------------------------------
# segment GEMM machinery (shared kernel, precision-mode multiplex)
# ---------------------------------------------------------------------------


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
    if GATHER:
        rowid = tl.load(TOK + rm, mask=rmask, other=0).to(tl.int64)
    else:
        rowid = rm
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
        if GATHER:
            rows = tl.load(TOK + rr, mask=rmask, other=0).to(tl.int64)
        else:
            rows = rr
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


# ---------------------------------------------------------------------------
# SwiGLU elementwise
# ---------------------------------------------------------------------------


@triton.jit
def _swiglu_fwd_kernel(GU, ACT, total, F, F2, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    o = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    m = o < total
    r = o // F
    f = o % F
    g = tl.load(GU + r * F2 + f, mask=m, other=0.0)
    u = tl.load(GU + r * F2 + F + f, mask=m, other=0.0)
    sg = tl.sigmoid(g)
    tl.store(ACT + o, g * sg * u, mask=m)


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


# ---------------------------------------------------------------------------
# combine (top-1: one contribution per kept token) and gradient path
# ---------------------------------------------------------------------------


@triton.jit
def _combine_fwd_kernel(YP, GATE, TOK, KEEP_T, Y, D, BLOCK: tl.constexpr):
    """Each program handles (kept_row q, D-tile). Writes gate[t]*yp[q] into
    y[t] for the token t=TOK[q]. Dropped tokens keep the zero-init in y."""
    q = tl.program_id(0).to(tl.int64)
    pd = tl.program_id(1).to(tl.int64)
    d = pd * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    dm = d < D
    t = tl.load(TOK + q).to(tl.int64)
    gv = tl.load(GATE + t)
    yv = tl.load(YP + q * D + d, mask=dm, other=0.0)
    tl.store(Y + t * D + d, gv * yv, mask=dm)


@triton.jit
def _dyp_kernel(DY, GATE, TOK, DYP, D, BLOCK: tl.constexpr):
    """dyp[q, :] = gate[t] * dy[t, :] for kept rows only (t = TOK[q])."""
    q = tl.program_id(0).to(tl.int64)
    pd = tl.program_id(1).to(tl.int64)
    d = pd * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    dm = d < D
    t = tl.load(TOK + q).to(tl.int64)
    gv = tl.load(GATE + t)
    dyv = tl.load(DY + t * D + d, mask=dm, other=0.0).to(tl.float32)
    tl.store(DYP + q * D + d, gv * dyv, mask=dm)


@triton.jit
def _dgate_kernel(DY, YP32, TOK, DG_T, D, BLOCK: tl.constexpr):
    """dgate[t] += <dy[t], yp32[q]> for kept rows only."""
    q = tl.program_id(0).to(tl.int64)
    t = tl.load(TOK + q).to(tl.int64)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for d0 in range(0, D, BLOCK):
        d = d0 + tl.arange(0, BLOCK).to(tl.int64)
        dm = d < D
        dyv = tl.load(DY + t * D + d, mask=dm, other=0.0).to(tl.float32)
        ypv = tl.load(YP32 + q * D + d, mask=dm, other=0.0)
        acc += dyv * ypv
    tl.store(DG_T + t, tl.sum(acc, axis=0))


# ---------------------------------------------------------------------------
# host side
# ---------------------------------------------------------------------------


def _dispatch(sel_kept_flat, E):
    """Sorted dispatch for the KEPT-only slot list."""
    dev = sel_kept_flat.device
    Q = sel_kept_flat.numel()
    order = torch.argsort(sel_kept_flat.to(torch.int64), stable=True)
    counts = torch.bincount(sel_kept_flat.to(torch.int64), minlength=E)
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
    def forward(ctx, x, router_w, w1, w2, capacity):
        capacity = int(capacity)
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

        # router matmul in fp32 (small tensor; keeps margin)
        logits = x.detach().float() @ router_w.detach().float()  # [T, E]
        p = torch.empty(T, E, dtype=torch.float32, device=dev)
        sel = torch.empty(T, dtype=torch.int32, device=dev)
        gate = torch.empty(T, dtype=torch.float32, device=dev)
        _select_top1_kernel[(T,)](
            logits, p, sel, gate, T, E, BE=max(triton.next_power_of_2(E), 2)
        )

        # per-expert capacity keep mask (kept[t]=1 iff t chosen AND top-cap)
        keep_e = torch.zeros(E, T, dtype=torch.int32, device=dev)
        tbits = max((T - 1).bit_length(), 1)
        _cap_keep_kernel[(E,)](
            p, sel, keep_e, T, E, capacity, BT=triton.next_power_of_2(T), TBITS=tbits
        )
        keep = keep_e.sum(dim=0).to(torch.bool)  # [T]

        # kept-only dispatch
        kept_idx = keep.nonzero(as_tuple=True)[0].to(torch.int64)
        Q = int(kept_idx.numel())
        if Q == 0:
            y = torch.zeros(T, D, dtype=cdt, device=dev)
            ctx.save_for_backward(
                x, router_w, w1, w2, p, sel, gate, kept_idx, kept_idx.to(torch.int32)
            )
            ctx.meta = (capacity, 0, ieee, F, F2)
            return y
        sel_kept = sel.to(torch.int64)[kept_idx]  # [Q]
        order, seg_start, seg_end, texp, trow, ntiles = _dispatch(sel_kept, E)
        toks = kept_idx[order].to(torch.int32)  # [Q] token id
        toks_c = toks.contiguous()

        prec_fwd = 1 if ieee else 0
        out1 = torch.empty(Q, F2, dtype=torch.float32, device=dev)
        _mm_launch(
            x,
            w1,
            out1,
            toks_c,
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

        act = torch.empty(Q, F, dtype=cdt, device=dev)
        total = Q * F
        _swiglu_fwd_kernel[(triton.cdiv(total, BD),)](out1, act, total, F, F2, BLOCK=BD)

        yp = torch.empty(Q, D, dtype=torch.float32, device=dev)
        _mm_launch(
            act,
            w2,
            yp,
            toks_c,
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

        y = torch.zeros(T, D, dtype=cdt, device=dev)
        _combine_fwd_kernel[(Q, triton.cdiv(D, BD))](
            yp, gate, toks_c, keep.to(torch.int32), y, D, BLOCK=BD
        )

        ctx.save_for_backward(
            x,
            router_w,
            w1,
            w2,
            p,
            sel,
            gate,
            kept_idx,
            toks_c,
            seg_start,
            seg_end,
            texp,
            trow,
            out1,
        )
        ctx.meta = (capacity, ntiles, ieee, F, F2)
        return y

    @staticmethod
    def backward(ctx, dy):
        saved = ctx.saved_tensors
        if len(saved) == 9:  # empty-Q corner case
            x, router_w, w1, w2, p, sel, gate, kept_idx, _ = saved
            zeros_x = torch.zeros_like(x)
            zeros_rw = torch.zeros_like(router_w)
            zeros_w1 = torch.zeros_like(w1)
            zeros_w2 = torch.zeros_like(w2)
            return zeros_x, zeros_rw, zeros_w1, zeros_w2, None
        (
            x,
            router_w,
            w1,
            w2,
            p,
            sel,
            gate,
            kept_idx,
            toks_c,
            seg_start,
            seg_end,
            texp,
            trow,
            out1,
        ) = saved
        capacity, ntiles, ieee, F, F2 = ctx.meta
        T, D = x.shape
        E = router_w.shape[1]
        Q = int(toks_c.numel())
        dev = x.device
        cdt = x.dtype
        dy = dy.contiguous()

        p_a = 1 if ieee else 2
        p_ab = 1 if ieee else 3
        p_b = 1 if ieee else 4

        # dyp = gate[t] * dy[t]  for kept rows only
        dyp = torch.empty(Q, D, dtype=torch.float32, device=dev)
        _dyp_kernel[(Q, triton.cdiv(D, BD))](dy, gate, toks_c, dyp, D, BLOCK=BD)

        # recompute activation at fp32 for weight-grad chain
        act32 = torch.empty(Q, F, dtype=torch.float32, device=dev)
        _swiglu_fwd_kernel[(triton.cdiv(Q * F, BD),)](
            out1, act32, Q * F, F, F2, BLOCK=BD
        )

        # dgate[t]: recomputed fp32 yp at kept rows
        yp32 = torch.empty(Q, D, dtype=torch.float32, device=dev)
        _mm_launch(
            act32,
            w2,
            yp32,
            toks_c,
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
        dgate_t = torch.zeros(T, dtype=torch.float32, device=dev)
        _dgate_kernel[(Q,)](dy, yp32, toks_c, dgate_t, D, BLOCK=BD)
        del yp32

        # dw2, dact
        dact = torch.empty(Q, F, dtype=torch.float32, device=dev)
        _mm_launch(
            dyp,
            w2,
            dact,
            toks_c,
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
        _dw_launch(act32, dyp, dw2, toks_c, seg_start, seg_end, E, F, D, F, 0, p_ab)
        del act32, dyp

        # activation bwd
        dgu = torch.empty(Q, F2, dtype=torch.float32, device=dev)
        _swiglu_bwd_kernel[(triton.cdiv(Q * F, BD),)](
            out1, dact, dgu, Q * F, F, F2, BLOCK=BD
        )
        del dact

        # dw1, dx_expert
        dw1 = torch.empty(E, D, F2, dtype=w1.dtype, device=dev)
        _dw_launch(x, dgu, dw1, toks_c, seg_start, seg_end, E, D, F2, D, 1, p_b)
        dgu_c = dgu.to(cdt)
        del dgu
        dxs = torch.empty(Q, D, dtype=torch.float32, device=dev)
        _mm_launch(
            dgu_c,
            w1,
            dxs,
            toks_c,
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
        # scatter dxs back to per-token dx-expert
        dxe = torch.zeros(T, D, dtype=torch.float32, device=dev)
        dxe.index_add_(0, toks_c.to(torch.int64), dxs)

        # router bwd: dense softmax Jacobian at chosen expert only (Switch)
        # gate[t] = p[t, e*]; grad ds[t, j] = dgate[t] * p[t, j] * (delta_{j,e*} - p[t, e*])
        # written as: ds_row = dgate[t] * p[t] * (onehot_{e*} - p[t, e*])
        # simpler: dlogits[t] = dgate[t] * (p[t] * onehot_{e*}) - dgate[t] * p[t, e*] * p[t]
        #                     = dgate[t] * p[t, e*] * (onehot_{e*} - p[t])
        # (softmax Jacobian at the argmax entry; dropped tokens have dgate=0 -> zero row)
        onehot = torch.zeros_like(p)
        onehot.scatter_(1, sel.to(torch.int64).unsqueeze(1), 1.0)
        dlogits = (dgate_t * gate).unsqueeze(1) * (onehot - p)
        xf = x.float()
        rwf = router_w.float()
        drw = (xf.t() @ dlogits).to(router_w.dtype)
        dx = (dxe + dlogits @ rwf.t()).to(x.dtype)
        return dx, drw, dw1, dw2, None


def fused_moe(
    x: torch.Tensor,  # [T, D] float32 or bfloat16
    router_w: torch.Tensor,  # [D, E] same dtype as x
    w1: torch.Tensor,  # [E, D, 2F]  gate|up
    w2: torch.Tensor,  # [E, F, D]
    capacity: int,  # per-expert token cap
) -> torch.Tensor:  # [T, D] same dtype as x
    return _FusedMoeOracle.apply(x, router_w, w1, w2, capacity)
# FORGE-CANARY-BEGIN forge-canary/v1
# FORGE-CANARY-SLOT-0 b173dc269c4b528e50f1815d7b6ee405819dac3b0022ede31c6e35a20ff21149
# FORGE-CANARY-SLOT-1 df6844fd145ff8951e4f4e42b79e0fad590894f957d98730c58f0d3a6be2b8bd
# FORGE-CANARY-SLOT-2 e0cae78f54d8feaa8bfa13a59ac9973cfdc0a2ff21bbdde6670a57e046a4911b
# FORGE-CANARY-SLOT-3 edc38a4f5fb6503f8d211c7cd378c974325417c475aa6aa7329c0e106497bf0a
# FORGE-CANARY-END
