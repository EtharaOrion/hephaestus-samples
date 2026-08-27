"""Private oracle -- from-scratch Triton routed fused-MoE, forward + backward.

Never shipped to an agent. Establishes that full reward is reachable through
every live gate (G1 symbol scan, G2 semantics incl. the exact tie-break, G3
timing, G4 adoption) with no grouped-matmul library call anywhere.

Structure (all deterministic; no atomics, no value-dependent branching):

  routing   router logits via one dense matmul, then ONE Triton kernel per
            token row: sigmoid, biased composite keys (order-preserving fp32
            bits << 10 | inverted expert index -- unique, so ties never
            exist), iterative top-A extraction in registers, raw-score
            normalization. Selection-only bias by construction: the bias
            enters the keys, never the stored weights.
  dispatch  stable argsort of the [T*A] expert ids (torch integer sort on
            routing metadata; the arithmetic stays in the kernels below),
            per-expert segment bounds, and a tile table: each 64-row tile of
            each expert segment becomes one program of the segment GEMMs.
  compute   one shared segment-GEMM kernel instantiated four ways (gate/up
            projection with fused token gather, down projection, and the two
            backward data GEMMs against transposed weights via stride swap),
            one shared segment-dW kernel instantiated twice, a SwiGLU
            elementwise pair, and ordered per-token combine kernels (slot
            order ascending -- reproducible by construction).
  router bw combine-weight grads -> normalization backward -> sigmoid
            backward at the selected entries only -> two small dense matmuls
            for drouter_w and the router half of dx.

  float32 accumulation everywhere. Precision policy, measured against the
  fp32 reference: the forward expert GEMMs and the dx data path keep
  bfloat16 operands (their error is bounded by short reductions and the
  output cast), but every WEIGHT-gradient operand -- dyp, the recomputed
  activations, dact, dgu -- carries float32-effective precision, because a
  bfloat16-rounded operand there turns the long per-expert row sums into
  noise orders of magnitude outside any sane element tolerance. On the
  bfloat16 surface that precision is delivered on TENSOR CORES by hi/lo
  splitting: a float32 operand becomes bf16(a) plus bf16(a - bf16(a)), two
  (or, when both operands are float32, three) bf16 dots whose float32
  accumulation reconstructs the product to ~2^-18 relative -- an order of
  magnitude below the graded tolerance and an order of magnitude faster
  than FMA-pipe "ieee" dots. float32 inputs run "ieee" dots throughout
  (tf32 is never allowed anywhere: its ~1e-3 error breaks both the output
  tolerance and the routing margin).

  PREC modes for the two GEMM kernels:
      0  native dot            (bf16 x bf16)
      1  cast both to fp32, input_precision="ieee"
      2  split-A x native-B    (A fp32 computed, B a bf16 leaf)
      3  split-A x split-B     (both fp32 computed)
      4  native-A x split-B    (A a bf16 leaf, B fp32 computed)
"""

import torch
import triton
import triton.language as tl

BM = 64            # segment tile rows (fixed: the tile table is built on it)
BD = 256           # combine/elementwise block
IDXB = 10          # index bits in the selection key (E <= 1024)


# ---------------------------------------------------------------------------
# routing
# ---------------------------------------------------------------------------

@triton.jit
def _select_kernel(LOG, RB, SEL, SSEL, W, SSUM, E,
                   A: tl.constexpr, BE: tl.constexpr):
    """Per token row: sigmoid scores, biased unique keys, iterative top-A.

    key = (ordered_bits(fp32(s + rb)) << 10) | (E - 1 - e); every key is
    unique because the expert index is embedded, so the running maximum
    realises (biased score desc, expert index asc) exactly and the tie-break
    is structural, not a special case. -0.0 canonicalises to +0.0 first.
    The stored combine weights normalize the RAW sigmoid scores of the
    selected experts; the bias influences selection only.
    """
    t = tl.program_id(0).to(tl.int64)
    e = tl.arange(0, BE)
    m = e < E
    lo = tl.load(LOG + t * E + e.to(tl.int64), mask=m, other=0.0)
    s = tl.sigmoid(lo)
    rb = tl.load(RB + e, mask=m, other=0.0)
    b = s + rb
    b = tl.where(b == 0.0, 0.0, b)
    ib = b.to(tl.int32, bitcast=True).to(tl.int64)
    u = tl.where(ib >= 0, ib ^ 0x80000000, (~ib) & 0xFFFFFFFF)
    key = (u << 10) | (E - 1 - e).to(tl.int64)
    key = tl.where(m, key, -1)
    ssum = 0.0
    for a in tl.static_range(A):
        mx = tl.max(key, axis=0)
        col = E - 1 - (mx & 1023).to(tl.int32)
        hit = e == col
        sv = tl.sum(tl.where(hit, s, 0.0), axis=0)
        tl.store(SEL + t * A + a, col)
        tl.store(SSEL + t * A + a, sv)
        ssum += sv
        key = tl.where(hit, -1, key)
    tl.store(SSUM + t, ssum)
    for a in tl.static_range(A):
        sv = tl.load(SSEL + t * A + a)
        wv = sv / ssum
        tl.store(W + t * A + a, wv)


# ---------------------------------------------------------------------------
# segment GEMMs over the sorted (token, slot) pairs
# ---------------------------------------------------------------------------

@triton.jit
def _prec_dot(a, b, acc, PREC: tl.constexpr):
    """One K-block contribution at the requested precision mode."""
    if PREC == 0:
        acc += tl.dot(a, b)
    elif PREC == 1:
        acc += tl.dot(a.to(tl.float32), b.to(tl.float32),
                      input_precision="ieee")
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
def _seg_mm_kernel(AP, BP, OP, TOK, TE, TR0, SEGEND,
                   K, N, sbe, sbk, sbn,
                   GATHER: tl.constexpr, BLK_M: tl.constexpr,
                   BLK_N: tl.constexpr, BLK_K: tl.constexpr,
                   PREC: tl.constexpr):
    """One 64-row tile of one expert segment times that expert's weight.

    A rows are sorted-pair rows (GATHER=0) or token rows fetched through the
    sorted token table (GATHER=1, the fused dispatch: x is never copied into
    a [T*A, D] buffer on the forward input side). B is one expert's weight
    matrix addressed by explicit strides, so the transposed backward-data
    GEMMs reuse this kernel with a stride swap. Accumulation is float32; the
    store casts to whatever OP holds.
    """
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
        a = tl.load(AP + rowid[:, None] * K + rk[None, :].to(tl.int64),
                    mask=rmask[:, None] & kmask[None, :], other=0.0)
        bmat = tl.load(bbase + rk[:, None].to(tl.int64) * sbk
                       + rn[None, :].to(tl.int64) * sbn,
                       mask=kmask[:, None] & nmask[None, :], other=0.0)
        acc = _prec_dot(a, bmat, acc, PREC)
    tl.store(OP + rm[:, None] * N + rn[None, :].to(tl.int64), acc,
             mask=rmask[:, None] & nmask[None, :])


@triton.jit
def _seg_dw_kernel(AP, BP, OP, TOK, SEGSTART, SEGEND,
                   M, N, K_ROWSTRIDE_A,
                   GATHER: tl.constexpr, BLK_M: tl.constexpr,
                   BLK_N: tl.constexpr, BLK_K: tl.constexpr,
                   PREC: tl.constexpr):
    """dW[e][M, N] = sum over expert-e segment rows r of a[r, M]^T b[r, N].

    One program per (expert, M-tile, N-tile); the segment reduction walks the
    rows in a fixed ascending order, so the accumulation order never depends
    on scheduling. Experts with empty segments store exact zeros, matching
    the reference's zero gradients for unused experts.
    """
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
        a = tl.load(AP + rows[:, None] * K_ROWSTRIDE_A + rm[None, :].to(tl.int64),
                    mask=rmask[:, None] & mmask[None, :], other=0.0)
        bmat = tl.load(BP + rr[:, None] * N + rn[None, :].to(tl.int64),
                       mask=rmask[:, None] & nmask[None, :], other=0.0)
        acc = _prec_dot(tl.trans(a), bmat, acc, PREC)
    tl.store(OP + pe * M * N + rm[:, None].to(tl.int64) * N
             + rn[None, :].to(tl.int64), acc,
             mask=mmask[:, None] & nmask[None, :])


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
# ordered combines (slot order ascending -- deterministic by construction)
# ---------------------------------------------------------------------------

@triton.jit
def _combine_fwd_kernel(YP, W, POS, Y, D, A_r, BLOCK: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    pd = tl.program_id(1).to(tl.int64)
    d = pd * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    dm = d < D
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for a in range(0, A_r):
        q = tl.load(POS + t * A_r + a).to(tl.int64)
        wv = tl.load(W + t * A_r + a)
        acc += wv * tl.load(YP + q * D + d, mask=dm, other=0.0)
    tl.store(Y + t * D + d, acc, mask=dm)


@triton.jit
def _dx_combine_kernel(DXS, POS, OUT, D, A_r, BLOCK: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    pd = tl.program_id(1).to(tl.int64)
    d = pd * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    dm = d < D
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for a in range(0, A_r):
        q = tl.load(POS + t * A_r + a).to(tl.int64)
        acc += tl.load(DXS + q * D + d, mask=dm, other=0.0)
    tl.store(OUT + t * D + d, acc, mask=dm)


@triton.jit
def _combine_bwd_dyp_kernel(DY, WS, TOKS, DYP, D, BLOCK: tl.constexpr):
    q = tl.program_id(0).to(tl.int64)
    pd = tl.program_id(1).to(tl.int64)
    d = pd * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    dm = d < D
    t = tl.load(TOKS + q).to(tl.int64)
    wv = tl.load(WS + q)
    dyv = tl.load(DY + t * D + d, mask=dm, other=0.0).to(tl.float32)
    tl.store(DYP + q * D + d, wv * dyv, mask=dm)


@triton.jit
def _combine_bwd_w_kernel(DY, YP, POS, DWC, D, A_r, BLOCK: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    a = tl.program_id(1).to(tl.int64)
    q = tl.load(POS + t * A_r + a).to(tl.int64)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for d0 in range(0, D, BLOCK):
        d = d0 + tl.arange(0, BLOCK).to(tl.int64)
        dm = d < D
        dyv = tl.load(DY + t * D + d, mask=dm, other=0.0).to(tl.float32)
        ypv = tl.load(YP + q * D + d, mask=dm, other=0.0)
        acc += dyv * ypv
    tl.store(DWC + t * A_r + a, tl.sum(acc, axis=0))


# ---------------------------------------------------------------------------
# host side
# ---------------------------------------------------------------------------

def _mm_launch(ap, bp, op, toks, texp, trow, seg_end,
               K, N, sbe, sbk, sbn, gather, prec, ntiles):
    """Segment GEMM launch with the measured per-precision tile config.

    Tensor-core modes (0/2/3/4) run 64x128x64 tiles at 8 warps; the "ieee"
    FMA mode runs 64x64x32 at 8 warps -- measured on the graded shapes, the
    wide-tile fp32 accumulator at low warp count register-spills two orders
    of magnitude off the roofline.
    """
    if prec == 1:
        bn, bk, w, s = 64, 32, 8, 3
    elif prec == 2:
        bn, bk, w, s = 256, 64, 8, 3
    else:
        bn, bk, w, s = 128, 64, 8, 3
    _seg_mm_kernel[(ntiles, triton.cdiv(N, bn))](
        ap, bp, op, toks, texp, trow, seg_end, K, N, sbe, sbk, sbn,
        GATHER=gather, BLK_M=BM, BLK_N=bn, BLK_K=bk, PREC=prec,
        num_warps=w, num_stages=s)


def _dw_launch(ap, bp, op, toks, seg_start, seg_end, E, M, N, rsa,
               gather, prec):
    """Segment dW launch with the measured per-precision tile config."""
    if prec == 3:
        bm, bn, bk, w = 128, 128, 32, 8
    else:
        bm, bn, bk, w = 64, 128, 32, 4
    _seg_dw_kernel[(E, triton.cdiv(M, bm), triton.cdiv(N, bn))](
        ap, bp, op, toks, seg_start, seg_end, M, N, rsa,
        GATHER=gather, BLK_M=bm, BLK_N=bn, BLK_K=bk, PREC=prec,
        num_warps=w)


def _dispatch(sel: torch.Tensor, E: int, A: int):
    """Sorted dispatch metadata from the [T, A] selection (all int32 on GPU).

    Stable integer argsort of the flattened expert ids gives the sorted pair
    order; each expert's segment is padded to BM-row tiles and every tile
    becomes one row of the tile table the segment GEMMs launch over.
    """
    dev = sel.device
    flat = sel.reshape(-1).to(torch.int64)
    TA = flat.numel()
    order = torch.argsort(flat, stable=True)
    pos = torch.empty(TA, dtype=torch.int64, device=dev)
    pos[order] = torch.arange(TA, dtype=torch.int64, device=dev)
    toks = torch.div(order, A, rounding_mode="floor").to(torch.int32)
    counts = torch.bincount(flat, minlength=E)
    seg_end = counts.cumsum(0)
    seg_start = seg_end - counts
    tiles_per = (counts + BM - 1) // BM
    tile_cum = tiles_per.cumsum(0) - tiles_per
    ntiles = int(tiles_per.sum().item())
    texp = torch.repeat_interleave(
        torch.arange(E, dtype=torch.int64, device=dev), tiles_per)
    tid = torch.arange(ntiles, dtype=torch.int64, device=dev)
    trow = (seg_start[texp] + (tid - tile_cum[texp]) * BM).to(torch.int32)
    return (order, pos.to(torch.int32), toks, seg_start.to(torch.int32),
            seg_end.to(torch.int32), texp.to(torch.int32), trow, ntiles)


class _FusedMoeOracle(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, router_w, router_b, w1, w2, A):
        A = int(A)
        x = x.contiguous()
        router_w = router_w.contiguous()
        rb = router_b.contiguous().float()
        w1 = w1.contiguous()
        w2 = w2.contiguous()
        T, D = x.shape
        E = router_w.shape[1]
        F2 = w1.shape[2]
        F = F2 // 2
        dev = x.device
        ieee = x.dtype == torch.float32
        cdt = x.dtype

        logits = x.detach().float() @ router_w.detach().float()   # [T, E] fp32
        sel = torch.empty(T, A, dtype=torch.int32, device=dev)
        ssel = torch.empty(T, A, dtype=torch.float32, device=dev)
        w = torch.empty(T, A, dtype=torch.float32, device=dev)
        ssum = torch.empty(T, dtype=torch.float32, device=dev)
        _select_kernel[(T,)](logits, rb, sel, ssel, w, ssum, E,
                             A=A, BE=max(triton.next_power_of_2(E), 2))

        (order, pos, toks, seg_start, seg_end,
         texp, trow, ntiles) = _dispatch(sel, E, A)
        TA = T * A
        ws = w.reshape(-1).index_select(0, order).contiguous()    # [TA] fp32

        prec_fwd = 1 if ieee else 0
        out1 = torch.empty(TA, F2, dtype=torch.float32, device=dev)
        _mm_launch(x, w1, out1, toks, texp, trow, seg_end,
                   D, F2, D * F2, F2, 1, 1, prec_fwd, ntiles)

        act = torch.empty(TA, F, dtype=cdt, device=dev)
        total = TA * F
        _swiglu_fwd_kernel[(triton.cdiv(total, BD),)](
            out1, act, total, F, F2, BLOCK=BD)

        yp = torch.empty(TA, D, dtype=torch.float32, device=dev)
        _mm_launch(act, w2, yp, toks, texp, trow, seg_end,
                   F, D, F * D, D, 1, 0, prec_fwd, ntiles)

        y = torch.empty(T, D, dtype=cdt, device=dev)
        _combine_fwd_kernel[(T, triton.cdiv(D, BD))](
            yp, w, pos, y, D, A, BLOCK=BD)

        ctx.save_for_backward(x, router_w, w1, w2, sel, ssel, w, ssum,
                              pos, toks, seg_start, seg_end, texp, trow,
                              ws, out1)
        ctx.meta = (A, ntiles, ieee)
        return y

    @staticmethod
    def backward(ctx, dy):
        (x, router_w, w1, w2, sel, ssel, w, ssum,
         pos, toks, seg_start, seg_end, texp, trow,
         ws, out1) = ctx.saved_tensors
        A, ntiles, ieee = ctx.meta
        T, D = x.shape
        E = router_w.shape[1]
        F2 = w1.shape[2]
        F = F2 // 2
        TA = T * A
        dev = x.device
        cdt = x.dtype
        dy = dy.contiguous()

        # combine backward: per-pair upstream grads (float32 -- these feed
        # the weight-gradient chain) and the fp32-recomputed activations
        dyp = torch.empty(TA, D, dtype=torch.float32, device=dev)
        _combine_bwd_dyp_kernel[(TA, triton.cdiv(D, BD))](
            dy, ws, toks, dyp, D, BLOCK=BD)
        act32 = torch.empty(TA, F, dtype=torch.float32, device=dev)
        total = TA * F
        _swiglu_fwd_kernel[(triton.cdiv(total, BD),)](
            out1, act32, total, F, F2, BLOCK=BD)

        # precision modes for the weight-grad chain: full fp32 effect --
        # "ieee" dots on the float32 surface, hi/lo split tensor-core dots
        # on the bfloat16 surface (see the module docstring)
        p_a = 1 if ieee else 2      # fp32-computed A x leaf B
        p_ab = 1 if ieee else 3     # fp32-computed A x fp32-computed B
        p_b = 1 if ieee else 4      # leaf A x fp32-computed B

        # combine-weight grads against the fp32-precision expert outputs
        yp32 = torch.empty(TA, D, dtype=torch.float32, device=dev)
        _mm_launch(act32, w2, yp32, toks, texp, trow, seg_end,
                   F, D, F * D, D, 1, 0, p_a, ntiles)
        dwc = torch.empty(T, A, dtype=torch.float32, device=dev)
        _combine_bwd_w_kernel[(T, A)](dy, yp32, pos, dwc, D, A, BLOCK=BD)
        del yp32

        # down projection backward (weight-grad chain keeps fp32 effect)
        dact = torch.empty(TA, F, dtype=torch.float32, device=dev)
        _mm_launch(dyp, w2, dact, toks, texp, trow, seg_end,
                   D, F, F * D, 1, D, 0, p_a, ntiles)
        dw2 = torch.empty(E, F, D, dtype=w2.dtype, device=dev)
        _dw_launch(act32, dyp, dw2, toks, seg_start, seg_end,
                   E, F, D, F, 0, p_ab)
        del act32, dyp

        # activation backward
        dgu = torch.empty(TA, F2, dtype=torch.float32, device=dev)
        _swiglu_bwd_kernel[(triton.cdiv(total, BD),)](
            out1, dact, dgu, total, F, F2, BLOCK=BD)
        del dact

        # gate/up projection backward
        dw1 = torch.empty(E, D, F2, dtype=w1.dtype, device=dev)
        _dw_launch(x, dgu, dw1, toks, seg_start, seg_end,
                   E, D, F2, D, 1, p_b)
        dgu_c = dgu.to(cdt)
        del dgu
        dxs = torch.empty(TA, D, dtype=torch.float32, device=dev)
        _mm_launch(dgu_c, w1, dxs, toks, texp, trow, seg_end,
                   F2, D, D * F2, 1, F2, 0, 1 if ieee else 0, ntiles)
        del dgu_c

        # expert half of dx: ordered per-token gather of the segment rows
        dxe = torch.empty(T, D, dtype=torch.float32, device=dev)
        _dx_combine_kernel[(T, triton.cdiv(D, BD))](
            dxs, pos, dxe, D, A, BLOCK=BD)

        # router backward: normalization -> sigmoid at selected entries only
        wsum = (dwc * w).sum(dim=-1, keepdim=True)
        ds_sel = (dwc - wsum) / ssum.unsqueeze(1)
        dlogit_sel = ds_sel * ssel * (1.0 - ssel)
        dlogits = torch.zeros(T, E, dtype=torch.float32, device=dev)
        dlogits.scatter_(1, sel.to(torch.int64), dlogit_sel)
        xf = x.float()
        rwf = router_w.float()
        drw = (xf.t() @ dlogits).to(router_w.dtype)
        dx = (dxe + dlogits @ rwf.t()).to(x.dtype)
        return dx, drw, None, dw1, dw2, None


def fused_moe(
    x: torch.Tensor,         # [T, D] float32 or bfloat16
    router_w: torch.Tensor,  # [D, E] same dtype as x
    router_b: torch.Tensor,  # [E] float32, selection-only bias, no gradient
    w1: torch.Tensor,        # [E, D, 2F]  [:, :, :F] gate, [:, :, F:] up
    w2: torch.Tensor,        # [E, F, D]
    A: int,                  # experts per token
) -> torch.Tensor:           # [T, D] same dtype as x
    return _FusedMoeOracle.apply(x, router_w, router_b, w1, w2, A)
# FORGE-CANARY-BEGIN forge-canary/v1
# FORGE-CANARY-SLOT-0 8215a22e4225cfc30e275ec814e7ddb124a14cd403662c0e21e1589362f4731a
# FORGE-CANARY-SLOT-1 f5410e9508bb5f2fef3c8124cea98af61d159d201a1609420842427ac38b4278
# FORGE-CANARY-SLOT-2 cdd60bafda970634ffb7c64018f6a7d1506f9afb2c3f710b5609f3c3b0444e4b
# FORGE-CANARY-SLOT-3 07b9e669b7aba77eae0eda15405545bc94fa27b78418c7644e3e61266f38da52
# FORGE-CANARY-END
