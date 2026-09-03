"""Private oracle -- from-scratch Triton expert-choice routed fused-MoE, fwd+bwd.

UNVERIFIED -- authored WITHOUT GPU calibration (CPU-only authoring pass; a live
KDA run owns the GPU). This file sets the eventual full-reward TARGET: a
from-scratch Triton implementation that should pass every live gate (G1 symbol
scan, G2 semantics incl. the exact per-expert token tie-break, G3 timing, G4
adoption) with no grouped-matmul library call anywhere. Its tile configs,
precision-mode routing, the CSR-combine loop lowering, and the target fraction
it reaches must ALL be re-measured on the H100 before this category is frozen;
treat every performance claim in the comments as design intent, not a measured
fact. Expert choice is the hard member -- expect the lowest reachable fraction
of the anchor in the family.

Never shipped to an agent. Structure (all deterministic; no atomics, no
value-dependent branching):

  routing   router logits via one dense matmul, softmax over experts in torch on
            the [T, E] affinity matrix, then per-expert top-C token selection via
            torch.topk on order-preserving keys (affinity bits << token_bits |
            inverted token index -- unique within an expert row, so ties resolve
            to the lower token index). The gathered affinities are the gates.
  dispatch  EXPERT-MAJOR and perfectly balanced: sel[E,C] flattens to E*C pairs
            already grouped by expert (segment e is [e*C, (e+1)*C)), so NO sort
            is needed for the compute. A separate stable argsort of the pair
            token-ids builds the per-token CSR (order, start, end) the combine
            reduces over -- this is the irregular part: a token gathers a
            data-dependent 0..E contributions.
  compute   one shared segment-GEMM kernel instantiated four ways (gate/up
            projection with fused token gather, down projection, and the two
            backward data GEMMs against transposed weights via stride swap), one
            shared segment-dW kernel instantiated twice, a SwiGLU elementwise
            pair, per-token CSR combine kernels for y and dx_expert, and per-pair
            gate/dyp/dgate kernels.
  router bw per-pair gate grads scattered into a dense [T, E] buffer -> DENSE
            softmax Jacobian over the expert row (dlogit_j = S_j (g_j - sum_k g_k
            S_k)) -> two small dense matmuls for drouter_w and the router half of
            dx.

  float32 accumulation everywhere. Precision policy (same as the siblings): the
  forward expert GEMMs and the dx data path keep bfloat16 operands, but every
  WEIGHT-gradient operand carries float32-effective precision via hi/lo
  tensor-core splitting on the bfloat16 surface. float32 inputs run "ieee" dots
  throughout (tf32 is never allowed: its ~1e-3 error breaks both the output
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


# ---------------------------------------------------------------------------
# routing (per-expert top-C over tokens; torch topk on the permitted metadata)
# ---------------------------------------------------------------------------

def _sel_topC(S: torch.Tensor, C: int) -> torch.Tensor:
    """[E, C] token indices: per expert, top-C tokens by affinity, ties to the
    LOWER token index. torch.topk on order-preserving integer keys is the
    permitted routing-step call; the heavy arithmetic stays in the kernels."""
    T, E = S.shape
    St = S.t().contiguous()
    b = torch.where(St == 0.0, torch.zeros_like(St), St).contiguous()
    ib = b.view(torch.int32).to(torch.int64)
    u = torch.where(ib >= 0, ib ^ 0x80000000, (~ib) & 0xFFFFFFFF)
    t_idx = torch.arange(T, device=S.device, dtype=torch.int64)
    tbits = max((T - 1).bit_length(), 1)
    keys = (u << tbits) | (T - 1 - t_idx).unsqueeze(0)
    return torch.topk(keys, C, dim=-1, largest=True, sorted=True).indices


# ---------------------------------------------------------------------------
# segment GEMMs over the expert-major (token, slot) pairs
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

    A rows are pair rows (GATHER=0) or token rows fetched through the token
    table (GATHER=1). B is one expert's weight matrix addressed by explicit
    strides, so the transposed backward-data GEMMs reuse this kernel with a
    stride swap. Accumulation is float32.
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

    The segment reduction walks rows in ascending order, so the accumulation
    order never depends on scheduling. Every expert has exactly C rows here.
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
# per-pair gate kernels and per-token CSR combines (deterministic reductions)
# ---------------------------------------------------------------------------

@triton.jit
def _dyp_kernel(DY, GATE, TOK, DYP, D, BLOCK: tl.constexpr):
    """dyp[j] = gate[j] * dy[tok[j]] (float32; feeds the weight-grad chain)."""
    j = tl.program_id(0).to(tl.int64)
    pd = tl.program_id(1).to(tl.int64)
    d = pd * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    dm = d < D
    t = tl.load(TOK + j).to(tl.int64)
    gv = tl.load(GATE + j)
    dyv = tl.load(DY + t * D + d, mask=dm, other=0.0).to(tl.float32)
    tl.store(DYP + j * D + d, gv * dyv, mask=dm)


@triton.jit
def _dgate_kernel(DY, YP, TOK, DGATE, D, BLOCK: tl.constexpr):
    """dgate[j] = <dy[tok[j]], yp[j]> (the affinity-gate grad, one scalar/pair)."""
    j = tl.program_id(0).to(tl.int64)
    t = tl.load(TOK + j).to(tl.int64)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for d0 in range(0, D, BLOCK):
        d = d0 + tl.arange(0, BLOCK).to(tl.int64)
        dm = d < D
        dyv = tl.load(DY + t * D + d, mask=dm, other=0.0).to(tl.float32)
        ypv = tl.load(YP + j * D + d, mask=dm, other=0.0)
        acc += dyv * ypv
    tl.store(DGATE + j, tl.sum(acc, axis=0))


@triton.jit
def _combine_ec_kernel(YP, GATE, ORDER, TSTART, TEND, Y, D, BLOCK: tl.constexpr):
    """y[t] = sum over t's pairs (ascending expert order) of gate[j] * yp[j].

    ORDER is the token-stable argsort of the pair token-ids, so [TSTART[t],
    TEND[t]) is t's contiguous run and the reduction order is fixed (=> bitwise
    deterministic). A token with an empty run stores exact zeros.
    """
    t = tl.program_id(0).to(tl.int64)
    pd = tl.program_id(1).to(tl.int64)
    d = pd * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    dm = d < D
    lo = tl.load(TSTART + t)
    hi = tl.load(TEND + t)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in range(lo, hi):
        j = tl.load(ORDER + k).to(tl.int64)
        gv = tl.load(GATE + j)
        acc += gv * tl.load(YP + j * D + d, mask=dm, other=0.0)
    tl.store(Y + t * D + d, acc, mask=dm)


@triton.jit
def _dx_combine_ec_kernel(DXS, ORDER, TSTART, TEND, OUT, D, BLOCK: tl.constexpr):
    """dx_expert[t] = sum over t's pairs of dxs[j] (ascending expert order)."""
    t = tl.program_id(0).to(tl.int64)
    pd = tl.program_id(1).to(tl.int64)
    d = pd * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    dm = d < D
    lo = tl.load(TSTART + t)
    hi = tl.load(TEND + t)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in range(lo, hi):
        j = tl.load(ORDER + k).to(tl.int64)
        acc += tl.load(DXS + j * D + d, mask=dm, other=0.0)
    tl.store(OUT + t * D + d, acc, mask=dm)


# ---------------------------------------------------------------------------
# host side
# ---------------------------------------------------------------------------

def _mm_launch(ap, bp, op, tok, texp, trow, seg_end,
               K, N, sbe, sbk, sbn, gather, prec, ntiles):
    """Segment GEMM launch with the (UNVERIFIED) per-precision tile config."""
    if prec == 1:
        bn, bk, w, s = 64, 32, 8, 3
    elif prec == 2:
        bn, bk, w, s = 256, 64, 8, 3
    else:
        bn, bk, w, s = 128, 64, 8, 3
    _seg_mm_kernel[(ntiles, triton.cdiv(N, bn))](
        ap, bp, op, tok, texp, trow, seg_end, K, N, sbe, sbk, sbn,
        GATHER=gather, BLK_M=BM, BLK_N=bn, BLK_K=bk, PREC=prec,
        num_warps=w, num_stages=s)


def _dw_launch(ap, bp, op, tok, seg_start, seg_end, E, M, N, rsa,
               gather, prec):
    """Segment dW launch with the (UNVERIFIED) per-precision tile config."""
    if prec == 3:
        bm, bn, bk, w = 128, 128, 32, 8
    else:
        bm, bn, bk, w = 64, 128, 32, 4
    _seg_dw_kernel[(E, triton.cdiv(M, bm), triton.cdiv(N, bn))](
        ap, bp, op, tok, seg_start, seg_end, M, N, rsa,
        GATHER=gather, BLK_M=bm, BLK_N=bn, BLK_K=bk, PREC=prec,
        num_warps=w)


def _dispatch_ec(sel: torch.Tensor, T: int, E: int, C: int):
    """Expert-major compute metadata (no sort) + per-token CSR for the combine."""
    dev = sel.device
    tok = sel.reshape(-1).to(torch.int32).contiguous()        # [E*C] expert-major
    tpe = triton.cdiv(C, BM)
    ntiles = E * tpe
    tid = torch.arange(ntiles, device=dev, dtype=torch.int64)
    texp = torch.div(tid, tpe, rounding_mode="floor")
    trow = (texp * C + (tid - texp * tpe) * BM).to(torch.int32)
    seg_start = (torch.arange(E, device=dev, dtype=torch.int64) * C).to(torch.int32)
    seg_end = (torch.arange(1, E + 1, device=dev, dtype=torch.int64) * C).to(torch.int32)
    tok64 = sel.reshape(-1).to(torch.int64)
    order = torch.argsort(tok64, stable=True).to(torch.int32)  # [E*C] token-sorted
    counts = torch.bincount(tok64, minlength=T)
    tend64 = counts.cumsum(0)
    tstart = (tend64 - counts).to(torch.int32)
    tend = tend64.to(torch.int32)
    return (tok, seg_start, seg_end, texp.to(torch.int32), trow, ntiles,
            order, tstart, tend)


class _FusedMoeOracle(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, router_w, w1, w2, C):
        C = int(C)
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
        EC = E * C

        logits = x.detach().float() @ router_w.detach().float()   # [T, E] fp32
        S = torch.softmax(logits, dim=-1)                         # [T, E] fp32
        sel = _sel_topC(S, C)                                     # [E, C]
        gate = torch.gather(S.t(), 1, sel).reshape(-1).contiguous()  # [E*C] fp32

        (tok, seg_start, seg_end, texp, trow, ntiles,
         order, tstart, tend) = _dispatch_ec(sel, T, E, C)

        prec_fwd = 1 if ieee else 0
        out1 = torch.empty(EC, F2, dtype=torch.float32, device=dev)
        _mm_launch(x, w1, out1, tok, texp, trow, seg_end,
                   D, F2, D * F2, F2, 1, 1, prec_fwd, ntiles)      # GATHER=1

        act = torch.empty(EC, F, dtype=cdt, device=dev)
        total = EC * F
        _swiglu_fwd_kernel[(triton.cdiv(total, BD),)](
            out1, act, total, F, F2, BLOCK=BD)

        yp = torch.empty(EC, D, dtype=torch.float32, device=dev)
        _mm_launch(act, w2, yp, tok, texp, trow, seg_end,
                   F, D, F * D, D, 1, 0, prec_fwd, ntiles)         # GATHER=0

        y = torch.empty(T, D, dtype=cdt, device=dev)
        _combine_ec_kernel[(T, triton.cdiv(D, BD))](
            yp, gate, order, tstart, tend, y, D, BLOCK=BD)

        ctx.save_for_backward(x, router_w, w1, w2, S, gate, tok,
                              seg_start, seg_end, texp, trow,
                              order, tstart, tend, out1)
        ctx.meta = (C, ntiles, ieee)
        return y

    @staticmethod
    def backward(ctx, dy):
        (x, router_w, w1, w2, S, gate, tok,
         seg_start, seg_end, texp, trow,
         order, tstart, tend, out1) = ctx.saved_tensors
        C, ntiles, ieee = ctx.meta
        T, D = x.shape
        E = router_w.shape[1]
        F2 = w1.shape[2]
        F = F2 // 2
        EC = E * C
        dev = x.device
        cdt = x.dtype
        dy = dy.contiguous()

        # combine backward: per-pair upstream grads (float32) and the
        # fp32-recomputed activations
        dyp = torch.empty(EC, D, dtype=torch.float32, device=dev)
        _dyp_kernel[(EC, triton.cdiv(D, BD))](dy, gate, tok, dyp, D, BLOCK=BD)
        act32 = torch.empty(EC, F, dtype=torch.float32, device=dev)
        total = EC * F
        _swiglu_fwd_kernel[(triton.cdiv(total, BD),)](
            out1, act32, total, F, F2, BLOCK=BD)

        p_a = 1 if ieee else 2      # fp32-computed A x leaf B
        p_ab = 1 if ieee else 3     # fp32-computed A x fp32-computed B
        p_b = 1 if ieee else 4      # leaf A x fp32-computed B

        # affinity-gate grads against the fp32-precision expert outputs
        yp32 = torch.empty(EC, D, dtype=torch.float32, device=dev)
        _mm_launch(act32, w2, yp32, tok, texp, trow, seg_end,
                   F, D, F * D, D, 1, 0, p_a, ntiles)             # GATHER=0
        dgate = torch.empty(EC, dtype=torch.float32, device=dev)
        _dgate_kernel[(EC,)](dy, yp32, tok, dgate, D, BLOCK=BD)
        del yp32

        # down projection backward (weight-grad chain keeps fp32 effect)
        dact = torch.empty(EC, F, dtype=torch.float32, device=dev)
        _mm_launch(dyp, w2, dact, tok, texp, trow, seg_end,
                   D, F, F * D, 1, D, 0, p_a, ntiles)             # GATHER=0, w2^T
        dw2 = torch.empty(E, F, D, dtype=w2.dtype, device=dev)
        _dw_launch(act32, dyp, dw2, tok, seg_start, seg_end,
                   E, F, D, F, 0, p_ab)                           # GATHER=0
        del act32, dyp

        # activation backward
        dgu = torch.empty(EC, F2, dtype=torch.float32, device=dev)
        _swiglu_bwd_kernel[(triton.cdiv(total, BD),)](
            out1, dact, dgu, total, F, F2, BLOCK=BD)
        del dact

        # gate/up projection backward
        dw1 = torch.empty(E, D, F2, dtype=w1.dtype, device=dev)
        _dw_launch(x, dgu, dw1, tok, seg_start, seg_end,
                   E, D, F2, D, 1, p_b)                           # GATHER=1 (x by tok)
        dgu_c = dgu.to(cdt)
        del dgu
        dxs = torch.empty(EC, D, dtype=torch.float32, device=dev)
        _mm_launch(dgu_c, w1, dxs, tok, texp, trow, seg_end,
                   F2, D, D * F2, 1, F2, 0, 1 if ieee else 0, ntiles)  # GATHER=0, w1^T
        del dgu_c

        # expert half of dx: per-token CSR reduction of the pair rows
        dxe = torch.empty(T, D, dtype=torch.float32, device=dev)
        _dx_combine_ec_kernel[(T, triton.cdiv(D, BD))](
            dxs, order, tstart, tend, dxe, D, BLOCK=BD)

        # router backward: scatter the per-pair gate grads into a dense [T, E]
        # buffer (each (token, expert) pair is unique), then the DENSE softmax
        # Jacobian spreads it across the expert row.
        eidx = torch.div(torch.arange(EC, device=dev), C,
                         rounding_mode="floor")
        gS = torch.zeros(T, E, dtype=torch.float32, device=dev)
        gS.index_put_((tok.to(torch.int64), eidx), dgate, accumulate=True)
        dlogits = S * (gS - (gS * S).sum(dim=-1, keepdim=True))
        xf = x.float()
        rwf = router_w.float()
        drw = (xf.t() @ dlogits).to(router_w.dtype)
        dx = (dxe + dlogits @ rwf.t()).to(x.dtype)
        return dx, drw, dw1, dw2, None


def fused_moe(
    x: torch.Tensor,         # [T, D] float32 or bfloat16
    router_w: torch.Tensor,  # [D, E] same dtype as x
    w1: torch.Tensor,        # [E, D, 2F]  [:, :, :F] gate, [:, :, F:] up
    w2: torch.Tensor,        # [E, F, D]
    C: int,                  # capacity: tokens per expert
) -> torch.Tensor:           # [T, D] same dtype as x
    return _FusedMoeOracle.apply(x, router_w, w1, w2, C)
# FORGE-CANARY-BEGIN forge-canary/v1
# FORGE-CANARY-SLOT-0 4e7e762d8b3dad1dcfb246d2edc3390b8a553be5b596e14b1c00221d229e5cae
# FORGE-CANARY-SLOT-1 f08bb6ec1361f90cd08417c20f95f8ec81e31dc6bcf31a3558867cdecfbd918b
# FORGE-CANARY-SLOT-2 8c5423ffa0358f6242ad22232d8bca7a153d783182ad9b217f7d2e58727be8fc
# FORGE-CANARY-SLOT-3 7fa891e696b67473713e10a3920721b855ce5dff2494d7bc9dff7fbe0536f295
# FORGE-CANARY-END
