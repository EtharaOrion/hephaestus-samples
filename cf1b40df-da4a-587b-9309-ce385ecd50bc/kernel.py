"""
Starter -- capacity-limited Switch-style routed fused MoE, fwd + bwd.

This file is yours. It is deliberately basic: the routing scores, per-expert
capacity clipping and every expert projection run through three small Triton
kernels that do the arithmetic the slowest reasonable way -- an unblocked,
untuned float32 matmul kernel launched once per expert per projection, plus
an elementwise silu*up kernel -- wrapped in a python loop over experts with
one boolean-mask gather each. Correct, deterministic, and slow by design.
Your job is to make it fast.

The entry point the harness calls is `fused_moe`, and its signature and
semantics must not change: softmax router `p = softmax(x @ router_w)`, top-1
selection with lower-expert-index tie-break, PER-EXPERT capacity clipping
(keep the top-`capacity` candidates by p, ties to LOWER token index; excess
tokens dropped with zero output), raw softmax gate g[t] = p[t, e*[t]] for
kept tokens, SwiGLU experts, float32 arithmetic with one final cast. Dropped
tokens have y[t] == 0 exactly. Autograd must work end to end: the harness
measures forward and backward and grades dx, drouter_w, dw1, dw2.
reference.py defines the semantics; agreement with it is graded under the
disclosed tolerances.
"""

import torch
import triton
import triton.language as tl

BM, BN, BK = 16, 16, 16
EBLOCK = 1024


@triton.jit
def _mm_kernel(
    A,
    B,
    C,
    M,
    N,
    K,
    sam,
    sak,
    sbk,
    sbn,
    scm,
    scn,
    BLK_M: tl.constexpr,
    BLK_N: tl.constexpr,
    BLK_K: tl.constexpr,
):
    """Plain float32 tiled matmul; one program per output tile. Strides are
    explicit so transposed views time exactly as badly as everything else."""
    pid_m = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1).to(tl.int64)
    rm = pid_m * BLK_M + tl.arange(0, BLK_M).to(tl.int64)
    rn = pid_n * BLK_N + tl.arange(0, BLK_N).to(tl.int64)
    acc = tl.zeros((BLK_M, BLK_N), dtype=tl.float32)
    for k0 in range(0, K, BLK_K):
        rk = k0 + tl.arange(0, BLK_K).to(tl.int64)
        a = tl.load(
            A + rm[:, None] * sam + rk[None, :] * sak,
            mask=(rm[:, None] < M) & (rk[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            B + rk[:, None] * sbk + rn[None, :] * sbn,
            mask=(rk[:, None] < K) & (rn[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b, input_precision="ieee")
    tl.store(
        C + rm[:, None] * scm + rn[None, :] * scn,
        acc,
        mask=(rm[:, None] < M) & (rn[None, :] < N),
    )


@triton.jit
def _silu_mul_fwd_kernel(G, U, OUT, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    o = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    m = o < n
    g = tl.load(G + o, mask=m, other=0.0)
    u = tl.load(U + o, mask=m, other=0.0)
    s = tl.sigmoid(g)
    tl.store(OUT + o, g * s * u, mask=m)


@triton.jit
def _silu_mul_bwd_kernel(G, U, DO, DG, DU, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    o = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    m = o < n
    g = tl.load(G + o, mask=m, other=0.0)
    u = tl.load(U + o, mask=m, other=0.0)
    do = tl.load(DO + o, mask=m, other=0.0)
    s = tl.sigmoid(g)
    silu = g * s
    tl.store(DG + o, do * u * (s + silu * (1.0 - s)), mask=m)
    tl.store(DU + o, do * silu, mask=m)


def _mm_launch(a, b):
    M, K = a.shape
    _, N = b.shape
    c = torch.empty(M, N, device=a.device, dtype=torch.float32)
    if M == 0 or N == 0:
        return c
    _mm_kernel[(triton.cdiv(M, BM), triton.cdiv(N, BN))](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLK_M=BM,
        BLK_N=BN,
        BLK_K=BK,
    )
    return c


class _MM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return _mm_launch(a, b)

    @staticmethod
    def backward(ctx, dc):
        a, b = ctx.saved_tensors
        dc = dc.contiguous()
        return _mm_launch(dc, b.t()), _mm_launch(a.t(), dc)


class _SiLUMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, g, u):
        ctx.save_for_backward(g, u)
        out = torch.empty_like(g)
        n = g.numel()
        _silu_mul_fwd_kernel[(triton.cdiv(n, EBLOCK),)](g, u, out, n, BLOCK=EBLOCK)
        return out

    @staticmethod
    def backward(ctx, dout):
        g, u = ctx.saved_tensors
        dout = dout.contiguous()
        dg, du = torch.empty_like(g), torch.empty_like(u)
        n = g.numel()
        _silu_mul_bwd_kernel[(triton.cdiv(n, EBLOCK),)](
            g, u, dout, dg, du, n, BLOCK=EBLOCK
        )
        return dg, du


def _select_top1(p):
    """Argmax by softmax weight; ties to LOWER expert index. Returns [T,1]."""
    E = p.shape[-1]
    b = torch.where(p == 0.0, torch.zeros_like(p), p).contiguous()
    ib = b.view(torch.int32).to(torch.int64)
    u = torch.where(ib >= 0, ib ^ 0x80000000, (~ib) & 0xFFFFFFFF)
    e_idx = torch.arange(E, device=p.device, dtype=torch.int64)
    keys = (u << 10) | (E - 1 - e_idx)
    return torch.topk(keys, 1, dim=-1, largest=True, sorted=True).indices


def _keep_topC(p_col, mask, capacity):
    """Bool [T] mask: keep top-`capacity` entries of p_col within `mask`.
    Ties broken to LOWER token index by the same int64-key trick."""
    T = p_col.shape[0]
    b = torch.where(p_col == 0.0, torch.zeros_like(p_col), p_col).contiguous()
    ib = b.view(torch.int32).to(torch.int64)
    u = torch.where(ib >= 0, ib ^ 0x80000000, (~ib) & 0xFFFFFFFF)
    t_idx = torch.arange(T, device=p_col.device, dtype=torch.int64)
    tbits = max((T - 1).bit_length(), 1)
    keys = (u << tbits) | (T - 1 - t_idx)
    keys = torch.where(mask, keys, torch.full_like(keys, -(1 << 62)))
    n_cand = int(mask.sum())
    k = min(int(capacity), n_cand)
    out = torch.zeros(T, dtype=torch.bool, device=p_col.device)
    if k == 0:
        return out
    picked = torch.topk(keys, k, largest=True, sorted=True).indices
    out.scatter_(0, picked, True)
    return out


class _FusedMoe(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, router_w, w1, w2, capacity):
        capacity = int(capacity)
        with torch.enable_grad():
            xf = x.detach().float().requires_grad_(True)
            rwf = router_w.detach().float().requires_grad_(True)
            w1f = w1.detach().float().requires_grad_(True)
            w2f = w2.detach().float().requires_grad_(True)
            T, D = xf.shape
            E = rwf.shape[1]
            F = w1f.shape[2] // 2
            logits = _MM.apply(xf, rwf)
            p = torch.softmax(logits, dim=-1)
            w1s = w1f.unbind(0)
            w2s = w2f.unbind(0)
            sel = _select_top1(p.detach()).squeeze(-1)  # [T]
            member = torch.zeros(T, E, dtype=torch.bool, device=x.device).scatter_(
                1, sel.unsqueeze(-1), True
            )  # [T, E]
            keep = torch.zeros(T, dtype=torch.bool, device=x.device)
            for e in range(E):
                cand = member[:, e]
                if not bool(cand.any()):
                    continue
                keep |= _keep_topC(p[:, e].detach(), cand, capacity)
            g_full = p.new_zeros(T, E).scatter(
                1, sel.unsqueeze(-1), torch.gather(p, 1, sel.unsqueeze(-1))
            )
            y = xf.new_zeros(T, D)
            # slow expert-by-expert loop; each token contributes to at most one
            # expert, so index_add is deterministic without atomics.
            for e in range(E):
                rows = (member[:, e] & keep).nonzero(as_tuple=True)[0]
                if rows.numel() == 0:
                    continue
                h = xf.index_select(0, rows)
                gu = _MM.apply(h, w1s[e])
                act = _SiLUMul.apply(gu[:, :F].contiguous(), gu[:, F:].contiguous())
                fe = _MM.apply(act, w2s[e])
                y = y.index_add(
                    0, rows, g_full.index_select(0, rows)[:, e : e + 1] * fe
                )
        ctx.save_for_backward(xf, rwf, w1f, w2f, y)
        ctx.dtypes = (x.dtype, router_w.dtype, w1.dtype, w2.dtype)
        return y.detach().to(x.dtype)

    @staticmethod
    def backward(ctx, dy):
        xf, rwf, w1f, w2f, y = ctx.saved_tensors
        gx, grw, gw1, gw2 = torch.autograd.grad(y, (xf, rwf, w1f, w2f), dy.float())
        dx_t, drw_t, dw1_t, dw2_t = ctx.dtypes
        return (gx.to(dx_t), grw.to(drw_t), gw1.to(dw1_t), gw2.to(dw2_t), None)


def fused_moe(
    x: torch.Tensor,  # [T, D] float32 or bfloat16
    router_w: torch.Tensor,  # [D, E] same dtype as x
    w1: torch.Tensor,  # [E, D, 2F]  gate|up
    w2: torch.Tensor,  # [E, F, D]
    capacity: int,  # per-expert token cap
) -> torch.Tensor:  # [T, D] same dtype as x
    """Entry point. Signature and semantics are fixed; the body is yours."""
    return _FusedMoe.apply(x, router_w, w1, w2, capacity)
