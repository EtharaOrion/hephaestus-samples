"""
Starter -- fused MoE + SwiGLU expert compute for wide-FFN, high-E shapes.

This file is yours. It is deliberately basic: a python loop over every
(slot, expert) pair, one boolean-mask gather per pair, a naive Triton matmul
kernel doing the arithmetic the slowest reasonable way, and a Triton silu*up
elementwise kernel. Correct, deterministic, and slow by design. The graded
shapes have E up to 256 (with strongly imbalanced token counts per expert)
and F up to ~3x D (wide FFN); a slow python-loop starter therefore launches
thousands of small kernels per forward, which is exactly the naivety this
task rewards you for erasing.

The entry point the harness calls is `wide_expert_swiglu_moe`; its signature
and semantics must not change. Autograd must work through it: the harness
grades four gradients -- dx, dw1, dw2 and dtopk_w. topk_idx has no gradient.
"""

import torch
import triton
import triton.language as tl

BM, BN, BK = 64, 64, 32
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
    """Plain float32 tiled matmul C[M,N] = A[M,K] @ B[K,N]."""
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


def _expert_ffn(h, w1e, w2e):
    F = w1e.shape[1] // 2
    gu = _MM.apply(h, w1e)
    act = _SiLUMul.apply(gu[:, :F].contiguous(), gu[:, F:].contiguous())
    return _MM.apply(act, w2e)


class _WideExpertSwigluMoe(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, w2, topk_idx, topk_w):
        with torch.enable_grad():
            xf = x.detach().float().requires_grad_(True)
            wf = topk_w.detach().float().requires_grad_(True)
            T, D = xf.shape
            E = w1.shape[0]
            A = topk_idx.shape[1]
            w1s = [w1[e].detach().float().requires_grad_(True) for e in range(E)]
            w2s = [w2[e].detach().float().requires_grad_(True) for e in range(E)]
            y = xf.new_zeros(T, D)
            for a in range(A):
                idx = topk_idx[:, a]
                contrib = xf.new_zeros(T, D)
                for e in range(E):
                    rows = (idx == e).nonzero(as_tuple=True)[0]
                    if rows.numel() == 0:
                        continue
                    fe = _expert_ffn(xf.index_select(0, rows), w1s[e], w2s[e])
                    contrib = contrib.index_copy(0, rows, fe)
                y = y + wf[:, a : a + 1] * contrib
        ctx.save_for_backward(xf, wf, y, *w1s, *w2s)
        ctx.n_experts = E
        ctx.dtypes = (x.dtype, w1.dtype, w2.dtype, topk_w.dtype)
        return y.detach().to(x.dtype)

    @staticmethod
    def backward(ctx, dy):
        E = ctx.n_experts
        xf, wf, y = ctx.saved_tensors[:3]
        w1s = list(ctx.saved_tensors[3 : 3 + E])
        w2s = list(ctx.saved_tensors[3 + E : 3 + 2 * E])
        grads = torch.autograd.grad(
            y, [xf, wf] + w1s + w2s, dy.float(), allow_unused=True
        )
        gx, gw = grads[0], grads[1]
        gw1 = torch.stack(
            [
                g if g is not None else torch.zeros_like(w1s[e])
                for e, g in enumerate(grads[2 : 2 + E])
            ]
        )
        gw2 = torch.stack(
            [
                g if g is not None else torch.zeros_like(w2s[e])
                for e, g in enumerate(grads[2 + E : 2 + 2 * E])
            ]
        )
        dx_t, dw1_t, dw2_t, dw_t = ctx.dtypes
        return gx.to(dx_t), gw1.to(dw1_t), gw2.to(dw2_t), None, gw.to(dw_t)


def wide_expert_swiglu_moe(
    x: torch.Tensor,  # [T, D]
    w1: torch.Tensor,  # [E, D, 2F]  E up to 256, F up to ~3D
    w2: torch.Tensor,  # [E, F, D]
    topk_idx: torch.Tensor,  # [T, A] int64, no gradient
    topk_w: torch.Tensor,  # [T, A] same dtype as x
) -> torch.Tensor:  # [T, D] same dtype as x
    """Entry point. Signature and semantics are fixed; the body is yours."""
    return _WideExpertSwigluMoe.apply(x, w1, w2, topk_idx, topk_w)
