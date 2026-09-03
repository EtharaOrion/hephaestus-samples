"""
Starter -- shared expert + routed SwiGLU experts, forward and backward.

This file is yours. It is deliberately basic. Two structurally different GEMM
regimes coexist in this one operator and both run the slow way here:

  * a DENSE always-on SHARED expert applied to EVERY token (its own weights
    ws1/ws2, weight 1, no routing), and
  * an IRREGULAR routed path -- a python loop over every (slot, expert) pair,
    one boolean-mask gather per pair -- feeding the routed experts w1/w2.

All the matmuls run in a naive, unblocked Triton kernel launched hundreds of
times per forward; the silu*up gate runs in a small Triton elementwise kernel.
Correct, deterministic, and slow by design. Your job is to make it fast.

The entry point the harness calls is `shared_plus_routed_swiglu`; its signature
and semantics must not change. Autograd must work through it: the harness
measures forward and backward together and grades SIX gradients -- dx, dws1,
dws2 (the shared expert), and dw1, dw2, dtopk_w (the routed path). topk_idx is
integer routing and has no gradient.

Structure worth keeping: the whole operator is one custom autograd.Function
whose forward builds a differentiable graph over detached float32 copies of the
inputs and whose backward replays it with torch.autograd.grad -- so dx picks up
BOTH paths automatically and the six-gradient bookkeeping is written once. The
combine order is fixed (ascending slot, shared added last), every routed scatter
hits disjoint rows, and there are no atomic adds anywhere: three runs on one
input produce bitwise-identical outputs and gradients, which the harness checks.
"""

import torch
import triton
import triton.language as tl

BM, BN, BK = 64, 64, 32     # naive matmul tile; one program per output tile
EBLOCK = 1024               # elementwise block


@triton.jit
def _mm_kernel(A, B, C, M, N, K,
               sam, sak, sbk, sbn, scm, scn,
               BLK_M: tl.constexpr, BLK_N: tl.constexpr, BLK_K: tl.constexpr):
    """Plain float32 tiled matmul C[M,N] = A[M,K] @ B[K,N]. Strides explicit, so
    transposed views time exactly as badly. input_precision="ieee" keeps full
    fp32. This one kernel carries the entire arithmetic load, fwd and bwd, for
    both the shared and routed paths -- launched once per (slot, expert) pair
    per projection plus once per shared-path projection."""
    pid_m = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1).to(tl.int64)
    rm = pid_m * BLK_M + tl.arange(0, BLK_M).to(tl.int64)
    rn = pid_n * BLK_N + tl.arange(0, BLK_N).to(tl.int64)
    acc = tl.zeros((BLK_M, BLK_N), dtype=tl.float32)
    for k0 in range(0, K, BLK_K):
        rk = k0 + tl.arange(0, BLK_K).to(tl.int64)
        a = tl.load(A + rm[:, None] * sam + rk[None, :] * sak,
                    mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
        b = tl.load(B + rk[:, None] * sbk + rn[None, :] * sbn,
                    mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b, input_precision="ieee")
    tl.store(C + rm[:, None] * scm + rn[None, :] * scn, acc,
             mask=(rm[:, None] < M) & (rn[None, :] < N))


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


def _mm_launch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    _, N = b.shape
    c = torch.empty(M, N, device=a.device, dtype=torch.float32)
    if M == 0 or N == 0:
        return c
    _mm_kernel[(triton.cdiv(M, BM), triton.cdiv(N, BN))](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        c.stride(0), c.stride(1), BLK_M=BM, BLK_N=BN, BLK_K=BK)
    return c


class _MM(torch.autograd.Function):
    """float32 matmul with the same naive kernel on every path of the chain
    rule: dA = dC @ B^T and dB = A^T @ dC, transposes expressed as stride
    swaps so nothing is ever copied (or fast)."""

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
    """act = silu(g) * u elementwise, float32, with its exact local backward."""

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
        _silu_mul_bwd_kernel[(triton.cdiv(n, EBLOCK),)](g, u, dout, dg, du, n,
                                                        BLOCK=EBLOCK)
        return dg, du


def _swiglu_ffn(h: torch.Tensor, w1e: torch.Tensor, w2e: torch.Tensor):
    """(silu(h@w1[:, :F]) * (h@w1[:, F:])) @ w2, all float32, all in the kernels
    above. Shared by the routed experts and the dense shared expert."""
    F = w1e.shape[1] // 2
    gu = _MM.apply(h, w1e)                                         # [n, 2F]
    act = _SiLUMul.apply(gu[:, :F].contiguous(), gu[:, F:].contiguous())
    return _MM.apply(act, w2e)                                     # [n, D]


class _SharedRoutedSwiglu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, ws1, ws2, w1, w2, topk_idx, topk_w):
        with torch.enable_grad():
            xf = x.detach().float().requires_grad_(True)
            wf = topk_w.detach().float().requires_grad_(True)
            ws1f = ws1.detach().float().requires_grad_(True)
            ws2f = ws2.detach().float().requires_grad_(True)
            T, D = xf.shape
            E = w1.shape[0]
            A = topk_idx.shape[1]
            # One float32 leaf PER routed expert so a branch's weight gradient
            # is a narrow [D, 2F] tensor, not a full-size scatter.
            w1s = [w1[e].detach().float().requires_grad_(True) for e in range(E)]
            w2s = [w2[e].detach().float().requires_grad_(True) for e in range(E)]
            # Routed path: every (slot, expert) pair is its own gather, matmul
            # chain and scatter. Combine order ascending slot; each index_copy
            # hits disjoint rows.
            y = xf.new_zeros(T, D)
            for a in range(A):
                idx = topk_idx[:, a]
                contrib = xf.new_zeros(T, D)
                for e in range(E):
                    rows = (idx == e).nonzero(as_tuple=True)[0]
                    if rows.numel() == 0:
                        continue
                    fe = _swiglu_ffn(xf.index_select(0, rows), w1s[e], w2s[e])
                    contrib = contrib.index_copy(0, rows, fe)
                y = y + wf[:, a:a + 1] * contrib
            # Shared path: dense SwiGLU over ALL tokens, always on (weight 1),
            # added last.
            shared = _swiglu_ffn(xf, ws1f, ws2f)
            y = y + shared
        ctx.save_for_backward(xf, wf, ws1f, ws2f, y, *w1s, *w2s)
        ctx.n_experts = E
        ctx.dtypes = (x.dtype, ws1.dtype, ws2.dtype, w1.dtype, w2.dtype,
                      topk_w.dtype)
        return y.detach().to(x.dtype)

    @staticmethod
    def backward(ctx, dy):
        E = ctx.n_experts
        xf, wf, ws1f, ws2f, y = ctx.saved_tensors[:5]
        w1s = list(ctx.saved_tensors[5:5 + E])
        w2s = list(ctx.saved_tensors[5 + E:5 + 2 * E])
        grads = torch.autograd.grad(
            y, [xf, wf, ws1f, ws2f] + w1s + w2s, dy.float(), allow_unused=True)
        gx, gw, gws1, gws2 = grads[0], grads[1], grads[2], grads[3]
        # Routed experts no token routed to are unused in the graph: exactly 0.
        gw1 = torch.stack([g if g is not None else torch.zeros_like(w1s[e])
                           for e, g in enumerate(grads[4:4 + E])])
        gw2 = torch.stack([g if g is not None else torch.zeros_like(w2s[e])
                           for e, g in enumerate(grads[4 + E:4 + 2 * E])])
        dx_t, dws1_t, dws2_t, dw1_t, dw2_t, dw_t = ctx.dtypes
        return (gx.to(dx_t), gws1.to(dws1_t), gws2.to(dws2_t),
                gw1.to(dw1_t), gw2.to(dw2_t), None, gw.to(dw_t))


def shared_plus_routed_swiglu(
    x: torch.Tensor,         # [T, D] float32 or bfloat16
    ws1: torch.Tensor,       # [D, 2F]  shared expert; [:, :F] gate, [:, F:] up
    ws2: torch.Tensor,       # [F, D]   shared expert output projection
    w1: torch.Tensor,        # [E, D, 2F]  routed; [:, :, :F] gate, [:, :, F:] up
    w2: torch.Tensor,        # [E, F, D]   routed output projections
    topk_idx: torch.Tensor,  # [T, A] int64, values in [0, E), no gradient
    topk_w: torch.Tensor,    # [T, A] same dtype as x, normalized
) -> torch.Tensor:           # [T, D] same dtype as x
    """Entry point. Signature and semantics are fixed; the body is yours."""
    return _SharedRoutedSwiglu.apply(x, ws1, ws2, w1, w2, topk_idx, topk_w)
