"""
Starter -- expert-choice routed fused mixture-of-experts layer, fwd + bwd.

This file is yours. It is deliberately basic: the routing scores, every expert
projection and the activation run through three small Triton kernels that do the
arithmetic the slowest reasonable way -- an unblocked, untuned float32 matmul
kernel launched once per expert per projection, plus an elementwise silu*up
kernel -- wrapped in a python loop over experts with one gather each. Correct,
deterministic, and slow by design. Your job is to make it fast.

The entry point the harness calls is `fused_moe`, and its signature and
semantics must not change: affinities `S = softmax(x @ router_w, dim=-1)` (the
softmax is over EXPERTS), each expert selects its top-C tokens by `S[:, e]`
(affinity descending, ties to the LOWER TOKEN index), each selected (token,
expert) contribution is scaled by the RAW affinity `S[token, e]` (no
renormalization) and summed into the token in ascending expert order, SwiGLU
experts, float32 arithmetic with one final cast. There is no router bias.
Autograd must work through it: the harness measures forward and backward
together and grades the gradients of x, router_w, w1 and w2 as well as the
output. reference.py defines the semantics; agreement with it is graded under
the disclosed tolerances.
"""

import torch
import triton
import triton.language as tl

BM, BN, BK = 16, 16, 16     # naive matmul tile; one program per output tile
EBLOCK = 1024               # elementwise block


@triton.jit
def _mm_kernel(A, B, C, M, N, K,
               sam, sak, sbk, sbn, scm, scn,
               BLK_M: tl.constexpr, BLK_N: tl.constexpr, BLK_K: tl.constexpr):
    """Plain float32 tiled matmul C[M,N] = A[M,K] @ B[K,N].

    Strides are passed explicitly, so transposed views time exactly as badly
    as everything else. input_precision="ieee" keeps it full fp32. This one
    kernel carries the entire arithmetic load of the starter and is launched
    once per expert per projection. That launch storm is the deliberate
    naivety.
    """
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
    rule: dA = dC @ B^T and dB = A^T @ dC, transposes as stride swaps."""

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


def _select_topC(S: torch.Tensor, C: int) -> torch.Tensor:
    """Per expert (column of S), top-C tokens by affinity, ties to the LOWER
    TOKEN index. Returns [E, C] token indices.

    Every (affinity, token index) pair becomes one unique int64 key: the
    float32 affinity's order-preserving bits (sign bit flipped for
    non-negatives, all bits flipped for negatives, -0.0 canonicalised to +0.0)
    shifted left by enough bits to hold the token index, OR'd with the inverted
    token index. Descending key order IS (affinity desc, token index asc), so
    `torch.topk` on the keys -- the permitted routing-step call -- returns
    exactly the reference selection with no tie left to chance.
    """
    T, E = S.shape
    St = S.t().contiguous()                                # [E, T]
    b = torch.where(St == 0.0, torch.zeros_like(St), St).contiguous()
    ib = b.view(torch.int32).to(torch.int64)
    u = torch.where(ib >= 0, ib ^ 0x80000000, (~ib) & 0xFFFFFFFF)
    t_idx = torch.arange(T, device=S.device, dtype=torch.int64)
    tbits = max((T - 1).bit_length(), 1)
    keys = (u << tbits) | (T - 1 - t_idx).unsqueeze(0)     # [E, T]
    return torch.topk(keys, C, dim=-1, largest=True, sorted=True).indices


class _FusedMoe(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, router_w, w1, w2, C):
        C = int(C)
        with torch.enable_grad():
            xf = x.detach().float().requires_grad_(True)
            rwf = router_w.detach().float().requires_grad_(True)
            w1f = w1.detach().float().requires_grad_(True)
            w2f = w2.detach().float().requires_grad_(True)
            T, D = xf.shape
            E = rwf.shape[1]
            F = w1f.shape[2] // 2
            logits = _MM.apply(xf, rwf)                      # [T, E]
            S = torch.softmax(logits, dim=-1)               # [T, E] affinities
            w1s = w1f.unbind(0)
            w2s = w2f.unbind(0)
            with torch.no_grad():
                sel = _select_topC(S.detach(), C)           # [E, C]
            y = xf.new_zeros(T, D)
            # The slow part: every expert is its own gather, three-kernel matmul
            # chain and ordered row-wise accumulate. Within an expert the C
            # selected tokens are distinct, so index_add hits each row once;
            # across experts the ascending-e order fixes the accumulation, so
            # everything is deterministic.
            for e in range(E):
                rows = sel[e]                               # [C] token indices
                g = S.index_select(0, rows)[:, e:e + 1]     # [C, 1] affinity gate
                h = xf.index_select(0, rows)                # [C, D]
                gu = _MM.apply(h, w1s[e])
                act = _SiLUMul.apply(gu[:, :F].contiguous(),
                                     gu[:, F:].contiguous())
                fe = _MM.apply(act, w2s[e])
                y = y.index_add(0, rows, g * fe)
        ctx.save_for_backward(xf, rwf, w1f, w2f, y)
        ctx.dtypes = (x.dtype, router_w.dtype, w1.dtype, w2.dtype)
        return y.detach().to(x.dtype)

    @staticmethod
    def backward(ctx, dy):
        xf, rwf, w1f, w2f, y = ctx.saved_tensors
        gx, grw, gw1, gw2 = torch.autograd.grad(y, (xf, rwf, w1f, w2f),
                                                dy.float())
        dx_t, drw_t, dw1_t, dw2_t = ctx.dtypes
        return (gx.to(dx_t), grw.to(drw_t), gw1.to(dw1_t), gw2.to(dw2_t), None)


def fused_moe(
    x: torch.Tensor,         # [T, D] float32 or bfloat16
    router_w: torch.Tensor,  # [D, E] same dtype as x
    w1: torch.Tensor,        # [E, D, 2F]  [:, :, :F] gate, [:, :, F:] up
    w2: torch.Tensor,        # [E, F, D]
    C: int,                  # capacity: tokens per expert
) -> torch.Tensor:           # [T, D] same dtype as x
    """Entry point. Signature and semantics are fixed; the body is yours."""
    return _FusedMoe.apply(x, router_w, w1, w2, C)
