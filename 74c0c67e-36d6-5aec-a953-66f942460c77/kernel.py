"""
Starter -- fp8-e4m3 weight-only quantized SwiGLU MoE, forward and backward.

Deliberately basic: dequantize weights per expert (multiply by the per-out-
channel scale), then run a python loop over every (slot, expert) pair with
naive Triton matmul and silu*up kernels. Correct, deterministic, and slow.

The entry point the harness calls is `fp8_e4m3_swiglu_moe`; its signature and
semantics are fixed. Autograd must work through it: the harness grades TWO
gradients, `dx` and `dtopk_w`. The quantized weight buffers `w1_q`, `w1_s`,
`w2_q`, `w2_s` are inference-shaped inputs and carry no gradient. `topk_idx`
is integer routing and has no gradient. The hardness lever is fusing the
per-out-channel dequant into the MMA prologue (or using fp8-e4m3 tensor cores
directly on the pre-snapped weights); this starter dequants eagerly.
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
    """float32 matmul; backward returns grads for both operands. dB is
    computed here even though the quantized weight buffers upstream carry no
    grad -- autograd discards it via `requires_grad_(False)` on the leaves."""

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


def _expert_ffn(h, w1dq, w2dq):
    F = w1dq.shape[1] // 2
    gu = _MM.apply(h, w1dq)
    act = _SiLUMul.apply(gu[:, :F].contiguous(), gu[:, F:].contiguous())
    return _MM.apply(act, w2dq)


class _Fp8E4m3SwigluMoe(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1_q, w1_s, w2_q, w2_s, topk_idx, topk_w):
        with torch.enable_grad():
            xf = x.detach().float().requires_grad_(True)
            wf = topk_w.detach().float().requires_grad_(True)
            T, D = xf.shape
            E = w1_q.shape[0]
            A = topk_idx.shape[1]
            # Dequantize once per expert; the dequantized weights are non-leaf
            # constants (no requires_grad) so no weight gradient is produced.
            w1_dq = [
                (w1_q[e].detach().float() * w1_s[e].detach().float()[None, :])
                for e in range(E)
            ]
            w2_dq = [
                (w2_q[e].detach().float() * w2_s[e].detach().float()[None, :])
                for e in range(E)
            ]
            y = xf.new_zeros(T, D)
            for a in range(A):
                idx = topk_idx[:, a]
                contrib = xf.new_zeros(T, D)
                for e in range(E):
                    rows = (idx == e).nonzero(as_tuple=True)[0]
                    if rows.numel() == 0:
                        continue
                    fe = _expert_ffn(xf.index_select(0, rows), w1_dq[e], w2_dq[e])
                    contrib = contrib.index_copy(0, rows, fe)
                y = y + wf[:, a : a + 1] * contrib
        ctx.save_for_backward(xf, wf, y)
        ctx.dtypes = (x.dtype, topk_w.dtype)
        return y.detach().to(x.dtype)

    @staticmethod
    def backward(ctx, dy):
        xf, wf, y = ctx.saved_tensors
        gx, gw = torch.autograd.grad(y, [xf, wf], dy.float())
        dx_t, dw_t = ctx.dtypes
        # order: x, w1_q, w1_s, w2_q, w2_s, topk_idx, topk_w
        return gx.to(dx_t), None, None, None, None, None, gw.to(dw_t)


def fp8_e4m3_swiglu_moe(
    x: torch.Tensor,  # [T, D]
    w1_q: torch.Tensor,  # [E, D, 2F] values on the fp8-e4m3 grid
    w1_s: torch.Tensor,  # [E, 2F]    per-out-channel scale
    w2_q: torch.Tensor,  # [E, F, D]
    w2_s: torch.Tensor,  # [E, D]
    topk_idx: torch.Tensor,  # [T, A] int64, no gradient
    topk_w: torch.Tensor,  # [T, A] same dtype as x
) -> torch.Tensor:  # [T, D] same dtype as x
    """Entry point. Signature and semantics are fixed; the body is yours."""
    return _Fp8E4m3SwigluMoe.apply(x, w1_q, w1_s, w2_q, w2_s, topk_idx, topk_w)
