"""
Starter -- int4 weight-only quantized (W4A16) SwiGLU MoE, forward and backward.

Deliberately basic: unpack the int4 nibbles per expert, subtract per-out-channel
zero-point, multiply per-out-channel scale, and run a python loop over every
(slot, expert) pair with naive Triton matmul and silu*up kernels. Correct,
deterministic, and slow. The hardness lever is fusing the int4 UNPACK plus
ASYMMETRIC dequant (subtract zp, multiply scale) into the MMA prologue -- ideally
with 2 nibbles feeding one 8-bit column-pair of the tensor-core input, keeping
weights half the bytes bf16 would consume. This starter unpacks eagerly.

Entry: `int4_wq_swiglu_moe`. Autograd: harness grades `dx` and `dtopk_w`; the
packed weight buffers, scales and zero-points carry no gradient.
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


def _unpack_last(packed: torch.Tensor) -> torch.Tensor:
    """[..., K] uint8 -> [..., 2K] int8 with nibbles in [0, 15]."""
    lo = (packed & 0x0F).to(torch.int8)
    hi = ((packed >> 4) & 0x0F).to(torch.int8)
    return torch.stack([lo, hi], dim=-1).flatten(-2)


def _expert_ffn(h, w1dq, w2dq):
    F = w1dq.shape[1] // 2
    gu = _MM.apply(h, w1dq)
    act = _SiLUMul.apply(gu[:, :F].contiguous(), gu[:, F:].contiguous())
    return _MM.apply(act, w2dq)


class _Int4WqSwigluMoe(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1_pack, w1_s, w1_zp, w2_pack, w2_s, w2_zp, topk_idx, topk_w):
        with torch.enable_grad():
            xf = x.detach().float().requires_grad_(True)
            wf = topk_w.detach().float().requires_grad_(True)
            T, D = xf.shape
            E = w1_pack.shape[0]
            A = topk_idx.shape[1]
            # Eager per-expert unpack + asymmetric dequant.
            w1_int = _unpack_last(w1_pack).float()  # [E, D, 2F]
            w2_int = _unpack_last(w2_pack).float()  # [E, F, D]
            w1_dq_all = (
                w1_int - w1_zp.detach().float().unsqueeze(1)
            ) * w1_s.detach().float().unsqueeze(1)
            w2_dq_all = (
                w2_int - w2_zp.detach().float().unsqueeze(1)
            ) * w2_s.detach().float().unsqueeze(1)
            y = xf.new_zeros(T, D)
            for a in range(A):
                idx = topk_idx[:, a]
                contrib = xf.new_zeros(T, D)
                for e in range(E):
                    rows = (idx == e).nonzero(as_tuple=True)[0]
                    if rows.numel() == 0:
                        continue
                    fe = _expert_ffn(
                        xf.index_select(0, rows), w1_dq_all[e], w2_dq_all[e]
                    )
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
        # order: x, w1_pack, w1_s, w1_zp, w2_pack, w2_s, w2_zp, topk_idx, topk_w
        return gx.to(dx_t), None, None, None, None, None, None, None, gw.to(dw_t)


def int4_wq_swiglu_moe(
    x: torch.Tensor,  # [T, D]
    w1_pack: torch.Tensor,  # [E, D, F] uint8, nibble-packed along last dim
    w1_s: torch.Tensor,  # [E, 2F]   per-out-channel scale
    w1_zp: torch.Tensor,  # [E, 2F]   per-out-channel zero-point, int8 [0, 15]
    w2_pack: torch.Tensor,  # [E, F, D/2] uint8
    w2_s: torch.Tensor,  # [E, D]    per-out-channel scale
    w2_zp: torch.Tensor,  # [E, D]    per-out-channel zero-point
    topk_idx: torch.Tensor,  # [T, A] int64, no gradient
    topk_w: torch.Tensor,  # [T, A] same dtype as x
) -> torch.Tensor:  # [T, D] same dtype as x
    """Entry point. Signature and semantics are fixed; the body is yours."""
    return _Int4WqSwigluMoe.apply(
        x, w1_pack, w1_s, w1_zp, w2_pack, w2_s, w2_zp, topk_idx, topk_w
    )
