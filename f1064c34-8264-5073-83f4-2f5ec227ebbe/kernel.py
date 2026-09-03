"""
Starter -- chunk-parallel DPLR generalized delta rule (RWKV-7 family),
forward and backward.

This file is yours. It is deliberately basic: a correct formulation written in
plain PyTorch with one Triton kernel doing per-chunk work, and nothing else
optimized. Your job is to make it fast.

The entry point the harness calls is `chunked_dplr_delta`, and its signature
and semantics must not change. Autograd must work through it: the harness
measures forward and backward together, and grades the gradients as well as
the output.

What the harness measures, in one sentence: the geometric mean over a hidden
shape set of (reference production kernel time) / (your time), gated on
agreeing with reference.py on the output AND all six input gradients (dq, dk,
dv, dalpha, dbeta, dgk).

NOTE (authored-draft): this starter is correct-by-construction (the differentiable
path is the reference recurrence in float32; gradients come from autograd), but
it has NOT been GPU-profiled -- adoption/timing calibration is a separate
follow-up. The single @triton.jit kernel below is the seed jit entry point to
grow from; the chunked per-K-channel cumulative-decay formulation, the intra-
chunk (A_qk, A_qb, A_ab, A_ak) matmul cascade of the DPLR chunkwise algorithm,
the long-context state carry, and the fused backward are yours.
"""

import torch
import triton
import triton.language as tl

CHUNK = 64


@triton.jit
def _gate_cumsum_channels_kernel(g_ptr, out_ptr, T, BT: tl.constexpr):
    """Per-chunk inclusive cumulative sum of the per-K-channel log-decay along time.

    Real work; only Triton in the starter.
    """
    pid_bhk = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs = tl.arange(0, BT)
    idx = pid_c * BT + offs
    mask = idx < T
    g = tl.load(g_ptr + pid_bhk * T + idx, mask=mask, other=0.0)
    tl.store(out_ptr + pid_bhk * T + idx, tl.cumsum(g, axis=0), mask=mask)


def _chunk_cumsum_channels(g: torch.Tensor) -> torch.Tensor:
    """g: [B, T, H, K] -> per-chunk inclusive cumsum along T, same shape."""
    B, T, H, K = g.shape
    gt = g.permute(0, 2, 3, 1).contiguous().view(B * H * K, T)
    out = torch.empty_like(gt)
    grid = (B * H * K, triton.cdiv(T, CHUNK))
    _gate_cumsum_channels_kernel[grid](gt, out, T, BT=CHUNK)
    return out.view(B, H, K, T).permute(0, 3, 1, 2).contiguous()


def _forward_math(q, k, v, alpha, beta, gk):
    """Correct, and deliberately slow.

    Walks the DPLR recurrence one timestep at a time, exactly as reference.py
    defines it. Every step is a set of einsums and the whole [K, V] state
    round-trips through global memory T times. On the long-context graded shape
    (T = 8192) that alone is a bandwidth wall the chunked form is meant to
    demolish; the intra-chunk (A_qk, A_qb, A_ab, A_ak) cascade of the DPLR
    chunkwise algorithm, the long-context carry, and the fused backward are
    yours to build.
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    q32 = q.float() * (K**-0.5)
    k32, v32 = k.float(), v.float()
    a32, b32, gk32 = alpha.float(), beta.float(), gk.float()
    # Seed Triton kernel: per-K-channel per-chunk cumulative decay.
    _ = _chunk_cumsum_channels(gk32)

    state = torch.zeros(B, H, K, V, dtype=torch.float32, device=q.device)
    outs = []
    for t in range(T):
        alpha_t, beta_t = a32[:, t], b32[:, t]
        kt, vt = k32[:, t], v32[:, t]
        sT_alpha = torch.einsum("bhk,bhkv->bhv", alpha_t, state)
        rank1 = torch.einsum("bhk,bhv->bhkv", beta_t, sT_alpha)
        kv = torch.einsum("bhk,bhv->bhkv", kt, vt)
        state = state * gk32[:, t].exp()[:, :, :, None] + kv + rank1
        outs.append(torch.einsum("bhk,bhkv->bhv", q32[:, t], state))
    return torch.stack(outs, dim=1).to(v.dtype)


class _Dplr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, alpha, beta, gk):
        with torch.enable_grad():
            qd = q.detach().requires_grad_()
            kd = k.detach().requires_grad_()
            vd = v.detach().requires_grad_()
            ad = alpha.detach().requires_grad_()
            bd = beta.detach().requires_grad_()
            gd = gk.detach().requires_grad_()
            out = _forward_math(qd, kd, vd, ad, bd, gd)
        ctx.save_for_backward(qd, kd, vd, ad, bd, gd, out)
        return out.detach()

    @staticmethod
    def backward(ctx, dout):
        qd, kd, vd, ad, bd, gd, out = ctx.saved_tensors
        return torch.autograd.grad(
            out, [qd, kd, vd, ad, bd, gd], dout, allow_unused=True
        )


def chunked_dplr_delta(
    q: torch.Tensor,  # [B, T, H, K]
    k: torch.Tensor,  # [B, T, H, K]
    v: torch.Tensor,  # [B, T, H, V]
    alpha: torch.Tensor,  # [B, T, H, K]    DPLR erasure basis
    beta: torch.Tensor,  # [B, T, H, K]    DPLR write basis
    gk: torch.Tensor,  # [B, T, H, K]    per-K-channel log-decay (<= 0)
) -> torch.Tensor:  # [B, T, H, V]
    """Entry point. Signature and semantics are fixed; the body is yours."""
    return _Dplr.apply(q, k, v, alpha, beta, gk)
