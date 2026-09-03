"""
Starter -- chunk-parallel COMBA (compositional basis attention) with an auxiliary
prediction key `p`, forward and backward.

This file is yours. It is deliberately basic: a correct formulation written in
plain PyTorch with one Triton kernel doing per-chunk work, and nothing else
optimized. Your job is to make it fast.

The entry point the harness calls is `chunked_comba`, and its signature and
semantics must not change. Autograd must work through it: the harness measures
forward and backward together, and grades the gradients as well as the output.

What the harness measures, in one sentence: the geometric mean over a hidden
shape set of (reference production kernel time) / (your time), gated on
agreeing with reference.py on the output AND all six input gradients (dq, dk,
dv, dp, dg, dbeta).

NOTE (authored-draft): this starter is correct-by-construction (the differentiable
path is the reference recurrence in float32; gradients come from autograd), but
it has NOT been GPU-profiled -- adoption/timing calibration is a separate
follow-up. The single @triton.jit kernel below is the seed jit entry point to
grow from; the chunked WY formulation on the CROSS basis `p k^T`, the intra-chunk
matmuls and the fused backward are yours.
"""

import torch
import triton
import triton.language as tl

CHUNK = 64


@triton.jit
def _gate_cumsum_kernel(g_ptr, out_ptr, T, BT: tl.constexpr):
    """Per-chunk inclusive cumulative sum of the scalar log-decay along time.

    Real work (a genuine per-chunk prefix sum). Only Triton in the starter.
    """
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs = tl.arange(0, BT)
    idx = pid_c * BT + offs
    mask = idx < T
    g = tl.load(g_ptr + pid_b * T + idx, mask=mask, other=0.0)
    tl.store(out_ptr + pid_b * T + idx, tl.cumsum(g, axis=0), mask=mask)


def _chunk_cumsum(g: torch.Tensor) -> torch.Tensor:
    """g: [B, T, H] -> per-chunk inclusive cumsum along T, same shape."""
    B, T, H = g.shape
    gt = g.permute(0, 2, 1).contiguous().view(B * H, T)
    out = torch.empty_like(gt)
    grid = (B * H, triton.cdiv(T, CHUNK))
    _gate_cumsum_kernel[grid](gt, out, T, BT=CHUNK)
    return out.view(B, H, T).permute(0, 2, 1).contiguous()


def _forward_math(q, k, v, p, g, beta):
    """Correct, and deliberately slow.

    Walks the COMBA recurrence one timestep at a time, exactly as reference.py
    defines it: gate applied BEFORE the write; erase term subtracts the state's
    projection on the AUXILIARY basis `p` (NOT `k`); write term adds the outer
    product on `k`; q scaled by K^-0.5. The chunked WY formulation on the cross
    basis `p k^T`, the intra-chunk matmuls and the fused backward are yours.
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    q32 = q.float() * (K**-0.5)
    k32, v32, p32, g32, b32 = k.float(), v.float(), p.float(), g.float(), beta.float()
    # Seed Triton kernel: the per-chunk cumulative gate. Real work.
    _ = _chunk_cumsum(g32)

    state = torch.zeros(B, H, K, V, dtype=torch.float32, device=q.device)
    outs = []
    for t in range(T):
        state = state * g32[:, t].exp()[:, :, None, None]
        pt, kt, vt = p32[:, t], k32[:, t], v32[:, t]
        pred = torch.einsum("bhk,bhkv->bhv", pt, state)  # erase against p
        delta = (vt - pred) * b32[:, t][:, :, None]
        state = state + torch.einsum("bhk,bhv->bhkv", kt, delta)  # write with k
        outs.append(torch.einsum("bhk,bhkv->bhv", q32[:, t], state))
    return torch.stack(outs, dim=1).to(v.dtype)


class _Comba(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, p, g, beta):
        with torch.enable_grad():
            qd = q.detach().requires_grad_()
            kd = k.detach().requires_grad_()
            vd = v.detach().requires_grad_()
            pd = p.detach().requires_grad_()
            gd = g.detach().requires_grad_()
            bd = beta.detach().requires_grad_()
            out = _forward_math(qd, kd, vd, pd, gd, bd)
        ctx.save_for_backward(qd, kd, vd, pd, gd, bd, out)
        return out.detach()

    @staticmethod
    def backward(ctx, dout):
        qd, kd, vd, pd, gd, bd, out = ctx.saved_tensors
        return torch.autograd.grad(
            out, [qd, kd, vd, pd, gd, bd], dout, allow_unused=True
        )


def chunked_comba(
    q: torch.Tensor,  # [B, T, H, K]
    k: torch.Tensor,  # [B, T, H, K]
    v: torch.Tensor,  # [B, T, H, V]
    p: torch.Tensor,  # [B, T, H, K]   auxiliary erase / prediction basis
    g: torch.Tensor,  # [B, T, H]      log-space scalar decay (<= 0)
    beta: torch.Tensor,  # [B, T, H]      write strength in (0, 1]
) -> torch.Tensor:  # [B, T, H, V]
    """Entry point. Signature and semantics are fixed; the body is yours."""
    return _Comba.apply(q, k, v, p, g, beta)
