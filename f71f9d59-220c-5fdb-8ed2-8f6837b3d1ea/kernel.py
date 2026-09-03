"""
Starter -- chunk-parallel gated delta-product (N Householder writes per token),
forward and backward.

This file is yours. It is deliberately basic: a correct formulation written in
plain PyTorch with one Triton kernel doing per-chunk work, and nothing else
optimized. Your job is to make it fast.

The entry point the harness calls is `chunked_gated_delta_product`, and its
signature and semantics must not change. Autograd must work through it: the
harness measures forward and backward together, and grades the gradients as
well as the output.

What the harness measures, in one sentence: the geometric mean over a hidden
shape set of (reference production kernel time) / (your time), gated on agreeing
with reference.py on the output AND all five input gradients (dq, dk, dv, dg,
dbeta).

NOTE (authored-draft): this starter is correct-by-construction (the differentiable
path is the reference recurrence in float32; gradients come from autograd), but it
has NOT been GPU-profiled -- adoption/timing calibration is a separate follow-up.
The single @triton.jit kernel below is the seed jit entry point to grow from; the
chunked WY formulation stacked N_householder times, the intra-chunk matmuls and
the fused backward are yours.
"""

import torch
import triton
import triton.language as tl

CHUNK = 64
NUM_HOUSEHOLDER = 2


@triton.jit
def _gate_cumsum_kernel(g_ptr, out_ptr, T, BT: tl.constexpr):
    """Per-chunk inclusive cumulative sum of the log-decay along time.

    Real work (a genuine per-chunk prefix sum) and the natural first step of a
    chunked gated delta rule: the intra-chunk cumulative gate. It is the only
    Triton in the starter; everything else below is plain PyTorch, and that is
    where the time goes.
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


def _forward_math(q, k, v, g, beta, num_householder):
    """Correct, and deliberately slow.

    Walks the recurrence one timestep at a time, exactly as reference.py defines
    it: gate applied ONCE before the N Householder writes; each Householder
    writes a rank-1 correction against the current state's own prediction; q
    scaled by K^-0.5. Every step is N sets of launches and the state round-trips
    through global memory; the chunked WY formulation (stacked N_householder
    times), the intra-chunk matmuls and the fused backward are yours to build.
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    N = int(num_householder)
    q32 = q.float() * (K**-0.5)
    k32, v32, g32, b32 = k.float(), v.float(), g.float(), beta.float()
    # Seed Triton kernel: the per-chunk cumulative gate. Real work; its result
    # is the first thing a chunked delta-product implementation needs.
    _ = _chunk_cumsum(g32)

    state = torch.zeros(B, H, K, V, dtype=torch.float32, device=q.device)
    outs = []
    for t in range(T):
        state = state * g32[:, t].exp()[:, :, None, None]  # gate BEFORE writes
        for j in range(N):
            k_tj = k32[:, t * N + j]
            v_tj = v32[:, t * N + j]
            b_tj = b32[:, t * N + j]
            pred = torch.einsum("bhk,bhkv->bhv", k_tj, state)
            delta = (v_tj - pred) * b_tj[:, :, None]
            state = state + torch.einsum("bhk,bhv->bhkv", k_tj, delta)
        outs.append(torch.einsum("bhk,bhkv->bhv", q32[:, t], state))
    return torch.stack(outs, dim=1).to(v.dtype)


class _GatedDeltaProduct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta, num_householder):
        with torch.enable_grad():
            qd = q.detach().requires_grad_()
            kd = k.detach().requires_grad_()
            vd = v.detach().requires_grad_()
            gd = g.detach().requires_grad_()
            bd = beta.detach().requires_grad_()
            out = _forward_math(qd, kd, vd, gd, bd, num_householder)
        ctx.save_for_backward(qd, kd, vd, gd, bd, out)
        return out.detach()

    @staticmethod
    def backward(ctx, dout):
        qd, kd, vd, gd, bd, out = ctx.saved_tensors
        grads = torch.autograd.grad(out, [qd, kd, vd, gd, bd], dout, allow_unused=True)
        return (*grads, None)  # num_householder has no gradient


def chunked_gated_delta_product(
    q: torch.Tensor,  # [B, T, H, K]
    k: torch.Tensor,  # [B, T*N, H, K]
    v: torch.Tensor,  # [B, T*N, H, V]
    g: torch.Tensor,  # [B, T, H]        log-space per-token decay (<= 0)
    beta: torch.Tensor,  # [B, T*N, H]      per (t, j, head) write strength in (0, 1]
    num_householder: int = NUM_HOUSEHOLDER,
) -> torch.Tensor:  # [B, T, H, V]
    """Entry point. Signature and semantics are fixed; the body is yours."""
    return _GatedDeltaProduct.apply(q, k, v, g, beta, num_householder)
