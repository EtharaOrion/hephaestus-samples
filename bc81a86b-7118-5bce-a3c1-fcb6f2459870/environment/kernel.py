"""
Starter -- chunk-parallel gated delta rule, forward and backward.

This file is yours. It is deliberately basic: a correct chunked formulation
written in plain PyTorch with one Triton kernel doing the intra-chunk work, and
nothing else optimized. Your job is to make it fast.

The entry point the harness calls is `gated_delta_rule`, and its signature and
semantics must not change. Autograd must work through it: the harness measures
forward and backward together, and grades the gradients as well as the output.

What the harness measures, in one sentence: the geometric mean over a hidden
shape set of (reference production kernel time) / (your time), gated on
agreeing with reference.py.
"""

import torch
import triton
import triton.language as tl

CHUNK = 64


@triton.jit
def _decay_cumsum_kernel(g_ptr, out_ptr, T, BT: tl.constexpr):
    """Per-chunk exclusive cumulative sum of the log-decay.

    The only Triton in the starter. Everything else below is plain PyTorch, and
    that is where the time goes.
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
    _decay_cumsum_kernel[grid](gt, out, T, BT=CHUNK)
    return out.view(B, H, T).permute(0, 2, 1).contiguous()


class _GatedDeltaRule(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta):
        # Plain autograd-through-the-chunk-loop. Correct, and slow: every chunk
        # round trips through global memory and nothing is fused.
        with torch.enable_grad():
            qd, kd, vd = q.detach().requires_grad_(), k.detach().requires_grad_(), v.detach().requires_grad_()
            gd, bd = g.detach().requires_grad_(), beta.detach().requires_grad_()
            out = _forward_math(qd, kd, vd, gd, bd)
        ctx.save_for_backward(qd, kd, vd, gd, bd, out)
        return out.detach()

    @staticmethod
    def backward(ctx, dout):
        qd, kd, vd, gd, bd, out = ctx.saved_tensors
        grads = torch.autograd.grad(out, [qd, kd, vd, gd, bd], dout, allow_unused=True)
        return grads


def _forward_math(q, k, v, g, beta):
    """Correct, and deliberately slow.

    This walks the recurrence one timestep at a time, exactly as reference.py
    defines it. It is the sub-optimal starting implementation: every step is a
    separate launch, the state round trips through global memory, and nothing is
    chunked, fused or tiled. The chunked formulation, the WY representation, the
    intra-chunk matmuls and the backward pass are all yours to build.
    """
    B, T, H, K = q.shape
    HV = v.shape[2]
    q32, k32, v32 = q.float() * (K ** -0.5), k.float(), v.float()
    if HV != H:
        rep = HV // H
        q32 = q32.repeat_interleave(rep, dim=2)
        k32 = k32.repeat_interleave(rep, dim=2)
        H = HV
    beta32 = beta.float()
    # The one Triton kernel in the starter. It is here so the file has a working
    # jit entry point to grow from, not because the cumsum is the bottleneck. Its
    # result is deliberately unused: the starter's decay is the plain elementwise
    # exp below, and turning `gc` into a per-chunk decay is one of the things you
    # are being asked to do.
    _ = _chunk_cumsum(g.float())
    dec = g.float().exp()

    state = torch.zeros(B, H, K, v.shape[-1], dtype=torch.float32, device=q.device)
    outs = []
    for t in range(T):
        state = state * dec[:, t][:, :, None, None]
        kt = k32[:, t]
        pred = torch.einsum("bhk,bhkv->bhv", kt, state)
        delta = (v32[:, t] - pred) * beta32[:, t][:, :, None]
        state = state + torch.einsum("bhk,bhv->bhkv", kt, delta)
        outs.append(torch.einsum("bhk,bhkv->bhv", q32[:, t], state))
    return torch.stack(outs, dim=1).to(v.dtype)


def gated_delta_rule(
    q: torch.Tensor,      # [B, T, H, K]
    k: torch.Tensor,      # [B, T, H, K]
    v: torch.Tensor,      # [B, T, H, V]
    g: torch.Tensor,      # [B, T, H]
    beta: torch.Tensor,   # [B, T, H]
) -> torch.Tensor:        # [B, T, H, V]
    """Entry point. Signature and semantics are fixed; the body is yours."""
    return _GatedDeltaRule.apply(q, k, v, g, beta)
