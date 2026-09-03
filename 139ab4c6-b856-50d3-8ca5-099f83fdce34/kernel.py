"""
Starter -- chunk-parallel GDN-2 (Gated DeltaNet 2), forward and backward.

This file is yours. It is deliberately basic: a correct formulation written in
plain PyTorch with one Triton kernel doing per-chunk work, and nothing else
optimized. Your job is to make it fast.

The entry point the harness calls is `chunked_gdn2`, and its signature and
semantics must not change. Autograd must work through it: the harness measures
forward and backward together, and grades the gradients as well as the output.

What the harness measures, in one sentence: the geometric mean over a hidden
shape set of (reference production kernel time) / (your time), gated on agreeing
with reference.py on the output AND all six input gradients (dq, dk, dv, dg,
db, dw).

NOTE (authored-draft): this starter is correct-by-construction (the differentiable
path is the reference recurrence in float32; gradients come from autograd), but
it has NOT been GPU-profiled -- adoption/timing calibration is a separate
follow-up. The single @triton.jit kernel below is the seed jit entry point to
grow from; the chunked per-K-channel cumulative-decay formulation, the intra-
chunk matmuls on the ASYMMETRIC (b*k, k) key pair, the K=V=256 state tiling,
and the fused backward are yours.
"""

import torch
import triton
import triton.language as tl

CHUNK = 64


@triton.jit
def _gate_cumsum_channels_kernel(g_ptr, out_ptr, T, BT: tl.constexpr):
    """Per-chunk inclusive cumulative sum of the per-K-channel log-decay along time.

    Real work (a genuine per-chunk prefix sum), one prefix per key channel. Only
    Triton in the starter.
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


def _forward_math(q, k, v, g, b, w):
    """Correct, and deliberately slow.

    Walks the GDN-2 recurrence one timestep at a time, exactly as reference.py
    defines it: per-K-channel decay of the K rows of the state, per-K erase
    through (b*k), per-V write through (w*v), q scaled by K^-0.5. The chunked
    per-channel cumulative-gate formulation, the intra-chunk matmuls on the
    asymmetric (b*k, k) key pair, and the fused backward are yours to build.
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    q32 = q.float() * (K**-0.5)
    k32, v32, g32, b32, w32 = k.float(), v.float(), g.float(), b.float(), w.float()
    # Seed Triton kernel: per-K-channel per-chunk cumulative decay.
    _ = _chunk_cumsum_channels(g32)

    state = torch.zeros(B, H, K, V, dtype=torch.float32, device=q.device)
    outs = []
    for t in range(T):
        state = state * g32[:, t].exp()[:, :, :, None]  # per-K decay
        bt_k = b32[:, t] * k32[:, t]
        erase = torch.einsum("bhk,bhkv->bhv", bt_k, state)  # erase read
        write = w32[:, t] * v32[:, t] - erase
        state = state + torch.einsum("bhk,bhv->bhkv", k32[:, t], write)
        outs.append(torch.einsum("bhk,bhkv->bhv", q32[:, t], state))
    return torch.stack(outs, dim=1).to(v.dtype)


class _Gdn2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, b, w):
        with torch.enable_grad():
            qd = q.detach().requires_grad_()
            kd = k.detach().requires_grad_()
            vd = v.detach().requires_grad_()
            gd = g.detach().requires_grad_()
            bd = b.detach().requires_grad_()
            wd = w.detach().requires_grad_()
            out = _forward_math(qd, kd, vd, gd, bd, wd)
        ctx.save_for_backward(qd, kd, vd, gd, bd, wd, out)
        return out.detach()

    @staticmethod
    def backward(ctx, dout):
        qd, kd, vd, gd, bd, wd, out = ctx.saved_tensors
        return torch.autograd.grad(
            out, [qd, kd, vd, gd, bd, wd], dout, allow_unused=True
        )


def chunked_gdn2(
    q: torch.Tensor,  # [B, T, H, K]
    k: torch.Tensor,  # [B, T, H, K]
    v: torch.Tensor,  # [B, T, H, V]
    g: torch.Tensor,  # [B, T, H, K]   per-K-channel log-decay (<= 0)
    b: torch.Tensor,  # [B, T, H, K]   per-K-channel erase gate
    w: torch.Tensor,  # [B, T, H, V]   per-V-channel write gate
) -> torch.Tensor:  # [B, T, H, V]
    """Entry point. Signature and semantics are fixed; the body is yours."""
    return _Gdn2.apply(q, k, v, g, b, w)
