"""
Starter -- chunk-parallel RWKV-6 with u-bonus, forward and backward.

This file is yours. It is deliberately basic: a correct formulation written in
plain PyTorch with one Triton kernel doing per-chunk work, and nothing else
optimized. Your job is to make it fast.

The entry point the harness calls is `chunked_rwkv6`, and its signature and
semantics must not change. Autograd must work through it: the harness measures
forward and backward together, and grades the gradients as well as the output.

What the harness measures, in one sentence: the geometric mean over a hidden shape
set of (reference production kernel time) / (your time), gated on agreeing with
reference.py on the output AND all FIVE input gradients (dr, dk, dv, dw, du).

NOTE (authored-draft): this starter is correct-by-construction (the differentiable
path is the reference recurrence in float32; gradients come from autograd), but it
has NOT been GPU-profiled -- adoption/timing calibration is a separate follow-up.
The single @triton.jit kernel below is the seed jit entry point to grow from; the
chunked cumulative-decay formulation, the intra-chunk matmuls, the u-bonus fused
on the diagonal, and the fused backward are yours.
"""

import torch
import triton
import triton.language as tl

CHUNK = 64


@triton.jit
def _w_cumsum_kernel(w_ptr, out_ptr, T, BT: tl.constexpr):
    """Per-chunk inclusive cumulative sum of the log-decay along time.

    Real work (a genuine per-chunk prefix sum), and the natural first step of a
    chunked RWKV-6: the intra-chunk cumulative forget gate, one prefix per key
    channel. It is the only Triton in the starter; everything else below is plain
    PyTorch, and that is where the time goes.
    """
    pid_bhk = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs = tl.arange(0, BT)
    idx = pid_c * BT + offs
    mask = idx < T
    w = tl.load(w_ptr + pid_bhk * T + idx, mask=mask, other=0.0)
    tl.store(out_ptr + pid_bhk * T + idx, tl.cumsum(w, axis=0), mask=mask)


def _chunk_cumsum_channels(w: torch.Tensor) -> torch.Tensor:
    """w: [B, T, H, K] -> per-chunk inclusive cumsum along T, same shape."""
    B, T, H, K = w.shape
    wt = w.permute(0, 2, 3, 1).contiguous().view(B * H * K, T)
    out = torch.empty_like(wt)
    grid = (B * H * K, triton.cdiv(T, CHUNK))
    _w_cumsum_kernel[grid](wt, out, T, BT=CHUNK)
    return out.view(B, H, K, T).permute(0, 3, 1, 2).contiguous()


def _forward_math(r, k, v, w, u):
    """Correct, and deliberately slow.

    Walks the RWKV-6 recurrence one timestep at a time, exactly as reference.py
    defines it: per-key-channel decay of the K rows of the state, the u-bonus
    added to the CURRENT-step outer product only in the output (state carried to
    t+1 does NOT see u), r scaled by K^-0.5. Every step is a separate set of
    launches and the whole [K, V] state round-trips through global memory; the
    chunked cumulative-gate formulation, the u-fused diagonal, and the fused
    backward are yours to build.
    """
    B, T, H, K = r.shape
    V = v.shape[-1]
    r32 = r.float() * (K**-0.5)
    k32, v32, w32, u32 = k.float(), v.float(), w.float(), u.float()
    # Seed Triton kernel: the per-chunk cumulative forget gate. Real work; its
    # result is the first thing a chunked RWKV-6 needs. The starter's own answer
    # below still applies the plain per-step exp(w), so turning this cumulative
    # gate into a per-chunk state transition is one of the things you are asked
    # to do.
    _ = _chunk_cumsum_channels(w32)

    state = torch.zeros(B, H, K, V, dtype=torch.float32, device=r.device)
    outs = []
    for t in range(T):
        kt = k32[:, t]  # [B, H, K]
        vt = v32[:, t]  # [B, H, V]
        kv = kt[:, :, :, None] * vt[:, :, None, :]  # [B, H, K, V]
        boosted = state + u32[None, :, :, None] * kv  # [B, H, K, V]
        outs.append((r32[:, t][:, :, :, None] * boosted).sum(2))
        state = state * w32[:, t].exp()[:, :, :, None] + kv  # decay AFTER readout
    return torch.stack(outs, dim=1).to(v.dtype)


class _RWKV6(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r, k, v, w, u):
        with torch.enable_grad():
            rd = r.detach().requires_grad_()
            kd = k.detach().requires_grad_()
            vd = v.detach().requires_grad_()
            wd = w.detach().requires_grad_()
            ud = u.detach().requires_grad_()
            out = _forward_math(rd, kd, vd, wd, ud)
        ctx.save_for_backward(rd, kd, vd, wd, ud, out)
        return out.detach()

    @staticmethod
    def backward(ctx, dout):
        rd, kd, vd, wd, ud, out = ctx.saved_tensors
        grads = torch.autograd.grad(out, [rd, kd, vd, wd, ud], dout, allow_unused=True)
        return grads


def chunked_rwkv6(
    r: torch.Tensor,  # [B, T, H, K]
    k: torch.Tensor,  # [B, T, H, K]
    v: torch.Tensor,  # [B, T, H, V]
    w: torch.Tensor,  # [B, T, H, K]   log-space per-channel decay (<= 0)
    u: torch.Tensor,  # [H, K]         per-head per-channel current-step bonus
) -> torch.Tensor:  # [B, T, H, V]
    """Entry point. Signature and semantics are fixed; the body is yours."""
    return _RWKV6.apply(r, k, v, w, u)
