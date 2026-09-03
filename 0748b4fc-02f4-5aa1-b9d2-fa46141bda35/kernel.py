"""
Starter -- per-channel-decay + beta-delta linear-attention decode step, forward only.

This file is yours. It is deliberately basic: one small Triton kernel does the
full per-step math -- per-channel decay of the state, pre-write prediction,
scalar delta write, post-write read -- launched once per timestep with the full
[K, V] state round-tripping through HBM every step. Correct, deterministic, and
slow by design. Your job is to make it fast.

The entry point the harness calls is `rwkv7_channel_delta_decode`, and its
signature and semantics must not change. Both outputs are graded; the state
must be carried in float32.

Why this starter is slow: (1) one launch per timestep pays S launch latencies;
(2) the state is re-read from HBM S times per (batch, head) even though it fits
in registers for the whole sequence; (3) two dense reads over the K axis per
step (pre-write prediction and post-write output) are performed as two separate
tl.sum reductions instead of any shared work.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _step_kernel(
    Q,
    Kk,
    Vv,
    Gk,
    Beta,
    St,
    O,
    t,
    S_len,
    H,
    scale,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
):
    """One step for one (batch*head, V-tile). Per-channel decay, delta write,
    read -- all in float32."""
    pid = tl.program_id(0).to(tl.int64)  # over B*H
    pv = tl.program_id(1).to(tl.int64)  # over V // BV
    b = pid // H
    h = pid % H

    o_k = tl.arange(0, K).to(tl.int64)
    o_v = pv * BV + tl.arange(0, BV).to(tl.int64)

    p_h = St + pid * K * V + o_k[:, None] * V + o_v[None, :]
    b_h = tl.load(p_h)  # [K, BV] float32

    base = (b * S_len + t) * H + h
    b_q = tl.load(Q + base * K + o_k).to(tl.float32) * scale
    b_k = tl.load(Kk + base * K + o_k).to(tl.float32)
    b_v = tl.load(Vv + base * V + o_v).to(tl.float32)
    b_g = tl.load(Gk + base * K + o_k).to(tl.float32)
    b_beta = tl.load(Beta + base).to(tl.float32)

    b_h = b_h * tl.exp(b_g)[:, None]  # per-channel decay before delta
    pred = tl.sum(b_h * b_k[:, None], 0)  # pre-write read [BV]
    delta = (b_v - pred) * b_beta  # [BV]
    b_h = b_h + b_k[:, None] * delta[None, :]  # delta write
    b_o = tl.sum(b_h * b_q[:, None], 0)  # post-write read [BV]

    tl.store(O + base * V + o_v, b_o.to(O.dtype.element_ty))
    tl.store(p_h, b_h)


def rwkv7_channel_delta_decode(
    q: torch.Tensor,  # [B, S, H, K]
    k: torch.Tensor,  # [B, S, H, K]
    v: torch.Tensor,  # [B, S, H, V]
    g: torch.Tensor,  # [B, S, H, K]  float32 per-channel log-forget gate
    beta: torch.Tensor,  # [B, S, H]                write strength
    state0: torch.Tensor,  # [B, H, K, V]  float32 initial state
) -> tuple:  # (o [B, S, H, V] in v.dtype, state_out [B, H, K, V] fp32)
    """Entry point. Signature and semantics are fixed; the body is yours."""
    B, S, H, K = q.shape
    V = v.shape[-1]
    if (K & (K - 1)) or (V & (V - 1)):
        raise ValueError(f"starter supports power-of-two K and V, got K={K}, V={V}")
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    g, beta = g.contiguous(), beta.contiguous()

    state = state0.to(torch.float32).contiguous().clone()
    o = torch.empty(B, S, H, V, dtype=v.dtype, device=v.device)
    scale = K**-0.5

    BV = 64 if V >= 64 else V
    grid = (B * H, V // BV)
    for t in range(S):
        _step_kernel[grid](q, k, v, g, beta, state, o, t, S, H, scale, K=K, V=V, BV=BV)
    return o, state
