"""
Starter -- int8-state delta rule decode step, forward only.

This file is yours. It is deliberately basic: one Triton kernel dequantizes and
runs one delta step per timestep in float32 (state round-tripped through HBM S
times), and a second kernel does the final per-(batch, head) absmax + int8
quantize at the end. Correct, deterministic, and slow by design. Your job is to
make it fast.

The entry point the harness calls is `int8_state_delta_decode`, and its
signature and semantics must not change: dequantize the int8 initial state,
run S delta-rule steps in fp32, quantize the final state back to int8 with a
per-(batch, head) symmetric scale. All THREE outputs are graded: o at v's
dtype, state_out_q int8, state_out_scale float32.

Why this starter is slow: (1) one launch per timestep pays S launch latencies;
(2) the fp32 scratch state round-trips through HBM S times per (batch, head)
even though it fits in registers for the whole sequence; (3) the final
quantization is a separate kernel that re-reads the full scratch state, when
the same per-(batch, head) absmax could be computed incrementally in registers.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _step_kernel(
    Q,
    Kk,
    Vv,
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
    """One delta-rule step for one (batch*head, V-tile) in float32 scratch
    state."""
    pid = tl.program_id(0).to(tl.int64)  # over B*H
    pv = tl.program_id(1).to(tl.int64)  # over V // BV
    b = pid // H
    h = pid % H

    o_k = tl.arange(0, K).to(tl.int64)
    o_v = pv * BV + tl.arange(0, BV).to(tl.int64)

    p_h = St + pid * K * V + o_k[:, None] * V + o_v[None, :]
    b_h = tl.load(p_h)  # [K, BV] float32 scratch state

    base = (b * S_len + t) * H + h
    b_q = tl.load(Q + base * K + o_k).to(tl.float32) * scale
    b_k = tl.load(Kk + base * K + o_k).to(tl.float32)
    b_v = tl.load(Vv + base * V + o_v).to(tl.float32)
    b_beta = tl.load(Beta + base).to(tl.float32)

    pred = tl.sum(b_h * b_k[:, None], 0)
    delta = (b_v - pred) * b_beta
    b_h = b_h + b_k[:, None] * delta[None, :]

    b_o = tl.sum(b_h * b_q[:, None], 0)
    tl.store(O + base * V + o_v, b_o.to(O.dtype.element_ty))
    tl.store(p_h, b_h)


@triton.jit
def _quantize_kernel(
    St,
    Sq,
    Sscale,
    K: tl.constexpr,
    V: tl.constexpr,
):
    """Per-(batch, head) symmetric int8 quantization: scale = absmax / 127,
    quantized value = round(x / scale) clamped to [-127, 127]. Zero state gets
    scale = 0 and q = 0."""
    pid = tl.program_id(0).to(tl.int64)  # over B*H
    o_k = tl.arange(0, K).to(tl.int64)
    o_v = tl.arange(0, V).to(tl.int64)

    p_h = St + pid * K * V + o_k[:, None] * V + o_v[None, :]
    b_h = tl.load(p_h)
    absmax = tl.max(tl.abs(b_h))
    scale = tl.where(absmax > 0.0, absmax / 127.0, 0.0)
    denom = tl.where(scale > 0.0, scale, 1.0)
    q = tl.extra.cuda.libdevice.round(b_h / denom)
    q = tl.minimum(tl.maximum(q, -127.0), 127.0)
    q = tl.where(absmax > 0.0, q, 0.0).to(tl.int8)

    p_q = Sq + pid * K * V + o_k[:, None] * V + o_v[None, :]
    tl.store(p_q, q)
    tl.store(Sscale + pid, scale)


def int8_state_delta_decode(
    q: torch.Tensor,  # [B, S, H, K]
    k: torch.Tensor,  # [B, S, H, K]
    v: torch.Tensor,  # [B, S, H, V]
    beta: torch.Tensor,  # [B, S, H]
    state0_q: torch.Tensor,  # [B, H, K, V]      int8
    state0_scale: torch.Tensor,  # [B, H]            float32
) -> tuple:  # (o [B,S,H,V], state_q [B,H,K,V] int8,
    #  state_scale [B,H] float32)
    """Entry point. Signature and semantics are fixed; the body is yours."""
    B, S, H, K = q.shape
    V = v.shape[-1]
    if (K & (K - 1)) or (V & (V - 1)):
        raise ValueError(f"starter supports power-of-two K and V, got K={K}, V={V}")
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    beta = beta.contiguous()
    state0_q = state0_q.contiguous()
    state0_scale = state0_scale.contiguous()

    # Host-side dequantize into the fp32 scratch state.
    state = (state0_q.float() * state0_scale[:, :, None, None]).contiguous()
    o = torch.empty(B, S, H, V, dtype=v.dtype, device=v.device)
    scale = K**-0.5

    BV = 64 if V >= 64 else V
    grid = (B * H, V // BV)
    for t in range(S):
        _step_kernel[grid](q, k, v, beta, state, o, t, S, H, scale, K=K, V=V, BV=BV)

    # Final per-(batch, head) int8 quantization.
    state_q = torch.empty(B, H, K, V, dtype=torch.int8, device=v.device)
    state_scale = torch.empty(B, H, dtype=torch.float32, device=v.device)
    _quantize_kernel[(B * H,)](state, state_q, state_scale, K=K, V=V)
    return o, state_q, state_scale
