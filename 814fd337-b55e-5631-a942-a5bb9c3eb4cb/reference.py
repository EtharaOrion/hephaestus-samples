"""
Reference implementation -- PyTorch-only ground truth for correctness verification.
DO NOT MODIFY. This is the oracle the benchmark harness checks every candidate against.

The operator is the DECODE regime of the DELTA RULE with the recurrent state
STORED in int8 (per-(batch, head) symmetric scale) at the boundary, computed in
float32 internally. The initial state is (state0_q [B, H, K, V] int8,
state0_scale [B, H] float32); the reference dequantizes state0 at entry, runs
the sequential fp32 scan for S steps, requantizes the final state to (state_out_q
int8, state_out_scale float32) at exit. The RECURRENCE runs in float32 so its
math is unchanged from the plain delta rule; the only observable change is the
QUANTIZATION at the output boundary, which is why the state comparison happens
on the DEQUANTIZED state (state_out_q * state_out_scale) at a looser tolerance
that accounts for the int8 round-trip (max relative error ~1/127).

    S_0    = state0_q * state0_scale                       (dequantize on entry)
    S_t    = S_{t-1} + k_t delta_t^T                       (delta = (v - S^T k)*beta)
    o_t    = S_t^T (q_t / sqrt(K))
    scale_out[b,h] = max(|S_S[b,h]|) / 127                 (per-(b,h) symmetric)
    state_out_q[b,h] = round(S_S[b,h] / scale_out[b,h])    (clamped to [-127,127])

BOTH tensor outputs graded: o [B, S, H, V] in v's dtype; state_out_q int8 and
state_out_scale float32 compared jointly via the dequantized state.
"""

import torch


def _quantize_int8_state(state_fp32: torch.Tensor):
    """Per-(batch, head) symmetric int8 quantization; scale = max(|x|)/127.

    Returns (q int8 [B, H, K, V], scale float32 [B, H]). A zero-state batch/head
    gets scale = 0 and q = 0 (avoids a divide-by-zero and keeps the round-trip
    equal to 0, which is exact)."""
    absmax = state_fp32.float().abs().amax(dim=(-2, -1))  # [B, H]
    scale = torch.where(absmax > 0, absmax / 127.0, torch.zeros_like(absmax))
    denom = scale[:, :, None, None].clamp(min=torch.finfo(torch.float32).tiny)
    q = (state_fp32.float() / denom).round().clamp(-127, 127).to(torch.int8)
    zero = (absmax == 0)[:, :, None, None].expand_as(q)
    q = torch.where(zero, torch.zeros_like(q), q)
    return q, scale


def _dequantize_int8_state(q: torch.Tensor, scale: torch.Tensor):
    """(int8 q [B, H, K, V], scale float32 [B, H]) -> float32 state [B, H, K, V]."""
    return q.float() * scale[:, :, None, None]


def int8_state_delta_decode_ref(
    q: torch.Tensor,  # [B, S, H, K]
    k: torch.Tensor,  # [B, S, H, K]
    v: torch.Tensor,  # [B, S, H, V]
    beta: torch.Tensor,  # [B, S, H]         write strength
    state0_q: torch.Tensor,  # [B, H, K, V]      int8 quantized state
    state0_scale: torch.Tensor,  # [B, H]            float32 per-(b,h) scale
):  # -> (o [B, S, H, V] in v.dtype,
    #     state_out_q [B, H, K, V] int8,
    #     state_out_scale [B, H] float32)
    """Sequential ground truth for the int8-boundary delta rule decode step."""
    B, S, H, K = q.shape
    V = v.shape[-1]
    q32 = q.float() * (K**-0.5)
    k32, v32, b32 = k.float(), v.float(), beta.float()

    state = _dequantize_int8_state(state0_q, state0_scale)  # [B, H, K, V]
    out = torch.empty(B, S, H, V, dtype=torch.float32, device=q.device)

    for t in range(S):
        kt = k32[:, t]
        pred = torch.einsum("bhk,bhkv->bhv", kt, state)
        delta = (v32[:, t] - pred) * b32[:, t][:, :, None]
        state = state + torch.einsum("bhk,bhv->bhkv", kt, delta)
        out[:, t] = torch.einsum("bhk,bhkv->bhv", q32[:, t], state)

    state_q, state_scale = _quantize_int8_state(state)
    return out.to(v.dtype), state_q, state_scale


if __name__ == "__main__":
    torch.manual_seed(0)
    B, S, H, K, V = 2, 4, 3, 8, 6
    q = torch.randn(B, S, H, K)
    k = torch.randn(B, S, H, K)
    v = torch.randn(B, S, H, V)
    beta = torch.rand(B, S, H) * 0.9 + 0.05
    st0 = torch.randn(B, H, K, V) * 0.5
    st0_q, st0_scale = _quantize_int8_state(st0)
    o, s_q, s_scale = int8_state_delta_decode_ref(q, k, v, beta, st0_q, st0_scale)
    assert o.shape == (B, S, H, V)
    assert s_q.shape == (B, H, K, V) and s_q.dtype == torch.int8
    assert s_scale.shape == (B, H) and s_scale.dtype == torch.float32

    # Round-trip on the dequantized initial state must be within ~scale/2.
    dq = _dequantize_int8_state(st0_q, st0_scale)
    err = (dq - st0).abs().amax(dim=(-2, -1))
    tol = st0_scale / 2 + 1e-5
    assert (err <= tol).all(), (float(err.max()), float(tol.min()))

    # Zero-state batch/head must round-trip to exactly zero.
    st_zero = torch.zeros(1, 1, K, V)
    q_z, s_z = _quantize_int8_state(st_zero)
    assert (q_z == 0).all() and float(s_z.item()) == 0.0
    assert torch.equal(_dequantize_int8_state(q_z, s_z), st_zero)

    # A fp32-state run (skipping the int8 boundary) matches within one
    # quantization step of the initial state.
    def _fp32_delta(q, k, v, beta, st):
        B, S, H, K = q.shape
        V = v.shape[-1]
        q32 = q.float() * (K**-0.5)
        k32, v32, b32 = k.float(), v.float(), beta.float()
        state = st.float().clone()
        out = torch.empty(B, S, H, V, dtype=torch.float32, device=q.device)
        for t in range(S):
            pred = torch.einsum("bhk,bhkv->bhv", k32[:, t], state)
            delta = (v32[:, t] - pred) * b32[:, t][:, :, None]
            state = state + torch.einsum("bhk,bhv->bhkv", k32[:, t], delta)
            out[:, t] = torch.einsum("bhk,bhkv->bhv", q32[:, t], state)
        return out, state

    o_fp32, _ = _fp32_delta(q, k, v, beta, dq)
    assert torch.allclose(o.float(), o_fp32, atol=1e-5)

    print(
        "int8_state_delta_decode_ref CPU sanity OK:",
        tuple(o.shape),
        o.dtype,
        "| state_q",
        tuple(s_q.shape),
        s_q.dtype,
        "| scale",
        tuple(s_scale.shape),
        s_scale.dtype,
    )
