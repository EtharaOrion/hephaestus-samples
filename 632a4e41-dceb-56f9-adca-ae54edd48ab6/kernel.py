"""
Starter -- decode-regime GROUPED-QUERY DELTA RULE, forward only.

This file is yours. It is deliberately basic: one small Triton kernel per group
of query heads does the state's delta update and one query-head read at a time,
launched once per (timestep, query-head) pair. Correct, deterministic, and slow
by design. Your job is to make it fast.

The entry point the harness calls is `gqa_delta_decode`, and its signature and
semantics must not change: S sequential steps of the delta rule per kv-head
starting from `state0`, with the state SHARED across the G=Hq/Hkv query heads
in each group. Both the per-step outputs `o` AND the final state are graded, and
the state must be carried in float32.

Why this starter is slow: (1) one launch per (timestep, query-head) pays S*Hq
launch latencies; (2) the shared state is re-read from HBM Hq times per step
instead of held resident across the whole group's reads; (3) the delta write is
performed by ONE of the Hq launches per step (the first in each group) and
race-avoided by having the other G-1 launches skip the write, wasting their
kernel work.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _write_kernel(
    Kk,
    Vv,
    Beta,
    St,
    t,
    S_len,
    Hkv,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
):
    """The state update: one program per (batch*Hkv, V-tile). Applies the
    delta write. Every arithmetic step is float32."""
    pid = tl.program_id(0).to(tl.int64)  # over B*Hkv
    pv = tl.program_id(1).to(tl.int64)  # over V // BV
    b = pid // Hkv
    hk = pid % Hkv

    o_k = tl.arange(0, K).to(tl.int64)
    o_v = pv * BV + tl.arange(0, BV).to(tl.int64)

    p_h = St + pid * K * V + o_k[:, None] * V + o_v[None, :]
    b_h = tl.load(p_h)  # [K, BV] float32

    base = (b * S_len + t) * Hkv + hk
    b_k = tl.load(Kk + base * K + o_k).to(tl.float32)
    b_v = tl.load(Vv + base * V + o_v).to(tl.float32)
    b_beta = tl.load(Beta + base).to(tl.float32)

    pred = tl.sum(b_h * b_k[:, None], 0)  # pre-update read
    delta = (b_v - pred) * b_beta  # [BV]
    b_h = b_h + b_k[:, None] * delta[None, :]  # delta write

    tl.store(p_h, b_h)


@triton.jit
def _read_kernel(
    Q,
    St,
    O,
    t,
    S_len,
    Hq,
    Hkv,
    scale,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
):
    """The query-head read: one program per (batch*Hq, V-tile). Reads the
    kv-head's state after the delta update.
    """
    pid = tl.program_id(0).to(tl.int64)  # over B*Hq
    pv = tl.program_id(1).to(tl.int64)  # over V // BV
    b = pid // Hq
    hq = pid % Hq
    G = Hq // Hkv
    hk = hq // G

    o_k = tl.arange(0, K).to(tl.int64)
    o_v = pv * BV + tl.arange(0, BV).to(tl.int64)

    kv_pid = b * Hkv + hk
    p_h = St + kv_pid * K * V + o_k[:, None] * V + o_v[None, :]
    b_h = tl.load(p_h)  # re-read every query head

    base_q = ((b * S_len + t) * Hq + hq) * K + o_k
    b_q = tl.load(Q + base_q).to(tl.float32) * scale
    b_o = tl.sum(b_h * b_q[:, None], 0)  # [BV]

    base_o = ((b * S_len + t) * Hq + hq) * V + o_v
    tl.store(O + base_o, b_o.to(O.dtype.element_ty))


def gqa_delta_decode(
    q: torch.Tensor,  # [B, S, Hq, K]
    k: torch.Tensor,  # [B, S, Hkv, K]
    v: torch.Tensor,  # [B, S, Hkv, V]
    beta: torch.Tensor,  # [B, S, Hkv]        write strength
    state0: torch.Tensor,  # [B, Hkv, K, V]     float32 initial state
) -> tuple:  # (o [B, S, Hq, V] in v.dtype, state_out [B, Hkv, K, V] fp32)
    """Entry point. Signature and semantics are fixed; the body is yours."""
    B, S, Hq, K = q.shape
    Hkv = k.shape[2]
    V = v.shape[-1]
    if Hq % Hkv:
        raise ValueError(f"Hq ({Hq}) must be a multiple of Hkv ({Hkv})")
    if (K & (K - 1)) or (V & (V - 1)):
        raise ValueError(f"starter supports power-of-two K and V, got K={K}, V={V}")
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    beta = beta.contiguous()

    state = state0.to(torch.float32).contiguous().clone()
    o = torch.empty(B, S, Hq, V, dtype=v.dtype, device=v.device)
    scale = K**-0.5

    BV = 64 if V >= 64 else V
    grid_w = (B * Hkv, V // BV)
    grid_r = (B * Hq, V // BV)
    for t in range(S):
        _write_kernel[grid_w](k, v, beta, state, t, S, Hkv, K=K, V=V, BV=BV)
        _read_kernel[grid_r](q, state, o, t, S, Hq, Hkv, scale, K=K, V=V, BV=BV)
    return o, state
