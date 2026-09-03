"""
Starter -- decode-regime PAGED GATED LINEAR ATTENTION, forward only.

This file is yours. It is deliberately basic: one small Triton kernel does all the
real work of one selective step for one (batch, head) pair -- gather the initial
state from the shared pool through the page table, decay by the per-channel forget
gate, add the outer product k v^T, read the state against the scaled query, store
back into a dense per-batch scratch state -- and it does it the slowest reasonable
way, launched once per timestep with the full [K, V] state round-tripping through
HBM on every one of the S steps. Correct, deterministic, and slow by design. Your
job is to make it fast.

The entry point the harness calls is `paged_gla_decode`, and its signature and
semantics must not change: S sequential GLA steps starting from `paged_state[
page_table[b]]` for each batch b, both the per-step outputs `o` AND the dense final
state returned. reference.py defines those semantics; agreement with it is graded on
both outputs, and the state must be carried in float32.

Why this starter is slow: (1) one launch per timestep pays S launch latencies; (2)
the per-step state round-trip through HBM is compulsory here but the WHOLE-SEQUENCE
state of one (batch, head) fits in registers, so the state should be resident from
the first step to the last; (3) the initial page gather is performed by torch on the
host through a fancy-index, forcing a full [B, H, K, V] materialization before the
first launch instead of a per-program gather inside the kernel from the page table
directly.
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
    """One GLA step for one (batch*head, V-tile) pair reading from a dense
    per-batch scratch state (populated once by the host from the paged pool).

    The state tile [K, BV] is loaded, per-row decayed by exp(gk), gets the outer
    product k v^T added, is read out against (q / sqrt(K)) and stored back --
    every step, the deliberate naivety. All arithmetic is float32; K and V are
    compile-time powers of two. The gate is PER K-CHANNEL: one factor per row.
    """
    pid = tl.program_id(0).to(tl.int64)  # over B*H
    pv = tl.program_id(1).to(tl.int64)  # over V // BV
    b = pid // H
    h = pid % H

    o_k = tl.arange(0, K).to(tl.int64)
    o_v = pv * BV + tl.arange(0, BV).to(tl.int64)

    p_h = St + pid * K * V + o_k[:, None] * V + o_v[None, :]
    b_h = tl.load(p_h)  # float32 state tile [K, BV]

    base = (b * S_len + t) * H + h
    b_q = tl.load(Q + base * K + o_k).to(tl.float32) * scale
    b_k = tl.load(Kk + base * K + o_k).to(tl.float32)
    b_v = tl.load(Vv + base * V + o_v).to(tl.float32)
    b_g = tl.load(Gk + base * K + o_k).to(tl.float32)

    b_h = b_h * tl.exp(b_g)[:, None]  # per-row decay before write
    b_h = b_h + b_k[:, None] * b_v[None, :]  # outer product write
    b_o = tl.sum(b_h * b_q[:, None], 0)  # S^T (q / sqrt(K))

    tl.store(O + base * V + o_v, b_o.to(O.dtype.element_ty))
    tl.store(p_h, b_h)


def paged_gla_decode(
    q: torch.Tensor,  # [B, S, H, K]
    k: torch.Tensor,  # [B, S, H, K]
    v: torch.Tensor,  # [B, S, H, V]
    g: torch.Tensor,  # [B, S, H, K]  float32  per-channel log-forget gate
    page_table: torch.Tensor,  # [B]           int32    physical page id per batch
    paged_state: torch.Tensor,  # [P, H, K, V]  float32  shared pool (P >= B)
) -> tuple:  # (o [B, S, H, V] in v.dtype, state_out [B, H, K, V] fp32)
    """Entry point. Signature and semantics are fixed; the body is yours."""
    B, S, H, K = q.shape
    V = v.shape[-1]
    if (K & (K - 1)) or (V & (V - 1)):
        raise ValueError(f"starter supports power-of-two K and V, got K={K}, V={V}")
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    g = g.contiguous()

    # Slow half A: host-side dense gather of the initial state from the pool.
    # A fast kernel would gather one (batch, head) tile per program directly from
    # the page table without ever materializing a dense [B, H, K, V] scratch.
    idx = page_table.long()
    state = paged_state[idx].to(torch.float32).contiguous().clone()
    o = torch.empty(B, S, H, V, dtype=v.dtype, device=v.device)
    scale = K**-0.5

    BV = 64 if V >= 64 else V
    grid = (B * H, V // BV)
    for t in range(S):  # slow half B: one launch per step
        _step_kernel[grid](q, k, v, g, state, o, t, S, H, scale, K=K, V=V, BV=BV)
    return o, state
