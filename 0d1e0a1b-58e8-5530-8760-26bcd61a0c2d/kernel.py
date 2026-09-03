"""
Starter -- speculative-decode regime of the gated delta rule, forward only.

This file is yours. It is deliberately basic: one small Triton kernel does one
speculative step for one (batch, head, V-tile), then a host-side Python loop
handles the per-batch commit snapshot by branching on commit_len[b]. Correct,
deterministic, and slow by design. Your job is to make it fast.

The entry point the harness calls is `speculative_gated_delta_decode`, and its
signature and semantics must not change: S speculative steps produce reads `o`
for every position, and `state_out[b]` is a per-batch snapshot at
commit_len[b]. Both graded, state carried in float32.

Why this starter is slow: (1) one launch per timestep pays S launch latencies;
(2) the commit snapshot is a host-side per-batch Python branch; (3) the state
of one (batch, head) pair fits in the registers of a single program for the
whole sequence but is re-read from HBM S times.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _step_kernel(
    Q,
    Kk,
    Vv,
    Gg,
    Beta,
    St,
    StSnap,
    O,
    CL,
    t,
    S_len,
    H,
    scale,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
):
    """One speculative step for one (batch*head, V-tile). Applies the gated
    delta write, computes the read, and if commit_len[b] == t + 1 also snaps
    the state into StSnap for that batch (branch inside the kernel keeps the
    snapshot path off the host)."""
    pid = tl.program_id(0).to(tl.int64)  # over B*H
    pv = tl.program_id(1).to(tl.int64)  # over V // BV
    b = pid // H
    h = pid % H

    o_k = tl.arange(0, K).to(tl.int64)
    o_v = pv * BV + tl.arange(0, BV).to(tl.int64)

    p_h = St + pid * K * V + o_k[:, None] * V + o_v[None, :]
    b_h = tl.load(p_h)

    base = (b * S_len + t) * H + h
    b_q = tl.load(Q + base * K + o_k).to(tl.float32) * scale
    b_k = tl.load(Kk + base * K + o_k).to(tl.float32)
    b_v = tl.load(Vv + base * V + o_v).to(tl.float32)
    b_g = tl.load(Gg + base).to(tl.float32)
    b_beta = tl.load(Beta + base).to(tl.float32)

    b_h = b_h * tl.exp(b_g)  # scalar gate before write
    pred = tl.sum(b_h * b_k[:, None], 0)
    delta = (b_v - pred) * b_beta
    b_h = b_h + b_k[:, None] * delta[None, :]

    b_o = tl.sum(b_h * b_q[:, None], 0)
    tl.store(O + base * V + o_v, b_o.to(O.dtype.element_ty))
    tl.store(p_h, b_h)

    # In-kernel per-batch snapshot: cheap for the starter (each program does
    # one load, one compare, one conditional store), still S launches per call.
    cl = tl.load(CL + b).to(tl.int32)
    if cl == (t + 1):
        p_snap = StSnap + pid * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_snap, b_h)


def speculative_gated_delta_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    commit_len: torch.Tensor,
    state0: torch.Tensor,
) -> tuple:
    """Entry point. Signature and semantics are fixed; the body is yours."""
    B, S, H, K = q.shape
    V = v.shape[-1]
    if (K & (K - 1)) or (V & (V - 1)):
        raise ValueError(f"starter supports power-of-two K and V, got K={K}, V={V}")
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    g, beta = g.contiguous(), beta.contiguous()
    cl = commit_len.to(torch.int32).contiguous()

    state = state0.to(torch.float32).contiguous().clone()
    # state_out starts as state0 so a batch with commit_len == 0 stays put.
    state_snap = state0.to(torch.float32).contiguous().clone()
    o = torch.empty(B, S, H, V, dtype=v.dtype, device=v.device)
    scale = K**-0.5

    BV = 64 if V >= 64 else V
    grid = (B * H, V // BV)
    for t in range(S):
        _step_kernel[grid](
            q, k, v, g, beta, state, state_snap, o, cl, t, S, H, scale, K=K, V=V, BV=BV
        )
    return o, state_snap
