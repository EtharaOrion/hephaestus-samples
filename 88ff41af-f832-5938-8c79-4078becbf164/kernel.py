"""
Starter -- decode-regime Mamba-2 SSD selective state update, forward only.

This file is yours. It is deliberately basic: one small Triton kernel does all the
real work of one selective-scan step -- discretize the time step, decay the state by
the selective scalar dA, add the rank-1 selective write, read out against C with the
D skip -- and it does it the slowest reasonable way, launched once per timestep with
the full [P, N] state round-tripping through HBM on every one of the S steps. Correct,
deterministic, and slow by design. Your job is to make it fast.

The entry point the harness calls is `mamba2_ssd_decode`, and its signature and
semantics must not change: S sequential selective-state-space steps starting from
`state0`, both the per-step outputs `o` AND the final state returned. reference.py
defines those semantics; agreement with it is graded on both outputs, and the state
must be carried in float32.

The state here is a full [P (head dim) x N (state dim)] matrix per head, and every
step consumes five selective parameter streams (dt, A, B, C, D) -- the most per-step
traffic and arithmetic of any operator in this family.

Why this starter is slow, in one sentence: a serial recurrence run as S separate
kernel launches pays launch latency S times and re-reads the entire state from HBM S
times, while the state of one (batch, head) pair fits in the registers of a single
program for the whole sequence.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _step_kernel(
    X, Dt, Alog, Bsel, Csel, Dskip, DtBias, St, O,
    t, S_len, H,
    P: tl.constexpr, N: tl.constexpr, BP: tl.constexpr,
):
    """One selective-scan step for one (batch*head, P-tile) pair.

    The state tile [BP, N] is loaded, decayed by the scalar dA, gets the rank-1
    selective write (dt*x) B^T added, is read out against C with the D skip, and
    stored back -- every step, the deliberate naivety. All arithmetic is float32;
    P and N are compile-time powers of two. The N axis is the in-program
    reduction axis for the y = state C read, so it is never tiled.
    """
    pid = tl.program_id(0).to(tl.int64)          # over B*H
    pp = tl.program_id(1).to(tl.int64)           # over P // BP
    b = pid // H
    h = pid % H

    o_p = pp * BP + tl.arange(0, BP).to(tl.int64)
    o_n = tl.arange(0, N).to(tl.int64)

    p_state = St + pid * P * N + o_p[:, None] * N + o_n[None, :]
    b_s = tl.load(p_state)                        # float32 state tile [BP, N]

    base = (b * S_len + t) * H + h
    b_x = tl.load(X + base * P + o_p).to(tl.float32)        # [BP]
    b_B = tl.load(Bsel + base * N + o_n).to(tl.float32)     # [N]
    b_C = tl.load(Csel + base * N + o_n).to(tl.float32)     # [N]
    b_dt = tl.load(Dt + base).to(tl.float32)               # scalar
    b_alog = tl.load(Alog + h).to(tl.float32)              # scalar per head
    b_D = tl.load(Dskip + h).to(tl.float32)                # scalar per head
    b_dtb = tl.load(DtBias + h).to(tl.float32)             # scalar per head

    # dt = softplus(dt_raw + dt_bias), stable & branchless: max(z,0)+log1p(exp(-|z|))
    z = b_dt + b_dtb
    az = tl.maximum(z, -z)                        # |z|
    dt_eff = tl.maximum(z, 0.0) + tl.log(1.0 + tl.exp(-az))
    A = -tl.exp(b_alog)                           # scalar per head, < 0
    dA = tl.exp(dt_eff * A)                        # scalar in (0, 1)
    dtx = dt_eff * b_x                            # [BP] discretized input

    b_s = b_s * dA + dtx[:, None] * b_B[None, :]  # [BP, N] decay + rank-1 write
    b_y = tl.sum(b_s * b_C[None, :], 1) + b_D * b_x   # [BP] read over N + skip

    tl.store(O + base * P + o_p, b_y.to(O.dtype.element_ty))
    tl.store(p_state, b_s)


def mamba2_ssd_decode(
    x: torch.Tensor,        # [B, S, H, P]  per-head input / value
    dt: torch.Tensor,       # [B, S, H]     float32 raw time-step (pre-softplus)
    A_log: torch.Tensor,    # [H]           float32 log of -A
    B: torch.Tensor,        # [B, S, H, N]  selective input vector
    C: torch.Tensor,        # [B, S, H, N]  selective output vector
    D: torch.Tensor,        # [H]           float32 input skip, scalar per head
    dt_bias: torch.Tensor,  # [H]           float32 time-step bias, scalar per head
    state0: torch.Tensor,   # [B, H, P, N]  float32 initial state
) -> tuple:                 # (o [B, S, H, P] in x.dtype, state_out [B, H, P, N] float32)
    """Entry point. Signature and semantics are fixed; the body is yours."""
    Bb, S, H, P = x.shape
    N = B.shape[-1]
    if (P & (P - 1)) or (N & (N - 1)):
        raise ValueError(f"starter supports power-of-two P and N, got P={P}, N={N}")
    x, B, C = x.contiguous(), B.contiguous(), C.contiguous()
    dt, A_log = dt.contiguous(), A_log.contiguous()
    D, dt_bias = D.contiguous(), dt_bias.contiguous()

    state = state0.to(torch.float32).contiguous().clone()   # never mutate the caller
    o = torch.empty(Bb, S, H, P, dtype=x.dtype, device=x.device)

    BP = 32 if N < 128 else 16
    if BP > P:
        BP = P
    grid = (Bb * H, P // BP)
    for t in range(S):                            # the slow part: one launch per step
        _step_kernel[grid](x, dt, A_log, B, C, D, dt_bias, state, o,
                           t, S, H, P=P, N=N, BP=BP)
    return o, state
