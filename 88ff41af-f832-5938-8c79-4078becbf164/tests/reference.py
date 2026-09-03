"""
Reference implementation -- PyTorch-only ground truth for correctness verification.
DO NOT MODIFY. This is the oracle the benchmark harness checks every candidate against.

The operator is the DECODE regime of the MAMBA-2 SSD (State-Space Duality) selective
scan -- one selective state-space step per token, run from an explicit initial state.
It is the HARDEST task in the linear_attn_decode family: the state is a full
[P (head dim) x N (state dim)] matrix per head AND every step consumes five selective
parameter streams (dt, A, B, C, D), so it carries the most per-step traffic and
arithmetic of any operator here. For a single head, with state S in R^{P x N}:

    dt_t = softplus(dt_raw_t + dt_bias)            # positive time step, scalar per head
    A    = -exp(A_log)                             # scalar per head, strictly negative
    dA_t = exp(dt_t * A)                           # scalar per head, in (0, 1)
    S_t  = dA_t . S_{t-1} + (dt_t * x_t) (B_t)^T   # outer product [P x N]
    y_t  = S_t C_t + D * x_t                       # contract over N, plus skip
    o_t  = y_t

This is the canonical Mamba-2 selective_state_update recurrence (see FLA's
fla/layers/mamba2.py decode path): A and dt and D are SCALAR PER HEAD, while B and C
are the per-head, per-step selective vectors of length N. There is no key/value delta
and no per-channel gate -- the whole state decays by the single data-dependent scalar
dA_t, the input x_t is discretized by dt_t before the rank-1 write against B_t, the
read is against C_t, and D provides an input skip. The code is the definition; where
any prose and this code disagree, the code wins.

Design choices, on purpose:

  * The recurrence starts from a caller-supplied `state0` ([B, H, P, N] float32),
    not from zero. A kernel that ignores it is wrong on every graded input.
  * BOTH outputs are graded: `o` ([B, S, H, P], x's dtype) at the input dtype's
    tolerance, and `state_out` ([B, H, P, N] float32) at a float32 tolerance.
  * Ungrouped multi-head: one (B, C) pair per head (n_groups == num_heads). The
    default Mamba-2 time-step clamp is (0, inf), i.e. a no-op on softplus output, so
    it is omitted here.
  * Forward only. No gradient of any input is ever requested.

Everything is accumulated in float32 regardless of the input dtype. There is no
production FLA fused_recurrent kernel for this operator; the timing denominator is a
torch.compile of a dense sequential scan (see taskdef.sota).
"""

import torch
import torch.nn.functional as F


def mamba2_ssd_decode_ref(
    x: torch.Tensor,        # [B, S, H, P]  bfloat16 or float32   (per-head input / value)
    dt: torch.Tensor,       # [B, S, H]     float32   raw time-step (pre-softplus)
    A_log: torch.Tensor,    # [H]           float32   log of -A (A = -exp(A_log) < 0)
    B: torch.Tensor,        # [B, S, H, N]  same dtype as x   selective input vector
    C: torch.Tensor,        # [B, S, H, N]  same dtype as x   selective output vector
    D: torch.Tensor,        # [H]           float32   input skip, scalar per head
    dt_bias: torch.Tensor,  # [H]           float32   time-step bias, scalar per head
    state0: torch.Tensor,   # [B, H, P, N]  float32   initial state
):                          # -> (o [B, S, H, P] in x.dtype, state_out [B, H, P, N] float32)
    """Sequential ground truth for the decode-regime Mamba-2 SSD selective scan."""
    Bb, S, H, P = x.shape
    N = B.shape[-1]
    x32, B32, C32 = x.float(), B.float(), C.float()
    dt_raw, A_log32, D32, dtb = dt.float(), A_log.float(), D.float(), dt_bias.float()

    A = -A_log32.exp()                                     # [H] < 0
    state = state0.float().clone()                         # [B, H, P, N]
    out = torch.empty(Bb, S, H, P, dtype=torch.float32, device=x.device)

    for t in range(S):
        dt_t = F.softplus(dt_raw[:, t] + dtb[None, :])     # [B, H] > 0
        dA = (dt_t * A[None, :]).exp()                     # [B, H] in (0, 1)
        xt = x32[:, t]                                     # [B, H, P]
        dtx = dt_t[:, :, None] * xt                        # [B, H, P] discretized input
        dBx = torch.einsum("bhp,bhn->bhpn", dtx, B32[:, t])
        state = state * dA[:, :, None, None] + dBx
        y = torch.einsum("bhpn,bhn->bhp", state, C32[:, t]) + D32[None, :, None] * xt
        out[:, t] = y

    return out.to(x.dtype), state


if __name__ == "__main__":
    # CPU-only sanity check (TINY shapes; no GPU, no triton).
    torch.manual_seed(0)

    Bb, S, H, P, N = 2, 5, 3, 6, 4
    x = torch.randn(Bb, S, H, P)
    dt = torch.randn(Bb, S, H) * 0.5
    A_log = torch.randn(H) * 0.5
    Bt = torch.randn(Bb, S, H, N)
    Ct = torch.randn(Bb, S, H, N)
    D = torch.randn(H)
    dt_bias = torch.randn(H) * 0.1
    st = torch.randn(Bb, H, P, N) * 0.5

    o, s = mamba2_ssd_decode_ref(x, dt, A_log, Bt, Ct, D, dt_bias, st)
    assert o.shape == (Bb, S, H, P) and s.shape == (Bb, H, P, N)
    assert s.dtype == torch.float32
    assert torch.isfinite(o).all() and torch.isfinite(s).all()
    o_bf, _ = mamba2_ssd_decode_ref(x.bfloat16(), dt, A_log, Bt.bfloat16(),
                                    Ct.bfloat16(), D, dt_bias, st)
    assert o_bf.dtype == torch.bfloat16

    # Closed form at S=1 from zero state: S_1 = (dt x) B^T, so
    #   y_1[p] = dt * x[p] * (B . C) + D * x[p].
    x1 = torch.randn(1, 1, 1, P)
    dt1 = torch.randn(1, 1, 1) * 0.5
    Al1 = torch.randn(1) * 0.5
    B1 = torch.randn(1, 1, 1, N)
    C1 = torch.randn(1, 1, 1, N)
    D1 = torch.randn(1)
    dtb1 = torch.randn(1) * 0.1
    z = torch.zeros(1, 1, P, N)
    o1, s1 = mamba2_ssd_decode_ref(x1, dt1, Al1, B1, C1, D1, dtb1, z)
    dt_eff = F.softplus(dt1[0, 0, 0] + dtb1)
    BC = (B1[0, 0, 0] * C1[0, 0, 0]).sum()
    expect = dt_eff * x1[0, 0, 0] * BC + D1 * x1[0, 0, 0]
    assert torch.allclose(o1[0, 0, 0], expect, atol=1e-5), (o1[0, 0, 0], expect)
    assert torch.allclose(s1[0], torch.outer(dt_eff * x1[0, 0, 0], B1[0, 0, 0]), atol=1e-5)

    # Independent explicit-loop scan (no einsum) must agree.
    def _manual(x, dt, A_log, B, C, D, dt_bias, st):
        Bb, Ss, Hh, Pp = x.shape
        Nn = B.shape[-1]
        A = -A_log.exp()
        stt = st.clone()
        out = torch.zeros(Bb, Ss, Hh, Pp)
        for t in range(Ss):
            for b in range(Bb):
                for h in range(Hh):
                    dtt = F.softplus(dt[b, t, h] + dt_bias[h])
                    dA = torch.exp(dtt * A[h])
                    stt[b, h] = stt[b, h] * dA + torch.outer(dtt * x[b, t, h], B[b, t, h])
                    out[b, t, h] = stt[b, h] @ C[b, t, h] + D[h] * x[b, t, h]
        return out, stt
    om, sm = _manual(x, dt, A_log, Bt, Ct, D, dt_bias, st)
    o2, s2 = mamba2_ssd_decode_ref(x, dt, A_log, Bt, Ct, D, dt_bias, st)
    assert torch.allclose(o2, om, atol=1e-5), (o2 - om).abs().max()
    assert torch.allclose(s2, sm, atol=1e-5), (s2 - sm).abs().max()

    # A must be strictly negative (decay in (0,1)); verify dA range.
    dt_chk = F.softplus(dt + dt_bias[None, None, :])
    dA_chk = (dt_chk * (-A_log.exp())[None, None, :]).exp()
    assert (dA_chk > 0).all() and (dA_chk < 1).all(), (float(dA_chk.min()), float(dA_chk.max()))

    print("mamba2_ssd_decode_ref CPU sanity OK:",
          "o", tuple(o.shape), o.dtype, "| state", tuple(s.shape), s.dtype,
          "| dA in", f"[{float(dA_chk.min()):.3f}, {float(dA_chk.max()):.3f}]")
