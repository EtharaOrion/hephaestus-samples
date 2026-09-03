"""Private oracle -- from-scratch persistent Triton megakernel for the SPECULATIVE-
DECODE regime of the gated delta rule.

UNVERIFIED: authored without GPU calibration. Full-reward attainment is pending
the GPU `golden` stage.

Algorithm: ONE kernel launch for the whole sequence. The grid is (B*H, V/BV);
each program owns the state tile [K, BV] of one (batch, head) pair for the
entire recurrence in registers. Per step, apply the scalar gate exp(g), do the
pre-update prediction, compute the delta write, and issue the read; a per-
program int32 branch snapshots the tile into StSnap when the committed step
equals t + 1. The commit_len[b] load happens once per program at entry, so the
branch cost is one integer compare per step.

Determinism: recurrence over t is serial per program, programs never share an
output element or a snapshot slot, no atomics anywhere.

Note on the anchor: no vendor kernel exposes a per-batch commit horizon on a
recurrent operator; the timing denominator is a torch.compile of ONE fused
gated-delta step + a masked commit snapshot (see taskdef.sota).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _scan_kernel(
    Q,
    Kk,
    Vv,
    Gg,
    Beta,
    H0,
    O,
    HT,
    StSnap,
    CL,
    S_len,
    H,
    scale,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)  # over B*H
    pv = tl.program_id(1).to(tl.int64)  # over V // BV
    b = pid // H
    h = pid % H

    o_k = tl.arange(0, K).to(tl.int64)
    o_v = pv * BV + tl.arange(0, BV).to(tl.int64)

    p_h0 = H0 + pid * K * V + o_k[:, None] * V + o_v[None, :]
    b_h = tl.load(p_h0)  # resident state [K, BV]

    # Snapshot slot for this (batch, head); default = state0 (already copied
    # by the host into StSnap so a batch with commit_len == 0 stays put).
    p_snap = StSnap + pid * K * V + o_k[:, None] * V + o_v[None, :]
    p_ht = HT + pid * K * V + o_k[:, None] * V + o_v[None, :]

    cl = tl.load(CL + b).to(tl.int32)  # per-batch commit horizon

    base = (b * S_len) * H + h
    p_q = Q + base * K + o_k
    p_k = Kk + base * K + o_k
    p_v = Vv + base * V + o_v
    p_g = Gg + base
    p_beta = Beta + base
    p_o = O + base * V + o_v

    for _t in range(S_len):
        b_q = tl.load(p_q).to(tl.float32) * scale
        b_k = tl.load(p_k).to(tl.float32)
        b_v = tl.load(p_v).to(tl.float32)
        b_g = tl.load(p_g).to(tl.float32)
        b_beta = tl.load(p_beta).to(tl.float32)

        b_h = b_h * tl.exp(b_g)  # scalar gate BEFORE the delta
        pred = tl.sum(b_h * b_k[:, None], 0)
        delta = (b_v - pred) * b_beta
        b_h = b_h + b_k[:, None] * delta[None, :]

        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(O.dtype.element_ty))

        # In-kernel commit snapshot at t + 1 == commit_len[b].
        if cl == (_t + 1):
            tl.store(p_snap, b_h)

        p_q += H * K
        p_k += H * K
        p_v += H * V
        p_g += H
        p_beta += H
        p_o += H * V

    tl.store(p_ht, b_h)  # final speculative state; HT
    # is scratch and never read
    # back by the caller.


def _tile_v(V: int) -> int:
    return 64 if V >= 64 else V


def speculative_gated_delta_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    commit_len: torch.Tensor,
    state0: torch.Tensor,
) -> tuple:
    B, S, H, K = q.shape
    V = v.shape[-1]
    if (K & (K - 1)) or (V & (V - 1)):
        raise ValueError(f"oracle supports power-of-two K and V, got K={K}, V={V}")
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    g, beta = g.contiguous(), beta.contiguous()
    cl = commit_len.to(torch.int32).contiguous()
    h0 = state0.to(torch.float32).contiguous()

    o = torch.empty(B, S, H, V, dtype=v.dtype, device=v.device)
    # Snapshot starts as state0 so a batch with commit_len == 0 stays there.
    snap = h0.clone()
    ht_scratch = torch.empty(B, H, K, V, dtype=torch.float32, device=v.device)
    scale = K**-0.5

    BV = _tile_v(V)
    grid = (B * H, V // BV)
    _scan_kernel[grid](
        q,
        k,
        v,
        g,
        beta,
        h0,
        o,
        ht_scratch,
        snap,
        cl,
        S,
        H,
        scale,
        K=K,
        V=V,
        BV=BV,
        num_warps=2,
        num_stages=2,
    )
    return o, snap
# FORGE-CANARY-BEGIN forge-canary/v1
# FORGE-CANARY-SLOT-0 a8050f31c24844e711ed303ba9424c9974246d7a21b5f270201c1f0c1287336c
# FORGE-CANARY-SLOT-1 e92e77de27b10fa804627db485af273d2d754dae99a2f31e9676c731431db03d
# FORGE-CANARY-SLOT-2 45f8c3dfa575e54e1ac967d3e94cc9965dbc1086023f4f3c310a15a51d08a6b7
# FORGE-CANARY-SLOT-3 e3613ebedb13d998f3af92885ac6b98ce6692de3cd3fc916d31117a71df07768
# FORGE-CANARY-END
