"""Private oracle -- from-scratch persistent Triton megakernel for the fused
PER-CHANNEL-DECAY + BETA-DELTA linear-attention decode operator.

UNVERIFIED: authored without GPU calibration. Full-reward attainment is pending
the GPU `golden` stage.

Algorithm: ONE kernel launch for the whole sequence. The grid is (B*H, V/BV);
each program owns the state tile [K, BV] of one (batch, head) pair for the
entire recurrence in registers. Per step: load q, k, v, g (length K, per-channel),
beta (scalar); apply per-channel decay `exp(g)` to each row of the state; pre-
write prediction reduces along K; delta = (v - pred) * beta; delta write; post-
write read against (q / sqrt(K)) and emit. The per-channel gate and both
reductions run on the SAME resident tile, sharing register state across the
pre-write and post-write reads.

Determinism: recurrence over t is serial per program, programs never share an
output element, no atomics anywhere.

Note on the anchor: no vendor kernel fuses per-channel decay AND beta-delta
in one step; the timing denominator is a torch.compile of the fused hybrid
step (see taskdef.sota).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _scan_kernel(
    Q,
    Kk,
    Vv,
    Gk,
    Beta,
    H0,
    O,
    HT,
    S_len,
    H,
    scale,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    pv = tl.program_id(1).to(tl.int64)
    b = pid // H
    h = pid % H

    o_k = tl.arange(0, K).to(tl.int64)
    o_v = pv * BV + tl.arange(0, BV).to(tl.int64)

    p_h0 = H0 + pid * K * V + o_k[:, None] * V + o_v[None, :]
    b_h = tl.load(p_h0)
    p_ht = HT + pid * K * V + o_k[:, None] * V + o_v[None, :]

    base = (b * S_len) * H + h
    p_q = Q + base * K + o_k
    p_k = Kk + base * K + o_k
    p_v = Vv + base * V + o_v
    p_g = Gk + base * K + o_k
    p_beta = Beta + base
    p_o = O + base * V + o_v

    for _t in range(S_len):
        b_q = tl.load(p_q).to(tl.float32) * scale
        b_k = tl.load(p_k).to(tl.float32)
        b_v = tl.load(p_v).to(tl.float32)
        b_g = tl.load(p_g).to(tl.float32)
        b_beta = tl.load(p_beta).to(tl.float32)

        b_h = b_h * tl.exp(b_g)[:, None]  # per-channel row decay
        pred = tl.sum(b_h * b_k[:, None], 0)  # pre-write read
        delta = (b_v - pred) * b_beta
        b_h = b_h + b_k[:, None] * delta[None, :]  # delta write
        b_o = tl.sum(b_h * b_q[:, None], 0)  # post-write read
        tl.store(p_o, b_o.to(O.dtype.element_ty))

        p_q += H * K
        p_k += H * K
        p_v += H * V
        p_g += H * K
        p_beta += H
        p_o += H * V

    tl.store(p_ht, b_h)


def _tile_v(V: int) -> int:
    return 64 if V >= 64 else V


def rwkv7_channel_delta_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state0: torch.Tensor,
) -> tuple:
    B, S, H, K = q.shape
    V = v.shape[-1]
    if (K & (K - 1)) or (V & (V - 1)):
        raise ValueError(f"oracle supports power-of-two K and V, got K={K}, V={V}")
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    g, beta = g.contiguous(), beta.contiguous()
    h0 = state0.to(torch.float32).contiguous()

    o = torch.empty(B, S, H, V, dtype=v.dtype, device=v.device)
    ht = torch.empty(B, H, K, V, dtype=torch.float32, device=v.device)
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
        ht,
        S,
        H,
        scale,
        K=K,
        V=V,
        BV=BV,
        num_warps=2,
        num_stages=2,
    )
    return o, ht
# FORGE-CANARY-BEGIN forge-canary/v1
# FORGE-CANARY-SLOT-0 6a93dd99f533b5c6800885bdc91abae3577c184ec6ac7d5d1538ba393cb1586b
# FORGE-CANARY-SLOT-1 51e9e8fc2683454d7f16402bbe50e1981b07ea084445591440ab51fa1846cdb2
# FORGE-CANARY-SLOT-2 8f204779bb9fc59e65b62fc9125a4a4eeec5d12fbc9215ea326b216233640f7c
# FORGE-CANARY-SLOT-3 cb4d5b3ad630abb8290b0578178e274afdcff3fc29cb5a83ec2322d6cf4fb6df
# FORGE-CANARY-END
