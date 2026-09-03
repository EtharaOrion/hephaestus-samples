"""Private oracle -- from-scratch persistent Triton megakernel for the decode-regime
GROUPED-QUERY DELTA RULE operator.

UNVERIFIED: authored without GPU calibration. Written on a CPU-only host as the
intended shape of the fast solution; it has NOT been run, timed, or numerically
validated against reference.py on the GPU. Full-reward attainment is pending the
GPU `golden` stage.

Algorithm: ONE kernel launch for the whole sequence. The grid is (B*Hkv, V/BV);
each program owns the state tile [K, BV] of one (batch, kv-head) pair for the
entire recurrence, held in registers from the first step to the last. Per step,
ONE delta write is performed (pre-update prediction, delta = (v - pred)*beta,
delta write), then all G = Hq/Hkv query heads of the group read the freshly
updated state tile with their own q vector before the loop advances. The
per-group read fan-out reuses the state tile in registers, so the state is not
re-read from HBM Hq times per step. Everything is float32 in registers.

Determinism is structural: the recurrence over t is serial per program,
programs never share an output element, no atomics anywhere. The group read
order is fixed (0..G-1) so the reduction order across the group is deterministic.

Note on the anchor: there is NO production Hq!=Hkv delta-rule decode kernel;
the timing denominator is a torch.compile of a fused delta step + broadcast
query read (see taskdef.sota).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _scan_kernel(
    Q,
    Kk,
    Vv,
    Beta,
    H0,
    O,
    HT,
    S_len,
    Hq,
    Hkv,
    scale,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    G: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)  # over B*Hkv
    pv = tl.program_id(1).to(tl.int64)  # over V // BV
    b = pid // Hkv
    hk = pid % Hkv

    o_k = tl.arange(0, K).to(tl.int64)
    o_v = pv * BV + tl.arange(0, BV).to(tl.int64)

    p_h0 = H0 + pid * K * V + o_k[:, None] * V + o_v[None, :]
    b_h = tl.load(p_h0)  # resident state [K, BV]
    p_ht = HT + pid * K * V + o_k[:, None] * V + o_v[None, :]

    # Per-step stride bases (kv-head-major streams for k, v, beta).
    base_kv = (b * S_len) * Hkv + hk
    p_k = Kk + base_kv * K + o_k
    p_v = Vv + base_kv * V + o_v
    p_beta = Beta + base_kv

    for _t in range(S_len):
        b_k = tl.load(p_k).to(tl.float32)
        b_v = tl.load(p_v).to(tl.float32)
        b_beta = tl.load(p_beta).to(tl.float32)

        pred = tl.sum(b_h * b_k[:, None], 0)  # pre-update read
        delta = (b_v - pred) * b_beta
        b_h = b_h + b_k[:, None] * delta[None, :]  # delta write

        # G query-head reads over the freshly updated state tile. Loop over G
        # (a compile-time constant), each iteration issues its own q load and
        # store; the state tile stays in registers across every iteration.
        for gi in tl.static_range(G):
            hq = hk * G + gi
            base_q = ((b * S_len + _t) * Hq + hq) * K + o_k
            b_q = tl.load(Q + base_q).to(tl.float32) * scale
            b_o = tl.sum(b_h * b_q[:, None], 0)
            base_o = ((b * S_len + _t) * Hq + hq) * V + o_v
            tl.store(O + base_o, b_o.to(O.dtype.element_ty))

        p_k += Hkv * K
        p_v += Hkv * V
        p_beta += Hkv

    tl.store(p_ht, b_h)


def _tile_v(V: int) -> int:
    return 64 if V >= 64 else V


def gqa_delta_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    state0: torch.Tensor,
) -> tuple:
    B, S, Hq, K = q.shape
    Hkv = k.shape[2]
    V = v.shape[-1]
    if Hq % Hkv:
        raise ValueError(f"Hq ({Hq}) must be a multiple of Hkv ({Hkv})")
    if (K & (K - 1)) or (V & (V - 1)):
        raise ValueError(f"oracle supports power-of-two K and V, got K={K}, V={V}")
    G = Hq // Hkv
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    beta = beta.contiguous()
    h0 = state0.to(torch.float32).contiguous()

    o = torch.empty(B, S, Hq, V, dtype=v.dtype, device=v.device)
    ht = torch.empty(B, Hkv, K, V, dtype=torch.float32, device=v.device)
    scale = K**-0.5

    BV = _tile_v(V)
    grid = (B * Hkv, V // BV)
    _scan_kernel[grid](
        q,
        k,
        v,
        beta,
        h0,
        o,
        ht,
        S,
        Hq,
        Hkv,
        scale,
        K=K,
        V=V,
        BV=BV,
        G=G,
        num_warps=2,
        num_stages=2,
    )
    return o, ht
# FORGE-CANARY-BEGIN forge-canary/v1
# FORGE-CANARY-SLOT-0 2020c1c5f3297c1c30599fd51a06c953764929922a2bd474bf0a5e627cbf8ca8
# FORGE-CANARY-SLOT-1 bfc12e636519eb56dfe1afba9612b77ddb42bdf99604dfc205fd53ba8b20a812
# FORGE-CANARY-SLOT-2 d08683cffec923750b75f3b4e1f88b74cce99be4bc0907474dee958915faf3c2
# FORGE-CANARY-SLOT-3 3fa9e5ac9e26368e5572621167a3d49453ab0411713300a41fdc249337c296ee
# FORGE-CANARY-END
