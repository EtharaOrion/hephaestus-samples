"""Private oracle -- from-scratch persistent Triton megakernel for the decode-regime
PAGED GATED LINEAR ATTENTION operator.

UNVERIFIED: authored without GPU calibration. Written on a CPU-only host as the
intended shape of the fast solution; it has NOT been run, timed, or numerically
validated against reference.py on the GPU. Full-reward attainment is pending the
GPU `golden` stage (measure -> golden -> smoke -> controls); the negative-control
anchor strings in taskdef.py target the STARTER's bytes here, so oracle-base
controls are not exercised for this operator.

Never shipped to an agent. Establishes the intended shape of the full-reward
solution through every live gate (G1 symbol scan, G2 dual-output correctness at
input-dtype and float32 tolerances, G3 timing against the torch.compile fused
step, G4 adoption floor) with no forbidden call.

Algorithm: ONE kernel launch for the whole sequence. The grid is (B*H, V/BV);
each program owns the state tile [K, BV] of one (batch, head) pair for the
entire recurrence, held in registers from the first step to the last. Per-batch
initial state is gathered IN-KERNEL from the paged pool via `page_table[b]`,
without any dense scratch materialization. The K axis is the read reduction
axis for y = state^T (q / sqrt(K)) so it is never tiled. Per step it loads q_t,
k_t (length K), v_t (length BV) and g_t (length K); applies the per-channel
row-decay `exp(g)`; adds the outer product k v^T; issues the read; emits the
step output; and advances its input pointers by one step's stride. The final
state is written exactly once, after the last step. Everything is float32 in
registers.

Determinism is structural: the recurrence over t is serial inside one program,
programs never share an output element, and there are no atomics anywhere.

Note on the anchor: there is NO production kernel of a paged-state linear-
attention decode operator, so the timing denominator is a torch.compile of a
fused GLA step in a dense decode loop with a fancy-index gather at entry (see
taskdef.sota). The tile heuristic below is an UNVERIFIED starting point for
the sweep.
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
    Pt,
    Pool,
    O,
    HT,
    S_len,
    H,
    P_pages,
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

    # In-kernel gather: read page id from page_table, then read the [K, BV]
    # state tile for (page_id, h) directly from the pool. No dense scratch.
    page_id = tl.load(Pt + b).to(tl.int64)
    p_pool = Pool + (page_id * H + h) * K * V + o_k[:, None] * V + o_v[None, :]
    b_h = tl.load(p_pool)  # float32 state tile [K, BV]

    # Per-batch final-state output slot (dense [B, H, K, V]).
    p_ht = HT + pid * K * V + o_k[:, None] * V + o_v[None, :]

    # Stride bases into the per-step streams.
    base = (b * S_len) * H + h
    p_q = Q + base * K + o_k
    p_k = Kk + base * K + o_k
    p_v = Vv + base * V + o_v
    p_g = Gk + base * K + o_k
    p_o = O + base * V + o_v

    for _t in range(S_len):
        b_q = tl.load(p_q).to(tl.float32) * scale
        b_k = tl.load(p_k).to(tl.float32)
        b_v = tl.load(p_v).to(tl.float32)
        b_g = tl.load(p_g).to(tl.float32)

        b_h = b_h * tl.exp(b_g)[:, None]  # per-channel row decay
        b_h = b_h + b_k[:, None] * b_v[None, :]  # outer-product write
        b_o = tl.sum(b_h * b_q[:, None], 0)  # read over K
        tl.store(p_o, b_o.to(O.dtype.element_ty))

        p_q += H * K
        p_k += H * K
        p_v += H * V
        p_g += H * K
        p_o += H * V

    tl.store(p_ht, b_h)


def _tile_v(V: int) -> int:
    bv = 64 if V >= 64 else V
    return bv


def paged_gla_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    page_table: torch.Tensor,
    paged_state: torch.Tensor,
) -> tuple:
    B, S, H, K = q.shape
    V = v.shape[-1]
    if (K & (K - 1)) or (V & (V - 1)):
        raise ValueError(f"oracle supports power-of-two K and V, got K={K}, V={V}")
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    g = g.contiguous()
    pt = page_table.to(torch.int32).contiguous()
    pool = paged_state.to(torch.float32).contiguous()

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
        pt,
        pool,
        o,
        ht,
        S,
        H,
        pool.shape[0],
        scale,
        K=K,
        V=V,
        BV=BV,
        num_warps=2,
        num_stages=2,
    )
    return o, ht
# FORGE-CANARY-BEGIN forge-canary/v1
# FORGE-CANARY-SLOT-0 835fa0061c214984a01a76f3a06dbedd4d863b282964d54b8dc3e00f4aeee3f1
# FORGE-CANARY-SLOT-1 b2e9295f205f11f4cbf7fb5808ed375d1bdfd797becab32e76a87104eb1f1dee
# FORGE-CANARY-SLOT-2 77935e574cff55c4ecc99d63886b3fccb324d9daa7b48c6454fb1f7d2d8b9903
# FORGE-CANARY-SLOT-3 cf6170bfc4ad891d6d0fb6a82a5b5669a0887be24c088ab861e09de947293fab
# FORGE-CANARY-END
