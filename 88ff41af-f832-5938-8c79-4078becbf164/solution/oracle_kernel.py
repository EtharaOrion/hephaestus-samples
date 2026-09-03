"""Private oracle -- from-scratch persistent Triton megakernel for the decode-regime
Mamba-2 SSD selective state update.

UNVERIFIED: authored without GPU calibration. Written on a CPU-only host as the
intended shape of the fast solution; it has NOT been run, timed, or numerically
validated against reference.py on the GPU. GPU calibration (measure / golden /
controls) is a required follow-up before any number here is trusted.

Never shipped to an agent. Establishes that full reward is reachable through every
live gate (G1 symbol scan, G2 dual-output correctness, G3 timing, G4 adoption) with
no library call.

Algorithm: ONE kernel launch for the whole sequence. The grid is (B*H, P/BP); each
program owns the state tile [BP, N] of one (batch, head) pair for the entire
recurrence, held in registers from the first step to the last. The N axis (the read
reduction axis for y = state C) is never tiled. Per step it loads x_t (the P-tile),
the two selective vectors B_t, C_t (length N), and the scalars dt_t / A / D / dt_bias;
discretizes the time step; decays the state by the selective scalar dA; adds the
rank-1 selective write; reads out against C with the D skip; emits the step output;
and advances its input pointers by one step's stride. The final state is written
exactly once, after the last step. Everything is float32 in registers.

Determinism is structural: the recurrence over t is serial inside one program,
programs never share an output element, and there are no atomics anywhere.

Note on the anchor: there is NO production FLA fused_recurrent kernel for this
operator, so the timing denominator is a torch.compile of a dense sequential scan
(see taskdef.sota). This is the hardest member of the family -- largest state, most
selective parameter streams -- so the reachable fraction is expected to be the
lowest; the tile heuristic below is an UNVERIFIED starting point for the sweep.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _scan_kernel(
    X, Dt, Alog, Bsel, Csel, Dskip, DtBias, H0, O, HT,
    S_len, H,
    P: tl.constexpr, N: tl.constexpr, BP: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)          # over B*H
    pp = tl.program_id(1).to(tl.int64)           # over P // BP
    b = pid // H
    h = pid % H

    o_p = pp * BP + tl.arange(0, BP).to(tl.int64)
    o_n = tl.arange(0, N).to(tl.int64)

    p_h0 = H0 + pid * P * N + o_p[:, None] * N + o_n[None, :]
    b_s = tl.load(p_h0)                           # resident for the whole sequence
    p_ht = HT + pid * P * N + o_p[:, None] * N + o_n[None, :]

    # Head-constant scalars: loaded once, outside the step loop.
    b_alog = tl.load(Alog + h).to(tl.float32)
    b_D = tl.load(Dskip + h).to(tl.float32)
    b_dtb = tl.load(DtBias + h).to(tl.float32)
    A = -tl.exp(b_alog)                           # scalar per head, < 0

    base = (b * S_len) * H + h
    p_x = X + base * P + o_p
    p_B = Bsel + base * N + o_n
    p_C = Csel + base * N + o_n
    p_dt = Dt + base
    p_o = O + base * P + o_p

    for _t in range(S_len):
        b_x = tl.load(p_x).to(tl.float32)
        b_B = tl.load(p_B).to(tl.float32)
        b_C = tl.load(p_C).to(tl.float32)
        b_dt = tl.load(p_dt).to(tl.float32)

        z = b_dt + b_dtb
        az = tl.maximum(z, -z)                    # |z|
        dt_eff = tl.maximum(z, 0.0) + tl.log(1.0 + tl.exp(-az))   # softplus, stable
        dA = tl.exp(dt_eff * A)
        dtx = dt_eff * b_x

        b_s = b_s * dA + dtx[:, None] * b_B[None, :]
        b_y = tl.sum(b_s * b_C[None, :], 1) + b_D * b_x
        tl.store(p_o, b_y.to(O.dtype.element_ty))

        p_x += H * P
        p_B += H * N
        p_C += H * N
        p_dt += H
        p_o += H * P

    tl.store(p_ht, b_s)


def _tile_p(P: int, N: int) -> int:
    bp = 32 if N < 128 else 16
    if bp > P:
        bp = P
    return bp


def mamba2_ssd_decode(x, dt, A_log, B, C, D, dt_bias, state0) -> tuple:
    Bb, S, H, P = x.shape
    N = B.shape[-1]
    if (P & (P - 1)) or (N & (N - 1)):
        raise ValueError(f"oracle supports power-of-two P and N, got P={P}, N={N}")
    x, B, C = x.contiguous(), B.contiguous(), C.contiguous()
    dt, A_log = dt.contiguous(), A_log.contiguous()
    D, dt_bias = D.contiguous(), dt_bias.contiguous()
    h0 = state0.to(torch.float32).contiguous()

    o = torch.empty(Bb, S, H, P, dtype=x.dtype, device=x.device)
    ht = torch.empty(Bb, H, P, N, dtype=torch.float32, device=x.device)

    BP = _tile_p(P, N)
    grid = (Bb * H, P // BP)
    _scan_kernel[grid](x, dt, A_log, B, C, D, dt_bias, h0, o, ht,
                       S, H, P=P, N=N, BP=BP,
                       num_warps=1, num_stages=2)
    return o, ht
# FORGE-CANARY-BEGIN forge-canary/v1
# FORGE-CANARY-SLOT-0 e91c89c5b933e54ead0235cd801ea84738f4c76ff3e70d93ad31d8ec4d4da3b0
# FORGE-CANARY-SLOT-1 7850bfa701dbab19a3499ffa9296d72b49359e91fb8cdc0eed9bed7f701ad10d
# FORGE-CANARY-SLOT-2 541f7cad29753300209c4f5d955e762fa54c4f331a1e999d1bb17073d8ffcfe9
# FORGE-CANARY-SLOT-3 d47c7437729060ca2301d0ea7a60dbc4ad9e0c8c02aafad5654d677d0f6d3194
# FORGE-CANARY-END
