"""Private oracle -- from-scratch persistent Triton megakernel for the INT8-STATE
delta-rule decode operator (int8 boundary, fp32 accumulation inside).

UNVERIFIED: authored without GPU calibration. Full-reward attainment is pending
the GPU `golden` stage.

Algorithm: ONE kernel launch for the whole sequence. The grid is (B*H, V/BV);
each program owns the state tile [K, BV] of one (batch, head) pair for the
entire recurrence in registers. On entry the program loads state0_q (int8) and
state0_scale (fp32 scalar per-(batch, head)) and dequantizes the tile into fp32
registers ONCE, with a zero-scale branch that dequantizes to a zero tile without
dividing by zero. Per step: plain delta rule (pre-write prediction, delta
write, post-write read). On the last step, the program computes tl.max(|tile|)
across the K and BV axes; a second kernel reduces the per-BV-tile absmaxes into
one per-(batch, head) scale and quantizes each BV tile of the resident state
into int8 with `round(x / scale)` clamped to [-127, 127], with a zero-state
branch that emits q = 0, scale = 0 exactly.

Determinism: recurrence over t is serial per program, programs never share an
output element; the requantization reduction is deterministic because it happens
in a fixed (BV-tile) order. No atomics anywhere.

Note on the anchor: no vendor kernel accepts an int8 recurrent-state boundary;
the timing denominator is a torch.compile of the fused fp32 step + a torch-side
dequant/requant (see taskdef.sota).
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
    H0q,
    H0scale,
    O,
    Htmp,
    Amax,
    S_len,
    H,
    scale,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
):
    """Fused int8-dequantize on entry, S delta steps in fp32 registers, and a
    per-BV-tile absmax emitted for the second kernel to reduce and quantize."""
    pid = tl.program_id(0).to(tl.int64)
    pv = tl.program_id(1).to(tl.int64)
    b = pid // H
    h = pid % H

    o_k = tl.arange(0, K).to(tl.int64)
    o_v = pv * BV + tl.arange(0, BV).to(tl.int64)

    # In-kernel dequant of the state0 tile. Zero-scale branch avoids /0.
    scale0 = tl.load(H0scale + pid).to(tl.float32)
    p_q0 = H0q + pid * K * V + o_k[:, None] * V + o_v[None, :]
    b_q0 = tl.load(p_q0).to(tl.float32)
    b_h = tl.where(scale0 > 0.0, b_q0 * scale0, 0.0)

    base = (b * S_len) * H + h
    p_q = Q + base * K + o_k
    p_k = Kk + base * K + o_k
    p_v = Vv + base * V + o_v
    p_beta = Beta + base
    p_o = O + base * V + o_v

    for _t in range(S_len):
        b_qv = tl.load(p_q).to(tl.float32) * scale
        b_k = tl.load(p_k).to(tl.float32)
        b_v = tl.load(p_v).to(tl.float32)
        b_beta = tl.load(p_beta).to(tl.float32)

        pred = tl.sum(b_h * b_k[:, None], 0)
        delta = (b_v - pred) * b_beta
        b_h = b_h + b_k[:, None] * delta[None, :]

        b_o = tl.sum(b_h * b_qv[:, None], 0)
        tl.store(p_o, b_o.to(O.dtype.element_ty))

        p_q += H * K
        p_k += H * K
        p_v += H * V
        p_beta += H
        p_o += H * V

    # Emit the final fp32 tile plus this program's per-BV-tile absmax for the
    # second-kernel reduction and quantization pass.
    p_ht = Htmp + pid * K * V + o_k[:, None] * V + o_v[None, :]
    tl.store(p_ht, b_h)
    tile_absmax = tl.max(tl.abs(b_h))
    tl.store(Amax + pid * (V // BV) + pv, tile_absmax)


@triton.jit
def _quant_kernel(
    Htmp,
    Amax,
    Sq,
    Sscale,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    NV: tl.constexpr,
):
    """Per-(batch, head) reduce the NV=V/BV per-tile absmaxes into one scale,
    then quantize each BV tile of the stored fp32 state into int8. Zero-state
    branch emits (q = 0, scale = 0) exactly."""
    pid = tl.program_id(0).to(tl.int64)  # over B*H

    # Reduce V/BV per-tile absmaxes -> one scale per (batch, head).
    o_t = tl.arange(0, NV).to(tl.int64)
    tile_absmaxes = tl.load(Amax + pid * NV + o_t)
    absmax = tl.max(tile_absmaxes)
    sc = tl.where(absmax > 0.0, absmax / 127.0, 0.0)
    tl.store(Sscale + pid, sc)

    # Requantize each BV tile of the stored fp32 state.
    denom = tl.where(sc > 0.0, sc, 1.0)
    o_k = tl.arange(0, K).to(tl.int64)
    for pv in range(NV):
        o_v = pv * BV + tl.arange(0, BV).to(tl.int64)
        p_h = Htmp + pid * K * V + o_k[:, None] * V + o_v[None, :]
        b_h = tl.load(p_h)
        # rint (half-to-even) matches torch.round; libdevice.round is half-away-from-zero
        # and differs on exact .5 boundaries (26.5 -> 27 vs torch's 26).
        q = tl.extra.cuda.libdevice.rint(b_h / denom)
        q = tl.minimum(tl.maximum(q, -127.0), 127.0)
        q = tl.where(absmax > 0.0, q, 0.0).to(tl.int8)
        p_q = Sq + pid * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_q, q)


def _tile_v(V: int) -> int:
    return 64 if V >= 64 else V


def int8_state_delta_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    state0_q: torch.Tensor,
    state0_scale: torch.Tensor,
) -> tuple:
    B, S, H, K = q.shape
    V = v.shape[-1]
    if (K & (K - 1)) or (V & (V - 1)):
        raise ValueError(f"oracle supports power-of-two K and V, got K={K}, V={V}")
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    beta = beta.contiguous()
    h0q = state0_q.contiguous()
    h0scale = state0_scale.to(torch.float32).contiguous()

    o = torch.empty(B, S, H, V, dtype=v.dtype, device=v.device)
    htmp = torch.empty(B, H, K, V, dtype=torch.float32, device=v.device)
    scale = K**-0.5

    BV = _tile_v(V)
    NV = V // BV
    amax = torch.empty(B * H * NV, dtype=torch.float32, device=v.device)

    _scan_kernel[(B * H, NV)](
        q,
        k,
        v,
        beta,
        h0q,
        h0scale,
        o,
        htmp,
        amax,
        S,
        H,
        scale,
        K=K,
        V=V,
        BV=BV,
        num_warps=2,
        num_stages=2,
    )

    state_q = torch.empty(B, H, K, V, dtype=torch.int8, device=v.device)
    state_scale = torch.empty(B, H, dtype=torch.float32, device=v.device)
    _quant_kernel[(B * H,)](
        htmp,
        amax,
        state_q,
        state_scale,
        K=K,
        V=V,
        BV=BV,
        NV=NV,
        num_warps=2,
        num_stages=1,
    )
    return o, state_q, state_scale
# FORGE-CANARY-BEGIN forge-canary/v1
# FORGE-CANARY-SLOT-0 613d1dd8b0c1d7e4655d28fa89c2ca836cfd992893236cc41d5ff73ed1dd6276
# FORGE-CANARY-SLOT-1 ddedab146e135cbfe7af9106f7c07ab4ade9ddcf30e449923b69e83b565b3b64
# FORGE-CANARY-SLOT-2 7bc132e9a66829b35dafd53d4e676a93033fe764f867322eeba98aca31219906
# FORGE-CANARY-SLOT-3 f35bb1bcf447de1931c459f34599ea1c809dc379dda80dc035dae5c3a86de873
# FORGE-CANARY-END
