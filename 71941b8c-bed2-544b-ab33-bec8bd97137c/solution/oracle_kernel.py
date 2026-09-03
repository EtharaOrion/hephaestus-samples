"""PRIVATE oracle -- from-scratch chunk-parallel gated delta rule (fwd+bwd).

    UNVERIFIED: authored CPU-only, WITHOUT GPU calibration. The chunk-parallel
    ALGORITHM below is CPU-verified against reference.py to ~1e-7 in float32
    (its own naming/structure -- NOT a copy of fla source; reading the library
    was allowed, vendoring is not). What is NOT yet verified, and what GPU
    calibration must establish before this can be trusted as the reward ceiling:
      * that the from-scratch @triton.jit cumulative-gate kernel matches the
        torch cumsum numerically on-device,
      * the tile choices, and the fraction-of-sota this reaches against the
        production `chunk_gated_delta_rule`,
      * G4 adoption: the chunk GEMMs/WY-inverse are torch.matmul here (fast
        cuBLAS, real device work but NOT this module's Triton). Clearing the
        0.60 written-kernel floor honestly needs those moved into Triton; until
        then the golden stage uses the ORACLE_ENV exemption (taskdef
        ORACLE_ENV_FOR_GOLDEN=True), the same lever the KDA oracle uses.

Never shipped to an agent. It exists to establish that the full-reward point of
the metric is reachable by a from-scratch implementation with no library call.

Algorithm (WY representation with gated decay, per chunk of length BT):
  * cumulative log-decay within each chunk (the @triton.jit kernel), g_i = sum g;
  * L[i,j] = exp(g_i - g_j) for j<=i (bounded, g decreasing);
  * T = (I - tril(diag(beta) K K^T . L))^{-1} by forward substitution;
  * U = T (beta v),  W = T (beta k . e^{g});  intra A = (q k^T . L) strictly-lower;
  * per chunk: v' = W S; o = (q . e^{g}) S + A (U - v'); carry S with the gated
    key-value outer products. q is scaled by K^-0.5. Decay is applied BEFORE the
    prediction, matching reference.py. Grouped value attention (HV>H) is handled
    by expanding q, k to the value-head count.
"""

import torch
import triton
import triton.language as tl

BT = 64  # chunk length


@triton.jit
def _chunk_cumsum_kernel(g_ptr, out_ptr, T, BLK: tl.constexpr):
    """Per-chunk inclusive cumulative sum of the scalar log-decay along time.
    From-scratch; one program per (row, chunk). Real device work: the intra-chunk
    cumulative gate the WY formulation is built on."""
    row = tl.program_id(0)
    c = tl.program_id(1)
    offs = tl.arange(0, BLK)
    idx = c * BLK + offs
    mask = idx < T
    g = tl.load(g_ptr + row * T + idx, mask=mask, other=0.0)
    tl.store(out_ptr + row * T + idx, tl.cumsum(g, axis=0), mask=mask)


def _cumgate_triton(g_bht: torch.Tensor) -> torch.Tensor:
    """g: [B, T, H] (scalar log-decay) -> per-chunk cumsum along T, [B, T, H]."""
    B, T, H = g_bht.shape
    gt = g_bht.permute(0, 2, 1).contiguous().view(B * H, T)
    out = torch.empty_like(gt)
    _chunk_cumsum_kernel[(B * H, triton.cdiv(T, BT))](gt, out, T, BLK=BT)
    return out.view(B, H, T).permute(0, 2, 1).contiguous()


def _pad_T(x, tdim):
    T = x.shape[tdim]
    pad = (BT - (T % BT)) % BT
    if pad == 0:
        return x, T
    shp = list(x.shape); shp[tdim] = pad
    return torch.cat([x, x.new_zeros(shp)], dim=tdim), T


def _chunk_core(q, k, v, g, beta, use_triton_cumgate):
    """Chunk-parallel gated delta rule. Differentiable (torch). When
    use_triton_cumgate is True the intra-chunk cumulative gate is produced by the
    from-scratch Triton kernel (forward device path); otherwise by torch.cumsum
    (the differentiable recompute path used for gradients)."""
    B, T, H, K = q.shape
    HV = v.shape[2]
    V = v.shape[-1]
    if HV != H:
        rep = HV // H
        q = q.repeat_interleave(rep, dim=2)
        k = k.repeat_interleave(rep, dim=2)
        H = HV
    scale = K ** -0.5
    qh = q.transpose(1, 2).float() * scale
    kh = k.transpose(1, 2).float()
    vh = v.transpose(1, 2).float()
    gh = g.transpose(1, 2).float()
    bh = beta.transpose(1, 2).float()
    qh, T0 = _pad_T(qh, 2); kh, _ = _pad_T(kh, 2); vh, _ = _pad_T(vh, 2)
    gh, _ = _pad_T(gh, 2); bh, _ = _pad_T(bh, 2)
    Tp = qh.shape[2]; nc = Tp // BT

    if use_triton_cumgate:
        # [B,H,Tp] -> per-chunk cumsum via Triton, then to [B,H,nc,BT]
        cg = _cumgate_triton(gh.transpose(1, 2).contiguous()).transpose(1, 2)
        cg = cg.reshape(B, H, nc, BT)
    else:
        cg = gh.reshape(B, H, nc, BT).cumsum(-1)

    qc = qh.reshape(B, H, nc, BT, K); kc = kh.reshape(B, H, nc, BT, K)
    vc = vh.reshape(B, H, nc, BT, V); bc = bh.reshape(B, H, nc, BT)
    vbc = vc * bc[..., None]
    kbc = kc * bc[..., None]
    gexp = cg.exp()[..., None]                                   # [B,H,nc,BT,1]
    Lm = ((cg.unsqueeze(-1) - cg.unsqueeze(-2)).tril().exp()).tril()
    m_diag = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), 0)
    # WY transition T = (I + M)^{-1}, M = strictly-lower(diag(beta) K K^T . L).
    # Solved as a unit lower-triangular system (autograd-safe; the in-place
    # forward-substitution that would build it row by row is not).
    M = ((kbc @ kc.transpose(-1, -2)) * Lm).masked_fill(m_diag, 0)
    eye = torch.eye(BT, device=q.device).expand_as(M)
    Tt = torch.linalg.solve_triangular(eye + M, eye, upper=False, unitriangular=True)
    Uv = Tt @ vbc                                                # k_cumsum
    Wk = Tt @ (kbc * gexp)                                       # k_cumdecay
    S = q.new_zeros(B, H, K, V).float()  # state accumulates across chunks: fp32, matching the fp32 chunk operands
    o = torch.zeros(B, H, nc, BT, V, device=q.device)
    m_strict = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), 1)
    for i in range(nc):
        qi, ki = qc[:, :, i], kc[:, :, i]
        A = ((qi @ ki.transpose(-1, -2)) * Lm[:, :, i]).masked_fill(m_strict, 0)
        v_new = Uv[:, :, i] - Wk[:, :, i] @ S
        o_inter = (qi * cg[:, :, i, :, None].exp()) @ S
        o[:, :, i] = o_inter + A @ v_new
        gl = cg[:, :, i, -1, None]
        S = S * gl[..., None].exp() + (ki * (gl - cg[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
    o = o.reshape(B, H, Tp, V)[:, :, :T0].transpose(1, 2).contiguous()
    return o.to(v.dtype)


class _GatedDeltaOracle(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta):
        with torch.no_grad():
            out = _chunk_core(q, k, v, g, beta, use_triton_cumgate=True)
        ctx.save_for_backward(q, k, v, g, beta)
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, g, beta = ctx.saved_tensors
        with torch.enable_grad():
            qd = q.detach().requires_grad_(); kd = k.detach().requires_grad_()
            vd = v.detach().requires_grad_(); gd = g.detach().requires_grad_()
            bd = beta.detach().requires_grad_()
            out = _chunk_core(qd, kd, vd, gd, bd, use_triton_cumgate=False)
        return torch.autograd.grad(out, [qd, kd, vd, gd, bd], dout, allow_unused=True)


def chunked_gated_delta(q, k, v, g, beta):
    """Entry point (same signature/semantics as the starter)."""
    return _GatedDeltaOracle.apply(q, k, v, g, beta)
# FORGE-CANARY-BEGIN forge-canary/v1
# FORGE-CANARY-SLOT-0 e1babe41f81202dabfe8359afd8b5de29f822b2471ef5b83e1a6ec13bfadd1c8
# FORGE-CANARY-SLOT-1 0226c24712e845c431dad141943e1c24d4e96ff960427d36ebcae525b45da5f5
# FORGE-CANARY-SLOT-2 f66c96128e13464a51db92505b855c2d3e45b9331e8d30843e7c68fa0fe97f49
# FORGE-CANARY-SLOT-3 3143df2c938f203c96533fb9165a5c0ac6d8607bda73a72c3ecaaf9af4f11605
# FORGE-CANARY-END
