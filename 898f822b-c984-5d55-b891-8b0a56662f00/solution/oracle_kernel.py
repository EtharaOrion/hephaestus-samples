"""PRIVATE oracle -- from-scratch chunk-parallel COMBA (fwd+bwd).

    UNVERIFIED: authored CPU-only, WITHOUT GPU calibration. The chunk-parallel
    ALGORITHM below is CPU-verified against reference.py to ~1e-7 in float32
    (its own naming/structure -- NOT a copy of fla source; reading the library
    was allowed, vendoring is not). What is NOT yet verified, and what GPU
    calibration must establish before this can be trusted as the reward ceiling:
      * that the from-scratch @triton.jit cumulative-gate kernel matches the
        torch cumsum numerically on-device,
      * the tile choices, the reached fraction of the FLA `chunk_comba` anchor,
      * G4 adoption: the intra-chunk cross-basis matmuls and the WY inverse are
        torch.matmul plus torch.linalg.solve_triangular here (fast cuBLAS, real
        device work but NOT this module's Triton). Clearing the 0.60 floor
        honestly needs those in Triton; until then the golden stage uses the
        ORACLE_ENV exemption (taskdef ORACLE_ENV_FOR_GOLDEN=True).

Never shipped to an agent. It exists to establish that the full-reward point of
the metric is REACHABLE by a from-scratch implementation with no library call.

Algorithm (per chunk of length BT, scalar per-step log-decay g, WY on the
CROSS basis p k^T rather than the symmetric k k^T):

  * per-chunk cumulative log-decay c[i] = sum_{m<=i} g[m];
  * L_mask[i,j]   = exp(c[i]      - c[j]) for j <= i    (bounded, g <= 0);
  * L_mask_0[i,j] = exp((c[i]-g[i]) - c[j]) for j <= i  (decay BEFORE current step);
  * intra-chunk CROSS matrix M[i,j] for j < i:
        M[i,j] = (p_beta[i] . k[j]) * L_mask_0[i,j]      (strictly lower)
    where p_beta = p * beta[..., None]. WY transition
        T_wy = (I + M)^{-1}                              (unit-lower triangular);
  * U = T_wy (beta * v),  W = T_wy (p_beta * exp(c - g))  (a.k.a. k_cumsum, k_cumdecay);
  * per chunk: v' = W S; o = (q * exp(c)) S + tril(q k^T . L_mask, 0) (U - v');
    state carry S = diag(exp(c_last)) S + (k * exp(c_last - c))^T (U - v').
  q is scaled by K^{-0.5}. No delta correction against k, no beta on the write
  key, no grouped value attention: the erase basis is p and the write basis is k.
"""

import torch
import triton
import triton.language as tl

BT = 64  # chunk length


@triton.jit
def _chunk_cumsum_kernel(g_ptr, out_ptr, T, BLK: tl.constexpr):
    """Per-chunk inclusive cumulative sum of the scalar log-decay along time.
    From-scratch; one program per (row, chunk)."""
    row = tl.program_id(0)
    c = tl.program_id(1)
    offs = tl.arange(0, BLK)
    idx = c * BLK + offs
    mask = idx < T
    g = tl.load(g_ptr + row * T + idx, mask=mask, other=0.0)
    tl.store(out_ptr + row * T + idx, tl.cumsum(g, axis=0), mask=mask)


def _cumgate_triton(g_bht: torch.Tensor) -> torch.Tensor:
    """g: [B, T, H] -> per-chunk cumsum along T, [B, T, H]."""
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
    shp = list(x.shape)
    shp[tdim] = pad
    return torch.cat([x, x.new_zeros(shp)], dim=tdim), T


def _chunk_core(q, k, v, p, g, beta, use_triton_cumgate):
    """Chunk-parallel COMBA. Differentiable (torch). use_triton_cumgate routes
    the intra-chunk cumulative gate through the from-scratch Triton kernel for
    forward and torch.cumsum for the differentiable recompute path."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    scale = K**-0.5
    device = q.device

    qh = q.transpose(1, 2).float() * scale  # [B,H,T,K]
    kh = k.transpose(1, 2).float()
    vh = v.transpose(1, 2).float()
    ph = p.transpose(1, 2).float()
    gh = g.transpose(1, 2).float()  # [B,H,T]
    bh = beta.transpose(1, 2).float()

    qh, T0 = _pad_T(qh, 2)
    kh, _ = _pad_T(kh, 2)
    vh, _ = _pad_T(vh, 2)
    ph, _ = _pad_T(ph, 2)
    gh, _ = _pad_T(gh, 2)
    bh, _ = _pad_T(bh, 2)
    Tp = qh.shape[2]
    nc = Tp // BT

    if use_triton_cumgate:
        cg = _cumgate_triton(gh.transpose(1, 2).contiguous()).transpose(1, 2)
        cg = cg.reshape(B, H, nc, BT)
    else:
        cg = gh.reshape(B, H, nc, BT).cumsum(-1)

    qc = qh.reshape(B, H, nc, BT, K)
    kc = kh.reshape(B, H, nc, BT, K)
    vc = vh.reshape(B, H, nc, BT, V)
    pc = ph.reshape(B, H, nc, BT, K)
    bc = bh.reshape(B, H, nc, BT)
    gc = gh.reshape(B, H, nc, BT)

    pbeta = pc * bc[..., None]
    vbeta = vc * bc[..., None]

    # Decay masks. cg = c_incl (includes current step's g). c_excl = cg - gc.
    c_excl = cg - gc
    Lm = (
        (cg.unsqueeze(-1) - cg.unsqueeze(-2)).tril().exp()
    ).tril()  # includes diag; L_mask
    # L_mask_0: inclusive cumgate on BOTH axes. The reference gates before the write at t,
    # so a write at j decays into position i by exp(cg[i] - cg[j]); c_excl here dropped exp(-g_i).
    Lm0 = ((cg.unsqueeze(-1) - cg.unsqueeze(-2)).tril().exp()).tril()
    # Strictly-lower CROSS matrix on (p_beta, k): M[i,j] = (pbeta_i . k_j) * Lm0[i,j], i>j.
    m_diag = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=device), 0)
    M = ((pbeta @ kc.transpose(-1, -2)) * Lm0).masked_fill(m_diag, 0)
    eye = torch.eye(BT, device=device).expand_as(M)
    # WY transition T_wy = (I + M)^{-1}, unit lower triangular.
    T_wy = torch.linalg.solve_triangular(eye + M, eye, upper=False, unitriangular=True)
    Uv = T_wy @ vbeta  # k_cumsum
    Wk = T_wy @ (
        pbeta * cg.exp()[..., None]
    )  # k_cumdecay: inclusive cumgate, same convention as Lm0 above (gate precedes the write at t)

    S = q.new_zeros(B, H, K, V).float()
    o = torch.zeros(B, H, nc, BT, V, device=device, dtype=torch.float32)
    m_strict_upper = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=device), 1)
    for i in range(nc):
        qi, ki = qc[:, :, i], kc[:, :, i]
        A = ((qi @ ki.transpose(-1, -2)) * Lm[:, :, i]).masked_fill(m_strict_upper, 0)
        v_new = Uv[:, :, i] - Wk[:, :, i] @ S
        o_inter = (qi * cg[:, :, i, :, None].exp()) @ S
        o[:, :, i] = o_inter + A @ v_new
        cl = cg[:, :, i, -1, None]
        S = (
            S * cl[..., None].exp()
            + (ki * (cl - cg[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    o = o.reshape(B, H, Tp, V)[:, :, :T0].transpose(1, 2).contiguous()
    return o.to(v.dtype)


class _CombaOracle(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, p, g, beta):
        with torch.no_grad():
            out = _chunk_core(q, k, v, p, g, beta, use_triton_cumgate=True)
        ctx.save_for_backward(q, k, v, p, g, beta)
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, p, g, beta = ctx.saved_tensors
        with torch.enable_grad():
            qd = q.detach().requires_grad_()
            kd = k.detach().requires_grad_()
            vd = v.detach().requires_grad_()
            pd = p.detach().requires_grad_()
            gd = g.detach().requires_grad_()
            bd = beta.detach().requires_grad_()
            out = _chunk_core(qd, kd, vd, pd, gd, bd, use_triton_cumgate=False)
        return torch.autograd.grad(
            out, [qd, kd, vd, pd, gd, bd], dout, allow_unused=True
        )


def chunked_comba(q, k, v, p, g, beta):
    """Entry point (same signature/semantics as the starter)."""
    return _CombaOracle.apply(q, k, v, p, g, beta)
# FORGE-CANARY-BEGIN forge-canary/v1
# FORGE-CANARY-SLOT-0 0dbcb37aa49e750934ca346e9728b2478ca3ac23a92fa990c606eaceeb387dc0
# FORGE-CANARY-SLOT-1 d74d65d8b8c6b275caac0adc871695e11373e8ef8a88eb37a9bb0ce1334a8523
# FORGE-CANARY-SLOT-2 b6dd38b39decebc93bb079d81367a0d7c5d9d1f5c9857133a93ba76b414a9333
# FORGE-CANARY-SLOT-3 d84ed212e57d92108a7c6f0019ccc29d9c4568f3c39780a2404ad876defbe034
# FORGE-CANARY-END
