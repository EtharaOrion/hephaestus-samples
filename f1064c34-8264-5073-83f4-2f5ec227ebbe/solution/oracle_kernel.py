"""PRIVATE oracle -- from-scratch chunk-parallel DPLR generalized delta rule
(fwd+bwd).

    UNVERIFIED: authored CPU-only, WITHOUT GPU calibration. The chunk-parallel
    ALGORITHM below is derived from the DPLR chunkwise recurrence in
    reference.py and is CPU-verified against it in float32 (its own naming/
    structure -- NOT a copy of fla source; reading the library was allowed,
    vendoring is not). What is NOT yet verified, and what GPU calibration must
    establish before this can be trusted as the reward ceiling:
      * that the from-scratch @triton.jit per-K-channel cumulative-gate kernel
        matches the torch cumsum numerically on-device AT T = 8192 (the graded
        long-context stress shape -- ~128 chunk-boundary state carries),
      * the tile choices, the reached fraction of the FLA
        `chunk_dplr_delta_rule` anchor,
      * G4 adoption: the intra-chunk (A_qk, A_qb, A_ab, A_ak) matmul cascade
        and the WY inverse are torch.matmul + torch.linalg.solve_triangular
        here (fast cuBLAS, real device work but NOT this module's Triton).
        Clearing the 0.60 floor honestly needs those in Triton; until then
        the golden stage uses the ORACLE_ENV exemption (taskdef
        ORACLE_ENV_FOR_GOLDEN=True).

Never shipped to an agent. It exists to establish that the full-reward point of
the metric is REACHABLE by a from-scratch implementation with no library call.

Algorithm (per chunk of length BT; per-K-channel log-decay gk; rank-1 DPLR
correction alpha/beta):

  * per-chunk per-channel cumulative log-decay c[i,k] = sum_{m<=i} gk[m,k];
    c_excl[i,k] = c[i,k] - gk[i,k] (decay entering step i);
  * A_qk[i,j] = einsum_k q[i,k] k[j,k] exp(c[i,k] - c[j,k])   for j <= i;
  * A_qb[i,j] = einsum_k q[i,k] beta[j,k] exp(c[i,k] - c[j,k]) for j <= i;
  * A_ab[i,j] = einsum_k alpha[i,k] beta[j,k] exp(c_excl[i,k] - c[j,k])
                                                              for j < i (strict);
  * A_ak[i,j] = einsum_k alpha[i,k] k[j,k]    exp(c_excl[i,k] - c[j,k])
                                                              for j < i (strict);
  * WY-like transition on the DPLR erasure recurrence:
        T_wy = (I - A_ab)^{-1}      (unit lower triangular);
    then u = T_wy (A_ak v),  w = T_wy (alpha * exp(c_excl));
  * per chunk: v2 = u + w S;  o = A_qk v + A_qb v2 + (q * exp(c)) S;
    state carry S = diag(exp(c_last)) S
                    + (k * exp(c_last - c))^T v
                    + (beta * exp(c_last - c))^T v2.
  q is scaled by K^{-0.5}. Every exponent is bounded (c is non-increasing when
  gk <= 0). The long-context T = 8192 graded shape amortizes into 128 chunk
  carries of length BT = 64; the state carry stays in float32 throughout.
"""

import torch
import triton
import triton.language as tl

BT = 64  # chunk length


@triton.jit
def _chunk_cumsum_kernel(g_ptr, out_ptr, T, BLK: tl.constexpr):
    """Per-chunk inclusive cumulative sum of the per-channel log-decay along
    time. From-scratch; one program per (channel-row, chunk)."""
    row = tl.program_id(0)
    c = tl.program_id(1)
    offs = tl.arange(0, BLK)
    idx = c * BLK + offs
    mask = idx < T
    g = tl.load(g_ptr + row * T + idx, mask=mask, other=0.0)
    tl.store(out_ptr + row * T + idx, tl.cumsum(g, axis=0), mask=mask)


def _cumgate_triton(g_bthk: torch.Tensor) -> torch.Tensor:
    """g: [B, T, H, K] -> per-chunk cumsum along T, [B, T, H, K]."""
    B, T, H, K = g_bthk.shape
    gt = g_bthk.permute(0, 2, 3, 1).contiguous().view(B * H * K, T)
    out = torch.empty_like(gt)
    _chunk_cumsum_kernel[(B * H * K, triton.cdiv(T, BT))](gt, out, T, BLK=BT)
    return out.view(B, H, K, T).permute(0, 3, 1, 2).contiguous()


def _pad_T(x, tdim):
    T = x.shape[tdim]
    pad = (BT - (T % BT)) % BT
    if pad == 0:
        return x, T
    shp = list(x.shape)
    shp[tdim] = pad
    return torch.cat([x, x.new_zeros(shp)], dim=tdim), T


def _chunk_core(q, k, v, alpha, beta, gk, use_triton_cumgate):
    """Chunk-parallel DPLR generalized delta rule. Differentiable (torch).
    use_triton_cumgate routes the intra-chunk cumulative gate through the
    from-scratch Triton kernel for forward, torch.cumsum for the recompute
    path used by autograd."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    scale = K**-0.5
    device = q.device

    qh = q.transpose(1, 2).float() * scale
    kh = k.transpose(1, 2).float()
    vh = v.transpose(1, 2).float()
    ah = alpha.transpose(1, 2).float()
    bh = beta.transpose(1, 2).float()
    gh = gk.transpose(1, 2).float()

    qh, T0 = _pad_T(qh, 2)
    kh, _ = _pad_T(kh, 2)
    vh, _ = _pad_T(vh, 2)
    ah, _ = _pad_T(ah, 2)
    bh, _ = _pad_T(bh, 2)
    gh, _ = _pad_T(gh, 2)
    Tp = qh.shape[2]
    nc = Tp // BT

    if use_triton_cumgate:
        cg = _cumgate_triton(gh.transpose(1, 2).contiguous()).transpose(1, 2)
        cg = cg.reshape(B, H, nc, BT, K)
    else:
        cg = gh.reshape(B, H, nc, BT, K).cumsum(-2)

    qc = qh.reshape(B, H, nc, BT, K)
    kc = kh.reshape(B, H, nc, BT, K)
    vc = vh.reshape(B, H, nc, BT, V)
    gc = gh.reshape(B, H, nc, BT, K)
    ac = ah.reshape(B, H, nc, BT, K)
    bc = bh.reshape(B, H, nc, BT, K)
    c_excl = cg - gc  # decay entering step i

    tril_incl = torch.tril(torch.ones(BT, BT, device=device))  # j <= i
    tril_strict = torch.tril(torch.ones(BT, BT, device=device), diagonal=-1)  # j < i

    # Intra-chunk pairwise decays. Both are bounded because c is non-increasing.
    #   P_incl[i,j,k] = exp(c[i,k] - c[j,k])              # for A_qk, A_qb
    #   P_excl[i,j,k] = exp(c_excl[i,k] - c[j,k])         # for A_ab, A_ak
    P_incl = (cg.unsqueeze(-2) - cg.unsqueeze(-3)).clamp(max=0.0).exp()
    P_excl = (c_excl.unsqueeze(-2) - cg.unsqueeze(-3)).clamp(max=0.0).exp()

    A_qk = torch.einsum("bhcik,bhcjk,bhcijk->bhcij", qc, kc, P_incl) * tril_incl
    A_qb = torch.einsum("bhcik,bhcjk,bhcijk->bhcij", qc, bc, P_incl) * tril_incl
    A_ab = torch.einsum("bhcik,bhcjk,bhcijk->bhcij", ac, bc, P_excl) * tril_strict
    A_ak = torch.einsum("bhcik,bhcjk,bhcijk->bhcij", ac, kc, P_excl) * tril_strict

    eye = torch.eye(BT, device=device).expand_as(A_ab)
    T_wy = torch.linalg.solve_triangular(
        eye - A_ab, eye, upper=False, unitriangular=True
    )
    u = T_wy @ (A_ak @ vc)  # [B,H,nc,BT,V]
    w_read = T_wy @ (ac * c_excl.exp())  # [B,H,nc,BT,K]

    S = q.new_zeros(B, H, K, V).float()
    o = torch.zeros(B, H, nc, BT, V, device=device, dtype=torch.float32)
    for i in range(nc):
        v2 = u[:, :, i] + w_read[:, :, i] @ S  # [B,H,BT,V]
        o_1 = A_qk[:, :, i] @ vc[:, :, i]
        o_2 = A_qb[:, :, i] @ v2
        o_3 = torch.einsum("bhik,bhkv->bhiv", qc[:, :, i] * cg[:, :, i].exp(), S)
        o[:, :, i] = o_1 + o_2 + o_3
        # State carry: decay-and-add both k v and beta v2 terms.
        cl = cg[:, :, i, -1, :]  # [B,H,K]
        decay = (cl.unsqueeze(-2) - cg[:, :, i]).exp()  # [B,H,BT,K], exponent <= 0
        S = S * cl[..., None].exp()
        S = S + (kc[:, :, i] * decay).transpose(-1, -2) @ vc[:, :, i]
        S = S + (bc[:, :, i] * decay).transpose(-1, -2) @ v2

    o = o.reshape(B, H, Tp, V)[:, :, :T0].transpose(1, 2).contiguous()
    return o.to(v.dtype)


class _DplrOracle(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, alpha, beta, gk):
        with torch.no_grad():
            out = _chunk_core(q, k, v, alpha, beta, gk, use_triton_cumgate=True)
        ctx.save_for_backward(q, k, v, alpha, beta, gk)
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, alpha, beta, gk = ctx.saved_tensors
        with torch.enable_grad():
            qd = q.detach().requires_grad_()
            kd = k.detach().requires_grad_()
            vd = v.detach().requires_grad_()
            ad = alpha.detach().requires_grad_()
            bd = beta.detach().requires_grad_()
            gd = gk.detach().requires_grad_()
            out = _chunk_core(qd, kd, vd, ad, bd, gd, use_triton_cumgate=False)
        return torch.autograd.grad(
            out, [qd, kd, vd, ad, bd, gd], dout, allow_unused=True
        )


def chunked_dplr_delta(q, k, v, alpha, beta, gk):
    """Entry point (same signature/semantics as the starter)."""
    return _DplrOracle.apply(q, k, v, alpha, beta, gk)
# FORGE-CANARY-BEGIN forge-canary/v1
# FORGE-CANARY-SLOT-0 bd75c14bbcecda2649446976c929276c87f84cb606f835b626dc13d237a7b3c1
# FORGE-CANARY-SLOT-1 7b198b9eceb17cf635390975a768934822f0dcde9db54e42af880cf21656f5a9
# FORGE-CANARY-SLOT-2 6f1b5d971b8cc6be8974bc414d7c5ab09158c1bfb08989b156ebe83e3c2b39f3
# FORGE-CANARY-SLOT-3 f349e824bc001431bdbc4b7f6830926c0b61185fb870fa19d266dcf1ab6f88b8
# FORGE-CANARY-END
