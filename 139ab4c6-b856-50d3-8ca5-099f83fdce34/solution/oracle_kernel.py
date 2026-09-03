"""PRIVATE oracle -- from-scratch chunk-parallel GDN-2 (fwd+bwd).

    UNVERIFIED: authored CPU-only, WITHOUT GPU calibration. The chunk-parallel
    ALGORITHM below is derived from the GDN-2 recurrence in reference.py and
    is CPU-verified against it in float32 (its own naming/structure -- NOT a
    copy of fla source; reading the library was allowed, vendoring is not).
    What is NOT yet verified, and what GPU calibration must establish before
    this can be trusted as the reward ceiling:
      * that the from-scratch @triton.jit per-K-channel cumulative-gate kernel
        matches the torch cumsum numerically on-device,
      * the tile choices for the K = V = 256 graded shape (state = 65k float32
        entries per (batch, head) -- past comfortable SRAM residency on the
        128-tile shapes the family exemplars are tuned around),
      * the reached fraction of the FLA `chunk_gdn2` anchor,
      * G4 adoption: the intra-chunk cross-basis matmuls plus the WY inverse
        are torch.matmul plus torch.linalg.solve_triangular here (fast cuBLAS,
        real device work but NOT this module's Triton). Clearing the 0.60
        floor honestly needs those in Triton; until then the golden stage
        uses the ORACLE_ENV exemption (taskdef ORACLE_ENV_FOR_GOLDEN=True).

Never shipped to an agent. It exists to establish that the full-reward point of
the metric is REACHABLE by a from-scratch implementation with no library call.

Algorithm (per chunk of length BT, per-K-channel gate, per-K erase gate b on
the erase basis (b*k), per-V write gate w on the write value (w*v)):

  * per-chunk per-channel cumulative log-decay c[i, k] = sum_{m<=i} g[m, k],
    and c_excl[i, k] = c[i, k] - g[i, k] (decay entering step i);
  * intra-chunk CROSS matrix, strictly lower (j < i), fully K-channel-decayed:
        M[i, j] = einsum_k (b*k)[i, k] * k[j, k] * exp(c_excl[i, k] - c[j, k])
    (the exponent stays <= 0 because c is non-increasing);
  * WY transition T_wy = (I + M)^{-1} on the strictly-lower matrix, so the
    N-fold intra-chunk erasure cascade collapses to one triangular solve;
  * write-value pre-image  Uw = T_wy (w * v);
  * decayed erase pre-image Wk[i, k, v] = sum_j T_wy[i, j] * (b*k)[j, k]
    * exp(c_excl[i, k] - c[j, k]) * (per-channel decay from j to i);
    computed as Wk = T_wy @ ((b*k) * exp(c_excl)), then multiplied on the
    read side by exp(-c[j, k]) inside the state carry;
  * per chunk:  v' = W_read S,  o = (q * exp(c_excl)) S + tril(...) (Uw - v')
    and state carry S = diag(exp(c_last)) S + (k * exp(c_last - c))^T (Uw - v').
  q is scaled by K^{-0.5}. Because both the erase basis and the state decay are
  per-K-channel, the pairwise (c[i] - c[j]) exponent must stay inside the einsum
  and never be split into a P @ Q^T with an unbounded exp(-c[j]) factor.
"""

import torch
import triton
import triton.language as tl

BT = 32  # smaller chunk for the K=V=256 state-tile budget on the graded g4


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


def _chunk_core(q, k, v, g, b, w, use_triton_cumgate):
    """Chunk-parallel GDN-2. Differentiable (torch). use_triton_cumgate routes
    the intra-chunk cumulative gate through the from-scratch Triton kernel for
    forward and torch.cumsum for the differentiable recompute path."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    scale = K**-0.5
    device = q.device

    qh = q.transpose(1, 2).float() * scale  # [B,H,T,K]
    kh = k.transpose(1, 2).float()
    vh = v.transpose(1, 2).float()
    gh = g.transpose(1, 2).float()  # [B,H,T,K]
    bh = b.transpose(1, 2).float()
    wh = w.transpose(1, 2).float()  # [B,H,T,V]

    qh, T0 = _pad_T(qh, 2)
    kh, _ = _pad_T(kh, 2)
    vh, _ = _pad_T(vh, 2)
    gh, _ = _pad_T(gh, 2)
    bh, _ = _pad_T(bh, 2)
    wh, _ = _pad_T(wh, 2)
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
    bc = bh.reshape(B, H, nc, BT, K)
    wc = wh.reshape(B, H, nc, BT, V)

    c_excl = cg - gc  # decay entering step i
    bk = bc * kc  # per-K erase key
    wv = wc * vc  # per-V write value

    # Strictly-lower intra-chunk cross matrix on (bk, k) with per-K-channel decay
    # baked in. Bounded exponent: c_excl[i,k] - c[j,k] <= 0 for i > j since c
    # is non-increasing when g <= 0. Pairwise (c[i-1] - c[j]) via einsum keeps it
    # numerically safe (never a naked exp(-c[j]) factor).
    # clamp: masked-out (i<j) entries carry large POSITIVE exponents that overflow to inf,
    # and inf*0 from the triangular mask becomes NaN. Valid (i>j) entries are always <= 0
    # because every gate is <= 0, so clamping cannot alter a graded value.
    Ppair = (cg.unsqueeze(-2) - cg.unsqueeze(-3)).clamp(max=0.0).exp()  # [B,H,nc,BT_i,BT_j,K]
    M = torch.einsum("bhcik,bhcjk,bhcijk->bhcij", bk, kc, Ppair)  # [B,H,nc,BT,BT]
    m_diag = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=device), 0)
    M = M.masked_fill(m_diag, 0)
    eye = torch.eye(BT, device=device).expand_as(M)
    T_wy = torch.linalg.solve_triangular(eye + M, eye, upper=False, unitriangular=True)
    Uw = T_wy @ wv  # [B,H,nc,BT,V]

    S = q.new_zeros(B, H, K, V).float()
    o = torch.zeros(B, H, nc, BT, V, device=device, dtype=torch.float32)
    m_strict_upper = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=device), 1)
    for i in range(nc):
        qi, ki, ci, cei, bki = (
            qc[:, :, i],
            kc[:, :, i],
            cg[:, :, i],
            c_excl[:, :, i],
            bk[:, :, i],
        )
        # Read-through-state term: pre-image of the erase read on S is
        # W_read[i, k, v] = sum_j T_wy[i,j] (bk)[j, k] exp(c_excl[i,k] - c[j,k]),
        # so v' = sum_k W_read[i, k, v] * S[k, v]. Compute as
        # v' = einsum_j T_wy[i,j] * (einsum_k (bk)[j,k] * exp(-c[j,k]) * S[k,v])
        # (bounded via cross with exp(c_excl[i,k]))): pair the exps inside einsum.
        # The S0-read of the erase at position j is sum_k bk[j,k]*exp(c[j,k])*S[k,v] --
        # a function of j ALONE, not an (i,j) pair. exp(+c[j]) cannot overflow (c <= 0),
        # so the previous pairwise exp(c_excl[i]-c[j]) factor was both unneeded and wrong.
        Sk_bkj = torch.einsum("bhjk,bhkv->bhjv", bki * ci.exp(), S)  # [B,H,BT_j,V]
        v_prime = T_wy[:, :, i] @ Sk_bkj  # [B,H,BT_i,V]
        v_new = Uw[:, :, i] - v_prime  # [B,H,BT,V]
        # Intra output diagonal-inclusive: A[i,j] = sum_k q[i,k] k[j,k] exp(c[i,k]-c[j,k]), j <= i.
        Ppair_qk = (ci.unsqueeze(-2) - ci.unsqueeze(-3)).clamp(max=0.0).exp()  # [B,H,BT_i,BT_j,K]
        A = torch.einsum("bhik,bhjk,bhijk->bhij", qi, ki, Ppair_qk).masked_fill(
            m_strict_upper, 0
        )
        # Inter contribution: q[i] . (state decayed by exp(c_excl[i]))^T
        o_inter = torch.einsum("bhik,bhkv->bhiv", qi * ci.exp(), S)
        o[:, :, i] = o_inter + A @ v_new
        # State carry to next chunk (per-K decay to c_last, then add write rank-1 outer products).
        cl = ci[:, :, -1, :]  # [B,H,K]
        S = S * cl[..., None].exp()
        # Add sum_j k_j (v_new_j - w*v decays via inclusive c) outer product decayed by (c_last - c_j).
        kdec = ki * (cl.unsqueeze(-2) - ci).exp()  # [B,H,BT,K], exponent <= 0
        S = S + kdec.transpose(-1, -2) @ v_new

    o = o.reshape(B, H, Tp, V)[:, :, :T0].transpose(1, 2).contiguous()
    return o.to(v.dtype)


class _Gdn2Oracle(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, b, w):
        with torch.no_grad():
            out = _chunk_core(q, k, v, g, b, w, use_triton_cumgate=True)
        ctx.save_for_backward(q, k, v, g, b, w)
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, g, b, w = ctx.saved_tensors
        with torch.enable_grad():
            qd = q.detach().requires_grad_()
            kd = k.detach().requires_grad_()
            vd = v.detach().requires_grad_()
            gd = g.detach().requires_grad_()
            bd = b.detach().requires_grad_()
            wd = w.detach().requires_grad_()
            out = _chunk_core(qd, kd, vd, gd, bd, wd, use_triton_cumgate=False)
        return torch.autograd.grad(
            out, [qd, kd, vd, gd, bd, wd], dout, allow_unused=True
        )


def chunked_gdn2(q, k, v, g, b, w):
    """Entry point (same signature/semantics as the starter)."""
    return _Gdn2Oracle.apply(q, k, v, g, b, w)
# FORGE-CANARY-BEGIN forge-canary/v1
# FORGE-CANARY-SLOT-0 5f502cffc6145ac72b5802e99e10768d5d4f786f8cc0834efd90a8a95b6c378e
# FORGE-CANARY-SLOT-1 f047a424b68f371e8611ef3f76c3eb166ae396a4daa9552e37a198ba573a4eba
# FORGE-CANARY-SLOT-2 738e3eba492a7096fff592ac9fd2c597e035e8adb88a0eb447fbac3d0f72894a
# FORGE-CANARY-SLOT-3 7d279c030e0d4a58f2bda4d383ff36c0a7d65c1eeb5d49ee6678dc49cf414c77
# FORGE-CANARY-END
