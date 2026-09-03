"""PRIVATE oracle -- from-scratch chunk-parallel gated delta-product (fwd+bwd).

    UNVERIFIED: authored CPU-only, WITHOUT GPU calibration. The chunk-parallel
    ALGORITHM below is CPU-verified against reference.py to ~1e-7 in float32
    (its own naming/structure -- NOT a copy of fla source; reading the library
    was allowed, vendoring is not). What is NOT yet verified, and what GPU
    calibration must establish before this can be trusted as the reward ceiling:
      * that the from-scratch @triton.jit cumulative-gate kernel matches the
        torch cumsum numerically on-device,
      * the tile choices for the (N*BT) x (N*BT) intra-chunk WY solve, and the
        fraction-of-sota this reaches against `chunk_gated_delta_product`,
      * G4 adoption: the chunk GEMMs / WY-inverse are torch.matmul plus
        torch.linalg.solve_triangular here (fast cuBLAS, real device work but
        NOT this module's Triton). Clearing the 0.60 written-kernel floor
        honestly needs those moved into Triton; until then the golden stage
        uses the ORACLE_ENV exemption (taskdef ORACLE_ENV_FOR_GOLDEN=True).

Never shipped to an agent. It exists to establish that the full-reward point of
the metric is REACHABLE by a from-scratch implementation with no library call.

Algorithm (num_householder = N, chunks of the OUTER length T; the whole intra-
chunk problem is expressed on the interleaved N*BT-long stream so the standard
WY factorization carries N Householder writes per outer token without a Python
for-loop over j):

  * interleave the per-token gate: g_ext[t*N + 0] = g[t], zero for j >= 1;
    the per-chunk cumulative decay is therefore a step function over the
    N-subblocks -- decay applied exactly ONCE per outer token, matching
    reference.py;
  * expand q to length N*T with q_ext[t*N + (N-1)] = q[t] and zero elsewhere,
    so the standard chunked-delta readout only fires at the end of each outer
    token's N-block;
  * standard WY chunked gated delta on the (q_ext, k, v, g_ext, beta) stream
    of length N*T: per-chunk cumulative log-decay c, L[i,j] = exp(c[i]-c[j])
    for j<=i, T_wy = (I + tril(diag(beta) K K^T . L, -1))^{-1}, U = T_wy(beta v),
    W = T_wy(beta k . e^{c}), intra A = (q k^T . L) strictly-lower;
    per chunk v' = W S, o = (q . e^{c}) S + A(U - v'), and state carry.
  q_ext is scaled by K^{-0.5}. Decay is applied BEFORE the prediction, matching
  reference.py; the u-bonus / beta / grouped-attention pieces do not apply here.
"""

import torch
import triton
import triton.language as tl

BT = 64  # OUTER chunk length; inner stream sees N*BT after expansion


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


def _cumgate_triton(g_bht: torch.Tensor, blk: int) -> torch.Tensor:
    """g: [B, L, H] -> per-chunk cumsum along L (block = blk)."""
    B, L, H = g_bht.shape
    gt = g_bht.permute(0, 2, 1).contiguous().view(B * H, L)
    out = torch.empty_like(gt)
    _chunk_cumsum_kernel[(B * H, triton.cdiv(L, blk))](gt, out, L, BLK=blk)
    return out.view(B, H, L).permute(0, 2, 1).contiguous()


def _pad_L(x, ldim, blk):
    L = x.shape[ldim]
    pad = (blk - (L % blk)) % blk
    if pad == 0:
        return x, L
    shp = list(x.shape)
    shp[ldim] = pad
    return torch.cat([x, x.new_zeros(shp)], dim=ldim), L


def _chunk_core(q, k, v, g, beta, num_householder, use_triton_cumgate):
    """Chunk-parallel gated delta-product on the interleaved N*T stream.
    Differentiable (torch). use_triton_cumgate routes the intra-chunk
    cumulative gate through the from-scratch Triton kernel for forward and
    through torch.cumsum for the differentiable recompute path."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    N = int(num_householder)
    L = T * N  # interleaved stream length
    blk = BT * N  # inner-chunk length
    scale = K**-0.5
    device = q.device

    # Interleave gate: length T -> L; g at j=0 sub-position, 0 elsewhere.
    g_ext = torch.zeros(B, T, N, H, dtype=torch.float32, device=device)
    g_ext[:, :, 0, :] = g.float()
    g_ext = g_ext.reshape(B, L, H)  # [B, L, H]

    # Expand q: length T -> L; q at j=N-1 sub-position, 0 elsewhere.
    q_ext = torch.zeros(B, T, N, H, K, dtype=torch.float32, device=device)
    q_ext[:, :, N - 1, :, :] = q.float() * scale
    q_ext = q_ext.reshape(B, L, H, K)

    k32 = k.float()
    v32 = v.float()
    b32 = beta.float()

    # Move to [B, H, L, D] and pad L to a multiple of blk.
    qh = q_ext.transpose(1, 2)  # [B, H, L, K]
    kh = k32.transpose(1, 2)
    vh = v32.transpose(1, 2)
    gh = g_ext.transpose(1, 2)  # [B, H, L]
    bh = b32.transpose(1, 2)
    qh, L0 = _pad_L(qh, 2, blk)
    kh, _ = _pad_L(kh, 2, blk)
    vh, _ = _pad_L(vh, 2, blk)
    gh, _ = _pad_L(gh, 2, blk)
    bh, _ = _pad_L(bh, 2, blk)
    Lp = qh.shape[2]
    nc = Lp // blk

    if use_triton_cumgate:
        cg = _cumgate_triton(gh.transpose(1, 2).contiguous(), blk).transpose(1, 2)
        cg = cg.reshape(B, H, nc, blk)
    else:
        cg = gh.reshape(B, H, nc, blk).cumsum(-1)

    qc = qh.reshape(B, H, nc, blk, K)
    kc = kh.reshape(B, H, nc, blk, K)
    vc = vh.reshape(B, H, nc, blk, V)
    bc = bh.reshape(B, H, nc, blk)

    vbc = vc * bc[..., None]
    kbc = kc * bc[..., None]
    gexp = cg.exp()[..., None]
    Lm = ((cg.unsqueeze(-1) - cg.unsqueeze(-2)).tril().exp()).tril()
    m_diag = torch.triu(torch.ones(blk, blk, dtype=torch.bool, device=device), 0)
    M = ((kbc @ kc.transpose(-1, -2)) * Lm).masked_fill(m_diag, 0)
    eye = torch.eye(blk, device=device).expand_as(M)
    Tt = torch.linalg.solve_triangular(eye + M, eye, upper=False, unitriangular=True)
    Uv = Tt @ vbc
    Wk = Tt @ (kbc * gexp)

    S = q.new_zeros(B, H, K, V).float()
    o = torch.zeros(B, H, nc, blk, V, device=device, dtype=torch.float32)
    m_strict = torch.triu(torch.ones(blk, blk, dtype=torch.bool, device=device), 1)
    for i in range(nc):
        qi, ki = qc[:, :, i], kc[:, :, i]
        A = ((qi @ ki.transpose(-1, -2)) * Lm[:, :, i]).masked_fill(m_strict, 0)
        v_new = Uv[:, :, i] - Wk[:, :, i] @ S
        o_inter = (qi * cg[:, :, i, :, None].exp()) @ S
        o[:, :, i] = o_inter + A @ v_new
        gl = cg[:, :, i, -1, None]
        S = (
            S * gl[..., None].exp()
            + (ki * (gl - cg[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    # Read outputs at end of each outer token (position t*N + N-1).
    o_full = o.reshape(B, H, Lp, V)[:, :, :L0]  # [B,H,L,V]
    o_full = o_full.reshape(B, H, T, N, V)[:, :, :, N - 1, :]  # [B,H,T,V]
    return o_full.transpose(1, 2).contiguous().to(v.dtype)


class _GatedDeltaProductOracle(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta, num_householder):
        with torch.no_grad():
            out = _chunk_core(
                q, k, v, g, beta, num_householder, use_triton_cumgate=True
            )
        ctx.save_for_backward(q, k, v, g, beta)
        ctx.num_householder = num_householder
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, g, beta = ctx.saved_tensors
        num_householder = ctx.num_householder
        with torch.enable_grad():
            qd = q.detach().requires_grad_()
            kd = k.detach().requires_grad_()
            vd = v.detach().requires_grad_()
            gd = g.detach().requires_grad_()
            bd = beta.detach().requires_grad_()
            out = _chunk_core(
                qd, kd, vd, gd, bd, num_householder, use_triton_cumgate=False
            )
        grads = torch.autograd.grad(out, [qd, kd, vd, gd, bd], dout, allow_unused=True)
        return (*grads, None)


def chunked_gated_delta_product(q, k, v, g, beta, num_householder=2):
    """Entry point (same signature/semantics as the starter)."""
    return _GatedDeltaProductOracle.apply(q, k, v, g, beta, num_householder)
# FORGE-CANARY-BEGIN forge-canary/v1
# FORGE-CANARY-SLOT-0 f6ab49de11d2b8c644e451dbbb006d8b7894acb6f0fe89d1db6403266091d78b
# FORGE-CANARY-SLOT-1 2b0c7224a9eef2572ceced4db57eaa02dfbf7cfcbce5d1d396f9a736f5b94e8d
# FORGE-CANARY-SLOT-2 dc193ff01b034159ce65ff851c3390033cdfe7d8de6fadf586f91fb6f8ad27dd
# FORGE-CANARY-SLOT-3 56f261abf5278f2651488981c78e88b4f0191965de952be4c2ff1efdc83c843e
# FORGE-CANARY-END
