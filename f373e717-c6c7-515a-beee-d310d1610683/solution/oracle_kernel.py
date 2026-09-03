"""PRIVATE oracle -- from-scratch chunk-parallel RWKV-6 with u-bonus (fwd+bwd).

    UNVERIFIED: authored CPU-only, WITHOUT GPU calibration. The chunk-parallel
    ALGORITHM below is derived from the RWKV-6 recurrence in reference.py and
    is CPU-verified against it in float32 (its own naming/structure -- NOT a
    copy of fla source; reading the library was allowed, vendoring is not).
    GPU calibration must still establish: the from-scratch @triton.jit
    cumulative-gate kernel's on-device numerics, the tile choices, the reached
    fraction of the FLA `chunk_rwkv6` anchor, and G4 adoption (the chunk GEMMs
    are torch.matmul here -- real cuBLAS device work but NOT this module's
    Triton; clearing the 0.60 floor honestly needs them in Triton, and until
    then the golden stage uses the ORACLE_ENV exemption (taskdef
    ORACLE_ENV_FOR_GOLDEN=True), the same lever the KDA / GLA oracles use).

Never shipped to an agent. It exists to establish that the full-reward point of
the metric is REACHABLE by a from-scratch implementation with no library call.

Algorithm (per chunk of length BT, gate is PER KEY CHANNEL, decay AFTER
readout, plus u-bonus applied ONLY to the current step's outer product in the
output; state carried to t+1 does NOT include u):

  * per-chunk per-channel cumulative log-decay,  c[i,k] = sum_{m<=i} w[m,k];
  * intra-chunk A[i,j] for j < i, over the OLD state contribution:
        A[i,j] = sum_k r[i,k] * k[j,k] * exp(c[i-1,k] - c[j,k])
    (implemented as pairwise (c[i-1] - c[j]) so the exponent stays <= 0);
  * intra-chunk u-diagonal (current step, added only to the output):
        D[i]   = sum_k r[i,k] * u[h,k] * k[i,k]
  * inter-chunk (state entering the chunk decays by exp(c[i-1])):
        o_inter[i] = sum_k r[i,k] * S_prev[k, :] * exp(c[i-1,k])
  * state carry to next chunk (u is NOT carried into the state):
        S_new = S_prev * exp(c[BT-1]) + sum_j k_j v_j^T * exp(c[BT-1] - c[j])
  r is scaled by K^-0.5. No delta, no beta, no grouped value attention.
"""

import torch
import triton
import triton.language as tl

BT = 64  # chunk length


@triton.jit
def _chunk_cumsum_kernel(g_ptr, out_ptr, T, BLK: tl.constexpr):
    """Per-chunk inclusive cumulative sum of the per-channel log-decay along
    time. From-scratch; one program per (channel-row, chunk). Real device work:
    the intra-chunk cumulative forget gate the RWKV-6 chunked form is built on.
    """
    row = tl.program_id(0)
    c = tl.program_id(1)
    offs = tl.arange(0, BLK)
    idx = c * BLK + offs
    mask = idx < T
    g = tl.load(g_ptr + row * T + idx, mask=mask, other=0.0)
    tl.store(out_ptr + row * T + idx, tl.cumsum(g, axis=0), mask=mask)


def _cumgate_triton(w_bthk: torch.Tensor) -> torch.Tensor:
    """w: [B, T, H, K] -> per-chunk cumsum along T, [B, T, H, K]."""
    B, T, H, K = w_bthk.shape
    wt = w_bthk.permute(0, 2, 3, 1).contiguous().view(B * H * K, T)
    out = torch.empty_like(wt)
    _chunk_cumsum_kernel[(B * H * K, triton.cdiv(T, BT))](wt, out, T, BLK=BT)
    return out.view(B, H, K, T).permute(0, 3, 1, 2).contiguous()


def _pad_T(x, tdim):
    T = x.shape[tdim]
    pad = (BT - (T % BT)) % BT
    if pad == 0:
        return x, T
    shp = list(x.shape)
    shp[tdim] = pad
    return torch.cat([x, x.new_zeros(shp)], dim=tdim), T


def _chunk_core(r, k, v, w, u, use_triton_cumgate):
    """Chunk-parallel RWKV-6. Differentiable (torch). When use_triton_cumgate
    is True the intra-chunk cumulative gate is produced by the from-scratch
    Triton kernel (forward device path); otherwise by torch.cumsum (the
    differentiable recompute path used for gradients).
    """
    B, T, H, K = r.shape
    V = v.shape[-1]
    scale = K**-0.5
    rh = r.transpose(1, 2).float() * scale  # [B,H,T,K]
    kh = k.transpose(1, 2).float()
    vh = v.transpose(1, 2).float()
    wh = w.transpose(1, 2).float()
    u32 = u.float()  # [H, K]
    rh, T0 = _pad_T(rh, 2)
    kh, _ = _pad_T(kh, 2)
    vh, _ = _pad_T(vh, 2)
    wh, _ = _pad_T(wh, 2)
    Tp = rh.shape[2]
    nc = Tp // BT

    if use_triton_cumgate:
        # [B,T,H,K] round-trip for the Triton kernel; then [B,H,Tp,K] -> [B,H,nc,BT,K]
        cg = _cumgate_triton(wh.transpose(1, 2).contiguous()).transpose(1, 2)
        cg = cg.reshape(B, H, nc, BT, K)
    else:
        cg = wh.reshape(B, H, nc, BT, K).cumsum(-2)

    rc = rh.reshape(B, H, nc, BT, K)
    kc = kh.reshape(B, H, nc, BT, K)
    vc = vh.reshape(B, H, nc, BT, V)
    wc = wh.reshape(B, H, nc, BT, K)

    # Bounded exponent basis for the intra-chunk term: pairwise (c[i-1] - c[j])
    # for j < i. c[i-1] = c[i] - w[i], stored as `c_minus_w[i,k]`, so the exponent
    # c_minus_w[i,k] - c[j,k] is <= 0 whenever j <= i-1 and w <= 0 along the run.
    c_minus_w = cg - wc  # [B,H,nc,BT,K]
    tril_strict = torch.tril(
        torch.ones(BT, BT, device=r.device), diagonal=-1
    )  # [BT,BT]

    S = r.new_zeros(B, H, K, V).float()
    o = torch.zeros(B, H, nc, BT, V, device=r.device, dtype=torch.float32)
    u_bcast = u32[None, :, None, :]  # [1,H,1,K] -- broadcasts against the 4-D per-chunk slices [B,H,BT,K]

    for i in range(nc):
        ri, ki, vi, ci, cmw = (
            rc[:, :, i],
            kc[:, :, i],
            vc[:, :, i],
            cg[:, :, i],
            c_minus_w[:, :, i],
        )
        # Intra, j < i pairs: exponent cmw[i,k] - c[j,k] is <= 0.
        # P[b,h,i,j,k] = exp(cmw[b,h,i,k] - c[b,h,j,k]).
        # clamp: masked-out (i<j) entries carry large POSITIVE exponents that overflow to inf,
        # and inf*0 from the triangular mask becomes NaN. Valid (i>j) entries are always <= 0
        # because every gate is <= 0, so clamping cannot alter a graded value.
        P = (cmw.unsqueeze(-2) - ci.unsqueeze(-3)).clamp(max=0.0).exp()  # [B,H,BT_i,BT_j,K]
        A = torch.einsum("bhik,bhjk,bhijk->bhij", ri, ki, P) * tril_strict
        # Inter (state entering this chunk decays by exp(cmw[i])).
        o_inter = torch.einsum("bhik,bhkv->bhiv", ri * cmw.exp(), S)
        # u-diagonal: u * (r_i^T k_i) * v_i, applied ONLY to the output.
        diag_rk = (ri * u_bcast * ki).sum(-1, keepdim=True)  # [B,H,BT,1]
        o_diag = diag_rk * vi  # [B,H,BT,V]
        o[:, :, i] = o_inter + A @ vi + o_diag
        # Advance state (u NOT included).
        cl = ci[:, :, -1, :]  # [B,H,K]
        kdec = ki * (cl.unsqueeze(-2) - ci).exp()  # exponent <= 0
        S = S * cl[..., None].exp() + kdec.transpose(-1, -2) @ vi

    o = o.reshape(B, H, Tp, V)[:, :, :T0].transpose(1, 2).contiguous()
    return o.to(v.dtype)


class _RWKV6Oracle(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r, k, v, w, u):
        with torch.no_grad():
            out = _chunk_core(r, k, v, w, u, use_triton_cumgate=True)
        ctx.save_for_backward(r, k, v, w, u)
        return out

    @staticmethod
    def backward(ctx, dout):
        r, k, v, w, u = ctx.saved_tensors
        with torch.enable_grad():
            rd = r.detach().requires_grad_()
            kd = k.detach().requires_grad_()
            vd = v.detach().requires_grad_()
            wd = w.detach().requires_grad_()
            ud = u.detach().requires_grad_()
            out = _chunk_core(rd, kd, vd, wd, ud, use_triton_cumgate=False)
        return torch.autograd.grad(out, [rd, kd, vd, wd, ud], dout, allow_unused=True)


def chunked_rwkv6(r, k, v, w, u):
    """Entry point (same signature/semantics as the starter)."""
    return _RWKV6Oracle.apply(r, k, v, w, u)
# FORGE-CANARY-BEGIN forge-canary/v1
# FORGE-CANARY-SLOT-0 beb3630eb2386931f3095668f3f06d7b4e2f0627b5e45a26f7797d6475c30a0a
# FORGE-CANARY-SLOT-1 95ad8a25a8c85262317d4d57b0fbd2237a8bb3ac27393641c70b056941aac641
# FORGE-CANARY-SLOT-2 1462a0a75eeda29612d75c913ed08043ce8515d77567866dcba50c9dca58e168
# FORGE-CANARY-SLOT-3 1dedf61af7593b69390069711c6dedc20133b4403d829b53e6875b0a1e3af5b3
# FORGE-CANARY-END
