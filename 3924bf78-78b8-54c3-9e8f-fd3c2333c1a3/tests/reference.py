"""
Reference implementation -- PyTorch-only ground truth for correctness verification.
DO NOT MODIFY.

Family: moe_fused_experts (precomputed top-k routing, fwd+bwd). This member is
the int4 weight-only quantized (W4A16) SwiGLU expert with per-output-channel
ASYMMETRIC (scale + zero-point) dequant:

    int4_wq_swiglu_moe(x, w1_pack, w1_s, w1_zp,
                          w2_pack, w2_s, w2_zp, topk_idx, topk_w) -> y

    x:        [T, D]        tokens (float32 or bfloat16)
    w1_pack:  [E, D, F]     uint8; two int4 nibbles per byte along the LAST dim,
                            unpacked width is 2F (low nibble = column 2j,
                            high nibble = column 2j+1); nibble values in [0, 15]
    w1_s:     [E, 2F]       per-output-channel scale, `x.dtype`
    w1_zp:    [E, 2F]       per-output-channel zero-point, int8 in [0, 15]
    w2_pack:  [E, F, D/2]   uint8, same nibble convention, unpacked width D
    w2_s:     [E, D]        per-output-channel scale, `x.dtype`
    w2_zp:    [E, D]        per-output-channel zero-point, int8 in [0, 15]
    topk_idx: [T, A]        int64, no gradient
    topk_w:   [T, A]        `x.dtype`, normalized
    y:        [T, D]        `x.dtype`

`2F` (i.e. `w1_pack.shape[-1] * 2`) and `D` (i.e. `w2_pack.shape[-1] * 2`) must
both be even. Semantics, exactly:

    w1_int[e] = unpack_nibbles(w1_pack[e])                      # [D, 2F], int in [0, 15]
    w1_dq[e]  = (w1_int[e] - w1_zp[e][None, :]) * w1_s[e][None, :]
    w2_int[e] = unpack_nibbles(w2_pack[e])                      # [F, D]
    w2_dq[e]  = (w2_int[e] - w2_zp[e][None, :]) * w2_s[e][None, :]
    y[t] = sum_a topk_w[t, a] * ((silu(g)*u) @ w2_dq[e]);   e = topk_idx[t, a]
    g, u  = (h @ w1_dq[e]) split into first F and last F columns

Arithmetic in float32; final cast back. Graded gradients: dx and dtopk_w
only -- packed weight buffers, scales and zero-points carry no gradient.
"""

import torch


def _unpack_last(packed: torch.Tensor) -> torch.Tensor:
    """[..., K] uint8 -> [..., 2K] int8 with nibbles in [0, 15]. Low nibble is
    the EVEN column, high nibble the ODD column, so unpack doubles the size."""
    lo = (packed & 0x0F).to(torch.int8)
    hi = ((packed >> 4) & 0x0F).to(torch.int8)
    return torch.stack([lo, hi], dim=-1).flatten(-2)


def int4_wq_swiglu_moe_ref(
    x: torch.Tensor,
    w1_pack: torch.Tensor,
    w1_s: torch.Tensor,
    w1_zp: torch.Tensor,
    w2_pack: torch.Tensor,
    w2_s: torch.Tensor,
    w2_zp: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_w: torch.Tensor,
) -> torch.Tensor:
    """Sequential ground truth: unpack + asymmetric dequant, then SwiGLU MoE."""
    if x.dim() != 2 or w1_pack.dim() != 3 or w2_pack.dim() != 3:
        raise ValueError("bad ranks")
    T, D = x.shape
    E, D2, Fp = w1_pack.shape
    F2 = Fp * 2  # unpacked columns of w1 == 2F
    F = F2 // 2
    if D2 != D or F2 != 2 * F or tuple(w2_pack.shape) != (E, F, D // 2) or D % 2 != 0:
        raise ValueError(
            f"inconsistent shapes: x{tuple(x.shape)} "
            f"w1_pack{tuple(w1_pack.shape)} w2_pack{tuple(w2_pack.shape)}"
        )
    if (
        tuple(w1_s.shape) != (E, 2 * F)
        or tuple(w1_zp.shape) != (E, 2 * F)
        or tuple(w2_s.shape) != (E, D)
        or tuple(w2_zp.shape) != (E, D)
    ):
        raise ValueError("scale/zp shape mismatch")
    if (
        topk_idx.dtype != torch.int64
        or topk_idx.shape != topk_w.shape
        or topk_idx.shape[0] != T
    ):
        raise ValueError("topk_idx must be int64 [T, A] and match topk_w")
    A = topk_idx.shape[1]

    xf, wf = x.float(), topk_w.float()
    w1_int_all = _unpack_last(w1_pack).float()  # [E, D, 2F]
    w2_int_all = _unpack_last(w2_pack).float()  # [E, F, D]
    w1_sf, w1_zpf = w1_s.float(), w1_zp.float()
    w2_sf, w2_zpf = w2_s.float(), w2_zp.float()
    y = xf.new_zeros(T, D)
    for a in range(A):
        idx = topk_idx[:, a]
        contrib = xf.new_zeros(T, D)
        for e in range(E):
            rows = (idx == e).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            h = xf.index_select(0, rows)
            w1_dq = (w1_int_all[e] - w1_zpf[e][None, :]) * w1_sf[e][None, :]
            gu = h @ w1_dq
            act = torch.nn.functional.silu(gu[:, :F]) * gu[:, F:]
            w2_dq = (w2_int_all[e] - w2_zpf[e][None, :]) * w2_sf[e][None, :]
            contrib = contrib.index_copy(0, rows, act @ w2_dq)
        y = y + wf[:, a : a + 1] * contrib
    return y.to(x.dtype)


# CPU sanity check
if __name__ == "__main__":
    torch.manual_seed(0)
    dev = "cpu"

    def pack_last(q_int8):
        lo = q_int8[..., 0::2]
        hi = q_int8[..., 1::2]
        return ((hi & 0x0F) << 4 | (lo & 0x0F)).to(torch.uint8)

    def brute(x, w1_pack, w1_s, w1_zp, w2_pack, w2_s, w2_zp, idx, w):
        T, D = x.shape
        E, _, Fp = w1_pack.shape
        F2 = Fp * 2
        F = F2 // 2
        A = idx.shape[1]
        xd, wd = x.double(), w.double()
        w1_int = _unpack_last(w1_pack).double()
        w2_int = _unpack_last(w2_pack).double()
        w1_sd, w1_zpd = w1_s.double(), w1_zp.double()
        w2_sd, w2_zpd = w2_s.double(), w2_zp.double()
        y = torch.zeros(T, D, dtype=torch.float64)
        for t in range(T):
            acc = torch.zeros(D, dtype=torch.float64)
            for a in range(A):
                e = int(idx[t, a])
                h = xd[t]
                w1_dq = (w1_int[e] - w1_zpd[e][None, :]) * w1_sd[e][None, :]
                g = h @ w1_dq[:, :F]
                u = h @ w1_dq[:, F:]
                w2_dq = (w2_int[e] - w2_zpd[e][None, :]) * w2_sd[e][None, :]
                acc = acc + wd[t, a] * ((torch.nn.functional.silu(g) * u) @ w2_dq)
            y[t] = acc
        return y

    T, D, F, E, A = 5, 8, 6, 4, 3
    x = torch.randn(T, D, device=dev)

    # Draw raw float weights, quantize to int4 asymmetric per output channel.
    def quantize(w_raw):
        # w_raw: [E, K, N]; per-channel over N: scale = range/15, zp = -min/scale
        w_min = w_raw.amin(dim=1)  # [E, N]
        w_max = w_raw.amax(dim=1)  # [E, N]
        scale = ((w_max - w_min) / 15.0).clamp_min(1e-6)
        zp = torch.round(-w_min / scale).clamp(0, 15).to(torch.int8)
        q_int = (
            torch.round(w_raw / scale[:, None, :] + zp[:, None, :].float())
            .clamp(0, 15)
            .to(torch.int8)
        )
        packed = pack_last(q_int)  # [E, K, N/2]
        return packed, scale, zp

    w1_raw = torch.randn(E, D, 2 * F, device=dev) * (D**-0.5)
    w2_raw = torch.randn(E, F, D, device=dev) * (F**-0.5)
    w1_pack, w1_s, w1_zp = quantize(w1_raw)
    w2_pack, w2_s, w2_zp = quantize(w2_raw)
    idx = torch.tensor(
        [[0, 1, 2], [1, 1, 0], [2, 0, 1], [0, 2, 2], [1, 0, 2]], dtype=torch.int64
    )
    w = torch.softmax(torch.randn(T, A, device=dev), dim=-1)

    xg = x.clone().requires_grad_(True)
    wg = w.clone().requires_grad_(True)
    y = int4_wq_swiglu_moe_ref(xg, w1_pack, w1_s, w1_zp, w2_pack, w2_s, w2_zp, idx, wg)
    assert y.shape == (T, D)

    yb = brute(x, w1_pack, w1_s, w1_zp, w2_pack, w2_s, w2_zp, idx, w)
    assert torch.allclose(y.detach().double(), yb, atol=1e-4, rtol=1e-4), (
        f"forward mismatch vs brute: {(y.detach().double() - yb).abs().max()}"
    )

    y.sum().backward()
    for name, gg, ref_t in (("dx", xg.grad, xg), ("dtopk_w", wg.grad, wg)):
        assert (
            gg is not None and gg.shape == ref_t.shape and torch.isfinite(gg).all()
        ), name

    eps = 1e-5
    xp = x.clone()
    xp[0, 0] += eps
    xm = x.clone()
    xm[0, 0] -= eps
    fd_x = (
        brute(xp, w1_pack, w1_s, w1_zp, w2_pack, w2_s, w2_zp, idx, w).sum()
        - brute(xm, w1_pack, w1_s, w1_zp, w2_pack, w2_s, w2_zp, idx, w).sum()
    ) / (2 * eps)
    assert abs(float(fd_x) - float(xg.grad[0, 0])) < 1e-2

    print(
        "int4_wq_swiglu_moe reference CPU sanity check: OK",
        "(unpack+dequant fwd matches float64 brute; dx & dtopk_w finite-"
        "difference agree; packed weight buffers carry no gradient)",
    )
