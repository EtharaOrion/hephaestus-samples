"""
Reference implementation -- PyTorch-only ground truth for correctness verification.
DO NOT MODIFY.

Family: moe_fused_experts (precomputed top-k routing, fwd+bwd). This member is
the fp8-e4m3 weight-only quantized SwiGLU expert:

    fp8_e4m3_swiglu_moe(x, w1_q, w1_s, w2_q, w2_s, topk_idx, topk_w) -> y

    x:        [T, D]     tokens (float32 or bfloat16)
    w1_q:     [E, D, 2F] "quantized" gate/up weights, stored in `x.dtype`
                         but with values snapped to the fp8-e4m3fn grid
    w1_s:     [E, 2F]    per-output-channel scale, `x.dtype`
    w2_q:     [E, F, D]  quantized down weights, `x.dtype`, on fp8-e4m3 grid
    w2_s:     [E, D]     per-output-channel scale, `x.dtype`
    topk_idx: [T, A]     int64, no gradient
    topk_w:   [T, A]     routing weights, `x.dtype`, normalized
    y:        [T, D]     `x.dtype`

Semantics, exactly:

    w1_dq[e] = w1_q[e] * w1_s[e][None, :]        (elementwise per output channel)
    w2_dq[e] = w2_q[e] * w2_s[e][None, :]
    y[t] = sum_a topk_w[t, a] * f(topk_idx[t, a], x[t])          (a ascending)
    f(e, h) = (silu(h @ w1_dq[e][:, :F]) * (h @ w1_dq[e][:, F:])) @ w2_dq[e]
    silu(z) = z * sigmoid(z)

Arithmetic is accumulated in float32; only the final y is cast back. Graded
gradients: `dx` and `dtopk_w` only -- quantized weight buffers `w1_q`, `w2_q`
and their scales `w1_s`, `w2_s` are inference-shaped inputs and carry no
gradient. `topk_idx` is integer and has no gradient. The hardness lever is
that a fast kernel fuses the per-out-channel dequant into the MMA prologue
of each expert GEMM (equivalently: uses fp8-e4m3 tensor cores directly on
the pre-snapped weights). A correctness-preserving bf16 GEMM on the same
values is legal but slow.
"""

import torch


def fp8_e4m3_swiglu_moe_ref(
    x: torch.Tensor,  # [T, D]
    w1_q: torch.Tensor,  # [E, D, 2F]
    w1_s: torch.Tensor,  # [E, 2F]
    w2_q: torch.Tensor,  # [E, F, D]
    w2_s: torch.Tensor,  # [E, D]
    topk_idx: torch.Tensor,  # [T, A] int64
    topk_w: torch.Tensor,  # [T, A] dtype of x
) -> torch.Tensor:  # [T, D] dtype of x
    """Sequential ground truth: dequantize then SwiGLU MoE."""
    if (
        x.dim() != 2
        or w1_q.dim() != 3
        or w2_q.dim() != 3
        or w1_s.dim() != 2
        or w2_s.dim() != 2
    ):
        raise ValueError("bad ranks")
    T, D = x.shape
    E, D2, F2 = w1_q.shape
    F = F2 // 2
    if (
        D2 != D
        or F2 != 2 * F
        or tuple(w2_q.shape) != (E, F, D)
        or tuple(w1_s.shape) != (E, 2 * F)
        or tuple(w2_s.shape) != (E, D)
    ):
        raise ValueError(
            f"inconsistent shapes: x{tuple(x.shape)} "
            f"w1_q{tuple(w1_q.shape)} w1_s{tuple(w1_s.shape)} "
            f"w2_q{tuple(w2_q.shape)} w2_s{tuple(w2_s.shape)}"
        )
    if (
        topk_idx.dtype != torch.int64
        or topk_idx.shape != topk_w.shape
        or topk_idx.shape[0] != T
    ):
        raise ValueError("topk_idx must be int64 [T, A] and match topk_w")
    A = topk_idx.shape[1]

    xf, w1qf, w1sf = x.float(), w1_q.float(), w1_s.float()
    w2qf, w2sf, wf = w2_q.float(), w2_s.float(), topk_w.float()
    y = xf.new_zeros(T, D)
    for a in range(A):
        idx = topk_idx[:, a]
        contrib = xf.new_zeros(T, D)
        for e in range(E):
            rows = (idx == e).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            h = xf.index_select(0, rows)
            w1dq = w1qf[e] * w1sf[e][None, :]
            gu = h @ w1dq
            act = torch.nn.functional.silu(gu[:, :F]) * gu[:, F:]
            w2dq = w2qf[e] * w2sf[e][None, :]
            contrib = contrib.index_copy(0, rows, act @ w2dq)
        y = y + wf[:, a : a + 1] * contrib
    return y.to(x.dtype)


# CPU sanity check
if __name__ == "__main__":
    torch.manual_seed(0)
    dev = "cpu"

    def snap_e4m3(w):
        """Snap values to the fp8-e4m3fn grid using torch's native dtype if
        available; otherwise fall back to a nearest-grid simulation."""
        if hasattr(torch, "float8_e4m3fn"):
            return w.to(torch.float8_e4m3fn).to(w.dtype)
        # simulation: e4m3 grid ~= 2^exp * (1 + mantissa/8), 448 max
        return torch.clamp(w, -448.0, 448.0)

    def brute(x, w1_q, w1_s, w2_q, w2_s, idx, w):
        T, D = x.shape
        E, _, F2 = w1_q.shape
        F = F2 // 2
        A = idx.shape[1]
        xd, w1qd, w1sd = x.double(), w1_q.double(), w1_s.double()
        w2qd, w2sd, wd = w2_q.double(), w2_s.double(), w.double()
        y = torch.zeros(T, D, dtype=torch.float64)
        for t in range(T):
            acc = torch.zeros(D, dtype=torch.float64)
            for a in range(A):
                e = int(idx[t, a])
                h = xd[t]
                w1dq = w1qd[e] * w1sd[e][None, :]
                g = h @ w1dq[:, :F]
                u = h @ w1dq[:, F:]
                w2dq = w2qd[e] * w2sd[e][None, :]
                acc = acc + wd[t, a] * ((torch.nn.functional.silu(g) * u) @ w2dq)
            y[t] = acc
        return y

    T, D, F, E, A = 5, 8, 6, 4, 3
    x = torch.randn(T, D, device=dev)
    # Symmetric per-channel quant: amax over each output column becomes scale.
    w1_raw = torch.randn(E, D, 2 * F, device=dev) * (D**-0.5)
    w1_s = w1_raw.abs().amax(dim=1).clamp_min(1e-6) / 448.0  # [E, 2F]
    w1_q = snap_e4m3(w1_raw / w1_s[:, None, :])
    w2_raw = torch.randn(E, F, D, device=dev) * (F**-0.5)
    w2_s = w2_raw.abs().amax(dim=1).clamp_min(1e-6) / 448.0  # [E, D]
    w2_q = snap_e4m3(w2_raw / w2_s[:, None, :])
    idx = torch.tensor(
        [[0, 1, 2], [1, 1, 0], [2, 0, 1], [0, 2, 2], [1, 0, 2]], dtype=torch.int64
    )
    w = torch.softmax(torch.randn(T, A, device=dev), dim=-1)

    xg = x.clone().requires_grad_(True)
    wg = w.clone().requires_grad_(True)
    y = fp8_e4m3_swiglu_moe_ref(xg, w1_q, w1_s, w2_q, w2_s, idx, wg)
    assert y.shape == (T, D)

    yb = brute(x, w1_q, w1_s, w2_q, w2_s, idx, w)
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
        brute(xp, w1_q, w1_s, w2_q, w2_s, idx, w).sum()
        - brute(xm, w1_q, w1_s, w2_q, w2_s, idx, w).sum()
    ) / (2 * eps)
    assert abs(float(fd_x) - float(xg.grad[0, 0])) < 1e-2

    print(
        "fp8_e4m3_swiglu_moe reference CPU sanity check: OK",
        "(dequant fwd matches float64 brute; dx & dtopk_w finite-difference "
        "agree; quantized weight buffers carry no gradient)",
    )
