"""
Reference implementation -- PyTorch-only ground truth for correctness verification.
DO NOT MODIFY.

Family: moe_fused_experts (precomputed top-k routing, fwd+bwd). This member is
the SwiGLU expert operator (silu-gated GLU) with the same math as swiglu_moe
but a shape distribution that pushes two hardness levers at once: (a) variable-
length grouped GEMM over an expert-major permutation with E up to 256 experts
and strongly imbalanced token counts per expert (many short/empty segments,
few long ones), and (b) wide-vs-tall aspect where the FFN intermediate F is
substantially larger than the model dim D (F ~= 2*D..3*D). The operator math
is identical to the family baseline; the reference below is the same
sequential (slot, expert)-loop oracle.

    wide_expert_swiglu_moe(x, w1, w2, topk_idx, topk_w) -> y

    x:        [T, D]
    w1:       [E, D, 2F]  gate = [:, :, :F], up = [:, :, F:]
    w2:       [E, F, D]
    topk_idx: [T, A] int64, values in [0, E); no gradient
    topk_w:   [T, A] same dtype as x, already normalized
    y:        [T, D] same dtype as x

    y[t] = sum_a topk_w[t, a] * f(topk_idx[t, a], x[t]),  a ascending;
    f(e, h) = (silu(h @ w1[e, :, :F]) * (h @ w1[e, :, F:])) @ w2[e];
    silu(z) = z * sigmoid(z).

Graded gradients: dx, dw1, dw2, dtopk_w (four). Experts no token routes to
receive exactly zero weight gradients. Combine order fixed (ascending slot).
"""

import torch


def wide_expert_swiglu_moe_ref(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_w: torch.Tensor,
) -> torch.Tensor:
    """Sequential ground truth for wide-FFN, high-E SwiGLU MoE."""
    if x.dim() != 2 or w1.dim() != 3 or w2.dim() != 3:
        raise ValueError(
            f"bad ranks: x{tuple(x.shape)} w1{tuple(w1.shape)} w2{tuple(w2.shape)}"
        )
    T, D = x.shape
    E, D2, F2 = w1.shape
    F = F2 // 2
    if D2 != D or F2 != 2 * F or tuple(w2.shape) != (E, F, D):
        raise ValueError(
            f"inconsistent shapes: x{tuple(x.shape)} "
            f"w1{tuple(w1.shape)} w2{tuple(w2.shape)}"
        )
    if (
        topk_idx.dtype != torch.int64
        or topk_idx.shape != topk_w.shape
        or topk_idx.shape[0] != T
    ):
        raise ValueError("topk_idx must be int64 [T, A] and match topk_w")
    A = topk_idx.shape[1]

    xf, w1f, w2f, wf = x.float(), w1.float(), w2.float(), topk_w.float()
    y = xf.new_zeros(T, D)
    for a in range(A):
        idx = topk_idx[:, a]
        contrib = xf.new_zeros(T, D)
        for e in range(E):
            rows = (idx == e).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            h = xf.index_select(0, rows)
            gu = h @ w1f[e]
            fe = (torch.nn.functional.silu(gu[:, :F]) * gu[:, F:]) @ w2f[e]
            contrib = contrib.index_copy(0, rows, fe)
        y = y + wf[:, a : a + 1] * contrib
    return y.to(x.dtype)


# CPU sanity check
if __name__ == "__main__":
    torch.manual_seed(0)
    dev = "cpu"

    def brute(x, w1, w2, idx, w):
        T, D = x.shape
        E, _, F2 = w1.shape
        F = F2 // 2
        A = idx.shape[1]
        xd, w1d, w2d, wd = (t.double() for t in (x, w1, w2, w))
        y = torch.zeros(T, D, dtype=torch.float64)
        for t in range(T):
            acc = torch.zeros(D, dtype=torch.float64)
            for a in range(A):
                e = int(idx[t, a])
                h = xd[t]
                g = h @ w1d[e, :, :F]
                u = h @ w1d[e, :, F:]
                fe = (torch.nn.functional.silu(g) * u) @ w2d[e]
                acc = acc + wd[t, a] * fe
            y[t] = acc
        return y

    # Tiny wide-FFN, many experts. Expert 3 unused so its grads should be zero.
    T, D, F, E, A = 5, 4, 8, 4, 2
    x = torch.randn(T, D, device=dev)
    w1 = torch.randn(E, D, 2 * F, device=dev) * (D**-0.5)
    w2 = torch.randn(E, F, D, device=dev) * (F**-0.5)
    idx = torch.tensor([[0, 1], [1, 2], [2, 0], [0, 2], [1, 0]], dtype=torch.int64)
    assert idx.shape == (T, A) and int(idx.max()) < E - 1
    w = torch.softmax(torch.randn(T, A, device=dev), dim=-1)

    xg = x.clone().requires_grad_(True)
    w1g = w1.clone().requires_grad_(True)
    w2g = w2.clone().requires_grad_(True)
    wg = w.clone().requires_grad_(True)
    y = wide_expert_swiglu_moe_ref(xg, w1g, w2g, idx, wg)
    assert y.shape == (T, D)

    yb = brute(x, w1, w2, idx, w)
    assert torch.allclose(y.detach().double(), yb, atol=1e-4, rtol=1e-4), (
        f"forward mismatch vs brute: {(y.detach().double() - yb).abs().max()}"
    )

    y.sum().backward()
    for name, gg, ref_t in (
        ("dx", xg.grad, xg),
        ("dw1", w1g.grad, w1g),
        ("dw2", w2g.grad, w2g),
        ("dtopk_w", wg.grad, wg),
    ):
        assert (
            gg is not None and gg.shape == ref_t.shape and torch.isfinite(gg).all()
        ), name
    assert (
        torch.count_nonzero(w1g.grad[3]) == 0 and torch.count_nonzero(w2g.grad[3]) == 0
    ), "unused expert grads must be zero"

    eps = 1e-5
    wp = w.clone()
    wp[0, 0] += eps
    wm = w.clone()
    wm[0, 0] -= eps
    fd_tw = (brute(x, w1, w2, idx, wp).sum() - brute(x, w1, w2, idx, wm).sum()) / (
        2 * eps
    )
    assert abs(float(fd_tw) - float(wg.grad[0, 0])) < 1e-2
    xp = x.clone()
    xp[0, 0] += eps
    xm = x.clone()
    xm[0, 0] -= eps
    fd_x = (brute(xp, w1, w2, idx, w).sum() - brute(xm, w1, w2, idx, w).sum()) / (
        2 * eps
    )
    assert abs(float(fd_x) - float(xg.grad[0, 0])) < 1e-2

    print(
        "wide_expert_swiglu_moe reference CPU sanity check: OK",
        "(fwd matches float64 brute; four grads finite; unused-expert grads "
        "zero; dtopk_w & dx finite-difference agree)",
    )
