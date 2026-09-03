"""
Reference implementation -- PyTorch-only ground truth for correctness verification.
DO NOT MODIFY. This is the oracle the benchmark harness checks every candidate against.

Family: moe_fused_experts (fused mixture-of-experts expert compute with
PRECOMPUTED top-k routing, forward + backward). This member is the ReGLU-squared
gated expert: the SwiGLU shape (gate + up projection sharing one w1, output
projection w2) with the gate nonlinearity switched from silu to squared-ReLU
(a.k.a. ReLU^2, the Primer/Squared-ReLU gate). The activation is nonstandard,
NOT covered by torch.nn.functional as a single fused op, and its epilogue is
piecewise: relu(g) is zero on half of g's support, so the fused MMA epilogue
that a fast kernel must implement is a masked square-and-multiply.

    relu2_glu_moe(x, w1, w2, topk_idx, topk_w) -> y

    x:        [T, D]     tokens (float32 or bfloat16)
    w1:       [E, D, 2F] expert input projections; columns [:, :, :F] are the
                         GATE projection, columns [:, :, F:] are the UP projection
    w2:       [E, F, D]  expert output projections
    topk_idx: [T, A]     int64 expert ids in [0, E); PRECOMPUTED routing, carries
                         no gradient. The same expert MAY appear in more than one
                         slot of a token (each slot contributes independently).
    topk_w:   [T, A]     routing weights, same dtype as x, already normalized
    y:        [T, D]     same dtype as x

Semantics, exactly:

    y[t] = sum_a topk_w[t, a] * f(topk_idx[t, a], x[t])        (a ascending)
    f(e, h) = ((relu(h @ w1[e, :, :F]))^2 * (h @ w1[e, :, F:])) @ w2[e]
    relu(z) = max(z, 0)

  * Arithmetic is accumulated in float32; only the final y is cast back to the
    input dtype. Combine order per token is fixed (ascending slot).
  * Forward AND backward. Graded gradients: dx, dw1, dw2, dtopk_w. topk_idx is
    integer and has no gradient. Experts no token routes to receive exactly
    zero weight gradients. The squared-relu backward has a piecewise derivative
    d/dg [relu(g)^2] = 2 * relu(g) * (g > 0) that a candidate must implement
    without smoothing.
"""

import torch


def _relu2(z: torch.Tensor) -> torch.Tensor:
    r = torch.nn.functional.relu(z)
    return r * r


def relu2_glu_moe_ref(
    x: torch.Tensor,  # [T, D]
    w1: torch.Tensor,  # [E, D, 2F]
    w2: torch.Tensor,  # [E, F, D]
    topk_idx: torch.Tensor,  # [T, A] int64
    topk_w: torch.Tensor,  # [T, A] dtype of x
) -> torch.Tensor:  # [T, D] dtype of x
    """Sequential ground truth: MoE experts with a ReLU^2 gate."""
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
            act = _relu2(gu[:, :F]) * gu[:, F:]
            contrib = contrib.index_copy(0, rows, act @ w2f[e])
        y = y + wf[:, a : a + 1] * contrib
    return y.to(x.dtype)


# ---------------------------------------------------------------------------
# CPU sanity check (authoring-time only). TINY shapes, CPU, independent float64
# brute-force + finite-difference spot check. NO GPU / Triton.
# ---------------------------------------------------------------------------
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
                r = torch.nn.functional.relu(g)
                act = (r * r) * u
                acc = acc + wd[t, a] * (act @ w2d[e])
            y[t] = acc
        return y

    T, D, F, E, A = 5, 8, 6, 4, 3
    x = torch.randn(T, D, device=dev)
    w1 = torch.randn(E, D, 2 * F, device=dev) * (D**-0.5)
    w2 = torch.randn(E, F, D, device=dev) * (F**-0.5)
    idx = torch.tensor(
        [[0, 1, 2], [1, 1, 0], [2, 0, 1], [0, 2, 2], [1, 0, 2]], dtype=torch.int64
    )
    assert idx.shape == (T, A) and int(idx.max()) < E - 1
    w = torch.softmax(torch.randn(T, A, device=dev), dim=-1)

    xg = x.clone().requires_grad_(True)
    w1g = w1.clone().requires_grad_(True)
    w2g = w2.clone().requires_grad_(True)
    wg = w.clone().requires_grad_(True)
    y = relu2_glu_moe_ref(xg, w1g, w2g, idx, wg)
    assert y.shape == (T, D)

    yb = brute(x, w1, w2, idx, w)
    assert torch.allclose(y.detach().double(), yb, atol=1e-4, rtol=1e-4), (
        f"forward mismatch vs brute: {(y.detach().double() - yb).abs().max()}"
    )

    y.sum().backward()
    for name, g, ref_t in (
        ("dx", xg.grad, xg),
        ("dw1", w1g.grad, w1g),
        ("dw2", w2g.grad, w2g),
        ("dtopk_w", wg.grad, wg),
    ):
        assert g is not None and g.shape == ref_t.shape and torch.isfinite(g).all(), (
            name
        )
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
        "relu2_glu_moe reference CPU sanity check: OK",
        "(fwd matches float64 brute; four grads finite; unused-expert grads "
        "zero; dtopk_w & dx finite-difference agree)",
    )
