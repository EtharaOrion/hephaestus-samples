"""
Reference implementation -- PyTorch-only ground truth for correctness verification.
DO NOT MODIFY. This is the oracle the benchmark harness checks every candidate against.

Family: moe_fused_experts (fused mixture-of-experts expert compute with
PRECOMPUTED top-k routing, forward + backward). This member is SwiGLU-MoE
augmented with a **per-expert output bias** `b2[E, D]` that is added INTO the
down-projection output BEFORE the routing weight multiply. The bias is why the
hardness lever of this variant is "fused DOWN-projection": a fast kernel must
fuse the bias-add AND the routing-weight scalar multiply into the epilogue of
the second (down) GEMM without materializing `(act @ w2[e])` as an intermediate.

    fused_downproj_swiglu_moe(x, w1, w2, b2, topk_idx, topk_w) -> y

    x:        [T, D]     tokens (float32 or bfloat16)
    w1:       [E, D, 2F] expert input projections; [:, :, :F] gate, [:, :, F:] up
    w2:       [E, F, D]  expert output (down) projections
    b2:       [E, D]     per-expert output bias, same dtype as x
    topk_idx: [T, A]     int64 expert ids in [0, E); no gradient
    topk_w:   [T, A]     routing weights, same dtype as x, already normalized
    y:        [T, D]     same dtype as x

Semantics, exactly:

    y[t] = sum_a topk_w[t, a] * ( f(topk_idx[t, a], x[t]) )        (a ascending)
    f(e, h) = (silu(h @ w1[e, :, :F]) * (h @ w1[e, :, F:])) @ w2[e]  +  b2[e]
    silu(z) = z * sigmoid(z)

Every routed slot adds the routed expert's output bias BEFORE its topk_w scalar
multiply. The bias is INSIDE the routing sum: with A slots active per token
and the same expert e appearing k times in that token's routing, b2[e] is
added k times, each scaled by its own topk_w[t, a]. Experts no token routes
to receive exactly zero weight gradients AND exactly zero bias gradients.
Graded grads: dx, dw1, dw2, db2, dtopk_w (FIVE -- the family's four plus db2).
"""

import torch


def fused_downproj_swiglu_moe_ref(
    x: torch.Tensor,  # [T, D]
    w1: torch.Tensor,  # [E, D, 2F]
    w2: torch.Tensor,  # [E, F, D]
    b2: torch.Tensor,  # [E, D]
    topk_idx: torch.Tensor,  # [T, A] int64
    topk_w: torch.Tensor,  # [T, A] dtype of x
) -> torch.Tensor:  # [T, D] dtype of x
    """Sequential ground truth: SwiGLU MoE with per-expert output bias."""
    if x.dim() != 2 or w1.dim() != 3 or w2.dim() != 3 or b2.dim() != 2:
        raise ValueError(
            f"bad ranks: x{tuple(x.shape)} w1{tuple(w1.shape)} "
            f"w2{tuple(w2.shape)} b2{tuple(b2.shape)}"
        )
    T, D = x.shape
    E, D2, F2 = w1.shape
    F = F2 // 2
    if (
        D2 != D
        or F2 != 2 * F
        or tuple(w2.shape) != (E, F, D)
        or tuple(b2.shape) != (E, D)
    ):
        raise ValueError(
            f"inconsistent shapes: x{tuple(x.shape)} "
            f"w1{tuple(w1.shape)} w2{tuple(w2.shape)} "
            f"b2{tuple(b2.shape)}"
        )
    if (
        topk_idx.dtype != torch.int64
        or topk_idx.shape != topk_w.shape
        or topk_idx.shape[0] != T
    ):
        raise ValueError("topk_idx must be int64 [T, A] and match topk_w")
    A = topk_idx.shape[1]

    xf, w1f, w2f, b2f, wf = (t.float() for t in (x, w1, w2, b2, topk_w))
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
            fe = fe + b2f[e]  # bias fused with the down proj
            contrib = contrib.index_copy(0, rows, fe)
        y = y + wf[:, a : a + 1] * contrib
    return y.to(x.dtype)


# CPU sanity check
if __name__ == "__main__":
    torch.manual_seed(0)
    dev = "cpu"

    def brute(x, w1, w2, b2, idx, w):
        T, D = x.shape
        E, _, F2 = w1.shape
        F = F2 // 2
        A = idx.shape[1]
        xd, w1d, w2d, b2d, wd = (t.double() for t in (x, w1, w2, b2, w))
        y = torch.zeros(T, D, dtype=torch.float64)
        for t in range(T):
            acc = torch.zeros(D, dtype=torch.float64)
            for a in range(A):
                e = int(idx[t, a])
                h = xd[t]
                g = h @ w1d[e, :, :F]
                u = h @ w1d[e, :, F:]
                fe = (torch.nn.functional.silu(g) * u) @ w2d[e] + b2d[e]
                acc = acc + wd[t, a] * fe
            y[t] = acc
        return y

    T, D, F, E, A = 5, 8, 6, 4, 3
    x = torch.randn(T, D, device=dev)
    w1 = torch.randn(E, D, 2 * F, device=dev) * (D**-0.5)
    w2 = torch.randn(E, F, D, device=dev) * (F**-0.5)
    b2 = torch.randn(E, D, device=dev) * 0.01
    idx = torch.tensor(
        [[0, 1, 2], [1, 1, 0], [2, 0, 1], [0, 2, 2], [1, 0, 2]], dtype=torch.int64
    )
    assert idx.shape == (T, A) and int(idx.max()) < E - 1
    w = torch.softmax(torch.randn(T, A, device=dev), dim=-1)

    xg = x.clone().requires_grad_(True)
    w1g = w1.clone().requires_grad_(True)
    w2g = w2.clone().requires_grad_(True)
    b2g = b2.clone().requires_grad_(True)
    wg = w.clone().requires_grad_(True)
    y = fused_downproj_swiglu_moe_ref(xg, w1g, w2g, b2g, idx, wg)
    assert y.shape == (T, D)

    yb = brute(x, w1, w2, b2, idx, w)
    assert torch.allclose(y.detach().double(), yb, atol=1e-4, rtol=1e-4), (
        f"forward mismatch vs brute: {(y.detach().double() - yb).abs().max()}"
    )

    y.sum().backward()
    for name, gg, ref_t in (
        ("dx", xg.grad, xg),
        ("dw1", w1g.grad, w1g),
        ("dw2", w2g.grad, w2g),
        ("db2", b2g.grad, b2g),
        ("dtopk_w", wg.grad, wg),
    ):
        assert (
            gg is not None and gg.shape == ref_t.shape and torch.isfinite(gg).all()
        ), name
    assert (
        torch.count_nonzero(w1g.grad[3]) == 0 and torch.count_nonzero(w2g.grad[3]) == 0
    ), "unused expert w1/w2 grads must be zero"
    assert torch.count_nonzero(b2g.grad[3]) == 0, "unused expert bias grad must be zero"

    eps = 1e-5
    wp = w.clone()
    wp[0, 0] += eps
    wm = w.clone()
    wm[0, 0] -= eps
    fd_tw = (
        brute(x, w1, w2, b2, idx, wp).sum() - brute(x, w1, w2, b2, idx, wm).sum()
    ) / (2 * eps)
    assert abs(float(fd_tw) - float(wg.grad[0, 0])) < 1e-2
    bp = b2.clone()
    bp[0, 0] += eps
    bm = b2.clone()
    bm[0, 0] -= eps
    fd_b = (brute(x, w1, w2, bp, idx, w).sum() - brute(x, w1, w2, bm, idx, w).sum()) / (
        2 * eps
    )
    assert abs(float(fd_b) - float(b2g.grad[0, 0])) < 1e-2

    print(
        "fused_downproj_swiglu_moe reference CPU sanity check: OK",
        "(fwd matches float64 brute; five grads finite; unused-expert w/b "
        "grads zero; dtopk_w & db2 finite-difference agree)",
    )
