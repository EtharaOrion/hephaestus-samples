"""
Reference implementation -- PyTorch-only ground truth for correctness verification.
DO NOT MODIFY. This is the oracle the benchmark harness checks every candidate against.

Family: moe_fused_experts (fused mixture-of-experts expert compute with
PRECOMPUTED top-k routing, forward + backward). This member is the HARDEST:
an ALWAYS-ON dense SHARED expert applied to EVERY token, ADDED to the routed
SwiGLU experts. Two structurally different GEMM regimes coexist in one op --
a dense [T,D]x[D,2F] shared path over all tokens, and an irregular
expert-grouped routed path -- and both are differentiated.

    shared_plus_routed_swiglu(x, ws1, ws2, w1, w2, topk_idx, topk_w) -> y

    x:        [T, D]     tokens (float32 or bfloat16)
    ws1:      [D, 2F]    SHARED expert input projection; [:, :F] gate, [:, F:] up
    ws2:      [F, D]     SHARED expert output projection
    w1:       [E, D, 2F] routed expert input projections; [:, :, :F] gate, [:, :, F:] up
    w2:       [E, F, D]  routed expert output projections
    topk_idx: [T, A]     int64 expert ids in [0, E); PRECOMPUTED routing, carries
                         no gradient. The same expert MAY appear in more than one
                         slot of a token (each slot contributes independently).
    topk_w:   [T, A]     routing weights, same dtype as x, already normalized
    y:        [T, D]     same dtype as x

Semantics, exactly:

    y[t] = f_shared(x[t]) + sum_a topk_w[t, a] * f(topk_idx[t, a], x[t])   (a ascending)
    f_shared(h) = (silu(h @ ws1[:, :F]) * (h @ ws1[:, F:])) @ ws2
    f(e, h)     = (silu(h @ w1[e, :, :F]) * (h @ w1[e, :, F:])) @ w2[e]
    silu(z)     = z * sigmoid(z)

  * The shared expert is applied to ALL tokens and is NOT weighted by any
    routing weight (it is always on, weight 1). It uses the SAME intermediate
    width F as the routed experts but its OWN weights ws1/ws2.
  * Arithmetic is accumulated in float32; only the final y is cast back to the
    input dtype. Combine order per token is fixed (ascending slot); the shared
    contribution is added last as written above (associativity within tol).
  * Forward AND backward. Graded gradients are dx, dws1, dws2, dw1, dw2 and
    topk_w -- the family's four PLUS the two shared-expert weight gradients,
    because the always-on shared path's backward is part of this operator.
    topk_idx is integer and has no gradient. Experts no token routes to receive
    exactly zero weight gradients (the shared expert always receives gradient).

The implementation is deliberately simple and slow (dense shared path + a
python loop over (slot, expert) pairs for the routed path), autograd-capable
end to end.
"""

import torch


def shared_plus_routed_swiglu_ref(x: torch.Tensor,        # [T, D]
                                  ws1: torch.Tensor,       # [D, 2F]
                                  ws2: torch.Tensor,       # [F, D]
                                  w1: torch.Tensor,        # [E, D, 2F]
                                  w2: torch.Tensor,        # [E, F, D]
                                  topk_idx: torch.Tensor,  # [T, A] int64
                                  topk_w: torch.Tensor,    # [T, A] dtype of x
                                  ) -> torch.Tensor:       # [T, D] dtype of x
    """Sequential ground truth: dense shared expert + routed SwiGLU experts."""
    if x.dim() != 2 or ws1.dim() != 2 or ws2.dim() != 2 \
            or w1.dim() != 3 or w2.dim() != 3:
        raise ValueError(f"bad ranks: x{tuple(x.shape)} ws1{tuple(ws1.shape)} "
                         f"ws2{tuple(ws2.shape)} w1{tuple(w1.shape)} w2{tuple(w2.shape)}")
    T, D = x.shape
    E, D2, F2 = w1.shape
    F = F2 // 2
    if D2 != D or F2 != 2 * F or tuple(w2.shape) != (E, F, D) \
            or tuple(ws1.shape) != (D, 2 * F) or tuple(ws2.shape) != (F, D):
        raise ValueError(f"inconsistent shapes: x{tuple(x.shape)} "
                         f"ws1{tuple(ws1.shape)} ws2{tuple(ws2.shape)} "
                         f"w1{tuple(w1.shape)} w2{tuple(w2.shape)}")
    if topk_idx.dtype != torch.int64 or topk_idx.shape != topk_w.shape \
            or topk_idx.shape[0] != T:
        raise ValueError("topk_idx must be int64 [T, A] and match topk_w")
    A = topk_idx.shape[1]

    xf, wf = x.float(), topk_w.float()
    ws1f, ws2f = ws1.float(), ws2.float()
    w1f, w2f = w1.float(), w2.float()

    # Routed path: python loop over (slot, expert) pairs.
    y = xf.new_zeros(T, D)
    for a in range(A):                      # combine order: ascending slot
        idx = topk_idx[:, a]
        contrib = xf.new_zeros(T, D)
        for e in range(E):
            rows = (idx == e).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            h = xf.index_select(0, rows)                       # [n, D]
            gu = h @ w1f[e]                                    # [n, 2F]
            fe = (torch.nn.functional.silu(gu[:, :F]) * gu[:, F:]) @ w2f[e]
            contrib = contrib.index_copy(0, rows, fe)
        y = y + wf[:, a:a + 1] * contrib

    # Shared path: dense over ALL tokens, always on (weight 1).
    gs = xf @ ws1f                                             # [T, 2F]
    shared = (torch.nn.functional.silu(gs[:, :F]) * gs[:, F:]) @ ws2f
    y = y + shared
    return y.to(x.dtype)


# ---------------------------------------------------------------------------
# CPU sanity check (authoring-time only). TINY shapes, CPU, independent float64
# brute-force recomputation + finite-difference spot check. NO GPU / Triton.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    dev = "cpu"

    def brute(x, ws1, ws2, w1, w2, idx, w):
        T, D = x.shape
        E, _, F2 = w1.shape
        F = F2 // 2
        A = idx.shape[1]
        xd, wd = x.double(), w.double()
        ws1d, ws2d = ws1.double(), ws2.double()
        w1d, w2d = w1.double(), w2.double()
        y = torch.zeros(T, D, dtype=torch.float64)
        for t in range(T):
            h = xd[t]
            gs = h @ ws1d
            acc = (torch.nn.functional.silu(gs[:F]) * gs[F:]) @ ws2d   # shared
            for a in range(A):
                e = int(idx[t, a])
                g = h @ w1d[e, :, :F]
                u = h @ w1d[e, :, F:]
                fe = (torch.nn.functional.silu(g) * u) @ w2d[e]
                acc = acc + wd[t, a] * fe
            y[t] = acc
        return y

    T, D, F, E, A = 5, 8, 6, 4, 3
    x = torch.randn(T, D, device=dev)
    ws1 = torch.randn(D, 2 * F, device=dev) * (D ** -0.5)
    ws2 = torch.randn(F, D, device=dev) * (F ** -0.5)
    w1 = torch.randn(E, D, 2 * F, device=dev) * (D ** -0.5)
    w2 = torch.randn(E, F, D, device=dev) * (F ** -0.5)
    idx = torch.tensor([[0, 1, 2], [1, 1, 0], [2, 0, 1],
                        [0, 2, 2], [1, 0, 2]], dtype=torch.int64)   # expert 3 unused
    assert idx.shape == (T, A) and int(idx.max()) < E - 1
    w = torch.softmax(torch.randn(T, A, device=dev), dim=-1)

    xg = x.clone().requires_grad_(True)
    ws1g = ws1.clone().requires_grad_(True)
    ws2g = ws2.clone().requires_grad_(True)
    w1g = w1.clone().requires_grad_(True)
    w2g = w2.clone().requires_grad_(True)
    wg = w.clone().requires_grad_(True)
    y = shared_plus_routed_swiglu_ref(xg, ws1g, ws2g, w1g, w2g, idx, wg)
    assert y.shape == (T, D) and y.dtype == torch.float32

    yb = brute(x, ws1, ws2, w1, w2, idx, w)
    assert torch.allclose(y.detach().double(), yb, atol=1e-4, rtol=1e-4), \
        f"forward mismatch vs brute: {(y.detach().double() - yb).abs().max()}"

    y.sum().backward()
    for name, g, ref_t in (("dx", xg.grad, xg), ("dws1", ws1g.grad, ws1g),
                           ("dws2", ws2g.grad, ws2g), ("dw1", w1g.grad, w1g),
                           ("dw2", w2g.grad, w2g), ("dtopk_w", wg.grad, wg)):
        assert g is not None and g.shape == ref_t.shape and torch.isfinite(g).all(), name
    # Routed expert 3 is unused -> zero routed grads; shared expert always used.
    assert torch.count_nonzero(w1g.grad[3]) == 0 and torch.count_nonzero(w2g.grad[3]) == 0, \
        "unused routed expert grads must be zero"
    assert torch.count_nonzero(ws1g.grad) > 0 and torch.count_nonzero(ws2g.grad) > 0, \
        "shared expert grads must be non-zero (always on)"

    eps = 1e-5
    wp = w.clone(); wp[0, 0] += eps
    wm = w.clone(); wm[0, 0] -= eps
    fd_tw = (brute(x, ws1, ws2, w1, w2, idx, wp).sum()
             - brute(x, ws1, ws2, w1, w2, idx, wm).sum()) / (2 * eps)
    assert abs(float(fd_tw) - float(wg.grad[0, 0])) < 1e-2
    # dx[0,0] picks up BOTH the shared and routed paths -- a good joint check.
    xp = x.clone(); xp[0, 0] += eps
    xm = x.clone(); xm[0, 0] -= eps
    fd_x = (brute(xp, ws1, ws2, w1, w2, idx, w).sum()
            - brute(xm, ws1, ws2, w1, w2, idx, w).sum()) / (2 * eps)
    assert abs(float(fd_x) - float(xg.grad[0, 0])) < 1e-2
    # dws1[0,0] exercises the shared-path backward specifically.
    s1p = ws1.clone(); s1p[0, 0] += eps
    s1m = ws1.clone(); s1m[0, 0] -= eps
    fd_s1 = (brute(x, s1p, ws2, w1, w2, idx, w).sum()
             - brute(x, s1m, ws2, w1, w2, idx, w).sum()) / (2 * eps)
    assert abs(float(fd_s1) - float(ws1g.grad[0, 0])) < 1e-2

    print("shared_plus_routed_swiglu reference CPU sanity check: OK",
          "(shared+routed fwd matches float64 brute; six grads finite; routed "
          "unused-expert grads zero; shared grads non-zero; dtopk_w, dx & dws1 "
          "finite-difference agree)")
