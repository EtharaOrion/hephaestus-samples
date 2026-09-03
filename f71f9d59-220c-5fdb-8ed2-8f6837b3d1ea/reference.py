"""
Reference implementation -- PyTorch-only ground truth for correctness verification.
DO NOT MODIFY. This is the oracle the benchmark harness checks every candidate against.

The operator is CHUNK-PARALLEL GATED DELTA-PRODUCT: a gated delta rule that
applies `num_householder` (N) delta updates PER TOKEN before the readout.
Distinct from chunked_gated_delta (single Householder) because each token
performs N nested `(I - k (beta*k)^T)` Householder transformations of the
state -- meaning the intra-chunk factorization must invert an N*BT x N*BT
lower-triangular system rather than a BT x BT one. This is the "generalized"
step from a single-write-per-token delta rule to a product-of-writes rule
described in the delta-product paper.

For a single (batch, head), with state S in R^{K x V}, per-token loop:

    S <- diag(exp(g_t)) . S                             (gate)
    for j in 0 .. N-1:                                  (N Householder steps)
        S <- S + (v_{t,j} - S^T k_{t,j})^T * k_{t,j} * beta_{t,j}
    o_t = S^T (q_t / sqrt(K))

Here `q` carries length T and `k, v, beta` carry length T*N: index `t*N + j`
is the j-th Householder tuple of token t. `g` carries length T (one gate per
token, applied ONCE before the N-fold write). `beta` is a per (step, head)
scalar write strength.

The N-fold intra-token write makes the intra-chunk delta term N stacked
`k (beta*k)^T` erasures, which chain multiplicatively along j -- the
lower-triangular `A_ab` matrix that captures the recurrence has (N*BT)
rows and columns instead of BT, and inverting it exactly is the WY term
whose size grows with num_householder.

This is written as a readable sequential scan in float32 and is deliberately slow.
The surface is forward+backward: autograd differentiates this scan, and the input
gradients dq, dk, dv, dg, dbeta ARE the graded gradient oracle. Correctness is
defined as agreement with this function, never with any Triton or CUDA
implementation of it. Semantics ported from the library's own ground truth,
fla/ops/gated_delta_product/naive.py::naive_recurrent_gated_delta_product
(read, never called or vendored).
"""

import torch


def chunked_gated_delta_product_ref(
    q: torch.Tensor,  # [B, T, H, K]
    k: torch.Tensor,  # [B, T*N, H, K]
    v: torch.Tensor,  # [B, T*N, H, V]
    g: torch.Tensor,  # [B, T, H]        log-space per-token decay (<= 0)
    beta: torch.Tensor,  # [B, T*N, H]      per (t, j, head) write strength in (0, 1]
    num_householder: int = 2,
) -> torch.Tensor:  # [B, T, H, V]
    """Sequential ground truth for the gated delta-product rule.

    Everything is accumulated in float32. q is scaled by 1/sqrt(K) to match the
    library's chunk_gated_delta_product default (scale=None). The N Householder
    writes chain multiplicatively per token; the gate is applied ONCE before the
    N writes.
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    N = int(num_householder)
    assert k.shape == (B, T * N, H, K), (k.shape, (B, T * N, H, K))
    assert v.shape == (B, T * N, H, V), v.shape
    assert beta.shape == (B, T * N, H), beta.shape
    q32 = q.float() * (K**-0.5)
    k32, v32, g32, b32 = k.float(), v.float(), g.float(), beta.float()

    state = torch.zeros(B, H, K, V, dtype=torch.float32, device=q.device)
    out = torch.empty(B, T, H, V, dtype=torch.float32, device=q.device)

    for t in range(T):
        # Single scalar per-token gate applied BEFORE the N Householder writes.
        state = state * g32[:, t].exp()[:, :, None, None]
        for j in range(N):
            k_tj = k32[:, t * N + j]  # [B, H, K]
            v_tj = v32[:, t * N + j]  # [B, H, V]
            b_tj = b32[:, t * N + j]  # [B, H]
            pred = torch.einsum("bhk,bhkv->bhv", k_tj, state)  # S^T k
            delta = (v_tj - pred) * b_tj[:, :, None]
            state = state + torch.einsum("bhk,bhv->bhkv", k_tj, delta)
        out[:, t] = torch.einsum("bhk,bhkv->bhv", q32[:, t], state)

    return out.to(v.dtype)


if __name__ == "__main__":
    # CPU-ONLY sanity check on TINY shapes. Asserts the scan runs, returns the
    # right shape/dtype, is finite, and is autograd-capable for all five inputs.
    # NO GPU / NO Triton here -- GPU calibration is a separate follow-up.
    torch.manual_seed(0)
    B, T, H, K, V, N = 2, 5, 3, 8, 8, 2
    q = torch.nn.functional.normalize(torch.randn(B, T, H, K), dim=-1).requires_grad_()
    k = torch.nn.functional.normalize(
        torch.randn(B, T * N, H, K), dim=-1
    ).requires_grad_()
    v = torch.randn(B, T * N, H, V, dtype=torch.float32).requires_grad_()
    g = (-torch.rand(B, T, H) * 0.5).requires_grad_()
    beta = (torch.rand(B, T * N, H) * 0.9 + 0.05).requires_grad_()

    out = chunked_gated_delta_product_ref(q, k, v, g, beta, num_householder=N)
    assert out.shape == (B, T, H, V), out.shape
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all(), "reference produced non-finite output"

    loss = out.float().sum()
    loss.backward()
    for name, t_ in (("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta)):
        assert t_.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(t_.grad).all(), f"non-finite gradient for {name}"

    # Scalar cross-check on N=1, T=1, H=K=V=1: reduces to a single Householder
    # delta step with gate applied to zero state (so gate has no effect on t=0).
    q1 = torch.tensor([[[[1.0]]]])  # [1,1,1,1]
    k1 = torch.tensor([[[[1.0]]]])
    v1 = torch.tensor([[[[2.0]]]])
    g1 = torch.tensor([[[-0.3]]])
    b1 = torch.tensor([[[0.7]]])
    got = chunked_gated_delta_product_ref(q1, k1, v1, g1, b1, num_householder=1)
    # gate on S=0 is 0; delta = (2 - 0)*0.7 = 1.4; S = 1*1.4 = 1.4; o = 1*K^-0.5*1.4
    expected = torch.tensor([[[[1.4]]]]) * (1**-0.5)
    assert torch.allclose(got, expected, atol=1e-6), (got, expected)
    print(
        "chunked_gated_delta_product reference: CPU sanity OK",
        "| out",
        tuple(out.shape),
        out.dtype,
        "| grads dq,dk,dv,dg,dbeta finite",
    )
