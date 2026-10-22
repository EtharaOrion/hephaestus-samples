"""
Reference implementation -- PyTorch-only ground truth for correctness verification.
DO NOT MODIFY. This is the oracle the benchmark harness checks every candidate against.

The operator is CHUNK-PARALLEL DPLR (Diagonal-Plus-Low-Rank) generalized delta
rule -- the recurrence at the core of RWKV-7. It generalizes gated linear
attention with a per-K-channel log-decay `gk` by ADDING a rank-1 correction
term `beta_t (S^T alpha_t)^T` per step. Distinct from every other member of
the chunked_linear_attn family: no other variant has BOTH a per-channel gate
AND a low-rank additive correction that mixes across the state's V axis.

For a single (batch, head), with state S in R^{K x V}:

    S_t = diag(exp(gk_t)) . S_{t-1}
          + k_t v_t^T
          + beta_t . ( S_{t-1}^T alpha_t )^T                (rank-1 DPLR correction)
    o_t = S_t^T ( q_t / sqrt(K) )

alpha_t, beta_t and k_t are all length-K vectors; the rank-1 update `beta_t
(S^T alpha_t)^T` reads a V-vector from the state through `alpha` and writes
it back through `beta`. Setting alpha == beta == 0 recovers plain gated linear
attention (GLA), and setting alpha == k, beta == -k*beta_scalar recovers the
delta-rule Householder form -- both are graded-distribution wrong. The intra-
chunk factorization must invert the (I + A_ab) matrix that captures the
alpha-beta erasure recurrence AND the standard A_qk gate-decay pattern; see
dplr_chunkwise in the pinned library's naive reference for the full algebra.

The graded distribution stresses this variant with LONG-CONTEXT recurrence
(T = 8192 on g1), where numerical error accumulates over many chunk boundaries
and the state carry across chunks matters more than the intra-chunk fusion. A
solver that keeps the chunk-boundary state in bf16 loses precision at every
carry and drifts far outside tolerance by t = 8192.

This is written as a readable sequential scan in float32 and is deliberately slow.
The surface is forward+backward: autograd differentiates this scan, and the input
gradients dq, dk, dv, dalpha, dbeta, dgk ARE the graded gradient oracle.
Correctness is defined as agreement with this function, never with any Triton or
CUDA implementation of it. Semantics ported from the library's own ground truth,
fla/ops/generalized_delta_rule/dplr/naive.py::dplr_recurrence (read, never
called or vendored).
"""

import torch


def chunked_dplr_delta_ref(
    q: torch.Tensor,  # [B, T, H, K]
    k: torch.Tensor,  # [B, T, H, K]
    v: torch.Tensor,  # [B, T, H, V]
    alpha: torch.Tensor,  # [B, T, H, K]    DPLR erasure basis
    beta: torch.Tensor,  # [B, T, H, K]    DPLR write basis
    gk: torch.Tensor,  # [B, T, H, K]    per-K-channel log-decay (<= 0)
) -> torch.Tensor:  # [B, T, H, V]
    """Sequential ground truth for chunk-parallel DPLR generalized delta rule.

    Everything is accumulated in float32. q is scaled by 1/sqrt(K) to match the
    library's chunk_dplr_delta_rule default (scale=None). The order matches
    the pinned naive dplr_recurrence: at each step, form the rank-1 alpha-beta
    outer product first, ADD the fresh k v^T, then decay-and-add.
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    q32 = q.float() * (K**-0.5)
    k32, v32 = k.float(), v.float()
    a32, b32, gk32 = alpha.float(), beta.float(), gk.float()

    state = torch.zeros(B, H, K, V, dtype=torch.float32, device=q.device)
    out = torch.empty(B, T, H, V, dtype=torch.float32, device=q.device)

    for t in range(T):
        alpha_t = a32[:, t]  # [B, H, K]
        beta_t = b32[:, t]  # [B, H, K]
        kt, vt = k32[:, t], v32[:, t]
        # Rank-1 DPLR update carried on the OLD state:
        #   (S^T alpha)^T is a [V]-vector read through alpha; beta writes it back.
        # In [K, V] tensor form: sum_k S[k, :] * alpha[k] -> [V]; then outer(beta, ...).
        sT_alpha = torch.einsum("bhk,bhkv->bhv", alpha_t, state)  # [B, H, V]
        rank1 = torch.einsum("bhk,bhv->bhkv", beta_t, sT_alpha)  # [B, H, K, V]
        kv = torch.einsum("bhk,bhv->bhkv", kt, vt)  # fresh outer product
        # decay . S_old + kv + rank1
        state = state * gk32[:, t].exp()[:, :, :, None] + kv + rank1
        out[:, t] = torch.einsum("bhk,bhkv->bhv", q32[:, t], state)

    return out.to(v.dtype)


if __name__ == "__main__":
    # CPU-ONLY sanity check on TINY shapes.
    torch.manual_seed(0)
    B, T, H, K, V = 2, 6, 3, 8, 8
    q = torch.nn.functional.normalize(torch.randn(B, T, H, K), dim=-1).requires_grad_()
    k = torch.nn.functional.normalize(torch.randn(B, T, H, K), dim=-1).requires_grad_()
    v = torch.randn(B, T, H, V).requires_grad_()
    alpha = torch.randn(B, T, H, K).requires_grad_()
    beta = torch.randn(B, T, H, K).requires_grad_()
    gk = torch.nn.functional.logsigmoid(torch.randn(B, T, H, K)).requires_grad_()

    out = chunked_dplr_delta_ref(q, k, v, alpha, beta, gk)
    assert out.shape == (B, T, H, V), out.shape
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()

    out.float().sum().backward()
    for name, t_ in (
        ("q", q),
        ("k", k),
        ("v", v),
        ("alpha", alpha),
        ("beta", beta),
        ("gk", gk),
    ):
        assert t_.grad is not None and torch.isfinite(t_.grad).all(), name

    # Scalar cross-check: T=1, H=K=V=1. S starts at 0, alpha/beta terms are zero,
    # so S = 0 + k*v = k*v; o = q * K^-0.5 * S.
    q1 = torch.tensor([[[[1.0]]]])
    k1 = torch.tensor([[[[1.0]]]])
    v1 = torch.tensor([[[[2.0]]]])
    a1 = torch.tensor([[[[3.0]]]])  # alpha_1 != 0 but S_0 == 0
    b1 = torch.tensor([[[[4.0]]]])  # so rank-1 term stays 0
    g1 = torch.tensor([[[[-0.3]]]])
    got = chunked_dplr_delta_ref(q1, k1, v1, a1, b1, g1)
    expected = torch.tensor([[[[2.0]]]])
    assert torch.allclose(got, expected, atol=1e-6), (got, expected)
    print(
        "chunked_dplr_delta reference: CPU sanity OK",
        "| out",
        tuple(out.shape),
        out.dtype,
        "| grads dq,dk,dv,dalpha,dbeta,dgk finite",
    )
