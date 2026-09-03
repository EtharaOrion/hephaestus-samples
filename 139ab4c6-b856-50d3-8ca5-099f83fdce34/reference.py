"""
Reference implementation -- PyTorch-only ground truth for correctness verification.
DO NOT MODIFY. This is the oracle the benchmark harness checks every candidate against.

The operator is CHUNK-PARALLEL GDN-2 (Gated DeltaNet 2): a per-key-channel gated
delta rule with SEPARATE per-channel ERASE gate `b` (on the K axis) and per-channel
WRITE gate `w` (on the V axis). Distinct from every other member of the
chunked_linear_attn family because BOTH gates are per-channel AND the state is
the largest of the family (K = V = 256 on the graded distribution). Collapsing
`b = w = beta` (scalar) recovers KDA / gated_delta_rule -- so a starter that
pattern-matches on that special case is wrong on the graded distribution.

For a single (batch, head), with state S in R^{K x V}:

    S_t = ( I - k_t (b_t * k_t)^T ) . Diag(exp(g_t)) . S_{t-1} + k_t (w_t * v_t)^T
    o_t = S_t^T ( q_t / sqrt(K) )

which is implemented per token as:

    S <- Diag(exp(g_t)) . S                                    (per-K-channel gate)
    erase = ((b_t * k_t)^T . S)                                (per-K erase read)  [B,H,V]
    S <- S + k_t . (w_t * v_t - erase)^T                        (per-V write)
    o_t = S^T ( q_t / sqrt(K) )

Everything is accumulated in float32 regardless of the input dtype. Semantics
ported from the library's own ground truth, fla/ops/gdn2/naive.py::
naive_recurrent_gdn2 (read, never called or vendored).
"""

import torch


def chunked_gdn2_ref(
    q: torch.Tensor,  # [B, T, H, K]
    k: torch.Tensor,  # [B, T, H, K]
    v: torch.Tensor,  # [B, T, H, V]
    g: torch.Tensor,  # [B, T, H, K]   per-K-channel log-decay (<= 0)
    b: torch.Tensor,  # [B, T, H, K]   per-K-channel ERASE gate  (~[0, 2])
    w: torch.Tensor,  # [B, T, H, V]   per-V-channel WRITE gate  (~[0, 2])
) -> torch.Tensor:  # [B, T, H, V]
    """Sequential ground truth for GDN-2."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    q32 = q.float() * (K**-0.5)
    k32, v32, g32, b32, w32 = k.float(), v.float(), g.float(), b.float(), w.float()

    state = torch.zeros(B, H, K, V, dtype=torch.float32, device=q.device)
    out = torch.empty(B, T, H, V, dtype=torch.float32, device=q.device)

    for t in range(T):
        bt_k = b32[:, t] * k32[:, t]  # [B, H, K]
        # Per-K-channel decay on the K rows of the state.
        state = state * g32[:, t].exp()[:, :, :, None]
        # Erase term reads through the (b * k) key.
        erase = torch.einsum("bhk,bhkv->bhv", bt_k, state)  # [B, H, V]
        write = w32[:, t] * v32[:, t] - erase  # [B, H, V]
        state = state + torch.einsum("bhk,bhv->bhkv", k32[:, t], write)
        out[:, t] = torch.einsum("bhk,bhkv->bhv", q32[:, t], state)

    return out.to(v.dtype)


if __name__ == "__main__":
    # CPU-ONLY sanity check on TINY shapes.
    torch.manual_seed(0)
    B, T, H, K, V = 2, 6, 3, 8, 8
    q = torch.nn.functional.normalize(torch.randn(B, T, H, K), dim=-1).requires_grad_()
    k = torch.nn.functional.normalize(torch.randn(B, T, H, K), dim=-1).requires_grad_()
    v = torch.randn(B, T, H, V).requires_grad_()
    g = torch.nn.functional.logsigmoid(torch.randn(B, T, H, K)).requires_grad_()
    b = torch.sigmoid(torch.randn(B, T, H, K)).requires_grad_()
    w = torch.sigmoid(torch.randn(B, T, H, V)).requires_grad_()

    out = chunked_gdn2_ref(q, k, v, g, b, w)
    assert out.shape == (B, T, H, V), out.shape
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()

    out.float().sum().backward()
    for name, t_ in (("q", q), ("k", k), ("v", v), ("g", g), ("b", b), ("w", w)):
        assert t_.grad is not None and torch.isfinite(t_.grad).all(), name

    # Scalar cross-check: T=1, H=K=V=1. S starts at 0, erase = 0,
    # write = w*v; S = k * w*v; o = q * K^-0.5 * S.
    q1 = torch.tensor([[[[1.0]]]])
    k1 = torch.tensor([[[[1.0]]]])
    v1 = torch.tensor([[[[2.0]]]])
    g1 = torch.tensor([[[[-0.3]]]])
    b1 = torch.tensor([[[[0.7]]]])
    w1 = torch.tensor([[[[0.5]]]])
    got = chunked_gdn2_ref(q1, k1, v1, g1, b1, w1)
    # S=0 -> S*exp(g)=0 -> erase=0*=0 -> write=0.5*2 - 0 = 1.0 -> S = 1*1 = 1
    # o = 1 * 1^-0.5 * 1 = 1
    expected = torch.tensor([[[[1.0]]]])
    assert torch.allclose(got, expected, atol=1e-6), (got, expected)
    print(
        "chunked_gdn2 reference: CPU sanity OK",
        "| out",
        tuple(out.shape),
        out.dtype,
        "| grads dq,dk,dv,dg,db,dw finite",
    )
