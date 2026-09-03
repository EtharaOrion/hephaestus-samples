"""
Reference implementation -- PyTorch-only ground truth for correctness verification.
DO NOT MODIFY. This is the oracle the benchmark harness checks every candidate against.

The operator is CHUNK-PARALLEL COMBA (Compositional Basis Attention): a gated
delta rule whose delta correction erases the state against an AUXILIARY key `p`,
distinct from the write key `k`. Distinct from every other member of the
chunked_linear_attn family: no other variant carries a separate prediction basis
`p` that decouples the erase from the write.

For a single (batch, head), with state S in R^{K x V}:

    S <- diag(exp(g_t)) . S                                       (gate)
    delta_t = ( v_t - S^T p_t ) * beta_t                          (erase against p)
    S <- S + k_t . delta_t^T                                      (write with k)
    o_t = S^T ( q_t / sqrt(K) )

`g` is a per (step, head) scalar log-decay (<= 0). `beta` is a per (step, head)
scalar write strength. `p` and `k` are BOTH length-K but semantically distinct:
`p` is the auxiliary basis the state is REGRESSED against (erase term uses `p`)
while `k` is the WRITE basis (the outer product added to the state uses `k`).
The two are always shipped together and the intra-chunk WY factorization uses
the cross basis `p k^T` (not `k k^T`), so a kernel that uses `k` for the erase
term is wrong on both the output and every gradient.

This is written as a readable sequential scan in float32 and is deliberately slow.
The surface is forward+backward: autograd differentiates this scan, and the input
gradients dq, dk, dv, dp, dg, dbeta ARE the graded gradient oracle. Correctness
is defined as agreement with this function, never with any Triton or CUDA
implementation of it. Semantics ported from the library's own ground truth,
fla/ops/comba/naive.py::naive_recurrent_comba (read, never called or vendored).
"""

import torch


def chunked_comba_ref(
    q: torch.Tensor,  # [B, T, H, K]
    k: torch.Tensor,  # [B, T, H, K]   write basis
    v: torch.Tensor,  # [B, T, H, V]
    p: torch.Tensor,  # [B, T, H, K]   auxiliary erase / prediction basis
    g: torch.Tensor,  # [B, T, H]      log-space per-step scalar decay (<= 0)
    beta: torch.Tensor,  # [B, T, H]      per-step write strength in (0, 1]
) -> torch.Tensor:  # [B, T, H, V]
    """Sequential ground truth for chunk-parallel COMBA.

    Everything is accumulated in float32. The erase term subtracts the state's
    projection on the AUXILIARY basis `p`, then the write term adds the outer
    product on `k`. Decoupling erase and write is the whole point of COMBA.
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    q32 = q.float() * (K**-0.5)
    k32, v32, p32, g32, b32 = k.float(), v.float(), p.float(), g.float(), beta.float()

    state = torch.zeros(B, H, K, V, dtype=torch.float32, device=q.device)
    out = torch.empty(B, T, H, V, dtype=torch.float32, device=q.device)

    for t in range(T):
        state = state * g32[:, t].exp()[:, :, None, None]  # gate BEFORE write
        pt = p32[:, t]  # [B, H, K]
        kt = k32[:, t]  # [B, H, K]
        vt = v32[:, t]  # [B, H, V]
        # Erase against p, not k.
        pred = torch.einsum("bhk,bhkv->bhv", pt, state)
        delta = (vt - pred) * b32[:, t][:, :, None]
        # Write with k, not p.
        state = state + torch.einsum("bhk,bhv->bhkv", kt, delta)
        out[:, t] = torch.einsum("bhk,bhkv->bhv", q32[:, t], state)

    return out.to(v.dtype)


if __name__ == "__main__":
    # CPU-ONLY sanity check on TINY shapes. Asserts the scan runs, returns the
    # right shape/dtype, is finite, and is autograd-capable for all six inputs.
    # NO GPU / NO Triton here -- GPU calibration is a separate follow-up.
    torch.manual_seed(0)
    B, T, H, K, V = 2, 6, 3, 8, 8
    q = torch.nn.functional.normalize(torch.randn(B, T, H, K), dim=-1).requires_grad_()
    k = torch.nn.functional.normalize(torch.randn(B, T, H, K), dim=-1).requires_grad_()
    v = torch.randn(B, T, H, V).requires_grad_()
    p = torch.nn.functional.normalize(torch.randn(B, T, H, K), dim=-1).requires_grad_()
    g = (-torch.rand(B, T, H) * 0.5).requires_grad_()
    beta = (torch.rand(B, T, H) * 0.9 + 0.05).requires_grad_()

    out = chunked_comba_ref(q, k, v, p, g, beta)
    assert out.shape == (B, T, H, V), out.shape
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all(), "reference produced non-finite output"

    loss = out.float().sum()
    loss.backward()
    for name, t_ in (("q", q), ("k", k), ("v", v), ("p", p), ("g", g), ("beta", beta)):
        assert t_.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(t_.grad).all(), f"non-finite gradient for {name}"

    # Scalar cross-check: T=1, H=K=V=1. S starts at 0, so pred=0, delta = v*beta,
    # S = k * v * beta; o = q * K^-0.5 * S.
    q1 = torch.tensor([[[[1.0]]]])
    k1 = torch.tensor([[[[1.0]]]])
    v1 = torch.tensor([[[[2.0]]]])
    p1 = torch.tensor([[[[0.5]]]])  # p differs from k on purpose
    g1 = torch.tensor([[[-0.3]]])
    b1 = torch.tensor([[[0.7]]])
    got = chunked_comba_ref(q1, k1, v1, p1, g1, b1)
    # pred = 0.5 * 0 = 0; delta = 2 * 0.7 = 1.4; S = 1 * 1.4 = 1.4; o = 1 * 1^-0.5 * 1.4
    expected = torch.tensor([[[[1.4]]]])
    assert torch.allclose(got, expected, atol=1e-6), (got, expected)
    print(
        "chunked_comba reference: CPU sanity OK",
        "| out",
        tuple(out.shape),
        out.dtype,
        "| grads dq,dk,dv,dp,dg,dbeta finite",
    )
