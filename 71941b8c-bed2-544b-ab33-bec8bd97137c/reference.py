"""
Reference implementation -- PyTorch-only ground truth for correctness verification.
DO NOT MODIFY. This is the oracle the benchmark harness checks every candidate against.

The operator is the chunk-parallel GATED delta rule, the recurrence at the core of
Gated DeltaNet and Kimi Delta Attention. For a single head, with state S in R^{K x V}:

    S'_t = diag(exp(g_t)) . S_{t-1}
    S_t  = S'_t + k_t (v_t - S'_t^T k_t)^T beta_t
    o_t  = S_t^T (q_t / sqrt(K))

The decay is applied BEFORE the prediction: the delta term subtracts the DECAYED
state's own prediction (`S'_t^T k_t`), not the pre-decay one (`S_{t-1}^T k_t`). The
two forms differ whenever the gate is non-zero, by roughly 4e-2 relative -- above
the bf16 tolerance -- so a solver implementing the pre-decay form fails the
correctness gate. The code below is the definition; where any prose disagrees with
it, the code wins.

`g` is a scalar (per (step, head)) log-space decay applied to the whole state, and
`beta` is a per (step, head) write strength. Grouped value attention is supported:
`HV` value heads may share `H` key heads (`HV // H` value heads per key head), so a
kernel that assumes a one-to-one head mapping breaks -- which is the point of
grading it.

This is written as a readable sequential scan in float32 and is deliberately slow.
The surface is forward+backward: autograd differentiates this scan, and the input
gradients dq, dk, dv, dg, dbeta ARE the graded gradient oracle. Correctness is
defined as agreement with this function, never with any Triton or CUDA
implementation of it.

Ported verbatim (semantics, scale, grouping, decay-before-prediction) from the
project's calibrated KDA exemplar at seed/forge/categories/kda/bundle/reference.py.
"""

import torch


def chunked_gated_delta_ref(
    q: torch.Tensor,      # [B, T, H, K]
    k: torch.Tensor,      # [B, T, H, K]
    v: torch.Tensor,      # [B, T, HV, V]   HV may exceed H (grouped value attention)
    g: torch.Tensor,      # [B, T, HV]      log-space per-step decay (<= 0)
    beta: torch.Tensor,   # [B, T, HV]      per-step write strength in (0, 1]
) -> torch.Tensor:        # [B, T, HV, V]
    """Sequential ground truth for the gated delta rule.

    Everything is accumulated in float32 regardless of the input dtype. The delta
    rule subtracts the current (decayed) state's own prediction before writing, so
    a low-precision accumulator moves the fixed point of the recurrence rather than
    merely rounding the answer.
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    # FLA scales q by 1/sqrt(K) inside the op; the reference must match it.
    q32, k32, v32 = q.float() * (K ** -0.5), k.float(), v.float()
    # Grouped value attention: HV value heads share H key heads, HV // H to one.
    # q and k are expanded to the value-head count so the recurrence is uniform.
    HV = v.shape[2]
    if HV != H:
        rep = HV // H
        q32 = q32.repeat_interleave(rep, dim=2)
        k32 = k32.repeat_interleave(rep, dim=2)
        H = HV
    g32, beta32 = g.float(), beta.float()

    state = torch.zeros(B, H, K, V, dtype=torch.float32, device=q.device)
    out = torch.empty(B, T, H, V, dtype=torch.float32, device=q.device)

    for t in range(T):
        decay = g32[:, t].exp()[:, :, None, None]          # [B, H, 1, 1]
        state = state * decay                               # decay BEFORE prediction
        kt = k32[:, t]                                      # [B, H, K]
        pred = torch.einsum("bhk,bhkv->bhv", kt, state)     # (decayed S)^T k
        delta = (v32[:, t] - pred) * beta32[:, t][:, :, None]
        state = state + torch.einsum("bhk,bhv->bhkv", kt, delta)
        out[:, t] = torch.einsum("bhk,bhkv->bhv", q32[:, t], state)

    return out.to(v.dtype)


if __name__ == "__main__":
    # CPU-ONLY sanity check on TINY shapes. Asserts the scan runs, returns the
    # right shape/dtype, is finite, and is autograd-capable for all five inputs.
    # NO GPU / NO Triton here -- GPU calibration is a separate follow-up.
    torch.manual_seed(0)
    B, T, H, HV, K, V = 2, 6, 2, 4, 8, 8      # HV != H exercises grouped attention
    dev = "cpu"
    q = torch.nn.functional.normalize(torch.randn(B, T, H, K), dim=-1).requires_grad_()
    k = torch.nn.functional.normalize(torch.randn(B, T, H, K), dim=-1).requires_grad_()
    v = torch.randn(B, T, HV, V, dtype=torch.float32).requires_grad_()
    g = (-torch.rand(B, T, HV) * 0.5).requires_grad_()
    beta = (torch.rand(B, T, HV) * 0.9 + 0.05).requires_grad_()

    out = chunked_gated_delta_ref(q, k, v, g, beta)
    assert out.shape == (B, T, HV, V), out.shape
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all(), "reference produced non-finite output"

    loss = out.float().sum()
    loss.backward()
    for name, t_ in (("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta)):
        assert t_.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(t_.grad).all(), f"non-finite gradient for {name}"

    # A one-step scalar cross-check (H == HV == 1) against the explicit formula.
    q1 = torch.randn(1, 1, 1, 3)
    k1 = torch.randn(1, 1, 1, 3)
    v1 = torch.randn(1, 1, 1, 2)
    g1 = torch.tensor([[[-0.3]]])
    b1 = torch.tensor([[[0.7]]])
    got = chunked_gated_delta_ref(q1, k1, v1, g1, b1)[0, 0, 0]
    # T == 1: S starts at 0, decay*0 == 0, pred == 0, delta == v*beta, S = k (x) delta
    kk = k1[0, 0, 0].float()
    delta = v1[0, 0, 0].float() * 0.7
    S = torch.outer(kk, delta)                     # [K, V]
    o_manual = (q1[0, 0, 0].float() * (3 ** -0.5)) @ S
    assert torch.allclose(got, o_manual, atol=1e-5), (got, o_manual)
    print("chunked_gated_delta reference: CPU sanity OK",
          "| out", tuple(out.shape), out.dtype, "| grads dq,dk,dv,dg,dbeta finite")
