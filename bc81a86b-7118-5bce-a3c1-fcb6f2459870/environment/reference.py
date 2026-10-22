"""
Reference implementation -- PyTorch-only ground truth for correctness verification.
DO NOT MODIFY. This is the oracle the benchmark harness checks every candidate against.

The operator is the chunk-parallel gated delta rule, the recurrence at the core of
Gated DeltaNet and Kimi Delta Attention. For a single head, with state S in R^{K x V}:

    S_t = diag(exp(g_t)) . S_{t-1} + k_t (v_t - (diag(exp(g_t)) . S_{t-1})^T k_t)^T beta_t
    o_t = S_t^T q_t

The decay is applied BEFORE the prediction: the delta term subtracts the DECAYED
state's own prediction, not the pre-decay one. An earlier revision of this
docstring wrote the prediction as S_{t-1}^T k_t, which is a different operator and
disagrees with the code below by about 4e-2 relative -- above the bf16 tolerance,
so a solver implementing that form fails the correctness gate. The code is the
oracle; this text now matches it.

written here as a readable sequential scan in float32. It is deliberately slow.
Correctness is defined as agreement with this function, never as agreement with
any Triton or CUDA implementation of it.
"""

import torch


def gated_delta_rule_ref(
    q: torch.Tensor,      # [B, T, H, K]
    k: torch.Tensor,      # [B, T, H, K]
    v: torch.Tensor,      # [B, T, HV, V]   HV may exceed H (grouped value attention)
    g: torch.Tensor,      # [B, T, HV]      log-space per-step decay
    beta: torch.Tensor,   # [B, T, HV]      per-step write strength
) -> torch.Tensor:        # [B, T, HV, V]
    """Sequential ground truth for the gated delta rule.

    Everything is accumulated in float32 regardless of the input dtype. The
    delta rule subtracts the current state's prediction before writing, so a
    low-precision accumulator changes the fixed point rather than merely
    rounding the answer.
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    # FLA scales q by 1/sqrt(K) inside the op; the reference must match it.
    q32, k32, v32 = q.float() * (K ** -0.5), k.float(), v.float()
    # Grouped value attention: HV value heads share H key heads, HV // H to one.
    # A kernel that assumes a one-to-one head mapping breaks here, which is the
    # point of grading it. q and k are expanded to the value-head count so the
    # recurrence below is uniform.
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
        state = state * decay
        kt = k32[:, t]                                     # [B, H, K]
        pred = torch.einsum("bhk,bhkv->bhv", kt, state)    # S^T k
        delta = (v32[:, t] - pred) * beta32[:, t][:, :, None]
        state = state + torch.einsum("bhk,bhv->bhkv", kt, delta)
        out[:, t] = torch.einsum("bhk,bhkv->bhv", q32[:, t], state)

    return out.to(v.dtype)
