"""
Reference implementation -- PyTorch-only ground truth for correctness verification.
DO NOT MODIFY. This is the oracle the benchmark harness checks every candidate against.

The operator is the SPECULATIVE-DECODE regime of the GATED DELTA RULE: on every
call the model proposes S candidate tokens per batch, the verifier accepts a
per-batch prefix of length `commit_len[b]` in [0, S], and the state advances
through those committed steps only. Outputs `o` are computed for ALL S positions
per batch (they are the speculative reads the acceptance check needs), but
`state_out[b]` is the state after step `commit_len[b] - 1` (or `state0[b]` when
commit_len[b] == 0). This is the standard tree-free linear speculative-decode
contract lifted onto a recurrent (linear-attention) operator, and no library
primitive implements it: fla's fused_recurrent_gated_delta_rule always advances
the state through every one of the S input steps.

For each batch b and head h, per step t:

    S_t = exp(g_t) * S_{t-1} + k_t delta_t^T   (with delta_t = (v_t - S_{t-1}^T k_t) beta_t)
    o_t = S_t^T (q_t / sqrt(K))                       (always computed for reads)
    state_out[b] = S_{commit_len[b] - 1}              (advance stops at commit)

The scalar gate `g_t` is applied BEFORE the delta write (family convention).
q is [B, S, H, K]; k, v are [B, S, H, {K, V}]; beta and g are [B, S, H];
commit_len is [B] int32 in [0, S]; state0 is [B, H, K, V] float32. BOTH outputs
graded: o [B, S, H, V] in v's dtype and state_out [B, H, K, V] float32.
"""

import torch
import torch.nn.functional as F


def speculative_gated_delta_decode_ref(
    q: torch.Tensor,  # [B, S, H, K]
    k: torch.Tensor,  # [B, S, H, K]
    v: torch.Tensor,  # [B, S, H, V]
    g: torch.Tensor,  # [B, S, H]       float32 log-decay (<= 0)
    beta: torch.Tensor,  # [B, S, H]                 write strength
    commit_len: torch.Tensor,  # [B]             int32 in [0, S]  per-batch commit
    state0: torch.Tensor,  # [B, H, K, V]    float32 initial state
):  # -> (o [B, S, H, V] in v.dtype, state_out [B, H, K, V] fp32)
    """Sequential ground truth: run S steps of the gated delta rule for reads,
    freeze the state at commit_len[b] per batch."""
    B, S, H, K = q.shape
    V = v.shape[-1]
    q32 = q.float() * (K**-0.5)
    k32, v32, g32, b32 = k.float(), v.float(), g.float(), beta.float()

    # Working per-batch state; advanced through every step so reads see the
    # speculative view, then a per-batch snapshot is taken at commit_len[b].
    speculative = state0.float().clone()
    committed = state0.float().clone()
    out = torch.empty(B, S, H, V, dtype=torch.float32, device=q.device)
    cl = commit_len.long()

    for t in range(S):
        speculative = speculative * g32[:, t].exp()[:, :, None, None]
        kt = k32[:, t]
        pred = torch.einsum("bhk,bhkv->bhv", kt, speculative)
        delta = (v32[:, t] - pred) * b32[:, t][:, :, None]
        speculative = speculative + torch.einsum("bhk,bhv->bhkv", kt, delta)
        out[:, t] = torch.einsum("bhk,bhkv->bhv", q32[:, t], speculative)
        # Batches whose commit horizon reaches t + 1 snap here.
        mask = cl == (t + 1)
        if mask.any():
            committed[mask] = speculative[mask]

    return out.to(v.dtype), committed


if __name__ == "__main__":
    torch.manual_seed(0)
    B, S, H, K, V = 3, 6, 2, 8, 6
    q = torch.randn(B, S, H, K)
    k = torch.randn(B, S, H, K)
    v = torch.randn(B, S, H, V)
    g = -torch.rand(B, S, H) * 0.5
    beta = torch.rand(B, S, H) * 0.9 + 0.05
    cl = torch.tensor([0, 3, S], dtype=torch.int32)
    st = torch.randn(B, H, K, V) * 0.5
    o, s = speculative_gated_delta_decode_ref(q, k, v, g, beta, cl, st)
    assert o.shape == (B, S, H, V) and s.shape == (B, H, K, V)
    assert s.dtype == torch.float32
    assert torch.isfinite(o).all() and torch.isfinite(s).all()

    # commit_len == 0 -> state_out == state0 (that batch never advances state).
    assert torch.allclose(s[0], st[0], atol=1e-6), (s[0] - st[0]).abs().max()

    # commit_len == S -> matches the plain gated delta rule on that batch.
    def _plain(q, k, v, g, beta, st):
        B, S, H, K = q.shape
        V = v.shape[-1]
        q32 = q.float() * (K**-0.5)
        k32, v32, g32, b32 = k.float(), v.float(), g.float(), beta.float()
        state = st.float().clone()
        out = torch.empty(B, S, H, V, dtype=torch.float32, device=q.device)
        for t in range(S):
            state = state * g32[:, t].exp()[:, :, None, None]
            pred = torch.einsum("bhk,bhkv->bhv", k32[:, t], state)
            delta = (v32[:, t] - pred) * b32[:, t][:, :, None]
            state = state + torch.einsum("bhk,bhv->bhkv", k32[:, t], delta)
            out[:, t] = torch.einsum("bhk,bhkv->bhv", q32[:, t], state)
        return out, state

    op, sp = _plain(q[2:3], k[2:3], v[2:3], g[2:3], beta[2:3], st[2:3])
    assert torch.allclose(s[2:3], sp, atol=1e-5)
    assert torch.allclose(o[2:3].float(), op, atol=1e-5)

    # Reads o are always the speculative reads; changing commit_len does not
    # change any element of o -- proves the read/commit split.
    cl2 = torch.tensor([S, 0, 2], dtype=torch.int32)
    o2, _ = speculative_gated_delta_decode_ref(q, k, v, g, beta, cl2, st)
    assert torch.equal(o, o2), "o must not depend on commit_len"

    # Snapshot at commit_len == 3 for batch 1 must be the third step's state.
    _, ssnap = _plain(
        q[1:2, :3], k[1:2, :3], v[1:2, :3], g[1:2, :3], beta[1:2, :3], st[1:2]
    )
    assert torch.allclose(s[1:2], ssnap, atol=1e-5)

    print(
        "speculative_gated_delta_decode_ref CPU sanity OK:",
        tuple(o.shape),
        o.dtype,
        "| state",
        tuple(s.shape),
        s.dtype,
        "| commit",
        cl.tolist(),
    )
