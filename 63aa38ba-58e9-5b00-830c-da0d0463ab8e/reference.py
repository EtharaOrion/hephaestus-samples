"""
Reference implementation -- PyTorch-only ground truth for correctness verification.
DO NOT MODIFY. This is the oracle the benchmark harness checks every candidate against.

The operator is the DECODE regime of GATED LINEAR ATTENTION with a PAGED recurrent
state pool. Instead of a contiguous per-batch state tensor, the recurrent state of
every sequence lives in a shared pool `paged_state[P, H, K, V]` (P >= B), and a
`page_table[B]` int32 vector tells which physical page each batch owns for this
call. The gather is non-contiguous by design (page_table is a random permutation of
distinct page ids on the graded distribution), so a kernel that assumes a stride-1
per-batch state address is wrong on every graded input.

Every step of every batch is the plain GATED LINEAR ATTENTION recurrence, per-channel
forget gate on the key-row axis of the state, applied BEFORE the outer product write:

    S_t = diag(exp(g_t)) . S_{t-1} + k_t v_t^T
    o_t = S_t^T (q_t / sqrt(K))

The initial state for batch b is the page `paged_state[page_table[b]]` (float32).
The returned final state is a DENSE [B, H, K, V] float32 tensor holding the per-batch
final state (NOT the mutated pool -- the pool is read-only in this operator). Both
outputs are graded.
"""

import torch
import torch.nn.functional as F


def paged_gla_decode_ref(
    q: torch.Tensor,  # [B, S, H, K]
    k: torch.Tensor,  # [B, S, H, K]
    v: torch.Tensor,  # [B, S, H, V]
    g: torch.Tensor,  # [B, S, H, K]  float32  per-channel log-forget gate
    page_table: torch.Tensor,  # [B]           int32    physical page id per batch
    paged_state: torch.Tensor,  # [P, H, K, V]  float32  shared pool (P >= B)
):  # -> (o [B, S, H, V] in v.dtype, state_out [B, H, K, V] fp32)
    """Sequential ground truth: gather per-batch initial state, run GLA scan."""
    B, S, H, K = q.shape
    V = v.shape[-1]
    q32 = q.float() * (K**-0.5)
    k32, v32, g32 = k.float(), v.float(), g.float()

    # Gather the initial state from the pool via the page table (non-contiguous).
    idx = page_table.long()
    state = paged_state[idx].contiguous().clone()  # [B, H, K, V] float32
    out = torch.empty(B, S, H, V, dtype=torch.float32, device=q.device)

    for t in range(S):
        state = state * g32[:, t].exp()[:, :, :, None]  # [B,H,K,1] per-row
        state = state + torch.einsum("bhk,bhv->bhkv", k32[:, t], v32[:, t])
        out[:, t] = torch.einsum("bhk,bhkv->bhv", q32[:, t], state)

    return out.to(v.dtype), state


if __name__ == "__main__":
    torch.manual_seed(0)
    B, S, H, K, V, P = 3, 4, 2, 8, 8, 7
    q = torch.randn(B, S, H, K)
    k = torch.randn(B, S, H, K)
    v = torch.randn(B, S, H, V)
    g = F.logsigmoid(torch.randn(B, S, H, K))
    # Random page permutation ensures the gather is non-contiguous.
    pt = torch.randperm(P, dtype=torch.int32)[:B]
    pool = torch.randn(P, H, K, V)
    o, s = paged_gla_decode_ref(q, k, v, g, pt, pool)
    assert o.shape == (B, S, H, V) and s.shape == (B, H, K, V)
    assert s.dtype == torch.float32
    assert torch.isfinite(o).all() and torch.isfinite(s).all()

    # An independent explicit-loop scan must agree bit-for-bit at fp32.
    def _manual(q, k, v, g, pt, pool):
        B, S, H, K = q.shape
        V = v.shape[-1]
        state = torch.stack([pool[int(pt[b])].clone() for b in range(B)], dim=0)
        out = torch.zeros(B, S, H, V)
        scale = K**-0.5
        for t in range(S):
            for b in range(B):
                for h in range(H):
                    state[b, h] = state[b, h] * g[b, t, h].exp()[:, None]
                    state[b, h] = state[b, h] + torch.outer(k[b, t, h], v[b, t, h])
                    out[b, t, h] = state[b, h].t() @ (q[b, t, h] * scale)
        return out, state

    om, sm = _manual(q, k, v, g, pt, pool)
    o2, s2 = paged_gla_decode_ref(q, k, v, g, pt, pool)
    assert torch.allclose(o2.float(), om, atol=1e-5)
    assert torch.allclose(s2, sm, atol=1e-5)

    # Pool must be read-only: same call again with an unchanged pool must repeat.
    pool_copy = pool.clone()
    o3, s3 = paged_gla_decode_ref(q, k, v, g, pt, pool)
    assert torch.equal(pool, pool_copy), "reference must not mutate paged_state"
    assert torch.equal(o3, o2) and torch.equal(s3, s2)

    print(
        "paged_gla_decode_ref CPU sanity OK:",
        tuple(o.shape),
        o.dtype,
        "| state",
        tuple(s.shape),
        s.dtype,
    )
