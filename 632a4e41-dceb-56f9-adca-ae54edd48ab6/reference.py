"""
Reference implementation -- PyTorch-only ground truth for correctness verification.
DO NOT MODIFY. This is the oracle the benchmark harness checks every candidate against.

The operator is the DECODE regime of the DELTA RULE run in the GROUPED-QUERY
regime: `Hq` query heads share `Hkv` state heads with a group ratio `G = Hq /
Hkv` (Hq is a multiple of Hkv). The recurrent state lives on the SMALLER (kv)
head axis; every query head reads and delta-writes the state of its group's kv
head. This is the standard grouped-query attention topology carried into a
recurrent (linear-attention) operator, where the state SHARING across query
heads is the source of extra structure -- an ordinary per-query-head kernel
either wastes GxHkv state slots (memory-quadratic in G) or races G query heads
writing to the same state without a designed accumulation order.

For each batch, kv-head hk and its G query heads {hq = hk*G, ..., hk*G+G-1},
with state S[hk] of shape [K, V] starting from state0[b, hk]:

    S[hk]_t = S[hk]_{t-1} + k_t v_t^T                              (write, no gate)
    pred_t  = (S[hk]_{t-1})^T k_t                                  (pre-update pred)
    delta_t = (v_t - pred_t) * beta_t                              (per-step scalar)
    S[hk]_t = S[hk]_{t-1} + k_t delta_t^T                          (delta write)
    o_t[hq] = (S[hk]_t)^T (q_t[hq] / sqrt(K))         for every hq in the group

The delta rule (with the pre-update prediction) is the OPERATOR's definition; the
state is per-kv-head and every query head in the group reads AFTER the delta
write. q is [B, S, Hq, K]; k, v are [B, S, Hkv, {K, V}]; beta is [B, S, Hkv].
state0 is [B, Hkv, K, V] float32. BOTH outputs are graded: o [B, S, Hq, V] in
v's dtype and state_out [B, Hkv, K, V] float32.
"""

import torch
import torch.nn.functional as F


def gqa_delta_decode_ref(
    q: torch.Tensor,  # [B, S, Hq, K]
    k: torch.Tensor,  # [B, S, Hkv, K]
    v: torch.Tensor,  # [B, S, Hkv, V]
    beta: torch.Tensor,  # [B, S, Hkv]        write strength
    state0: torch.Tensor,  # [B, Hkv, K, V]     float32 initial state
):  # -> (o [B, S, Hq, V] in v.dtype, state_out [B, Hkv, K, V] fp32)
    """Sequential ground truth for the grouped-query delta rule decode step."""
    B, S, Hq, K = q.shape
    Hkv = k.shape[2]
    V = v.shape[-1]
    if Hq % Hkv:
        raise ValueError(f"Hq ({Hq}) must be a multiple of Hkv ({Hkv})")
    G = Hq // Hkv
    q32 = q.float() * (K**-0.5)
    k32, v32, beta32 = k.float(), v.float(), beta.float()

    state = state0.float().clone()  # [B, Hkv, K, V]
    out = torch.empty(B, S, Hq, V, dtype=torch.float32, device=q.device)

    for t in range(S):
        kt = k32[:, t]  # [B, Hkv, K]
        vt = v32[:, t]  # [B, Hkv, V]
        bt = beta32[:, t]  # [B, Hkv]
        pred = torch.einsum("bhk,bhkv->bhv", kt, state)  # pre-update read
        delta = (vt - pred) * bt[:, :, None]  # [B, Hkv, V]
        state = state + torch.einsum("bhk,bhv->bhkv", kt, delta)  # delta write
        # Every query head in the group reads its kv-head's state.
        qt = q32[:, t].view(B, Hkv, G, K)  # regroup
        out[:, t] = torch.einsum("bhgk,bhkv->bhgv", qt, state).reshape(B, Hq, V)

    return out.to(v.dtype), state


if __name__ == "__main__":
    torch.manual_seed(0)
    B, S, Hq, Hkv, K, V = 2, 4, 8, 2, 8, 6
    q = torch.randn(B, S, Hq, K)
    k = torch.randn(B, S, Hkv, K)
    v = torch.randn(B, S, Hkv, V)
    beta = torch.rand(B, S, Hkv) * 0.9 + 0.05
    st = torch.randn(B, Hkv, K, V) * 0.5
    o, s = gqa_delta_decode_ref(q, k, v, beta, st)
    assert o.shape == (B, S, Hq, V) and s.shape == (B, Hkv, K, V)
    assert s.dtype == torch.float32
    assert torch.isfinite(o).all() and torch.isfinite(s).all()

    # Query heads in the same group must see the SAME final state.
    G = Hq // Hkv
    for hk in range(Hkv):
        # A closed check: with G=1 the operator degenerates to the plain
        # delta rule; with G>1, o[:, :, hk*G] and o[:, :, hk*G+1] read the
        # SAME state but different queries -- the reads differ only in q.
        for g_ in range(G):
            for gp in range(G):
                if g_ == gp:
                    continue
                # Swap the queries of two heads within the group; the outputs
                # at those slots must swap too because the state is shared.
                q_swap = q.clone()
                q_swap[:, :, hk * G + g_], q_swap[:, :, hk * G + gp] = (
                    q[:, :, hk * G + gp].clone(),
                    q[:, :, hk * G + g_].clone(),
                )
                o_swap, _ = gqa_delta_decode_ref(q_swap, k, v, beta, st)
                assert torch.allclose(o_swap[:, :, hk * G + g_], o[:, :, hk * G + gp])
                assert torch.allclose(o_swap[:, :, hk * G + gp], o[:, :, hk * G + g_])

    # G=1 degenerate: matches the plain delta rule (no gate).
    q1 = torch.randn(B, S, Hkv, K)
    o1, s1 = gqa_delta_decode_ref(q1, k, v, beta, st)

    def _plain(q, k, v, beta, st):
        B, S, H, K = q.shape
        V = v.shape[-1]
        q32 = q.float() * (K**-0.5)
        k32, v32, b32 = k.float(), v.float(), beta.float()
        state = st.float().clone()
        out = torch.empty(B, S, H, V, dtype=torch.float32, device=q.device)
        for t in range(S):
            pred = torch.einsum("bhk,bhkv->bhv", k32[:, t], state)
            delta = (v32[:, t] - pred) * b32[:, t][:, :, None]
            state = state + torch.einsum("bhk,bhv->bhkv", k32[:, t], delta)
            out[:, t] = torch.einsum("bhk,bhkv->bhv", q32[:, t], state)
        return out, state

    op, sp = _plain(q1, k, v, beta, st)
    assert torch.allclose(o1.float(), op, atol=1e-5)
    assert torch.allclose(s1, sp, atol=1e-5)

    print(
        "gqa_delta_decode_ref CPU sanity OK:",
        tuple(o.shape),
        o.dtype,
        "| state",
        tuple(s.shape),
        s.dtype,
        "| G =",
        G,
    )
