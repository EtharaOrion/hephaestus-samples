"""
Reference implementation -- PyTorch-only ground truth for correctness verification.
DO NOT MODIFY. This is the oracle the benchmark harness checks every candidate against.

The operator is a hybrid decode-regime linear-attention step that fuses two
per-step levers no single existing library kernel exposes together: a PER-CHANNEL
log-forget gate on the key axis of the [K, V] state (GLA-style) AND a per-step
scalar write-strength `beta` on a delta-corrected write (delta-rule-style).
Neither fla.ops.gla nor fla.ops.gated_delta_rule handles both at once: GLA has no
delta correction and no beta; the gated delta rule has no per-channel gate.

Per batch, head and step, with state `S` of shape `[K, V]` starting from state0:

    S_t         = diag(exp(g_t)) . S_{t-1}                (per-channel decay)
    pred_t      = S_t^T k_t                                (pre-write read)
    delta_t     = (v_t - pred_t) * beta_t                  (scalar write strength)
    S_t         = S_t + k_t delta_t^T                      (delta write)
    o_t         = S_t^T (q_t / sqrt(K))                    (read AFTER delta write)

q, k are [B, S, H, K]; v is [B, S, H, V]; g is [B, S, H, K] float32; beta is
[B, S, H]; state0 is [B, H, K, V] float32. BOTH outputs are graded: o
[B, S, H, V] in v's dtype and state_out [B, H, K, V] float32.
"""

import torch
import torch.nn.functional as F


def rwkv7_channel_delta_decode_ref(
    q: torch.Tensor,  # [B, S, H, K]
    k: torch.Tensor,  # [B, S, H, K]
    v: torch.Tensor,  # [B, S, H, V]
    g: torch.Tensor,  # [B, S, H, K]     float32 per-channel log-forget gate
    beta: torch.Tensor,  # [B, S, H]                 scalar write strength
    state0: torch.Tensor,  # [B, H, K, V]     float32 initial state
):  # -> (o [B, S, H, V] in v.dtype, state_out [B, H, K, V] fp32)
    """Sequential ground truth for the per-channel-decay + beta-delta operator."""
    B, S, H, K = q.shape
    V = v.shape[-1]
    q32 = q.float() * (K**-0.5)
    k32, v32 = k.float(), v.float()
    g32, b32 = g.float(), beta.float()

    state = state0.float().clone()
    out = torch.empty(B, S, H, V, dtype=torch.float32, device=q.device)

    for t in range(S):
        state = state * g32[:, t].exp()[:, :, :, None]  # per-row decay
        kt = k32[:, t]
        pred = torch.einsum("bhk,bhkv->bhv", kt, state)
        delta = (v32[:, t] - pred) * b32[:, t][:, :, None]
        state = state + torch.einsum("bhk,bhv->bhkv", kt, delta)
        out[:, t] = torch.einsum("bhk,bhkv->bhv", q32[:, t], state)

    return out.to(v.dtype), state


if __name__ == "__main__":
    torch.manual_seed(0)
    B, S, H, K, V = 2, 5, 3, 8, 6
    q = torch.randn(B, S, H, K)
    k = torch.randn(B, S, H, K)
    v = torch.randn(B, S, H, V)
    g = F.logsigmoid(torch.randn(B, S, H, K))
    beta = torch.rand(B, S, H) * 0.9 + 0.05
    st = torch.randn(B, H, K, V) * 0.5
    o, s = rwkv7_channel_delta_decode_ref(q, k, v, g, beta, st)
    assert o.shape == (B, S, H, V) and s.shape == (B, H, K, V)
    assert s.dtype == torch.float32
    assert torch.isfinite(o).all() and torch.isfinite(s).all()

    # Explicit triple-loop scan must agree bit-for-bit at fp32.
    def _manual(q, k, v, g, beta, st):
        B, S, H, K = q.shape
        V = v.shape[-1]
        state = st.float().clone()
        out = torch.zeros(B, S, H, V)
        scale = K**-0.5
        for t in range(S):
            for b in range(B):
                for h in range(H):
                    state[b, h] = state[b, h] * g[b, t, h].exp()[:, None]
                    kt = k[b, t, h]
                    pred = state[b, h].t() @ kt
                    delta = (v[b, t, h] - pred) * beta[b, t, h]
                    state[b, h] = state[b, h] + torch.outer(kt, delta)
                    out[b, t, h] = state[b, h].t() @ (q[b, t, h] * scale)
        return out, state

    om, sm = _manual(q, k, v, g, beta, st)
    o2, s2 = rwkv7_channel_delta_decode_ref(q, k, v, g, beta, st)
    assert torch.allclose(o2.float(), om, atol=1e-5)
    assert torch.allclose(s2, sm, atol=1e-5)

    # beta = 0 -> no delta write, degenerates to GLA (per-channel decay + read).
    o0, s0 = rwkv7_channel_delta_decode_ref(q, k, v, g, torch.zeros_like(beta), st)

    def _gla(q, k, v, g, st):
        B, S, H, K = q.shape
        V = v.shape[-1]
        q32 = q.float() * (K**-0.5)
        k32, v32, g32 = k.float(), v.float(), g.float()
        state = st.float().clone()
        out = torch.empty(B, S, H, V, dtype=torch.float32, device=q.device)
        for t in range(S):
            state = state * g32[:, t].exp()[:, :, :, None]
            out[:, t] = torch.einsum("bhk,bhkv->bhv", q32[:, t], state)
        return out, state

    ogla, sgla = _gla(q, k, v, g, st)
    assert torch.allclose(s0, sgla, atol=1e-5)
    assert torch.allclose(o0.float(), ogla, atol=1e-5)

    # g = 0 -> no decay, degenerates to the plain delta rule (no gate).
    o_ng, _ = rwkv7_channel_delta_decode_ref(q, k, v, torch.zeros_like(g), beta, st)

    def _delta(q, k, v, beta, st):
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

    od, _ = _delta(q, k, v, beta, st)
    assert torch.allclose(o_ng.float(), od, atol=1e-5)

    print(
        "rwkv7_channel_delta_decode_ref CPU sanity OK:",
        tuple(o.shape),
        o.dtype,
        "| state",
        tuple(s.shape),
        s.dtype,
    )
