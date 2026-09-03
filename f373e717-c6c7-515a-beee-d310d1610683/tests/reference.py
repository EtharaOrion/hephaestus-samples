"""
Reference implementation -- PyTorch-only ground truth for correctness verification.
DO NOT MODIFY. This is the oracle the benchmark harness checks every candidate against.

The operator is CHUNK-PARALLEL RWKV-6 (a.k.a. WKV-6): a linear-attention recurrence
with a PER-KEY-CHANNEL log-decay `w` AND a data-independent per-(head, channel)
BONUS `u` applied to the CURRENT step's outer product. Distinct from every other
member of the chunked_linear_attn family: no other variant carries a current-step
bonus that breaks the standard chunk-parallel factorization at the diagonal.

For a single (batch, head), with state S in R^{K x V}:

    S_t = diag(exp(w_t)) . S_{t-1} + k_t v_t^T
    o_t = r_t^T ( S_{t-1}  +  diag(u) . (k_t v_t^T) )

`w_t` is a length-K log-decay vector (typically <= 0; the state row for key channel
k is multiplied by exp(w_t[k]) at every step). `u[h, k]` is a per-head, per-key
learned SCALAR that boosts the CURRENT-step outer product ONLY in the output --
the recurrent state carried into t+1 does NOT see u. This asymmetry is the
distinguishing lever: the intra-chunk kernel must compute two closely-related
terms per position (one weighted by u on the diagonal, one not), and the standard
chunked lower-triangular attention pattern no longer captures the diagonal on its
own -- see naive_chunk_rwkv6 in the pinned library, which is written that way.

The recurrence emits the output on the OLD state (S_{t-1}) plus the u-boosted
current outer product; the state is updated AFTER. Applying the update BEFORE the
readout is a different operator that disagrees with this one by ~u * (k v^T)
per step -- above the bf16 tolerance -- so a solver that swaps the order fails
the correctness gate.

This is written as a readable sequential scan in float32 and is deliberately slow.
The surface is forward+backward: autograd differentiates this scan, and the input
gradients dr, dk, dv, dw, du ARE the graded gradient oracle. Correctness is
defined as agreement with this function, never with any Triton or CUDA
implementation of it. Semantics ported from the library's own ground truth,
fla/ops/rwkv6/recurrent_naive.py::naive_recurrent_rwkv6 (read, never called or
vendored); shape convention is the [B, T, H, ...] layout the library's chunk_rwkv6
wrapper uses.
"""

import torch


def chunked_rwkv6_ref(
    r: torch.Tensor,  # [B, T, H, K]
    k: torch.Tensor,  # [B, T, H, K]
    v: torch.Tensor,  # [B, T, H, V]
    w: torch.Tensor,  # [B, T, H, K]   log-space per-channel decay (<= 0)
    u: torch.Tensor,  # [H, K]         per-head per-channel current-step bonus
) -> torch.Tensor:  # [B, T, H, V]
    """Sequential ground truth for RWKV-6 with u-bonus.

    Everything is accumulated in float32. The u-bonus is applied only to the
    CURRENT-step outer product in the output; it is NOT persisted into the state
    that flows to t+1. r is scaled by 1/sqrt(K) to match the library's
    chunk_rwkv6 default scale.
    """
    B, T, H, K = r.shape
    V = v.shape[-1]
    r32 = r.float() * (K**-0.5)
    k32, v32, w32 = k.float(), v.float(), w.float()
    u32 = u.float()  # [H, K]

    state = torch.zeros(B, H, K, V, dtype=torch.float32, device=r.device)
    out = torch.empty(B, T, H, V, dtype=torch.float32, device=r.device)

    for t in range(T):
        kt = k32[:, t]  # [B, H, K]
        vt = v32[:, t]  # [B, H, V]
        kv = kt[:, :, :, None] * vt[:, :, None, :]  # [B, H, K, V]
        # Read from OLD state plus u-boosted current outer product.
        boosted = state + u32[None, :, :, None] * kv  # [B, H, K, V]
        out[:, t] = (r32[:, t][:, :, :, None] * boosted).sum(2)  # [B, H, V]
        # THEN advance state (decay per key channel, add fresh outer product).
        decay = w32[:, t].exp()[:, :, :, None]  # [B, H, K, 1]
        state = state * decay + kv

    return out.to(v.dtype)


if __name__ == "__main__":
    # CPU-ONLY sanity check on TINY shapes. Asserts the scan runs, returns the
    # right shape/dtype, is finite, and is autograd-capable for all five inputs.
    # NO GPU / NO Triton here -- GPU calibration is a separate follow-up.
    torch.manual_seed(0)
    B, T, H, K, V = 2, 6, 3, 8, 8
    r = torch.randn(B, T, H, K).requires_grad_()
    k = torch.randn(B, T, H, K).requires_grad_()
    v = torch.randn(B, T, H, V).requires_grad_()
    w = torch.nn.functional.logsigmoid(torch.randn(B, T, H, K)).requires_grad_()
    u = torch.randn(H, K).requires_grad_()

    out = chunked_rwkv6_ref(r, k, v, w, u)
    assert out.shape == (B, T, H, V), out.shape
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all(), "reference produced non-finite output"

    loss = out.float().sum()
    loss.backward()
    for name, t_ in (("r", r), ("k", k), ("v", v), ("w", w), ("u", u)):
        assert t_.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(t_.grad).all(), f"non-finite gradient for {name}"

    # Two-step scalar cross-check (H=K=V=1) against the explicit formula, with
    # the u-bonus contribution isolated so a wrong-place u shows up.
    ri = torch.tensor([1.0, 1.0]).reshape(1, 2, 1, 1)
    ki = torch.tensor([1.0, 1.0]).reshape(1, 2, 1, 1)
    vi = torch.tensor([2.0, 3.0]).reshape(1, 2, 1, 1)
    wi = torch.log(torch.tensor([0.5, 0.5])).reshape(1, 2, 1, 1)
    ui = torch.tensor([0.25]).reshape(1, 1)
    got = chunked_rwkv6_ref(ri, ki, vi, wi, ui).reshape(-1)
    # scale = K^-0.5 = 1.0. Step 0: S_{-1}=0, kv=1*2=2, boosted=0+0.25*2=0.5, o_0=1*0.5=0.5
    #                        state=0*0.5 + 2 = 2
    # Step 1: kv=1*3=3, boosted=2+0.25*3=2.75, o_1=1*2.75=2.75
    scale = 1.0**-0.5
    expected = torch.tensor([0.5, 2.75]) * scale
    assert torch.allclose(got, expected, atol=1e-6), (got, expected)
    print(
        "chunked_rwkv6 reference: CPU sanity OK",
        "| out",
        tuple(out.shape),
        out.dtype,
        "| grads dr,dk,dv,dw,du finite",
    )
