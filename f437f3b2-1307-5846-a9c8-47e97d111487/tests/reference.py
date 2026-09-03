"""
Reference implementation -- PyTorch-only ground truth for correctness verification.
DO NOT MODIFY. This is the oracle the benchmark harness checks every candidate against.

Family: moe_routed (routing computed INSIDE the graded op; forward + backward).
This member is EXPERT-CHOICE routing (Zhou et al. 2022): instead of each token
picking its experts, each EXPERT picks its top-C tokens by affinity. Routing is
irregular in the token dimension -- a token may be chosen by any number of
experts from 0 to E -- which is what makes this the hardest member of the family.

    fused_moe(x, router_w, w1, w2, C) -> y

    x:        [T, D]     tokens (float32 or bfloat16)
    router_w: [D, E]     router projection, same dtype as x
    w1:       [E, D, 2F] expert input projections; [:, :, :F] is the GATE
                         projection, [:, :, F:] is the UP projection
    w2:       [E, F, D]  expert output projections
    C:        int        capacity: tokens each expert selects, 1 <= C <= T
    y:        [T, D]     same dtype as x

Semantics, exactly (all arithmetic in float32; inputs are cast up, only the
final y is cast back to the input dtype):

    logits = x @ router_w                             # [T, E]
    S      = softmax(logits, dim=-1)                  # [T, E] affinities;
                                                      # softmax is over EXPERTS,
                                                      # so each token's row sums
                                                      # to 1 (token-to-expert
                                                      # affinity)
    for each expert e:
        sel[e] = the C tokens with the largest S[:, e], ties broken by LOWER
                 TOKEN index first; -0.0 and +0.0 compare equal
        g[e,c] = S[sel[e,c], e]                        # the affinity gate, used
                                                        # WITHOUT renormalization
        f_e    = (silu(x[sel[e]] @ w1[e,:,:F]) * (x[sel[e]] @ w1[e,:,F:])) @ w2[e]
    y[t]   = sum over every (e, c) with sel[e,c] == t of  g[e,c] * f_e[c]
    silu(z) = z * sigmoid(z)

  * The selection is over TOKENS for each fixed expert, so the deterministic
    tie-break is by LOWER TOKEN index (the structural analog of the
    lower-expert-index rule used by the token-choice siblings). It is realised
    exactly by packing each affinity's order-preserving float32 bits with the
    inverted token index into one int64 key per (expert, token) entry -- equal
    affinities never compete, key order IS selection order. -0.0 is
    canonicalised to +0.0 first.
  * The gate g[e,c] = S[sel[e,c], e] is the RAW softmax affinity and is used
    without renormalization: a token chosen by k experts sums k contributions,
    each scaled by that expert's affinity, so its total scale is data-dependent
    (unlike the token-choice gates, which renormalize to 1 per token).
  * The combine is irregular: each token gathers a variable number of expert
    contributions (0 .. E). They are summed in ascending expert order, so the
    accumulation order is fixed and the result is deterministic.
  * The graded surface is forward AND backward: gradients for x, router_w, w1
    and w2 are graded (there is no router_b). Selection indices are almost-
    everywhere locally constant; gradients flow through the affinity gates
    S[sel[e], e] (the softmax Jacobian over the whole expert row, into router_w
    and x) and through the expert computation. This function's autograd graph is
    the definition of those gradients.
  * A token selected by NO expert receives y[t] = 0 and, since its row of S is
    then unused, exactly zero input gradient. Experts are always full (exactly
    C tokens each), so no expert has empty weight gradients unless C == 0
    (excluded: C >= 1).

Correctness is defined as agreement with this function under the disclosed
per-dtype tolerances -- and where prose and this code disagree, the code wins.

The implementation below is deliberately simple and deliberately slow: a python
loop over experts with per-expert gathers. It is autograd-capable end to end,
which is how the harness obtains the reference gradients.
"""

import torch


def _selection_ec(S: torch.Tensor, C: int) -> torch.Tensor:
    """Per expert (column of S), the top-C tokens by affinity, ties to the
    LOWER token index, exactly. Returns [E, C] token indices.

    Packs (order-preserving bits of the float32 affinity, inverted token index)
    into one int64 key per (expert, token). Keys are unique within an expert
    row, so plain descending key order realises (affinity desc, token index
    asc) with no separate tie handling. -0.0 is canonicalised to +0.0 first.
    """
    T, E = S.shape
    St = S.t().contiguous()                                # [E, T]
    b = torch.where(St == 0.0, torch.zeros_like(St), St).contiguous()
    ib = b.view(torch.int32).to(torch.int64)
    u = torch.where(ib >= 0, ib ^ 0x80000000, (~ib) & 0xFFFFFFFF)
    t_idx = torch.arange(T, device=S.device, dtype=torch.int64)
    tbits = max((T - 1).bit_length(), 1)                   # bits for token index
    keys = (u << tbits) | (T - 1 - t_idx).unsqueeze(0)     # [E, T]
    return torch.topk(keys, C, dim=-1, largest=True, sorted=True).indices


def fused_moe_ref(x: torch.Tensor,         # [T, D] float32 or bfloat16
                  router_w: torch.Tensor,  # [D, E] same dtype as x
                  w1: torch.Tensor,        # [E, D, 2F] gate|up
                  w2: torch.Tensor,        # [E, F, D]
                  C: int,                  # capacity: tokens per expert
                  ) -> torch.Tensor:       # [T, D] same dtype as x
    """Sequential ground truth for the expert-choice routed fused-MoE layer."""
    if x.dim() != 2 or router_w.dim() != 2 or w1.dim() != 3 or w2.dim() != 3:
        raise ValueError(f"bad ranks: x{tuple(x.shape)} router_w{tuple(router_w.shape)} "
                         f"w1{tuple(w1.shape)} w2{tuple(w2.shape)}")
    T, D = x.shape
    D2, E = router_w.shape
    E2, D3, F2 = w1.shape
    F = F2 // 2
    if D2 != D or E2 != E or D3 != D or F2 != 2 * F or tuple(w2.shape) != (E, F, D):
        raise ValueError(f"inconsistent shapes: x{tuple(x.shape)} "
                         f"router_w{tuple(router_w.shape)} w1{tuple(w1.shape)} "
                         f"w2{tuple(w2.shape)}")
    C = int(C)
    if not 1 <= C <= T:
        raise ValueError(f"C={C} out of range for T={T}")
    if E > 1024:
        raise ValueError(f"reference supports E <= 1024, got {E}")

    xf, rwf = x.float(), router_w.float()
    w1f, w2f = w1.float(), w2.float()

    logits = xf @ rwf                                      # [T, E]
    S = torch.softmax(logits, dim=-1)                     # [T, E] affinities
    with torch.no_grad():
        sel = _selection_ec(S.detach(), C)                 # [E, C] token indices

    y = xf.new_zeros(T, D)
    for e in range(E):                                    # slow on purpose
        rows = sel[e]                                     # [C] token indices
        g = S[rows, e]                                    # [C] affinity gate (grads flow)
        h = xf.index_select(0, rows)                      # [C, D]
        gu = h @ w1f[e]                                   # [C, 2F]
        fe = (torch.nn.functional.silu(gu[:, :F]) * gu[:, F:]) @ w2f[e]  # [C, D]
        y = y.index_add(0, rows, g.unsqueeze(1) * fe)     # ascending-e accumulate
    return y.to(x.dtype)


if __name__ == "__main__":
    # CPU-ONLY sanity check on TINY shapes. No GPU/triton here: this validates
    # the reference forward against an independent double-precision brute force
    # and confirms the graded gradients populate with the right shapes/dtypes.
    torch.manual_seed(3)

    def _brute(x, rw, w1, w2, C):
        xf, rwf = x.detach().double(), rw.detach().double()
        w1f, w2f = w1.detach().double(), w2.detach().double()
        Tn, Dn = xf.shape
        En = rwf.shape[1]
        Fn = w1f.shape[2] // 2
        S = torch.softmax(xf @ rwf, dim=-1)               # [T, E]
        out = torch.zeros(Tn, Dn, dtype=torch.double)
        for e in range(En):
            order = sorted(range(Tn), key=lambda t: (-float(S[t, e]), t))
            sel = order[:C]
            for t in sel:
                gu = xf[t] @ w1f[e]
                fe = (torch.nn.functional.silu(gu[:Fn]) * gu[Fn:]) @ w2f[e]
                out[t] += float(S[t, e]) * fe             # raw affinity, no renorm
        return out

    for (T, D, F, E, C) in [(8, 5, 4, 6, 3), (7, 6, 3, 4, 1), (5, 4, 3, 3, 5)]:
        x = torch.randn(T, D).requires_grad_(True)
        rw = (torch.randn(D, E) * D ** -0.5).requires_grad_(True)
        w1 = (torch.randn(E, D, 2 * F) * D ** -0.5).requires_grad_(True)
        w2 = (torch.randn(E, F, D) * F ** -0.5).requires_grad_(True)

        y = fused_moe_ref(x, rw, w1, w2, C)
        assert y.shape == (T, D) and y.dtype == torch.float32
        loss = (y * torch.linspace(-1, 1, T * D).reshape(T, D)).sum()
        loss.backward()
        assert x.grad.shape == (T, D) and rw.grad.shape == (D, E)
        assert w1.grad.shape == (E, D, 2 * F) and w2.grad.shape == (E, F, D)
        assert float(rw.grad.abs().max()) > 0, "router grad must be non-trivial"

        err = float((y.detach().double() - _brute(x, rw, w1, w2, C)).abs().max())
        assert err < 1e-6, f"reference disagrees with brute force (C={C}): {err:.3e}"

    # Irregularity check: with C=1 and E<T some tokens are unselected -> y row 0.
    T, D, F, E, C = 10, 4, 3, 3, 1
    x = torch.randn(T, D)
    rw = torch.randn(D, E) * D ** -0.5
    w1 = torch.randn(E, D, 2 * F) * D ** -0.5
    w2 = torch.randn(E, F, D) * F ** -0.5
    y = fused_moe_ref(x, rw, w1, w2, C)
    unselected = int((y.abs().sum(dim=-1) == 0).sum())
    assert unselected > 0, "expected some unselected (all-zero) token rows at C=1"
    print(f"[expert_choice] reference CPU sanity OK  brute-force match on 3 shapes "
          f"incl. C=1 and C=T; grads dx/drw/dw1/dw2 populated (no router_b); "
          f"{unselected}/{T} tokens unselected at C=1 (irregular combine)")
