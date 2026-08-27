"""
Reference implementation -- PyTorch-only ground truth for correctness verification.
DO NOT MODIFY. This is the oracle the benchmark harness checks every candidate against.

The operator is a ROUTED fused mixture-of-experts layer: the routing decision --
scores, expert selection, combine weights -- happens INSIDE the graded surface,
GLM-4.5/5-style sigmoid routing with a selection-only bias:

    fused_moe(x, router_w, router_b, w1, w2, A) -> y

    x:        [T, D]     tokens (float32 or bfloat16)
    router_w: [D, E]     router projection, same dtype as x
    router_b: [E]        float32 selection bias: it biases WHICH experts are
                         picked and never enters the combine weights; it
                         carries no gradient (selection is piecewise-constant
                         in it)
    w1:       [E, D, 2F] expert input projections; [:, :, :F] is the GATE
                         projection, [:, :, F:] is the UP projection
    w2:       [E, F, D]  expert output projections
    A:        int        experts selected per token, 1 <= A <= E
    y:        [T, D]     same dtype as x

Semantics, exactly (all arithmetic in float32; inputs are cast up, only the
final y is cast back to the input dtype):

    s      = sigmoid(x @ router_w)                    # [T, E] raw scores
    sel    = top-A experts per token by (s + router_b), ties broken by
             LOWER expert index first; -0.0 and +0.0 compare equal
    w_a    = s[t, sel_a] / sum_a s[t, sel_a]          # normalize the RAW
                                                      # scores of the selected
                                                      # experts; router_b is
                                                      # NOT in these weights
    y[t]   = sum_a w_a * f(sel_a, x[t])
    f(e,h) = (silu(h @ w1[e, :, :F]) * (h @ w1[e, :, F:])) @ w2[e]
    silu(z) = z * sigmoid(z)

  * (biased score descending, expert index ascending) is a total order per
    token, so the selected set and its order are uniquely determined. The
    implementation below realises the tie-break exactly by packing each
    biased score's order-preserving integer bits together with the inverted
    expert index into one int64 key, so equal scores never compete: the key
    order IS the selection order.
  * The graded surface is forward AND backward: gradients for x, router_w,
    w1 and w2 are graded. router_b and the integer A have no gradient.
    Selection indices are almost-everywhere locally constant in the inputs;
    gradients flow through the GATHERED raw scores of the selected experts
    (through the sigmoid, into router_w and x) and through the expert
    computation. This function's autograd graph is the definition of those
    gradients.
  * Experts no token selects receive exactly zero weight gradients.
  * The same expert can never appear twice for one token (top-A of E distinct
    experts).

Correctness is defined as agreement with this function under the disclosed
per-dtype tolerances -- and where any prose description and this code
disagree, this code wins.

The implementation below is deliberately simple and deliberately slow: a
python loop over experts with boolean-mask gathers, every expert a separate
dense matmul chain. It is autograd-capable end to end, which is how the
harness obtains the reference gradients.
"""

import torch


def _selection(biased: torch.Tensor, A: int) -> torch.Tensor:
    """Top-A of the biased scores, ties to the LOWER expert index, exactly.

    Packs (order-preserving bits of the float32 score, inverted expert index)
    into one int64 key per entry. Keys are unique, so plain descending key
    order realises (score desc, index asc) with no separate tie handling.
    -0.0 is canonicalised to +0.0 first so both zeros form one tie group.
    """
    E = biased.shape[-1]
    b = torch.where(biased == 0.0, torch.zeros_like(biased), biased).contiguous()
    ib = b.view(torch.int32).to(torch.int64)
    u = torch.where(ib >= 0, ib ^ 0x80000000, (~ib) & 0xFFFFFFFF)
    e_idx = torch.arange(E, device=biased.device, dtype=torch.int64)
    keys = (u << 10) | (E - 1 - e_idx)
    return torch.topk(keys, A, dim=-1, largest=True, sorted=True).indices


def fused_moe_ref(x: torch.Tensor,         # [T, D] float32 or bfloat16
                  router_w: torch.Tensor,  # [D, E] same dtype as x
                  router_b: torch.Tensor,  # [E] float32, selection-only bias
                  w1: torch.Tensor,        # [E, D, 2F] gate|up
                  w2: torch.Tensor,        # [E, F, D]
                  A: int,                  # experts per token
                  ) -> torch.Tensor:       # [T, D] same dtype as x
    """Sequential ground truth for the routed fused-MoE layer."""
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
    if router_b.shape != (E,) or router_b.dtype != torch.float32:
        raise ValueError("router_b must be float32 [E]")
    A = int(A)
    if not 1 <= A <= E:
        raise ValueError(f"A={A} out of range for E={E}")
    if E > 1024:
        raise ValueError(f"reference supports E <= 1024, got {E}")

    xf, rwf = x.float(), router_w.float()
    w1f, w2f = w1.float(), w2.float()

    s = torch.sigmoid(xf @ rwf)                              # [T, E] raw scores
    with torch.no_grad():
        sel = _selection(s.detach() + router_b, A)           # [T, A] int64
    s_sel = torch.gather(s, 1, sel)                          # grads flow here
    w = s_sel / s_sel.sum(dim=-1, keepdim=True)              # RAW-score weights
    w_full = s.new_zeros(T, E).scatter(1, sel, w)            # [T, E]
    member = torch.zeros(T, E, dtype=torch.bool, device=x.device).scatter(
        1, sel, True)

    y = xf.new_zeros(T, D)
    for e in range(E):                                       # slow on purpose
        rows = member[:, e].nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            continue
        h = xf.index_select(0, rows)                         # [n, D]
        gu = h @ w1f[e]                                      # [n, 2F]
        fe = (torch.nn.functional.silu(gu[:, :F]) * gu[:, F:]) @ w2f[e]
        y = y.index_add(0, rows,
                        w_full.index_select(0, rows)[:, e:e + 1] * fe)
    return y.to(x.dtype)
