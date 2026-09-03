"""
Reference implementation -- PyTorch-only ground truth for correctness verification.
DO NOT MODIFY. This is the oracle the benchmark harness checks every candidate against.

Family: moe_routed (routing computed INSIDE the graded op; forward + backward).
This member is SIGMOID top-K routing with SUM-RENORMALIZED combine weights
(DeepSeek-V3-adjacent, distinct from the sibling sigmoid_topk_glm which uses
a raw sigmoid combine and a selection bias): sigmoid affinity, top-K by
sigmoid, weights = s_sel / s_sel.sum() combine. NO selection bias.

    fused_moe(x, router_w, w1, w2, K) -> y

    x:        [T, D]     tokens (float32 or bfloat16)
    router_w: [D, E]     router projection, same dtype as x
    w1:       [E, D, 2F] expert input projections; [:, :, :F] gate, [:, :, F:] up
    w2:       [E, F, D]  expert output projections
    K:        int        experts selected per token, 1 <= K <= E
    y:        [T, D]     same dtype as x

Semantics, exactly (all arithmetic in float32; inputs are cast up, only the
final y is cast back to the input dtype):

    logits = x @ router_w                              # [T, E]
    s      = sigmoid(logits)                           # [T, E] independent
                                                       # per-expert affinities
    sel    = top-K experts per token by s, ties broken by LOWER expert index
             first; -0.0 and +0.0 compare equal
    w      = s[t, sel] / s[t, sel].sum(-1)             # sum-renormalized
                                                       # combine (distinct from
                                                       # sigmoid_topk_glm's raw
                                                       # sigmoid combine and
                                                       # from softmax_topk_
                                                       # mixtral's p_sel/sum
                                                       # over softmax scores)
    y[t]   = sum_k w[t, k] * f(sel[t, k], x[t])
    f(e,h) = (silu(h @ w1[e, :, :F]) * (h @ w1[e, :, F:])) @ w2[e]

  * Sigmoid is monotone in the logit, so top-K by s and top-K by logit pick
    the same set with the same tie structure. Selection is by (s desc,
    index asc) via the int64-key trick.
  * The sum-renorm combine keeps the router gradient alive on every selected
    logit (each selected s enters both the numerator and the denominator of
    a token's weight vector). A candidate that uses raw sigmoid (no renorm)
    or the softmax_topk_mixtral rule (p_sel/sum on softmax scores) will fail
    every graded gradient tolerance.
  * The graded surface is forward AND backward: gradients for x, router_w,
    w1 and w2 are graded (there is no router_b). Selection indices are
    almost-everywhere locally constant. Gradients flow through the sigmoid
    of the selected experts (NOT dense over the whole row -- sigmoid is
    elementwise, so unselected experts get zero from the router-side path
    of the routed contribution) and through the expert compute AND through
    the sum-renorm coupling of the K selected weights.
  * Experts no token selects receive exactly zero weight gradients.
  * The same expert never appears twice for one token (top-K of E distinct).

Correctness is defined as agreement with this function under the disclosed
per-dtype tolerances -- and where prose and this code disagree, the code wins.

The implementation below is deliberately simple and deliberately slow: a
python loop over experts with boolean-mask gathers. It is autograd-capable
end to end, which is how the harness obtains the reference gradients.
"""

import torch


def _selection(scores: torch.Tensor, K: int) -> torch.Tensor:
    """Top-K of the scores, ties to the LOWER expert index, exactly."""
    E = scores.shape[-1]
    b = torch.where(scores == 0.0, torch.zeros_like(scores), scores).contiguous()
    ib = b.view(torch.int32).to(torch.int64)
    u = torch.where(ib >= 0, ib ^ 0x80000000, (~ib) & 0xFFFFFFFF)
    e_idx = torch.arange(E, device=scores.device, dtype=torch.int64)
    ebits = max((E - 1).bit_length(), 1)
    keys = (u << ebits) | (E - 1 - e_idx)
    return torch.topk(keys, K, dim=-1, largest=True, sorted=True).indices


def fused_moe_ref(
    x: torch.Tensor,
    router_w: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    K: int,
) -> torch.Tensor:
    """Ground truth for sigmoid top-K routed MoE with sum-renorm combine."""
    if x.dim() != 2 or router_w.dim() != 2 or w1.dim() != 3 or w2.dim() != 3:
        raise ValueError("bad ranks")
    T, D = x.shape
    D2, E = router_w.shape
    E2, D3, F2 = w1.shape
    F = F2 // 2
    if D2 != D or E2 != E or D3 != D or F2 != 2 * F or tuple(w2.shape) != (E, F, D):
        raise ValueError("inconsistent shapes")
    K = int(K)
    if not 1 <= K <= E:
        raise ValueError(f"K={K} out of range for E={E}")
    if E > 4096:
        raise ValueError(f"reference supports E <= 4096, got {E}")

    xf, rwf = x.float(), router_w.float()
    w1f, w2f = w1.float(), w2.float()

    logits = xf @ rwf  # [T, E]
    s = torch.sigmoid(logits)  # [T, E]
    with torch.no_grad():
        sel = _selection(s.detach(), K)  # [T, K]
    s_sel = torch.gather(s, 1, sel)  # [T, K] grads flow
    w = s_sel / s_sel.sum(dim=-1, keepdim=True)  # [T, K] sum-renorm

    w_full = s.new_zeros(T, E).scatter(1, sel, w)
    member = torch.zeros(T, E, dtype=torch.bool, device=x.device).scatter(1, sel, True)

    y = xf.new_zeros(T, D)
    for e in range(E):
        rows = member[:, e].nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            continue
        h = xf.index_select(0, rows)
        gu = h @ w1f[e]
        fe = (torch.nn.functional.silu(gu[:, :F]) * gu[:, F:]) @ w2f[e]
        y = y.index_add(0, rows, w_full.index_select(0, rows)[:, e : e + 1] * fe)
    return y.to(x.dtype)


if __name__ == "__main__":
    torch.manual_seed(11)
    T, D, F, E, K = 7, 5, 4, 8, 3
    x = torch.randn(T, D).requires_grad_(True)
    rw = (torch.randn(D, E) * D**-0.5).requires_grad_(True)
    w1 = (torch.randn(E, D, 2 * F) * D**-0.5).requires_grad_(True)
    w2 = (torch.randn(E, F, D) * F**-0.5).requires_grad_(True)

    y = fused_moe_ref(x, rw, w1, w2, K)
    assert y.shape == (T, D) and y.dtype == torch.float32
    loss = (y * torch.linspace(-1, 1, T * D).reshape(T, D)).sum()
    loss.backward()
    assert x.grad.shape == (T, D) and rw.grad.shape == (D, E)
    assert w1.grad.shape == (E, D, 2 * F) and w2.grad.shape == (E, F, D)
    assert float(rw.grad.abs().max()) > 0

    def _brute(x, rw, w1, w2, K):
        xf, rwf = x.detach().double(), rw.detach().double()
        w1f, w2f = w1.detach().double(), w2.detach().double()
        Tn, Dn = xf.shape
        En = rwf.shape[1]
        Fn = w1f.shape[2] // 2
        s = torch.sigmoid(xf @ rwf)
        out = torch.zeros(Tn, Dn, dtype=torch.double)
        for t in range(Tn):
            order = sorted(range(En), key=lambda e: (-float(s[t, e]), e))
            sel = order[:K]
            ss = torch.tensor([float(s[t, e]) for e in sel], dtype=torch.double)
            w = ss / ss.sum()
            for k, e in enumerate(sel):
                gu = xf[t] @ w1f[e]
                fe = (torch.nn.functional.silu(gu[:Fn]) * gu[Fn:]) @ w2f[e]
                out[t] += float(w[k]) * fe
        return out

    err = float((y.detach().double() - _brute(x, rw, w1, w2, K)).abs().max())
    assert err < 1e-6, f"reference disagrees with brute force: {err:.3e}"

    # Sanity: sum-renorm weights sum to exactly 1 per token.
    with torch.no_grad():
        s = torch.sigmoid(x.double() @ rw.double())
        sel = _selection(s, K)
        ss = torch.gather(s, 1, sel)
        w = ss / ss.sum(dim=-1, keepdim=True)
        wsum = float(w.sum(dim=-1).max() - w.sum(dim=-1).min())
    assert wsum < 1e-12
    print(
        f"[sigmoid_topk_sumrenorm] reference CPU sanity OK  err={err:.3e}  "
        f"weight-sum drift {wsum:.3e}"
    )
