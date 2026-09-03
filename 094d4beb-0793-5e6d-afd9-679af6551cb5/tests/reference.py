"""
Reference implementation -- PyTorch-only ground truth for correctness verification.
DO NOT MODIFY. This is the oracle the benchmark harness checks every candidate against.

Family: moe_routed (routing computed INSIDE the graded op; forward + backward).
This member is FINE-GRAINED softmax top-K routing with a re-softmax combine
rule (DeepSeek-V3-adjacent, distinct from the mixtral p_sel/sum rule): very
large expert counts (E in the hundreds) with a small K (e.g. K=8) per token,
and the combine weights are softmax over the SELECTED logits, not the
softmax weights divided by their sum.

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
    p      = softmax(logits, dim=-1)                   # [T, E] scoring softmax
    sel    = top-K experts per token by p, ties broken by LOWER expert index
             first; -0.0 and +0.0 compare equal
    w      = softmax(logits[t, sel], dim=-1)           # RE-SOFTMAX over the
                                                       # SELECTED LOGITS
                                                       # (distinct from
                                                       # p_sel/sum: order-
                                                       # preserving but not
                                                       # linearly equal to the
                                                       # renormalized weights)
    y[t]   = sum_k w[t, k] * f(sel[t, k], x[t])
    f(e,h) = (silu(h @ w1[e, :, :F]) * (h @ w1[e, :, F:])) @ w2[e]
    silu(z) = z * sigmoid(z)

  * softmax is monotone in logits, so top-K by p and top-K by logit pick the
    same set with the same tie structure. Selection is by (p desc, index asc)
    via the int64-key trick used in the sibling variants.
  * The re-softmax combine rule is what makes gradients flow through EVERY
    selected column (through softmax over K terms) AND through the scoring
    softmax (through the top-1 selection ..). A candidate that combines by
    p_sel/sum(p_sel) will fail every graded gradient tolerance.
  * The graded surface is forward AND backward: gradients for x, router_w,
    w1 and w2 are graded (there is no router_b). Selection indices are
    almost-everywhere locally constant. Gradients flow through the scoring
    softmax over the whole expert row (into router_w and x) AND through the
    re-softmax over the K selected logits AND through the expert compute.
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
    x: torch.Tensor,  # [T, D]
    router_w: torch.Tensor,  # [D, E]
    w1: torch.Tensor,  # [E, D, 2F]
    w2: torch.Tensor,  # [E, F, D]
    K: int,  # experts per token
) -> torch.Tensor:  # [T, D]
    """Ground truth for fine-grained softmax top-K routed MoE with re-softmax."""
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
    p = torch.softmax(logits, dim=-1)  # [T, E]
    with torch.no_grad():
        sel = _selection(p.detach(), K)  # [T, K]
    sel_logits = torch.gather(logits, 1, sel)  # [T, K]
    w = torch.softmax(sel_logits, dim=-1)  # [T, K]
    # Membership -> per-expert token lists and per-(token, expert) weight.
    w_full = p.new_zeros(T, E).scatter(1, sel, w)  # [T, E]
    member = torch.zeros(T, E, dtype=torch.bool, device=x.device).scatter(1, sel, True)

    y = xf.new_zeros(T, D)
    for e in range(E):  # slow on purpose
        rows = member[:, e].nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            continue
        h = xf.index_select(0, rows)
        gu = h @ w1f[e]
        fe = (torch.nn.functional.silu(gu[:, :F]) * gu[:, F:]) @ w2f[e]
        y = y.index_add(0, rows, w_full.index_select(0, rows)[:, e : e + 1] * fe)
    return y.to(x.dtype)


if __name__ == "__main__":
    torch.manual_seed(7)
    T, D, F, E, K = 6, 5, 4, 8, 3
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
        logits = xf @ rwf
        p = torch.softmax(logits, dim=-1)
        out = torch.zeros(Tn, Dn, dtype=torch.double)
        for t in range(Tn):
            order = sorted(range(En), key=lambda e: (-float(p[t, e]), e))
            sel = order[:K]
            sl = torch.tensor([float(logits[t, e]) for e in sel], dtype=torch.double)
            w = torch.softmax(sl, dim=-1)
            for k, e in enumerate(sel):
                gu = xf[t] @ w1f[e]
                fe = (torch.nn.functional.silu(gu[:Fn]) * gu[Fn:]) @ w2f[e]
                out[t] += float(w[k]) * fe
        return out

    err = float((y.detach().double() - _brute(x, rw, w1, w2, K)).abs().max())
    assert err < 1e-6, f"reference disagrees with brute force: {err:.3e}"

    # Re-softmax is distinct from p_sel/sum: pick a random token, verify
    # the two combine schemes disagree on the fp64 reference.
    with torch.no_grad():
        logits = x.double() @ rw.double()
        p64 = torch.softmax(logits, dim=-1)
        sel = _selection(p64, K)
        p_sel = torch.gather(p64, 1, sel)
        w_ren = p_sel / p_sel.sum(dim=-1, keepdim=True)
        sl = torch.gather(logits, 1, sel)
        w_resm = torch.softmax(sl, dim=-1)
        max_gap = float((w_ren - w_resm).abs().max())
    assert max_gap > 1e-6, (
        "combine rules are indistinguishable on this fixture; "
        "make_inputs distribution needs a richer draw"
    )
    print(
        f"[fine_grained_softmax_topk] reference CPU sanity OK  err={err:.3e}  "
        f"re-softmax vs p_sel/sum disagreement (fp64) {max_gap:.3e}"
    )
