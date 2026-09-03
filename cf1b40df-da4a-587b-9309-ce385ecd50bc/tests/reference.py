"""
Reference implementation -- PyTorch-only ground truth for correctness verification.
DO NOT MODIFY. This is the oracle the benchmark harness checks every candidate against.

Family: moe_routed (routing computed INSIDE the graded op; forward + backward).
This member is CAPACITY-LIMITED Switch top-1 routing (GShard / Switch-Transformer
overflow discipline): each token picks its single expert by argmax softmax
probability, but each expert accepts at most `capacity` tokens; overflow tokens
are DROPPED (their output row is zero and they contribute no weight gradient).
The irregularity comes from a data-dependent per-expert drop mask.

    fused_moe(x, router_w, w1, w2, capacity) -> y

    x:        [T, D]     tokens (float32 or bfloat16)
    router_w: [D, E]     router projection, same dtype as x
    w1:       [E, D, 2F] expert input projections; [:, :, :F] gate, [:, :, F:] up
    w2:       [E, F, D]  expert output projections
    capacity: int        max tokens each expert accepts, 1 <= capacity <= T
    y:        [T, D]     same dtype as x; dropped tokens carry an exact-zero row

There is NO router bias and NO A: top-1 is intrinsic; overflow discipline is
the whole point of this member.

Semantics, exactly (all arithmetic in float32; inputs are cast up, only the
final y is cast back to the input dtype):

    logits = x @ router_w                              # [T, E]
    p      = softmax(logits, dim=-1)                   # [T, E]
    e*[t]  = argmax_e p[t, e]  ties -> LOWER expert index; -0.0 == +0.0
    for each expert e:
        candidates = {t : e*[t] == e}
        keep the top-`capacity` candidates by p[t, e], ties broken to LOWER
        token index; if |candidates| <= capacity, keep them all; excess
        candidates are DROPPED.
    g[t]   = p[t, e*[t]]                               # raw Switch gate
    y[t]   = g[t] * f(e*[t], x[t])  if t is kept, else 0
    f(e,h) = (silu(h @ w1[e,:,:F]) * (h @ w1[e,:,F:])) @ w2[e]

  * The token-per-expert selection uses the same order-preserving int64 key
    trick as the family siblings: pack the softmax weight's float32 bits with
    the inverted token index into one int64, so within an expert's candidate
    set the top-`capacity` are picked with a stable, structural tie-break.
  * The gate is the RAW softmax probability p[t, e*], NOT renormalized. Kept
    tokens receive the full gate scale; dropped tokens receive zero output.
  * The graded surface is forward AND backward. Gradients flow through the
    Switch gate p[t, e*] (softmax Jacobian over the whole expert row, into
    router_w and x) and through the expert compute -- ONLY for kept tokens.
    Dropped tokens have y[t] == 0, so they carry zero gradient into x[t],
    w1[e*], w2[e*] AND into router_w through their softmax row. This is
    load-bearing: a candidate that silently keeps a dropped token in the
    backward path will fail dw1/dw2/drouter_w on the graded tolerance.
  * An expert that no token picks contributes exact-zero weight gradients.

Correctness is defined as agreement with this function under the disclosed
per-dtype tolerances -- and where prose and this code disagree, the code wins.

The implementation below is deliberately simple and deliberately slow: a python
loop over experts with per-expert candidate masking and per-expert top-`capacity`
gathers. It is autograd-capable end to end, which is how the harness obtains
the reference gradients.
"""

import torch


def _select_top1(scores: torch.Tensor) -> torch.Tensor:
    """Argmax by softmax weight; ties to LOWER expert index. Returns [T,1]."""
    E = scores.shape[-1]
    b = torch.where(scores == 0.0, torch.zeros_like(scores), scores).contiguous()
    ib = b.view(torch.int32).to(torch.int64)
    u = torch.where(ib >= 0, ib ^ 0x80000000, (~ib) & 0xFFFFFFFF)
    e_idx = torch.arange(E, device=scores.device, dtype=torch.int64)
    keys = (u << 10) | (E - 1 - e_idx)
    return torch.topk(keys, 1, dim=-1, largest=True, sorted=True).indices


def _keep_topC(
    scores_col: torch.Tensor, mask: torch.Tensor, capacity: int
) -> torch.Tensor:
    """Return a bool [T] mask that keeps the top-`capacity` entries of
    scores_col that lie inside `mask`, ties broken to LOWER token index.
    Uses the same order-preserving int64 key trick, with masked-out entries
    given a sentinel key that always loses.
    """
    T = scores_col.shape[0]
    b = torch.where(
        scores_col == 0.0, torch.zeros_like(scores_col), scores_col
    ).contiguous()
    ib = b.view(torch.int32).to(torch.int64)
    u = torch.where(ib >= 0, ib ^ 0x80000000, (~ib) & 0xFFFFFFFF)
    t_idx = torch.arange(T, device=scores_col.device, dtype=torch.int64)
    tbits = max((T - 1).bit_length(), 1)
    keys = (u << tbits) | (T - 1 - t_idx)
    keys = torch.where(mask, keys, torch.full_like(keys, -(1 << 62)))
    n_cand = int(mask.sum())
    k = min(capacity, n_cand)
    out = torch.zeros(T, dtype=torch.bool, device=scores_col.device)
    if k == 0:
        return out
    picked = torch.topk(keys, k, largest=True, sorted=True).indices
    out.scatter_(0, picked, True)
    return out


def fused_moe_ref(
    x: torch.Tensor,  # [T, D] float32 or bfloat16
    router_w: torch.Tensor,  # [D, E] same dtype as x
    w1: torch.Tensor,  # [E, D, 2F] gate|up
    w2: torch.Tensor,  # [E, F, D]
    capacity: int,  # per-expert token cap
) -> torch.Tensor:  # [T, D] same dtype as x
    """Sequential ground truth for capacity-limited Switch-style routed MoE."""
    if x.dim() != 2 or router_w.dim() != 2 or w1.dim() != 3 or w2.dim() != 3:
        raise ValueError(
            f"bad ranks: x{tuple(x.shape)} rw{tuple(router_w.shape)} "
            f"w1{tuple(w1.shape)} w2{tuple(w2.shape)}"
        )
    T, D = x.shape
    D2, E = router_w.shape
    E2, D3, F2 = w1.shape
    F = F2 // 2
    if D2 != D or E2 != E or D3 != D or F2 != 2 * F or tuple(w2.shape) != (E, F, D):
        raise ValueError("inconsistent shapes")
    capacity = int(capacity)
    if not 1 <= capacity <= T:
        raise ValueError(f"capacity={capacity} out of range for T={T}")
    if E > 1024:
        raise ValueError(f"reference supports E <= 1024, got {E}")

    xf, rwf = x.float(), router_w.float()
    w1f, w2f = w1.float(), w2.float()

    logits = xf @ rwf  # [T, E]
    p = torch.softmax(logits, dim=-1)  # [T, E]
    with torch.no_grad():
        sel = _select_top1(p.detach()).squeeze(-1)  # [T]
        # per-expert candidate masks
        member = torch.zeros(T, E, dtype=torch.bool, device=x.device).scatter_(
            1, sel.unsqueeze(-1), True
        )  # [T, E]
        keep = torch.zeros(T, dtype=torch.bool, device=x.device)
        for e in range(E):
            cand = member[:, e]
            if not bool(cand.any()):
                continue
            keep_e = _keep_topC(p[:, e].detach(), cand, capacity)
            keep |= keep_e

    g_full = p.new_zeros(T, E).scatter(
        1, sel.unsqueeze(-1), torch.gather(p, 1, sel.unsqueeze(-1))
    )
    y = xf.new_zeros(T, D)
    for e in range(E):
        rows = (member[:, e] & keep).nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            continue
        h = xf.index_select(0, rows)  # [n, D]
        gu = h @ w1f[e]  # [n, 2F]
        fe = (torch.nn.functional.silu(gu[:, :F]) * gu[:, F:]) @ w2f[e]
        y = y.index_add(0, rows, g_full.index_select(0, rows)[:, e : e + 1] * fe)
    return y.to(x.dtype)


if __name__ == "__main__":
    torch.manual_seed(5)
    T, D, F, E, C = 12, 5, 4, 4, 2
    x = torch.randn(T, D).requires_grad_(True)
    rw = (torch.randn(D, E) * D**-0.5).requires_grad_(True)
    w1 = (torch.randn(E, D, 2 * F) * D**-0.5).requires_grad_(True)
    w2 = (torch.randn(E, F, D) * F**-0.5).requires_grad_(True)

    y = fused_moe_ref(x, rw, w1, w2, C)
    assert y.shape == (T, D) and y.dtype == torch.float32
    loss = (y * torch.linspace(-1, 1, T * D).reshape(T, D)).sum()
    loss.backward()
    assert x.grad.shape == (T, D) and rw.grad.shape == (D, E)
    assert w1.grad.shape == (E, D, 2 * F) and w2.grad.shape == (E, F, D)
    assert float(rw.grad.abs().max()) > 0

    # Capacity actually drops tokens: with T=12, E=4, C=2 the routing floor
    # is E*C=8, so at least T-E*C=4 tokens must have y-row equal to zero.
    zero_rows = int((y.detach().abs().sum(dim=-1) == 0).sum())
    assert zero_rows >= T - E * C, (
        f"expected at least {T - E * C} dropped tokens, got {zero_rows}"
    )

    # Independent double-precision brute force.
    def _brute(x, rw, w1, w2, C):
        xf, rwf = x.detach().double(), rw.detach().double()
        w1f, w2f = w1.detach().double(), w2.detach().double()
        Tn, Dn = xf.shape
        En = rwf.shape[1]
        Fn = w1f.shape[2] // 2
        p = torch.softmax(xf @ rwf, dim=-1)
        sel = [max(range(En), key=lambda e: (float(p[t, e]), -e)) for t in range(Tn)]
        kept = [False] * Tn
        for e in range(En):
            cand = [t for t in range(Tn) if sel[t] == e]
            cand.sort(key=lambda t: (-float(p[t, e]), t))
            for t in cand[:C]:
                kept[t] = True
        out = torch.zeros(Tn, Dn, dtype=torch.double)
        for t in range(Tn):
            if not kept[t]:
                continue
            e = sel[t]
            gu = xf[t] @ w1f[e]
            fe = (torch.nn.functional.silu(gu[:Fn]) * gu[Fn:]) @ w2f[e]
            out[t] += float(p[t, e]) * fe
        return out

    err = float((y.detach().double() - _brute(x, rw, w1, w2, C)).abs().max())
    assert err < 1e-6, f"reference disagrees with brute force: {err:.3e}"
    print(
        f"[capacity_dropped_switch] reference CPU sanity OK  err={err:.3e}  "
        f"dropped {zero_rows}/{T} tokens at C={C}"
    )
