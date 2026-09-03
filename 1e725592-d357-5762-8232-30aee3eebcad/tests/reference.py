"""
Reference implementation -- PyTorch-only ground truth for correctness verification.
DO NOT MODIFY. This is the oracle the benchmark harness checks every candidate against.

Family: moe_routed (routing computed INSIDE the graded op; forward + backward).
This member is HIERARCHICAL group-then-expert top-K routing (DeepSeek-V2
device-limited routing): experts are partitioned into G equal-size groups,
each token selects the single top-1 group by its GROUP SCORE (sum of the
group's top-2 expert softmax weights), then selects top-K experts within
that chosen group by expert p; combine by p_sel/p_sel.sum(-1). This is a
TWO-STAGE selection with two independent discontinuity boundaries.

    fused_moe(x, router_w, w1, w2, G, K) -> y

    x:        [T, D]     tokens (float32 or bfloat16)
    router_w: [D, E]     router projection; E must be divisible by G
    w1:       [E, D, 2F] expert input projections; :F gate, F: up
    w2:       [E, F, D]  expert output projections
    G:        int        number of groups (E % G == 0)
    K:        int        top-K experts within selected group, 1 <= K <= E/G
    y:        [T, D]     same dtype as x

Semantics, exactly (all arithmetic in float32; inputs cast up, only final y cast back):

    logits = x @ router_w                              # [T, E]
    p      = softmax(logits, dim=-1)                   # [T, E]
    reshape p to [T, G, S] where S = E / G             # per-group [T, S] slabs
    # stage 1: group score = sum of top-2 p within each group
    gs[t, g] = top2_sum(p[t, g, :])                    # [T, G]
    grp[t]   = argmax_g gs[t, g]  ties -> LOWER GROUP index; -0.0 == +0.0
    # stage 2: top-K experts within the chosen group by p
    sel_local[t] = top-K positions in p[t, grp[t], :]  ties LOWER LOCAL index
    sel[t, k]    = grp[t] * S + sel_local[t, k]        # global expert index
    # combine
    w[t] = p[t, sel[t]] / p[t, sel[t]].sum(-1)         # p_sel/sum
    y[t] = sum_k w[t, k] * f(sel[t, k], x[t])
    f(e, h) = (silu(h @ w1[e, :, :F]) * (h @ w1[e, :, F:])) @ w2[e]

  * Both boundaries decided by the int64-key trick: the group score is a
    small [G] softmax-of-sums, and its tie-break rule is LOWER group index;
    the intra-group top-K is over [S] entries and its tie-break rule is
    LOWER LOCAL expert index (which is also the LOWER GLOBAL index within
    the chosen group).
  * Only experts inside the chosen group receive gradients. Every logit
    inside that group's slab receives a scoring-softmax gradient AND a
    group-score gradient (top-2 of that group's p sums into gs[t, grp]);
    logits outside the chosen group receive only the scoring-softmax
    gradient through gs[t, g!=grp[t]] flowing back into p, but since the
    group selection is argmax (locally constant), those grads land only
    on the scoring softmax's Jacobian into the WHOLE row (as always).
  * The graded surface is fwd AND bwd; 4 grads (dx, drouter_w, dw1, dw2).
    Experts outside the chosen group of every token receive exact-zero
    weight gradients (both w1 and w2).

Correctness is defined as agreement with this function under the disclosed
per-dtype tolerances -- and where prose and this code disagree, the code wins.
"""

import torch


def _top2_sum(pg: torch.Tensor) -> torch.Tensor:
    """Sum of top-2 along the last axis of `pg`. If last axis is size 1,
    returns pg squeezed."""
    S = pg.shape[-1]
    if S == 1:
        return pg.squeeze(-1)
    v = torch.topk(pg, min(2, S), dim=-1).values
    return v.sum(dim=-1)


def _sel_lowest_index(scores: torch.Tensor, k: int, ibits: int) -> torch.Tensor:
    """Top-k of `scores` along last axis with ties broken to LOWER index."""
    b = torch.where(scores == 0.0, torch.zeros_like(scores), scores).contiguous()
    ib = b.view(torch.int32).to(torch.int64)
    u = torch.where(ib >= 0, ib ^ 0x80000000, (~ib) & 0xFFFFFFFF)
    N = scores.shape[-1]
    idx = torch.arange(N, device=scores.device, dtype=torch.int64)
    keys = (u << ibits) | (N - 1 - idx)
    return torch.topk(keys, k, dim=-1, largest=True, sorted=True).indices


def fused_moe_ref(
    x: torch.Tensor,  # [T, D]
    router_w: torch.Tensor,  # [D, E]
    w1: torch.Tensor,  # [E, D, 2F]
    w2: torch.Tensor,  # [E, F, D]
    G: int,  # number of groups
    K: int,  # top-K within chosen group
) -> torch.Tensor:  # [T, D]
    """Ground truth for hierarchical group-then-expert routed MoE."""
    if x.dim() != 2 or router_w.dim() != 2 or w1.dim() != 3 or w2.dim() != 3:
        raise ValueError("bad ranks")
    T, D = x.shape
    D2, E = router_w.shape
    E2, D3, F2 = w1.shape
    F = F2 // 2
    G = int(G)
    K = int(K)
    if D2 != D or E2 != E or D3 != D or F2 != 2 * F or tuple(w2.shape) != (E, F, D):
        raise ValueError("inconsistent shapes")
    if E % G:
        raise ValueError(f"E={E} not divisible by G={G}")
    S = E // G
    if not 1 <= K <= S:
        raise ValueError(f"K={K} out of range for S={S}")
    if E > 4096:
        raise ValueError(f"reference supports E <= 4096, got {E}")

    xf, rwf = x.float(), router_w.float()
    w1f, w2f = w1.float(), w2.float()

    logits = xf @ rwf  # [T, E]
    p = torch.softmax(logits, dim=-1)  # [T, E]
    pg = p.view(T, G, S)  # per-group slabs

    gbits = max((G - 1).bit_length(), 1)
    sbits = max((S - 1).bit_length(), 1)

    with torch.no_grad():
        gs = _top2_sum(pg.detach())  # [T, G]
        grp = _sel_lowest_index(gs, 1, gbits).squeeze(-1)  # [T]
        # gather each token's chosen-group p slab
        row_idx = torch.arange(T, device=x.device, dtype=torch.int64)
        p_chosen = pg.detach()[row_idx, grp]  # [T, S]
        sel_local = _sel_lowest_index(p_chosen, K, sbits)  # [T, K]
        sel = grp.unsqueeze(-1) * S + sel_local  # [T, K] global expert

    p_sel = torch.gather(p, 1, sel)  # [T, K] grads flow
    w = p_sel / p_sel.sum(dim=-1, keepdim=True)  # [T, K]
    w_full = p.new_zeros(T, E).scatter(1, sel, w)
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
    torch.manual_seed(17)
    T, D, F, G, S, K = 6, 5, 4, 4, 3, 2
    E = G * S
    x = torch.randn(T, D).requires_grad_(True)
    rw = (torch.randn(D, E) * D**-0.5).requires_grad_(True)
    w1 = (torch.randn(E, D, 2 * F) * D**-0.5).requires_grad_(True)
    w2 = (torch.randn(E, F, D) * F**-0.5).requires_grad_(True)

    y = fused_moe_ref(x, rw, w1, w2, G, K)
    assert y.shape == (T, D) and y.dtype == torch.float32
    loss = (y * torch.linspace(-1, 1, T * D).reshape(T, D)).sum()
    loss.backward()
    assert x.grad is not None and rw.grad is not None
    assert w1.grad is not None and w2.grad is not None
    assert float(rw.grad.abs().max()) > 0

    def _brute(x, rw, w1, w2, G, K):
        xf, rwf = x.detach().double(), rw.detach().double()
        w1f, w2f = w1.detach().double(), w2.detach().double()
        Tn, Dn = xf.shape
        En = rwf.shape[1]
        Fn = w1f.shape[2] // 2
        Sn = En // G
        p = torch.softmax(xf @ rwf, dim=-1)
        out = torch.zeros(Tn, Dn, dtype=torch.double)
        for t in range(Tn):
            pg = p[t].reshape(G, Sn)
            gs = []
            for g in range(G):
                vals = sorted(
                    [(float(pg[g, s]), s) for s in range(Sn)],
                    key=lambda z: (-z[0], z[1]),
                )
                gs.append(vals[0][0] + (vals[1][0] if Sn > 1 else 0.0))
            grp = max(range(G), key=lambda g: (gs[g], -g))
            row = pg[grp]
            local = sorted(range(Sn), key=lambda s: (-float(row[s]), s))[:K]
            sel = [grp * Sn + s for s in local]
            ps = torch.tensor([float(p[t, e]) for e in sel], dtype=torch.double)
            w = ps / ps.sum()
            for k, e in enumerate(sel):
                gu = xf[t] @ w1f[e]
                fe = (torch.nn.functional.silu(gu[:Fn]) * gu[Fn:]) @ w2f[e]
                out[t] += float(w[k]) * fe
        return out

    err = float((y.detach().double() - _brute(x, rw, w1, w2, G, K)).abs().max())
    assert err < 1e-6, f"reference disagrees with brute force: {err:.3e}"
    # verify grads outside the chosen groups' experts are zero
    zeroed = 0
    with torch.no_grad():
        for t in range(T):
            pg = torch.softmax(x[t].double() @ rw.double(), dim=-1).reshape(G, S)
            gs_ = [
                (float(torch.topk(pg[g], min(2, S)).values.sum()), g) for g in range(G)
            ]
            grp = max(range(G), key=lambda g: (gs_[g][0], -g))
            for g in range(G):
                if g != grp:
                    zeroed += 1
    assert zeroed > 0
    print(
        f"[hierarchical_group_topk] reference CPU sanity OK  err={err:.3e}  "
        f"grads dx/drw/dw1/dw2 populated; {zeroed} (token, off-group) pairs "
        f"receive zero weight-grad"
    )
