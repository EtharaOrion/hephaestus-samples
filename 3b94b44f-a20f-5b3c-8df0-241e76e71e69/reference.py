"""
Reference implementation -- PyTorch-only ground truth for correctness verification.
DO NOT MODIFY. This is the oracle the benchmark harness checks every candidate against.

Family: moe_routed (routing computed INSIDE the graded op; forward + backward).
This member is the SHARED-plus-ROUTED composition (DeepSeek/Qwen-MoE style):
an always-on DENSE shared expert plus a softmax-routed top-K sparse expert
pool, summed together. The routing decision (softmax + top-K + p_sel/sum
combine, mixtral-style) lives INSIDE the operator; the shared path is dense
and unconditional. SIX gradients are graded.

    fused_moe(x, router_w, w1, w2, w1_s, w2_s, K) -> y

    x:        [T, D]     tokens (float32 or bfloat16)
    router_w: [D, E]     router projection, same dtype as x
    w1:       [E, D, 2F] routed expert input projections; :F gate, F: up
    w2:       [E, F, D]  routed expert output projections
    w1_s:     [D, 2Fs]   shared expert input projection; :Fs gate, Fs: up
    w2_s:     [Fs, D]    shared expert output projection
    K:        int        experts selected per token from the routed pool, 1..E
    y:        [T, D]     same dtype as x

Semantics, exactly (all arithmetic in float32; inputs are cast up, only the
final y is cast back to the input dtype):

    # routed contribution
    logits = x @ router_w                              # [T, E]
    p      = softmax(logits, dim=-1)                   # [T, E]
    sel    = top-K experts per token by p, ties broken by LOWER expert index
             first; -0.0 and +0.0 compare equal
    w      = p[t, sel] / p[t, sel].sum(-1)             # p_sel/sum combine
    y_r[t] = sum_k w[t, k] * f_r(sel[t, k], x[t])
    f_r(e, h) = (silu(h @ w1[e, :, :F]) * (h @ w1[e, :, F:])) @ w2[e]

    # shared contribution (always-on, dense, unconditional)
    y_s[t] = (silu(x[t] @ w1_s[:, :Fs]) * (x[t] @ w1_s[:, Fs:])) @ w2_s

    # composed output
    y[t]   = y_r[t] + y_s[t]

  * Selection is by (p desc, index asc) via the int64-key trick; ties broken
    to the LOWER expert index; -0.0 canonicalised to +0.0 first.
  * The combine over routed experts uses the mixtral rule (p_sel/sum). A
    variant that drops the shared path or moves the shared path into the
    combine gate is a different operator and will fail G2.
  * SIX gradients are graded: dx (both paths), drouter_w (routed only),
    dw1, dw2 (routed), dw1_s, dw2_s (shared). Shared-path gradients into x
    are added to routed-path gradients into x; both contribute.
  * Experts no token selects receive exactly zero weight gradients.
  * The same expert never appears twice for one token (top-K of E distinct).

Correctness is defined as agreement with this function under the disclosed
per-dtype tolerances -- and where prose and this code disagree, the code wins.

The implementation below is deliberately simple and deliberately slow: a
python loop over routed experts with boolean-mask gathers, plus a dense
shared-expert path. It is autograd-capable end to end, which is how the
harness obtains the reference gradients.
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
    w1_s: torch.Tensor,  # [D, 2Fs]
    w2_s: torch.Tensor,  # [Fs, D]
    K: int,  # top-K routed
) -> torch.Tensor:  # [T, D]
    """Ground truth for shared-expert routed top-K SwiGLU MoE."""
    if any(t.dim() not in (2, 3) for t in (x, router_w, w1, w2, w1_s, w2_s)):
        raise ValueError("bad ranks")
    T, D = x.shape
    D2, E = router_w.shape
    E2, D3, F2 = w1.shape
    F = F2 // 2
    D4, Fs2 = w1_s.shape
    Fs = Fs2 // 2
    if D2 != D or E2 != E or D3 != D or F2 != 2 * F or tuple(w2.shape) != (E, F, D):
        raise ValueError("inconsistent routed shapes")
    if D4 != D or Fs2 != 2 * Fs or tuple(w2_s.shape) != (Fs, D):
        raise ValueError("inconsistent shared shapes")
    K = int(K)
    if not 1 <= K <= E:
        raise ValueError(f"K={K} out of range for E={E}")
    if E > 4096:
        raise ValueError(f"reference supports E <= 4096, got {E}")

    xf, rwf = x.float(), router_w.float()
    w1f, w2f = w1.float(), w2.float()
    w1sf, w2sf = w1_s.float(), w2_s.float()

    # routed
    logits = xf @ rwf
    p = torch.softmax(logits, dim=-1)
    with torch.no_grad():
        sel = _selection(p.detach(), K)
    p_sel = torch.gather(p, 1, sel)
    w = p_sel / p_sel.sum(dim=-1, keepdim=True)
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

    # shared (dense, unconditional)
    gs = xf @ w1sf
    y = y + (torch.nn.functional.silu(gs[:, :Fs]) * gs[:, Fs:]) @ w2sf
    return y.to(x.dtype)


if __name__ == "__main__":
    torch.manual_seed(13)
    T, D, F, Fs, E, K = 6, 5, 4, 3, 8, 3
    x = torch.randn(T, D).requires_grad_(True)
    rw = (torch.randn(D, E) * D**-0.5).requires_grad_(True)
    w1 = (torch.randn(E, D, 2 * F) * D**-0.5).requires_grad_(True)
    w2 = (torch.randn(E, F, D) * F**-0.5).requires_grad_(True)
    w1_s = (torch.randn(D, 2 * Fs) * D**-0.5).requires_grad_(True)
    w2_s = (torch.randn(Fs, D) * Fs**-0.5).requires_grad_(True)

    y = fused_moe_ref(x, rw, w1, w2, w1_s, w2_s, K)
    assert y.shape == (T, D) and y.dtype == torch.float32
    loss = (y * torch.linspace(-1, 1, T * D).reshape(T, D)).sum()
    loss.backward()
    for t, name in [
        (x, "x"),
        (rw, "rw"),
        (w1, "w1"),
        (w2, "w2"),
        (w1_s, "w1_s"),
        (w2_s, "w2_s"),
    ]:
        assert t.grad is not None and float(t.grad.abs().max()) > 0, name

    def _brute(x, rw, w1, w2, w1_s, w2_s, K):
        xf, rwf = x.detach().double(), rw.detach().double()
        w1f, w2f = w1.detach().double(), w2.detach().double()
        w1sf, w2sf = w1_s.detach().double(), w2_s.detach().double()
        Tn, Dn = xf.shape
        En = rwf.shape[1]
        Fn = w1f.shape[2] // 2
        Fsn = w1sf.shape[1] // 2
        p = torch.softmax(xf @ rwf, dim=-1)
        out = torch.zeros(Tn, Dn, dtype=torch.double)
        for t in range(Tn):
            order = sorted(range(En), key=lambda e: (-float(p[t, e]), e))
            sel = order[:K]
            ps = torch.tensor([float(p[t, e]) for e in sel], dtype=torch.double)
            w = ps / ps.sum()
            for k, e in enumerate(sel):
                gu = xf[t] @ w1f[e]
                fe = (torch.nn.functional.silu(gu[:Fn]) * gu[Fn:]) @ w2f[e]
                out[t] += float(w[k]) * fe
            # shared
            gs = xf[t] @ w1sf
            out[t] += (torch.nn.functional.silu(gs[:Fsn]) * gs[Fsn:]) @ w2sf
        return out

    err = float(
        (y.detach().double() - _brute(x, rw, w1, w2, w1_s, w2_s, K)).abs().max()
    )
    assert err < 1e-6, f"reference disagrees with brute force: {err:.3e}"
    print(
        f"[shared_expert_routed_topk] reference CPU sanity OK  err={err:.3e}  "
        f"6 grads populated (x, rw, w1, w2, w1_s, w2_s)"
    )
