"""Taskdef: capacity-limited Switch-style top-1 routed fused-MoE (SURFACE fwdbwd).

AUTHORED DRAFT (status 'authored-draft: needs GPU calibration'). Family
moe_routed: the routing decision -- scores, selection, per-expert capacity
clipping, gate -- is INSIDE the graded surface, and gradients flow through
the router. This member is marked hard:true.

    fused_moe(x[T,D], router_w[D,E], w1[E,D,2F], w2[E,F,D], capacity) -> y[T,D]
    logits = x @ router_w; p = softmax(logits); e* = argmax_e p (ties to the
    LOWER expert index); for each expert e keep the top-`capacity` candidates
    of {t: e*[t]==e} by p[t,e] (ties LOWER token index); excess DROPPED.
    y[t] = p[t, e*] * f(e*, x[t]) if kept else 0. SwiGLU experts; fp32
    arithmetic, one final cast. Gradients graded: dx, drouter_w, dw1, dw2.
    reference.py is the definition.

WHY HARD (KernelBench-hard lever, adapted to a ROUTED family): the correct
kernel must implement per-expert capacity clipping structurally -- a
data-dependent per-token drop mask that decouples selection (top-1 by softmax)
from delivery (top-`capacity` within each expert bucket). NO torch or vendor
builtin covers "top-1-then-per-bucket-top-C-with-index-tie-break-and-zero-out-
overflow-with-router-gradient-only-through-kept-tokens" as a single primitive;
the token-level DROP forces the backward path to be selective in a way plain
grouped-GEMM MoE code paths never are. bf16 router-arithmetic drift moves the
selection boundary AND the capacity boundary; a capacity-boundary flip changes
which of two ties is kept vs dropped, silently corrupting dw1[e*] and drouter_w
on the graded tolerance.

UNCALIBRATED SEAMS (must be re-measured by a golden/controls run in this
folder before freeze): TOL, OUT_TOL, TARGET_FRACTION_OF_SOTA, _MARGIN, the
hidden shape sweep, _SOTA_ROUTE, and the negative-control anchor strings.
Values below are informed placeholders carried from the sibling calibrated
moe_routed task.

Selection-stability design (load-bearing): TWO discontinuities must survive
selection changes -- the top-1 argmax AND the per-expert top-`capacity` cut.
make_inputs enforces a decision margin on BOTH boundaries via an fp64 shadow
softmax over the dtype-cast tensors: rows whose top-1/top-2 gap is positive
but below _MARGIN are redrawn, and per-expert candidate lists whose C/(C+1)
p-gap is positive but below _MARGIN are redrawn on their two boundary tokens.
Exact ties (gap == 0) are legal and manufactured by tie-forcing draws.
"""

import torch

import taskdef_api  # noqa: F401  (protocol module; defaults resolved via cfg)

ENTRY_NAME = "fused_moe"
REF_NAME = "fused_moe_ref"
SIGNATURE = (
    "fused_moe(x[T,D], router_w[D,E], w1[E,D,2F], w2[E,F,D], capacity) -> y[T,D]"
)
SURFACE = "fwdbwd"
TASK_NAME = "hephaestus/capacity_dropped_switch"

# PLACEHOLDER tolerances; the capacity clip adds a second discontinuity, so
# the gradient knees may be looser than switch_top1 -- re-measure on the
# oracle before freeze. TOL gates ALL gradients; compare() gates output at
# the tighter OUT_TOL.
TOL = {
    "float32": (4.0e-3, 4.0e-3),
    "bfloat16": (4.5e-2, 4.5e-2),
}

OUT_TOL = {
    "float32": (2.0e-4, 2.0e-4),
    "bfloat16": (4.5e-2, 4.5e-2),
}


def compare(got, want, dtype_name, shape):
    """Output comparison at the tighter OUT_TOL budgets (grads use TOL)."""
    return taskdef_api.default_compare(got, want, dtype_name, shape, OUT_TOL)


FORBIDDEN = [
    "torch._grouped_mm",
    "_grouped_mm",
    "vllm",
    "sglang",
    "grouped_gemm",
    "transformer_engine",
    "flashinfer",
]

VENDOR_LIB_PACKAGE = ""
VENDOR_LIB_SUBDIRS = ()

ADOPTION_FLOOR = 0.60

# PLACEHOLDER: capacity clipping introduces an extra irregularity vs plain
# switch, so the reachable fraction is lower. Never raise above what the
# private oracle repeatably reaches; set from a golden run before freeze.
TARGET_FRACTION_OF_SOTA = 0.28

_MARGIN = 1.0e-4  # decision-margin floor: top-1/top-2 AND cap/cap+1
_IDXB = 10


def loss(out):
    """Fixed pseudorandom +-uniform cotangent, deterministic per output shape.
    See switch_top1: an all-ones cotangent is adversarially coherent for a
    routed MoE and a fixed +-uniform cotangent stays bitwise across the three
    determinism runs while exposing the real backward difficulty."""
    g = torch.Generator(device=out.device)
    g.manual_seed((0x5EEDC0DE ^ (out.shape[0] * 31 + out.shape[1])) & 0x7FFFFFFF)
    v = (
        torch.rand(out.shape, generator=g, device=out.device, dtype=torch.float32) * 2.0
        - 1.0
    ).to(out.dtype)
    return (out.float() * v.float()).sum()


# ---------------------------------------------------------------------------
# Selection helpers (verifier-side twins of the reference's exact tie-break)
# ---------------------------------------------------------------------------


def _sel_top1(scores):
    E = scores.shape[-1]
    b = torch.where(scores == 0.0, torch.zeros_like(scores), scores).contiguous()
    ib = b.view(torch.int32).to(torch.int64)
    u = torch.where(ib >= 0, ib ^ 0x80000000, (~ib) & 0xFFFFFFFF)
    e_idx = torch.arange(E, device=scores.device, dtype=torch.int64)
    keys = (u << _IDXB) | (E - 1 - e_idx)
    return torch.topk(keys, 1, dim=-1, largest=True, sorted=True).indices


# ---------------------------------------------------------------------------
# Input draws
# ---------------------------------------------------------------------------


def _margin_fix(x32, xq, rw_q, capacity, scale, g, device):
    """Two discontinuities to shore up:
    (1) rows whose top-1/top-2 softmax p gap is (0, _MARGIN) -> redraw the row;
    (2) per-expert candidate sets with a C/(C+1) p gap in (0, _MARGIN) among
    their assigned tokens -> redraw the two boundary tokens. Both use fp64
    shadow softmax over the DTYPE-CAST tensors (what the op actually sees).
    """
    T = x32.shape[0]
    E = rw_q.shape[1]
    if E < 2:
        return xq
    rw64 = rw_q.double()
    for _ in range(64):
        logits = xq.double() @ rw64
        p = torch.softmax(logits, dim=-1)
        v = torch.topk(p, min(2, E), dim=-1).values
        gap1 = v[:, 0] - v[:, 1] if v.shape[-1] > 1 else torch.ones(T, device=device)
        bad_row = (gap1 > 0) & (gap1 < _MARGIN)
        # capacity boundary per expert
        sel = torch.argmax(p, dim=-1)  # [T] fp64 top-1
        touched = torch.zeros(T, dtype=torch.bool, device=device)
        for e in range(E):
            cand = (sel == e).nonzero(as_tuple=True)[0]
            n = int(cand.numel())
            if n <= capacity or capacity == 0:
                continue
            vals, idx = torch.topk(p[cand, e], capacity + 1, largest=True)
            gap = float(vals[capacity - 1] - vals[capacity])
            if 0.0 < gap < _MARGIN:
                touched[cand[idx[capacity - 1]]] = True
                touched[cand[idx[capacity]]] = True
        bad = bad_row | touched
        n = int(bad.sum())
        if n == 0:
            break
        fresh = torch.randn(n, x32.shape[1], generator=g, device=device) * float(scale)
        x32[bad] = fresh
        xq[bad] = fresh.to(xq.dtype)
    return xq


def make_inputs(shape, seed, scale, dtype, device):
    """One reproducible draw; NaN-free by construction.

    Optional verifier-side shape key (never published):
      tie_pool: int G -> router_w columns from a pool of G distinct columns
                (duplicated across experts) -> exact softmax ties in every
                dtype, exercising the lower-expert-index rule for real. The
                capacity clip is exercised by the always-there per-expert
                overflow when T > capacity * E.
    """
    T, D = int(shape["T"]), int(shape["D"])
    F, E = int(shape["F"]), int(shape["E"])
    capacity = int(shape["capacity"])
    g = torch.Generator(device=device)
    g.manual_seed(seed & 0x7FFFFFFF)

    pool = int(shape.get("tie_pool", 0))
    if pool:
        cols = torch.randn(D, pool, generator=g, device=device) * D**-0.5
        assign = torch.randint(0, pool, (E,), generator=g, device=device)
        rw32 = cols.index_select(1, assign).contiguous()
    else:
        rw32 = torch.randn(D, E, generator=g, device=device) * D**-0.5
    w1 = torch.randn(E, D, 2 * F, generator=g, device=device) * D**-0.5
    w2 = torch.randn(E, F, D, generator=g, device=device) * F**-0.5
    x32 = torch.randn(T, D, generator=g, device=device) * float(scale)

    rw_q = rw32.to(dtype).contiguous()
    xq = x32.to(dtype).contiguous()
    if pool == 0:
        xq = _margin_fix(x32, xq, rw_q, capacity, scale, g, device)

    args = {
        "x": xq.requires_grad_(True),
        "router_w": rw_q.requires_grad_(True),
        "w1": w1.to(dtype).contiguous().requires_grad_(True),
        "w2": w2.to(dtype).contiguous().requires_grad_(True),
        "capacity": capacity,
    }
    return args, ("x", "router_w", "w1", "w2")


# ---------------------------------------------------------------------------
# Anchor: production composition (top-1 sort + capacity clip + grouped-GEMM)
# ---------------------------------------------------------------------------


def _anchor_route(x, router_w, capacity):
    T = x.shape[0]
    E = router_w.shape[1]
    p = torch.softmax(x.float() @ router_w.float(), dim=-1)
    with torch.no_grad():
        sel = _sel_top1(p.detach()).squeeze(-1)  # [T]
        # per-expert candidate sort with lower-token-index tie break using the
        # same int64-key packing, sentinel key for non-candidates.
        St = p.detach().t().contiguous()  # [E, T]
        b = torch.where(St == 0.0, torch.zeros_like(St), St).contiguous()
        ib = b.view(torch.int32).to(torch.int64)
        u = torch.where(ib >= 0, ib ^ 0x80000000, (~ib) & 0xFFFFFFFF)
        tbits = max((T - 1).bit_length(), 1)
        t_idx = torch.arange(T, device=x.device, dtype=torch.int64)
        keys = (u << tbits) | (T - 1 - t_idx).unsqueeze(0)  # [E, T]
        member = torch.zeros(E, T, dtype=torch.bool, device=x.device).scatter_(
            0, sel.unsqueeze(0), True
        )
        keys = torch.where(member, keys, torch.full_like(keys, -(1 << 62)))
        k = min(capacity, T)
        picked = torch.topk(keys, k, dim=-1, largest=True, sorted=True).indices
        # count real candidates per expert; clip capacity per expert to it
        counts = member.sum(dim=-1)  # [E]
        keep_mask = torch.arange(k, device=x.device).unsqueeze(0) < counts.unsqueeze(-1)
    gate = torch.gather(p, 1, sel.unsqueeze(-1)).squeeze(-1)  # [T]
    return sel, picked, keep_mask, gate


def _sota_loop(x, router_w, w1, w2, capacity):
    """Expert-loop cuBLAS composition; each expert owns a contiguous C-row
    block of candidate tokens, then the drop mask zeroes overflow rows."""
    T, D = x.shape
    E = router_w.shape[1]
    F = w1.shape[2] // 2
    sel, picked, keep_mask, gate = _anchor_route(x, router_w, capacity)
    y = x.new_zeros(T, D)
    for e in range(E):
        rows = picked[e][keep_mask[e]]
        if rows.numel() == 0:
            continue
        h = x.index_select(0, rows)
        gu = h @ w1[e]
        fe = (torch.nn.functional.silu(gu[:, :F]) * gu[:, F:]) @ w2[e]
        ws = gate.index_select(0, rows).to(x.dtype).unsqueeze(1)
        y = y.index_add(0, rows, ws * fe)
    return y


def _sota_grouped(x, router_w, w1, w2, capacity):
    """torch._grouped_mm composition. Kept tokens sorted by expert with each
    expert's block padded to `capacity` (extras are dropped BEFORE the GEMM by
    a compact selection)."""
    T, D = x.shape
    E = router_w.shape[1]
    F = w1.shape[2] // 2
    sel, picked, keep_mask, gate = _anchor_route(x, router_w, capacity)
    # compact per-expert kept tokens
    tok_lists = []
    for e in range(E):
        tok_lists.append(picked[e][keep_mask[e]])
    if not any(int(t.numel()) for t in tok_lists):
        return x.new_zeros(T, D)
    tok = torch.cat(tok_lists)  # [sum_e kept_e]
    counts = torch.tensor(
        [int(t.numel()) for t in tok_lists], device=x.device, dtype=torch.int64
    )
    offs = counts.cumsum(0).to(torch.int32)
    xg = x.index_select(0, tok)
    h = torch._grouped_mm(xg, w1, offs=offs)
    act = torch.nn.functional.silu(h[:, :F]) * h[:, F:]
    yp = torch._grouped_mm(act, w2, offs=offs)
    ws = gate.index_select(0, tok).to(x.dtype).unsqueeze(1)
    return x.new_zeros(T, D).index_add(0, tok, yp * ws)


# PLACEHOLDER selection (calibrate per dtype on the graded shapes before
# freeze). Grouped-mm is expected faster for wide E; the loop wins for tiny E.
_SOTA_ROUTE = {"torch.bfloat16": "grouped_mm", "torch.float32": "grouped_mm"}


def sota(args):
    """Production anchor: the measured-faster composition per dtype."""
    x = args["x"]
    route = _SOTA_ROUTE[str(x.dtype)]
    fn = _sota_grouped if route == "grouped_mm" else _sota_loop
    return fn(x, args["router_w"], args["w1"], args["w2"], int(args["capacity"]))


# ---------------------------------------------------------------------------
# Stability modes (finiteness-and-not-raising gate; drift recorded only)
# ---------------------------------------------------------------------------


def _stab_common(D, F, E, g, device):
    rw = torch.randn(D, E, generator=g, device=device) * D**-0.5
    w1 = torch.randn(E, D, 2 * F, generator=g, device=device) * D**-0.5
    w2 = torch.randn(E, F, D, generator=g, device=device) * F**-0.5
    return rw, w1, w2


def _finish(x32, rw32, w1, w2, capacity, dtype):
    return {
        "x": x32.to(dtype).contiguous().requires_grad_(True),
        "router_w": rw32.to(dtype).contiguous().requires_grad_(True),
        "w1": w1.to(dtype).contiguous().requires_grad_(True),
        "w2": w2.to(dtype).contiguous().requires_grad_(True),
        "capacity": capacity,
    }, ("x", "router_w", "w1", "w2")


def _mode_router_saturated(shape, seed, scale, dtype, device):
    T, D = int(shape["T"]), int(shape["D"])
    F, E = int(shape["F"]), int(shape["E"])
    C = int(shape["capacity"])
    g = torch.Generator(device=device)
    g.manual_seed(seed & 0x7FFFFFFF)
    rw, w1, w2 = _stab_common(D, F, E, g, device)
    x = torch.randn(T, D, generator=g, device=device) * 30.0
    return _finish(x, rw, w1, w2, C, dtype)


def _mode_capacity_zero(shape, seed, scale, dtype, device):
    """capacity=1 while T >> E: nearly every candidate list overflows and
    every expert keeps exactly one token; drop rate is maximal. Finiteness
    only, output rows for dropped tokens must be exact zero."""
    T, D = int(shape["T"]), int(shape["D"])
    F, E = int(shape["F"]), int(shape["E"])
    g = torch.Generator(device=device)
    g.manual_seed(seed & 0x7FFFFFFF)
    rw, w1, w2 = _stab_common(D, F, E, g, device)
    x = torch.randn(T, D, generator=g, device=device) * float(scale)
    return _finish(x, rw, w1, w2, 1, dtype)


def _mode_capacity_full(shape, seed, scale, dtype, device):
    """capacity=T: no overflow anywhere; the drop mask must be all-True."""
    T, D = int(shape["T"]), int(shape["D"])
    F, E = int(shape["F"]), int(shape["E"])
    g = torch.Generator(device=device)
    g.manual_seed(seed & 0x7FFFFFFF)
    rw, w1, w2 = _stab_common(D, F, E, g, device)
    x = torch.randn(T, D, generator=g, device=device) * float(scale)
    return _finish(x, rw, w1, w2, T, dtype)


def _mode_x_huge(shape, seed, scale, dtype, device):
    T, D = int(shape["T"]), int(shape["D"])
    F, E = int(shape["F"]), int(shape["E"])
    C = int(shape["capacity"])
    g = torch.Generator(device=device)
    g.manual_seed(seed & 0x7FFFFFFF)
    rw, w1, w2 = _stab_common(D, F, E, g, device)
    mag = torch.rand(T, D, generator=g, device=device) * 6.0 - 3.0
    x = (
        torch.randn(T, D, generator=g, device=device)
        * torch.pow(10.0, mag)
        * float(scale)
    )
    return _finish(x, rw, w1, w2, C, dtype)


def _mode_near_uniform(shape, seed, scale, dtype, device):
    T, D = int(shape["T"]), int(shape["D"])
    F, E = int(shape["F"]), int(shape["E"])
    C = int(shape["capacity"])
    g = torch.Generator(device=device)
    g.manual_seed(seed & 0x7FFFFFFF)
    _, w1, w2 = _stab_common(D, F, E, g, device)
    rw = torch.randn(D, E, generator=g, device=device) * (D**-0.5) * 1.0e-3
    x = torch.randn(T, D, generator=g, device=device) * float(scale)
    return _finish(x, rw, w1, w2, C, dtype)


STABILITY_MODES = {
    "router_saturated": _mode_router_saturated,
    "capacity_zero": _mode_capacity_zero,
    "capacity_full": _mode_capacity_full,
    "x_huge": _mode_x_huge,
    "near_uniform": _mode_near_uniform,
}

SMOKE_SHAPE = {"T": 1024, "D": 1024, "F": 704, "E": 32, "capacity": 40}
# tie_pool + a capacity that would drop tokens: a scheduling-dependent tie
# winner routes a different token to the kept slot, so bitwise identity across
# three determinism runs cannot survive a racy selection.
DETERMINISM_SHAPE = {
    "T": 2048,
    "D": 1024,
    "F": 704,
    "E": 64,
    "capacity": 40,
    "tie_pool": 8,
}
STABILITY_SHAPE = {"T": 1024, "D": 1024, "F": 704, "E": 32, "capacity": 40}


# ---------------------------------------------------------------------------
# bench.py published surface (DISJOINT from and smaller than the graded set).
# ---------------------------------------------------------------------------

BENCH_PUBLISHED_SHAPES = {
    "p1": {
        "T": 4096,
        "D": 1024,
        "F": 704,
        "E": 32,
        "capacity": 160,
        "dtypes": ["bfloat16"],
    },
    "p2": {
        "T": 2048,
        "D": 2048,
        "F": 1408,
        "E": 64,
        "capacity": 40,
        "dtypes": ["bfloat16"],
    },
}
BENCH_CHECK_SHAPE = {"T": 512, "D": 512, "F": 256, "E": 16, "capacity": 40}
BENCH_CHECK_SHAPE_NPOT = {"T": 301, "D": 200, "F": 96, "E": 11, "capacity": 31}
BENCH_CHECK_SHAPE_TIES = {
    "T": 256,
    "D": 256,
    "F": 128,
    "E": 16,
    "capacity": 20,
    "tie_pool": 4,
}


# ---------------------------------------------------------------------------
# Negative controls (Invariant 17, both halves). AUTHORED DRAFT: re-verify
# with controls_runner on GPU before freeze.
# ---------------------------------------------------------------------------


def _mut_dp(src):
    anchor = "                keep |= _keep_topC(p[:, e].detach(), cand, capacity)\n"
    if anchor not in src:
        raise RuntimeError("INERT: capacity-clip anchor not found in starter")
    repl = (
        "                if float(p.detach().amax(dim=-1).min()) >= 0.999:\n"
        "                    keep |= _keep_topC(p[:, e].detach(), cand, capacity)\n"
        "                else:\n"
        "                    keep |= cand\n"
    )
    return src.replace(anchor, repl, 1)


_FALLBACK_SRC = '''\
"""Pure-torch fallback candidate (negative control): no Triton anywhere."""
import torch


def _pick(scores):
    E = scores.shape[-1]
    b = torch.where(scores == 0.0, torch.zeros_like(scores), scores).contiguous()
    ib = b.view(torch.int32).to(torch.int64)
    u = torch.where(ib >= 0, ib ^ 0x80000000, (~ib) & 0xFFFFFFFF)
    e_idx = torch.arange(E, device=scores.device, dtype=torch.int64)
    keys = (u << 10) | (E - 1 - e_idx)
    return torch.topk(keys, 1, dim=-1, largest=True, sorted=True).indices


def _keep_topC(p_col, mask, cap):
    T = p_col.shape[0]
    b = torch.where(p_col == 0.0, torch.zeros_like(p_col), p_col).contiguous()
    ib = b.view(torch.int32).to(torch.int64)
    u = torch.where(ib >= 0, ib ^ 0x80000000, (~ib) & 0xFFFFFFFF)
    t_idx = torch.arange(T, device=p_col.device, dtype=torch.int64)
    tbits = max((T - 1).bit_length(), 1)
    keys = (u << tbits) | (T - 1 - t_idx)
    keys = torch.where(mask, keys, torch.full_like(keys, -(1 << 62)))
    n_cand = int(mask.sum())
    k = min(int(cap), n_cand)
    out = torch.zeros(T, dtype=torch.bool, device=p_col.device)
    if k == 0:
        return out
    picked = torch.topk(keys, k, largest=True, sorted=True).indices
    out.scatter_(0, picked, True)
    return out


class _Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, router_w, w1, w2, capacity):
        capacity = int(capacity)
        with torch.enable_grad():
            xf = x.detach().float().requires_grad_(True)
            rwf = router_w.detach().float().requires_grad_(True)
            w1f = w1.detach().float().requires_grad_(True)
            w2f = w2.detach().float().requires_grad_(True)
            T, D = xf.shape
            E = rwf.shape[1]
            F = w1f.shape[2] // 2
            p = torch.softmax(xf @ rwf, dim=-1)
            with torch.no_grad():
                sel = _pick(p.detach()).squeeze(-1)
                member = torch.zeros(T, E, dtype=torch.bool,
                                     device=x.device).scatter_(
                    1, sel.unsqueeze(-1), True)
                keep = torch.zeros(T, dtype=torch.bool, device=x.device)
                for e in range(E):
                    cand = member[:, e]
                    if not bool(cand.any()):
                        continue
                    keep |= _keep_topC(p[:, e].detach(), cand, capacity)
            g_full = p.new_zeros(T, E).scatter(
                1, sel.unsqueeze(-1),
                torch.gather(p, 1, sel.unsqueeze(-1)))
            y = xf.new_zeros(T, D)
            for e in range(E):
                rows = (member[:, e] & keep).nonzero(as_tuple=True)[0]
                if rows.numel() == 0:
                    continue
                h = xf.index_select(0, rows)
                gu = h @ w1f[e]
                fe = (torch.nn.functional.silu(gu[:, :F]) * gu[:, F:]) @ w2f[e]
                y = y.index_add(0, rows,
                                g_full.index_select(0, rows)[:, e:e + 1] * fe)
        ctx.save_for_backward(xf, rwf, w1f, w2f, y)
        ctx.dtypes = (x.dtype, router_w.dtype, w1.dtype, w2.dtype)
        return y.detach().to(x.dtype)

    @staticmethod
    def backward(ctx, dy):
        xf, rwf, w1f, w2f, y = ctx.saved_tensors
        gx, grw, gw1, gw2 = torch.autograd.grad(y, (xf, rwf, w1f, w2f),
                                                dy.float())
        dx_t, drw_t, dw1_t, dw2_t = ctx.dtypes
        return (gx.to(dx_t), grw.to(drw_t), gw1.to(dw1_t), gw2.to(dw2_t), None)


def fused_moe(x, router_w, w1, w2, capacity):
    return _Fn.apply(x, router_w, w1, w2, capacity)
'''


NEGATIVE_CONTROLS = [
    {
        "name": "nc_forbidden_call",
        "base": "starter",
        "target_file": "kernel.py",
        "mutate": ("import triton\n", "import triton\n_SHORTCUT = torch._grouped_mm\n"),
        "bound_gate": "G1",
        "oracle_env": False,
        "expectation": "the forbidden-symbol scan names torch._grouped_mm and "
        "zeroes at G1 before the candidate is ever imported",
    },
    {
        "name": "nc_drop_ignored",
        "base": "starter",
        "target_file": "kernel.py",
        "mutate": (
            "rows = (member[:, e] & keep).nonzero(as_tuple=True)[0]",
            "rows = member[:, e].nonzero(as_tuple=True)[0]",
        ),
        "bound_gate": "G2",
        "oracle_env": False,
        "expectation": "the capacity drop is silently ignored: every candidate "
        "flows through the expert path, so overflow tokens "
        "receive nonzero output rows and the dense G2 "
        "comparison fails at the smoke stage",
    },
    {
        "name": "nc_tiebreak_reversed",
        "base": "starter",
        "target_file": "kernel.py",
        "mutate": (
            "keys = (u << tbits) | (T - 1 - t_idx)",
            "keys = (u << tbits) | t_idx",
        ),
        "bound_gate": "G2",
        "oracle_env": False,
        "expectation": "capacity-boundary ties broken toward the HIGHER token "
        "index: identical on tie-free draws, but on the "
        "tie-forcing hidden shape a wrong tie winner keeps the "
        "wrong token and drops another -> G2 edge-case failure",
    },
    {
        "name": "nc_differential_path",
        "base": "starter",
        "target_file": "kernel.py",
        "mutate": _mut_dp,
        "bound_gate": "G2",
        "oracle_env": True,
        "expectation": "the regime-switch exploit: perform capacity clipping "
        "only when the softmax is saturated (a stability probe) "
        "and keep every candidate on nominal inputs. Because "
        "timed tensors ARE the checked tensors, the cheap path "
        "runs on graded draws and G2 fails at smoke even with "
        "FORGE_ORACLE=1",
    },
    {
        "name": "nc_torch_fallback",
        "base": "starter",
        "target_file": "kernel.py",
        "mutate": lambda src: _FALLBACK_SRC,
        "bound_gate": "G4",
        "oracle_env": False,
        "expectation": "correct and deterministic pure-torch composition, 100% "
        "of device time outside candidate Triton kernels: G2 "
        "passes and the written-kernel adoption floor zeroes at "
        "G4",
    },
]
