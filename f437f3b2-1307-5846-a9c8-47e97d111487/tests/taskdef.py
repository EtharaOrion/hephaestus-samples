"""Taskdef: expert-choice routed fused-MoE (SURFACE fwdbwd). THE HARD MEMBER.

AUTHORED DRAFT (status 'authored-draft: needs GPU calibration'). Family
moe_routed: the routing decision -- affinities, selection, gate -- is INSIDE the
graded surface, and gradients flow through the router. This member is
expert-choice routing (Zhou et al. 2022) and is marked hard:true: each EXPERT
picks its top-C tokens, so a token is chosen by a data-dependent number of
experts (0..E) and the combine is irregular in the token dimension.

    fused_moe(x[T,D], router_w[D,E], w1[E,D,2F], w2[E,F,D], C) -> y[T,D]
    logits = x @ router_w; S = softmax(logits, dim=-1) (softmax over EXPERTS, so
    each token's affinity row sums to 1); for each expert e, sel[e] = the top-C
    tokens by S[:,e] (ties to the LOWER TOKEN index); the gate g[e,c] =
    S[sel[e,c], e] is the RAW affinity (no renormalization); y[t] = sum over
    every (e,c) with sel[e,c]==t of g[e,c] * f(e, x[t]), summed in ascending
    expert order. SwiGLU experts; fp32 arithmetic, one final cast. Gradients
    graded: dx, drouter_w, dw1, dw2 -- all four flow through the softmax
    affinities (a DENSE Jacobian over the expert row) and the expert compute.
    reference.py is the definition.

WHY HARD (most headroom / least reachable): the selection is over the TOKEN axis
(a top-C over T, transposed from the token-choice siblings' top-A over E), the
per-token combine has variable fan-in, and the softmax-over-experts gate coupled
with a token-axis selection resists the token-sorted grouped-GEMM dispatch the
other members reuse. The dispatch itself is perfectly load-balanced (exactly C
tokens per expert, E*C pairs), but a correct deterministic combine must reduce a
data-dependent number of contributions per output row.

UNCALIBRATED SEAMS (must be re-measured by a golden/controls run in this folder
before freeze): TOL, OUT_TOL, TARGET_FRACTION_OF_SOTA, _MARGIN (the affinity
scale is ~1/E, unlike the [0,1] sigmoid siblings, so the margin floor may need
per-E adjustment), the hidden shape sweep, _SOTA_ROUTE, and the negative-control
anchor strings (which target this folder's kernel.py starter). Values below are
informed placeholders carried from the sibling calibrated moe_routed task and
flagged where set.

Selection-stability design (load-bearing): expert-choice selection is
discontinuous, so make_inputs enforces a decision margin PER EXPERT -- experts
whose C-th/(C+1)-th affinity gap is positive but below _MARGIN have their two
boundary tokens redrawn (fp64 shadow softmax over the dtype-cast tensors,
iterated). Exact ties (gap == 0) are legal and manufactured by the tie-forcing
draw, which duplicates TOKEN rows (bit-equal x -> bit-equal affinities in both
dtypes): a boundary that falls inside a duplicate group is split by the lower-
token-index rule, and because the two tied tokens are DIFFERENT output rows, a
wrong tie winner moves the contribution to the wrong row and changes y.
"""

import torch

import taskdef_api  # noqa: F401  (protocol module; defaults resolved via cfg)

ENTRY_NAME = "fused_moe"
REF_NAME = "fused_moe_ref"
SIGNATURE = ("fused_moe(x[T,D], router_w[D,E], w1[E,D,2F], w2[E,F,D], C) "
             "-> y[T,D]")
SURFACE = "fwdbwd"
TASK_NAME = "hephaestus/expert_choice"

# PLACEHOLDER tolerances carried from the calibrated moe_routed task. Expert
# choice adds a variable-fan-in combine and a softmax-over-experts backward, so
# the gradient knees may be looser -- re-measure on the oracle before freeze.
# TOL gates ALL gradients; compare() gates the output at the tighter OUT_TOL.
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


# Grouped-GEMM libraries and MoE serving stacks that ship fused-MoE kernels.
# torch.topk/torch.sort on the [T,E] (or its [E,T] transpose) affinity matrix
# or the [E*C] routing metadata are permitted (disclosed in instruction.md); G4
# still requires the heavy arithmetic in candidate-declared kernels.
FORBIDDEN = [
    "torch._grouped_mm", "_grouped_mm",
    "vllm", "sglang", "grouped_gemm", "transformer_engine", "flashinfer",
]

# No vendorable python source tree for the anchor (torch builtin + cuBLAS):
# shingle scan skipped (recorded available:false); the FORBIDDEN symbol scan
# still runs.
VENDOR_LIB_PACKAGE = ""
VENDOR_LIB_SUBDIRS = ()

ADOPTION_FLOOR = 0.60

# PLACEHOLDER, and LOWER than the token-choice siblings on purpose: expert
# choice is the hard member (most headroom / least reachable), so the oracle is
# expected to reach a smaller fraction of the anchor. Never raise this above
# what the private oracle repeatably reaches; set from a golden run in this
# folder before freeze.
TARGET_FRACTION_OF_SOTA = 0.25

_MARGIN = 1.0e-4          # per-expert C/(C+1) affinity gap floor (UNCALIBRATED)
_IDXB = 10               # (unused here; token-index bits are sized to T)


# ---------------------------------------------------------------------------
# Backward cotangent (overrides taskdef_api.default_loss)
# ---------------------------------------------------------------------------

def loss(out):
    """Fixed pseudorandom +-uniform cotangent, deterministic per output shape.

    The protocol default backpropagates an all-ones cotangent, which is
    adversarially coherent for a routed MoE (bf16 rounding of the backward
    intermediates becomes a same-sign bias accumulating over the selected
    tokens in every weight gradient). A fixed +-uniform cotangent -- a pure
    function of the output shape -- keeps the gradient comparison at the
    operator's true numerical difficulty and stays bitwise across determinism
    runs for every party (candidate, reference, anchor).
    """
    g = torch.Generator(device=out.device)
    g.manual_seed((0x5EEDC0DE ^ (out.shape[0] * 31 + out.shape[1])) & 0x7FFFFFFF)
    v = (torch.rand(out.shape, generator=g, device=out.device,
                    dtype=torch.float32) * 2.0 - 1.0).to(out.dtype)
    return (out.float() * v.float()).sum()


# ---------------------------------------------------------------------------
# Selection helper (verifier-side twin of the reference's exact tie-break)
# ---------------------------------------------------------------------------

def _sel_topC(S: torch.Tensor, C: int) -> torch.Tensor:
    """Per expert (column of S), top-C tokens by affinity, ties to the LOWER
    token index, exactly. Returns [E, C] token indices."""
    T, E = S.shape
    St = S.t().contiguous()
    b = torch.where(St == 0.0, torch.zeros_like(St), St).contiguous()
    ib = b.view(torch.int32).to(torch.int64)
    u = torch.where(ib >= 0, ib ^ 0x80000000, (~ib) & 0xFFFFFFFF)
    t_idx = torch.arange(T, device=S.device, dtype=torch.int64)
    tbits = max((T - 1).bit_length(), 1)
    keys = (u << tbits) | (T - 1 - t_idx).unsqueeze(0)
    return torch.topk(keys, C, dim=-1, largest=True, sorted=True).indices


# ---------------------------------------------------------------------------
# Input draws
# ---------------------------------------------------------------------------

def _margin_fix(x32, xq, rw_q, C, scale, g, device):
    """Redraw the boundary tokens of any expert whose C/(C+1) affinity gap is
    positive but < _MARGIN. The gap is computed by fp64 shadow softmax over the
    DTYPE-CAST tensors (the values the op actually sees). Redrawing a token
    changes its affinity for every expert, so this iterates."""
    T = x32.shape[0]
    if C >= T:
        return xq
    rw64 = rw_q.double()
    for _ in range(64):
        S = torch.softmax(xq.double() @ rw64, dim=-1)          # [T, E]
        vals, idx = torch.topk(S, C + 1, dim=0)                # [C+1, E] over tokens
        gap = vals[C - 1, :] - vals[C, :]                      # [E]
        bad = (gap > 0) & (gap < _MARGIN)                      # [E]
        if not bool(bad.any()):
            break
        toks = torch.cat([idx[C - 1, bad], idx[C, bad]]).unique()
        n = int(toks.numel())
        fresh = torch.randn(n, x32.shape[1], generator=g,
                            device=device) * float(scale)
        x32[toks] = fresh
        xq[toks] = fresh.to(xq.dtype)
    return xq


def make_inputs(shape, seed, scale, dtype, device):
    """One reproducible draw; NaN-free by construction.

    Optional verifier-side shape key (never published):
      tie_pool: int G -> x is built from a pool of G distinct token rows
                (duplicated across the T tokens), so affinities carry EXACT ties
                in every dtype: a per-expert selection boundary that lands inside
                a duplicate group is split by the lower-token-index rule. The two
                tied tokens are different OUTPUT rows, so a wrong tie winner moves
                the contribution to the wrong row and changes y. Router and
                expert weights stay distinct.
    """
    T, D = int(shape["T"]), int(shape["D"])
    F, E, C = int(shape["F"]), int(shape["E"]), int(shape["C"])
    g = torch.Generator(device=device)
    g.manual_seed(seed & 0x7FFFFFFF)

    rw32 = torch.randn(D, E, generator=g, device=device) * D ** -0.5
    w1 = torch.randn(E, D, 2 * F, generator=g, device=device) * D ** -0.5
    w2 = torch.randn(E, F, D, generator=g, device=device) * F ** -0.5

    pool = int(shape.get("tie_pool", 0))
    if pool:
        base = torch.randn(pool, D, generator=g, device=device) * float(scale)
        assign = torch.randint(0, pool, (T,), generator=g, device=device)
        x32 = base.index_select(0, assign).contiguous()
        rw_q = rw32.to(dtype).contiguous()
        xq = x32.to(dtype).contiguous()
        # exact ties are intended: do NOT run the margin fix (it would break the
        # token duplication).
    else:
        x32 = torch.randn(T, D, generator=g, device=device) * float(scale)
        rw_q = rw32.to(dtype).contiguous()
        xq = x32.to(dtype).contiguous()
        xq = _margin_fix(x32, xq, rw_q, C, scale, g, device)

    args = {
        "x": xq.requires_grad_(True),
        "router_w": rw_q.requires_grad_(True),
        "w1": w1.to(dtype).contiguous().requires_grad_(True),
        "w2": w2.to(dtype).contiguous().requires_grad_(True),
        "C": C,
    }
    return args, ("x", "router_w", "w1", "w2")


# ---------------------------------------------------------------------------
# Anchor: the "moe_fused_experts" production composition (expert-choice)
# ---------------------------------------------------------------------------

def _anchor_route(x, router_w, C):
    S = torch.softmax(x.float() @ router_w.float(), dim=-1)    # [T, E]
    with torch.no_grad():
        sel = _sel_topC(S.detach(), C)                         # [E, C]
    gate = torch.gather(S.t(), 1, sel)                         # [E, C] affinity gate
    return sel, gate


def _sota_grouped(x, router_w, w1, w2, C):
    """torch._grouped_mm composition; the dispatch is already expert-major
    (exactly C tokens per expert), so offsets are the constant multiples of C."""
    T, D = x.shape
    E = router_w.shape[1]
    F = w1.shape[2] // 2
    sel, gate = _anchor_route(x, router_w, C)
    tok = sel.reshape(-1)                                      # [E*C] expert-major
    offs = (torch.arange(1, E + 1, device=x.device) * C).to(torch.int32)
    xg = x.index_select(0, tok)
    h = torch._grouped_mm(xg, w1, offs=offs)
    act = torch.nn.functional.silu(h[:, :F]) * h[:, F:]
    yp = torch._grouped_mm(act, w2, offs=offs)
    ws = gate.reshape(-1).to(x.dtype).unsqueeze(1)
    return x.new_zeros(T, D).index_add(0, tok, yp * ws)


def _sota_loop(x, router_w, w1, w2, C):
    """Expert-loop cuBLAS composition; each expert owns a contiguous C-row block."""
    T, D = x.shape
    E = router_w.shape[1]
    F = w1.shape[2] // 2
    sel, gate = _anchor_route(x, router_w, C)
    tok = sel.reshape(-1)
    xg = x.index_select(0, tok)
    ws = gate.reshape(-1).to(x.dtype).unsqueeze(1)
    y = x.new_zeros(T, D)
    for e in range(E):
        lo, hi = e * C, (e + 1) * C
        h = xg[lo:hi] @ w1[e]
        act = torch.nn.functional.silu(h[:, :F]) * h[:, F:]
        y = y.index_add(0, tok[lo:hi], (act @ w2[e]) * ws[lo:hi])
    return y


# PLACEHOLDER selection (calibrate per dtype on the graded shapes before
# freeze). The calibrated sibling measured grouped-mm faster on every row.
_SOTA_ROUTE = {"torch.bfloat16": "grouped_mm", "torch.float32": "grouped_mm"}


def sota(args):
    """Production anchor: the measured-faster composition per dtype."""
    x = args["x"]
    route = _SOTA_ROUTE[str(x.dtype)]
    fn = _sota_grouped if route == "grouped_mm" else _sota_loop
    return fn(x, args["router_w"], args["w1"], args["w2"], int(args["C"]))


# ---------------------------------------------------------------------------
# Stability modes (finiteness-and-not-raising gate; drift recorded only)
# ---------------------------------------------------------------------------

def _stab_common(T, D, F, E, g, device):
    rw = torch.randn(D, E, generator=g, device=device) * D ** -0.5
    w1 = torch.randn(E, D, 2 * F, generator=g, device=device) * D ** -0.5
    w2 = torch.randn(E, F, D, generator=g, device=device) * F ** -0.5
    return rw, w1, w2


def _finish(x32, rw32, w1, w2, C, dtype):
    return {
        "x": x32.to(dtype).contiguous().requires_grad_(True),
        "router_w": rw32.to(dtype).contiguous().requires_grad_(True),
        "w1": w1.to(dtype).contiguous().requires_grad_(True),
        "w2": w2.to(dtype).contiguous().requires_grad_(True),
        "C": C,
    }, ("x", "router_w", "w1", "w2")


def _mode_router_saturated(shape, seed, scale, dtype, device):
    """Logits ~N(0, 30^2): affinities pinned near one-hot per token."""
    T, D = int(shape["T"]), int(shape["D"])
    F, E, C = int(shape["F"]), int(shape["E"]), int(shape["C"])
    g = torch.Generator(device=device)
    g.manual_seed(seed & 0x7FFFFFFF)
    rw, w1, w2 = _stab_common(T, D, F, E, g, device)
    x = torch.randn(T, D, generator=g, device=device) * 30.0
    return _finish(x, rw, w1, w2, C, dtype)


def _mode_score_plateau(shape, seed, scale, dtype, device):
    """Rank-1 router: every token's logits take two values, so within an expert
    column many tokens share an affinity and boundaries fall in tie groups."""
    T, D = int(shape["T"]), int(shape["D"])
    F, E, C = int(shape["F"]), int(shape["E"]), int(shape["C"])
    g = torch.Generator(device=device)
    g.manual_seed(seed & 0x7FFFFFFF)
    _, w1, w2 = _stab_common(T, D, F, E, g, device)
    u = torch.randn(D, 1, generator=g, device=device) * D ** -0.5
    signs = torch.where(torch.rand(1, E, generator=g, device=device) < 0.5,
                        -1.0, 1.0)
    rw = (u @ signs).contiguous()
    x = torch.randn(T, D, generator=g, device=device) * float(scale)
    return _finish(x, rw, w1, w2, C, dtype)


def _mode_x_tiny(shape, seed, scale, dtype, device):
    """x ~ 1e-30: logits round to 0, affinities are a uniform 1/E plateau, and
    the expert arithmetic runs at the bottom of the exponent range."""
    T, D = int(shape["T"]), int(shape["D"])
    F, E, C = int(shape["F"]), int(shape["E"]), int(shape["C"])
    g = torch.Generator(device=device)
    g.manual_seed(seed & 0x7FFFFFFF)
    rw, w1, w2 = _stab_common(T, D, F, E, g, device)
    x = torch.randn(T, D, generator=g, device=device) * 1.0e-30
    return _finish(x, rw, w1, w2, C, dtype)


def _mode_x_huge(shape, seed, scale, dtype, device):
    """Six decades of magnitude in one tensor: saturated router plus expert
    activations pushed to ~1e8, still finite in fp32."""
    T, D = int(shape["T"]), int(shape["D"])
    F, E, C = int(shape["F"]), int(shape["E"]), int(shape["C"])
    g = torch.Generator(device=device)
    g.manual_seed(seed & 0x7FFFFFFF)
    rw, w1, w2 = _stab_common(T, D, F, E, g, device)
    mag = torch.rand(T, D, generator=g, device=device) * 6.0 - 3.0
    x = (torch.randn(T, D, generator=g, device=device)
         * torch.pow(10.0, mag) * float(scale))
    return _finish(x, rw, w1, w2, C, dtype)


def _mode_near_uniform(shape, seed, scale, dtype, device):
    """router_w scaled to ~1e-3: logits ~ 0, affinities ~ uniform 1/E, so every
    per-expert boundary sits in a near-plateau. Finiteness only."""
    T, D = int(shape["T"]), int(shape["D"])
    F, E, C = int(shape["F"]), int(shape["E"]), int(shape["C"])
    g = torch.Generator(device=device)
    g.manual_seed(seed & 0x7FFFFFFF)
    _, w1, w2 = _stab_common(T, D, F, E, g, device)
    rw = torch.randn(D, E, generator=g, device=device) * (D ** -0.5) * 1.0e-3
    x = torch.randn(T, D, generator=g, device=device) * float(scale)
    return _finish(x, rw, w1, w2, C, dtype)


STABILITY_MODES = {
    "router_saturated": _mode_router_saturated,
    "score_plateau": _mode_score_plateau,
    "x_tiny": _mode_x_tiny,
    "x_huge": _mode_x_huge,
    "near_uniform": _mode_near_uniform,
}

SMOKE_SHAPE = {"T": 1024, "D": 1024, "F": 704, "E": 32, "C": 32}
# Tie-forcing via token duplication (tie_pool): guaranteed exact affinity ties,
# so a scheduling-dependent tie winner routes a contribution to the wrong output
# row and cannot repeat bitwise across the three determinism runs.
DETERMINISM_SHAPE = {"T": 2048, "D": 1024, "F": 704, "E": 64, "C": 32,
                     "tie_pool": 8}
STABILITY_SHAPE = {"T": 1024, "D": 1024, "F": 704, "E": 32, "C": 32}


# ---------------------------------------------------------------------------
# bench.py published surface (DISJOINT from and smaller than the graded set).
# ---------------------------------------------------------------------------

BENCH_PUBLISHED_SHAPES = {
    "p1": {"T": 4096, "D": 1024, "F": 704, "E": 32, "C": 128,
           "dtypes": ["bfloat16"]},
    "p2": {"T": 2048, "D": 2048, "F": 1408, "E": 64, "C": 32,
           "dtypes": ["bfloat16"]},
}
BENCH_CHECK_SHAPE = {"T": 512, "D": 512, "F": 256, "E": 16, "C": 32}
BENCH_CHECK_SHAPE_NPOT = {"T": 301, "D": 200, "F": 96, "E": 11, "C": 20}
BENCH_CHECK_SHAPE_TIES = {"T": 256, "D": 256, "F": 128, "E": 16, "C": 16,
                          "tie_pool": 4}


# ---------------------------------------------------------------------------
# Negative controls (Invariant 17, both halves). AUTHORED DRAFT: these target
# THIS folder's kernel.py starter (base:"starter"); the starter is correct and
# G2-clean, so mutating it and checking the zero lands at exactly bound_gate
# proves each gate bites. Re-verify with controls_runner on GPU before freeze.
# ---------------------------------------------------------------------------

def _mut_dp(src: str) -> str:
    anchor = ("                g = S.index_select(0, rows)[:, e:e + 1]     "
              "# [C, 1] affinity gate\n")
    if anchor not in src:
        raise RuntimeError("INERT: affinity-gate anchor not found in starter")
    repl = ("                if float(S.detach().amax(dim=-1).min()) >= 0.999:\n"
            "                    g = S.index_select(0, rows)[:, e:e + 1]\n"
            "                else:\n"
            "                    g = torch.full_like(\n"
            "                        S.index_select(0, rows)[:, e:e + 1], 1.0 / E)\n")
    return src.replace(anchor, repl, 1)


# NC-FALLBACK body: correct, deterministic, zero Triton -- the reference math,
# standalone.
_FALLBACK_SRC = '''\
"""Pure-torch fallback candidate (negative control): no Triton anywhere."""
import torch


def _pick(S, C):
    T, E = S.shape
    St = S.t().contiguous()
    b = torch.where(St == 0.0, torch.zeros_like(St), St).contiguous()
    ib = b.view(torch.int32).to(torch.int64)
    u = torch.where(ib >= 0, ib ^ 0x80000000, (~ib) & 0xFFFFFFFF)
    t_idx = torch.arange(T, device=S.device, dtype=torch.int64)
    tbits = max((T - 1).bit_length(), 1)
    keys = (u << tbits) | (T - 1 - t_idx).unsqueeze(0)
    return torch.topk(keys, C, dim=-1, largest=True, sorted=True).indices


class _Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, router_w, w1, w2, C):
        C = int(C)
        with torch.enable_grad():
            xf = x.detach().float().requires_grad_(True)
            rwf = router_w.detach().float().requires_grad_(True)
            w1f = w1.detach().float().requires_grad_(True)
            w2f = w2.detach().float().requires_grad_(True)
            T, D = xf.shape
            E = rwf.shape[1]
            F = w1f.shape[2] // 2
            S = torch.softmax(xf @ rwf, dim=-1)
            with torch.no_grad():
                sel = _pick(S.detach(), C)
            y = xf.new_zeros(T, D)
            for e in range(E):
                rows = sel[e]
                g = S.index_select(0, rows)[:, e:e + 1]
                h = xf.index_select(0, rows)
                gu = h @ w1f[e]
                fe = (torch.nn.functional.silu(gu[:, :F]) * gu[:, F:]) @ w2f[e]
                y = y.index_add(0, rows, g * fe)
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


def fused_moe(x, router_w, w1, w2, C):
    return _Fn.apply(x, router_w, w1, w2, C)
'''


NEGATIVE_CONTROLS = [
    {
        "name": "nc_forbidden_call",
        "base": "starter",
        "target_file": "kernel.py",
        "mutate": ("import triton\n",
                   "import triton\n_SHORTCUT = torch._grouped_mm\n"),
        "bound_gate": "G1",
        "oracle_env": False,
        "expectation": "the forbidden-symbol scan names torch._grouped_mm and "
                       "zeroes at G1 before the candidate is ever imported",
    },
    {
        "name": "nc_gate_dropped",
        "base": "starter",
        "target_file": "kernel.py",
        "mutate": ("g = S.index_select(0, rows)[:, e:e + 1]",
                   "g = torch.ones_like(S.index_select(0, rows)[:, e:e + 1])"),
        "bound_gate": "G2",
        "oracle_env": False,
        "expectation": "the affinity gate is replaced by a constant 1: the "
                       "output loses the S[t,e] scale on every contribution, so "
                       "the dense G2 comparison fails at the smoke stage",
    },
    {
        "name": "nc_tiebreak_reversed",
        "base": "starter",
        "target_file": "kernel.py",
        "mutate": ("keys = (u << tbits) | (T - 1 - t_idx).unsqueeze(0)",
                   "keys = (u << tbits) | t_idx.unsqueeze(0)"),
        "bound_gate": "G2",
        "oracle_env": False,
        "expectation": "per-expert ties broken toward the HIGHER token index: "
                       "identical on tie-free draws, but on the tie-forcing "
                       "hidden shape a boundary inside a duplicate group routes "
                       "the contribution to the wrong output row -> G2 edge-case "
                       "failure in both dtypes",
    },
    {
        "name": "nc_differential_path",
        "base": "starter",
        "target_file": "kernel.py",
        "mutate": _mut_dp,
        "bound_gate": "G2",
        "oracle_env": True,
        "expectation": "the regime-switch exploit: keep the exact affinity gate "
                       "only where the softmax is saturated (a stability probe), "
                       "and take a cheap uniform 1/E gate on nominal inputs. "
                       "Because timed tensors ARE the checked tensors, the cheap "
                       "path runs on the graded draws and G2 fails at smoke even "
                       "with FORGE_ORACLE=1",
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
