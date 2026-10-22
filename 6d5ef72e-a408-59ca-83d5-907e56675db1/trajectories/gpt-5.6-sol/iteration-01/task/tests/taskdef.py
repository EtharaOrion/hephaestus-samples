"""Taskdef: routed fused mixture-of-experts layer (SURFACE fwdbwd).

Category of the '01_glm52_fused_moe' frontier-gap family (opus-5 reached
0.0788 of roofline on this operator family on H100;
seed/frontier-defeat-analysis.md 4.2/5.1). Distinct from moe_swiglu: the
routing decision -- scores, selection, combine weights -- is INSIDE the
graded surface, so a candidate must fuse an irregular data-dependent
gather/scatter with the grouped expert compute, and gradients flow through
the router. Everything category-specific the generic verifier consumes lives
here (protocol: core/taskdef_api.py).

Operator (reference.py is the definition; summary):
    fused_moe(x[T,D], router_w[D,E], router_b[E] fp32, w1[E,D,2F],
              w2[E,F,D], A) -> y[T,D]
    s = sigmoid(x @ router_w); select top-A by (s + router_b), ties to the
    LOWER expert index; combine weights normalize the RAW selected scores
    (the bias picks experts, never weights them); SwiGLU experts; fp32
    arithmetic, one final cast. Gradients graded: dx, drouter_w, dw1, dw2.
    router_b carries no grad (selection is piecewise-constant in it).

Selection-stability design (the load-bearing part): expert selection is
discontinuous, so a draw whose A-th/(A+1)-th biased scores are separated by
less than fp32 GEMM noise would make G2 a coin flip for every honest
candidate. make_inputs therefore enforces a decision margin: rows whose
boundary gap is positive but below 1e-4 are redrawn (fp64 shadow routing).
Exact ties (gap == 0) are legal and deliberately manufactured by the
tie-forcing draw below -- duplicated router columns and quantized bias
levels make biased scores bit-equal in BOTH dtypes, so the lower-index rule
is exercised for real (arbitrary-order GEMMs map equal columns to equal
outputs). The margin also fixes the required router precision: fp32 score
error ~1e-6 sits far inside 1e-4; TF32/bf16 score error ~1e-3 does not, and
flips selections. This is disclosed in instruction.md.

Distribution invariance (frontier-defeat-analysis 7.1): the hidden scale
multiplies x per draw, router_b is REDRAWN per invocation from a hidden
zipf-flavoured popularity prior (so expert popularity is imbalanced but
never fixed -- a kernel specialized to one popularity layout meets a
different one on every draw), and the tensors that are timed are the
tensors that are checked. NC 'nc_differential_path' proves the
regime-switch exploit (5.4: exact math only on detected stress inputs)
zeroes at G2.

Anchor policy: the faster of a torch._grouped_mm composition and an
expert-loop cuBLAS composition over tokens sorted by expert, plus the
router matmul and integer-key top-A in torch -- measured per dtype on the
graded shapes, provenance recorded in _SOTA_ROUTE below and in
runs/moe_routed/measurement.json.
"""

import torch

import taskdef_api  # noqa: F401  (protocol module; defaults resolved via cfg)

ENTRY_NAME = "fused_moe"
REF_NAME = "fused_moe_ref"
SIGNATURE = ("fused_moe(x[T,D], router_w[D,E], router_b[E] fp32, "
             "w1[E,D,2F], w2[E,F,D], A) -> y[T,D]")
SURFACE = "fwdbwd"
TASK_NAME = "hephaestus/moe_routed_fwdbwd"

# MEASURED 2026-08-15 on the H100, calibrated on the private oracle AND the
# starter vs the frozen reference across every hidden shape/dtype with the
# hidden-scale draws live. Two seams, per the protocol:
#   TOL (this dict)  gates ALL GRADIENT comparisons. Worst honest knee
#                    1.61e-3 (fp32, drouter_w: the normalization backward
#                    divides a cancelling difference by the score sum, so
#                    even fp32-exact pipelines amplify GEMM-order noise) and
#                    8.3e-3 (bf16, weight grads with an fp32-precision
#                    operand chain -- a bf16-rounded operand chain measures
#                    ~1.1 and is unpassable by design). Set ~2.5x the knee.
#   compare() below  gates the OUTPUT with tighter per-dtype budgets:
#                    fp32 knee 1.31e-5 -> (2e-4, 2e-4), which forbids
#                    TF32-precision expert GEMMs (~1e-3) outright; bf16 knee
#                    2.07e-2 (bf16 GEMM operands + output cast) -> (4.5e-2,
#                    4.5e-2).
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

# The name half of G1: grouped-GEMM libraries and the MoE serving stacks
# that ship fused-MoE kernels. Word-boundary scan over comment-stripped
# candidate source (strings and docstrings count). The package names cover
# the import vectors (`from vllm... import fused_moe` cannot be written
# without naming vllm), so the candidate's own mandatory `def fused_moe`
# never false-positives. torch.topk / torch.sort on the small [T,E] routing
# metadata are deliberately permitted (disclosed in instruction.md); G4
# still requires the heavy arithmetic in the candidate's own kernels.
FORBIDDEN = [
    "torch._grouped_mm", "_grouped_mm",
    "vllm", "sglang", "grouped_gemm", "transformer_engine", "flashinfer",
]

# No vendorable python source tree for the anchor (torch builtin + cuBLAS):
# shingle scan is skipped (recorded available:false, never gated). The
# FORBIDDEN symbol scan still runs.
VENDOR_LIB_PACKAGE = ""
VENDOR_LIB_SUBDIRS = ()

ADOPTION_FLOOR = 0.60

# MEASURED 2026-08-15: three golden runs of the private oracle through the
# real bundle path (fractions in spec.yaml notes); the target sits below the
# repeated worst case because the min(r,1) clip turns symmetric timing noise
# into downward bias, the GPU is co-tenanted, and the oracle must reach
# outcome 1.0 on its own control arm. Never raise this above what the oracle
# repeatably achieves.
TARGET_FRACTION_OF_SOTA = 0.34

_MARGIN = 1.0e-4          # decision-margin floor for nominal draws (fp32 units)
_IDXB = 10                # index bits in selection keys (E <= 1024)


# ---------------------------------------------------------------------------
# Backward cotangent (overrides taskdef_api.default_loss)
# ---------------------------------------------------------------------------

def loss(out):
    """Fixed pseudorandom cotangent, deterministic per output shape.

    The protocol default backpropagates an all-ones cotangent, which is
    adversarially COHERENT for this operator: every (token, slot) pair's
    bf16 rounding of the backward intermediates becomes a same-sign bias
    that accumulates linearly over T in each weight gradient, so even a
    production-grade bf16 kernel lands orders of magnitude outside any
    sane tolerance while no real training step (noise-like dy) behaves
    that way. A fixed +-uniform cotangent keeps the gradient comparison at
    the operator's true numerical difficulty. The draw is a pure function
    of the output shape, so determinism runs stay bitwise and every party
    (candidate, reference, anchor) receives the identical cotangent.
    """
    g = torch.Generator(device=out.device)
    g.manual_seed((0x5EEDC0DE ^ (out.shape[0] * 31 + out.shape[1])) & 0x7FFFFFFF)
    v = (torch.rand(out.shape, generator=g, device=out.device,
                    dtype=torch.float32) * 2.0 - 1.0).to(out.dtype)
    return (out.float() * v.float()).sum()


# ---------------------------------------------------------------------------
# Selection helper (verifier-side twin of the reference's exact tie-break)
# ---------------------------------------------------------------------------

def _sel_topA(biased: torch.Tensor, A: int) -> torch.Tensor:
    """Top-A of biased fp32 scores by (score desc, index asc), exactly."""
    E = biased.shape[-1]
    b = torch.where(biased == 0.0, torch.zeros_like(biased), biased).contiguous()
    ib = b.view(torch.int32).to(torch.int64)
    u = torch.where(ib >= 0, ib ^ 0x80000000, (~ib) & 0xFFFFFFFF)
    e_idx = torch.arange(E, device=biased.device, dtype=torch.int64)
    keys = (u << _IDXB) | (E - 1 - e_idx)
    return torch.topk(keys, A, dim=-1, largest=True, sorted=True).indices


# ---------------------------------------------------------------------------
# Input draws
# ---------------------------------------------------------------------------

def _zipf_bias(E: int, g, device) -> torch.Tensor:
    """Hidden popularity prior, redrawn per invocation.

    Popularity rank is a fresh random permutation, popularity follows
    (rank+1)^-alpha with hidden alpha, and the bias is the centered
    log-popularity under a hidden temperature -- so which experts are hot,
    how hot, and how sharply popularity decays all change every draw."""
    alpha = 0.6 + 0.8 * float(torch.rand((), generator=g, device=device))
    tau = 0.25 + 0.65 * float(torch.rand((), generator=g, device=device))
    ranks = torch.empty(E, dtype=torch.float32, device=device)
    ranks[torch.randperm(E, generator=g, device=device)] = torch.arange(
        E, dtype=torch.float32, device=device)
    logpop = -alpha * torch.log1p(ranks)
    return (tau * (logpop - logpop.mean())).float().contiguous()


def _margin_fix(x32, xq, rw_q, rb, A, scale, g, device):
    """Redraw x rows whose boundary gap is positive but below _MARGIN.

    The gap is computed by fp64 shadow routing over the DTYPE-CAST tensors
    (the values the op actually sees), so the enforced margin is the margin
    the graded fp32 computation experiences. Exact ties (gap == 0, the
    tie-forcing draw) are legal and kept: equal inputs stay equal under any
    per-column-uniform GEMM, so they are decided by the tie-break rule, not
    by numerics."""
    E = rw_q.shape[1]
    if A >= E:
        return xq
    rw64 = rw_q.double()
    rb64 = rb.double()
    logits = xq.double() @ rw64
    for _ in range(64):
        biased = torch.sigmoid(logits) + rb64
        v = torch.topk(biased, A + 1, dim=-1).values
        gap = v[:, A - 1] - v[:, A]
        bad = (gap > 0) & (gap < _MARGIN)
        n = int(bad.sum())
        if n == 0:
            break
        fresh = torch.randn(n, x32.shape[1], generator=g,
                            device=device) * float(scale)
        x32[bad] = fresh
        xq[bad] = fresh.to(xq.dtype)
        logits[bad] = xq[bad].double() @ rw64
    return xq


def make_inputs(shape, seed, scale, dtype, device):
    """One reproducible draw; NaN-free by construction.

    Optional shape key (verifier-side only, never published):
      tie_pool: int G -> router_w columns are drawn from a pool of G distinct
                columns (duplicated across experts) and router_b from two
                quantized levels, so biased scores carry massive EXACT ties
                in every dtype and the boundary tie-break is exercised for
                real. Expert weights stay distinct, so a wrong tie winner
                changes the output.
    """
    T, D = int(shape["T"]), int(shape["D"])
    F, E, A = int(shape["F"]), int(shape["E"]), int(shape["A"])
    g = torch.Generator(device=device)
    g.manual_seed(seed & 0x7FFFFFFF)

    pool = int(shape.get("tie_pool", 0))
    if pool:
        cols = torch.randn(D, pool, generator=g, device=device) * D ** -0.5
        assign = torch.randint(0, pool, (E,), generator=g, device=device)
        rw32 = cols.index_select(1, assign).contiguous()
        rb = (torch.randint(0, 2, (E,), generator=g, device=device)
              .float() * 0.25).contiguous()
    else:
        rw32 = torch.randn(D, E, generator=g, device=device) * D ** -0.5
        rb = _zipf_bias(E, g, device)
    w1 = torch.randn(E, D, 2 * F, generator=g, device=device) * D ** -0.5
    w2 = torch.randn(E, F, D, generator=g, device=device) * F ** -0.5
    x32 = torch.randn(T, D, generator=g, device=device) * float(scale)

    rw_q = rw32.to(dtype).contiguous()
    xq = x32.to(dtype).contiguous()
    xq = _margin_fix(x32, xq, rw_q, rb, A, scale, g, device)

    args = {
        "x": xq.requires_grad_(True),
        "router_w": rw_q.requires_grad_(True),
        "router_b": rb,
        "w1": w1.to(dtype).contiguous().requires_grad_(True),
        "w2": w2.to(dtype).contiguous().requires_grad_(True),
        "A": A,
    }
    return args, ("x", "router_w", "w1", "w2")


# ---------------------------------------------------------------------------
# Anchor: measured-faster of grouped-mm composition vs expert-loop cuBLAS
# ---------------------------------------------------------------------------

def _anchor_route(x, router_w, router_b, A):
    s = torch.sigmoid(x.float() @ router_w.float())
    with torch.no_grad():
        sel = _sel_topA(s.detach() + router_b, A)
    s_sel = torch.gather(s, 1, sel)
    w = s_sel / s_sel.sum(dim=-1, keepdim=True)
    return sel, w


def _sota_grouped(x, router_w, router_b, w1, w2, A):
    """torch._grouped_mm composition over tokens sorted by expert."""
    T, D = x.shape
    E = router_w.shape[1]
    F = w1.shape[2] // 2
    sel, w = _anchor_route(x, router_w, router_b, A)
    flat = sel.reshape(-1)
    order = torch.argsort(flat)
    tok = torch.div(order, A, rounding_mode="floor")
    offs = torch.bincount(flat, minlength=E).cumsum(0).to(torch.int32)
    xg = x.index_select(0, tok)
    h = torch._grouped_mm(xg, w1, offs=offs)
    act = torch.nn.functional.silu(h[:, :F]) * h[:, F:]
    yp = torch._grouped_mm(act, w2, offs=offs)
    ws = w.reshape(-1).index_select(0, order).to(x.dtype).unsqueeze(1)
    return x.new_zeros(T, D).index_add(0, tok, yp * ws)


def _sota_loop(x, router_w, router_b, w1, w2, A):
    """Expert-loop cuBLAS composition over tokens sorted by expert."""
    T, D = x.shape
    E = router_w.shape[1]
    F = w1.shape[2] // 2
    sel, w = _anchor_route(x, router_w, router_b, A)
    flat = sel.reshape(-1)
    order = torch.argsort(flat)
    tok = torch.div(order, A, rounding_mode="floor")
    bounds = torch.bincount(flat, minlength=E).cumsum(0).tolist()
    xg = x.index_select(0, tok)
    ws = w.reshape(-1).index_select(0, order).to(x.dtype).unsqueeze(1)
    y = x.new_zeros(T, D)
    lo = 0
    for e in range(E):
        hi = int(bounds[e])
        if hi > lo:
            h = xg[lo:hi] @ w1[e]
            act = torch.nn.functional.silu(h[:, :F]) * h[:, F:]
            y = y.index_add(0, tok[lo:hi], (act @ w2[e]) * ws[lo:hi])
        lo = hi
    return y


# MEASURED 2026-08-15 (H100, locked clocks, paired fwdbwd methodology, all
# six graded shape/dtype rows, GPU shared with a co-tenant): the grouped-mm
# composition beat the expert-loop composition on every row -- bf16 by
# 1.9x/20.0x/3.0x (g3/g2/g1), fp32 by 1.2x/4.1x/3.3x. Provenance recorded
# per shape in runs/moe_routed/measurement.json.
_SOTA_ROUTE = {"torch.bfloat16": "grouped_mm", "torch.float32": "grouped_mm"}


def sota(args):
    """Production anchor: the measured-faster composition per dtype."""
    x = args["x"]
    route = _SOTA_ROUTE[str(x.dtype)]
    fn = _sota_grouped if route == "grouped_mm" else _sota_loop
    return fn(x, args["router_w"], args["router_b"],
              args["w1"], args["w2"], int(args["A"]))


# ---------------------------------------------------------------------------
# Stability modes (finiteness-and-not-raising gate; drift recorded only)
# ---------------------------------------------------------------------------

def _stab_common(T, D, F, E, g, device):
    rw = torch.randn(D, E, generator=g, device=device) * D ** -0.5
    w1 = torch.randn(E, D, 2 * F, generator=g, device=device) * D ** -0.5
    w2 = torch.randn(E, F, D, generator=g, device=device) * F ** -0.5
    return rw, w1, w2


def _finish(x32, rw32, rb, w1, w2, A, dtype):
    return {
        "x": x32.to(dtype).contiguous().requires_grad_(True),
        "router_w": rw32.to(dtype).contiguous().requires_grad_(True),
        "router_b": rb.float().contiguous(),
        "w1": w1.to(dtype).contiguous().requires_grad_(True),
        "w2": w2.to(dtype).contiguous().requires_grad_(True),
        "A": A,
    }, ("x", "router_w", "w1", "w2")


def _mode_router_saturated(shape, seed, scale, dtype, device):
    """Logits ~N(0, 30^2): sigmoid pinned to exact 0.0/1.0 over most of the
    row. Selection keeps at least one saturated-high expert per token, so
    the raw-score normalizer stays positive and finite."""
    T, D = int(shape["T"]), int(shape["D"])
    F, E, A = int(shape["F"]), int(shape["E"]), int(shape["A"])
    g = torch.Generator(device=device)
    g.manual_seed(seed & 0x7FFFFFFF)
    rw, w1, w2 = _stab_common(T, D, F, E, g, device)
    x = torch.randn(T, D, generator=g, device=device) * 30.0
    rb = torch.zeros(E, device=device)
    return _finish(x, rw, rb, w1, w2, A, dtype)


def _mode_score_plateau(shape, seed, scale, dtype, device):
    """Rank-1 router: every row's scores take two values, so almost every
    expert is in an exact tie group and the boundary falls inside one.
    Gates finiteness only; the tie-break itself is graded by the
    tie-forcing hidden shape."""
    T, D = int(shape["T"]), int(shape["D"])
    F, E, A = int(shape["F"]), int(shape["E"]), int(shape["A"])
    g = torch.Generator(device=device)
    g.manual_seed(seed & 0x7FFFFFFF)
    _, w1, w2 = _stab_common(T, D, F, E, g, device)
    u = torch.randn(D, 1, generator=g, device=device) * D ** -0.5
    signs = torch.where(torch.rand(1, E, generator=g, device=device) < 0.5,
                        -1.0, 1.0)
    rw = (u @ signs).contiguous()
    x = torch.randn(T, D, generator=g, device=device) * float(scale)
    rb = torch.zeros(E, device=device)
    return _finish(x, rw, rb, w1, w2, A, dtype)


def _mode_x_tiny(shape, seed, scale, dtype, device):
    """x ~ 1e-30: every score rounds to exactly 0.5 -- a full plateau -- and
    the expert arithmetic runs at the bottom of the exponent range."""
    T, D = int(shape["T"]), int(shape["D"])
    F, E, A = int(shape["F"]), int(shape["E"]), int(shape["A"])
    g = torch.Generator(device=device)
    g.manual_seed(seed & 0x7FFFFFFF)
    rw, w1, w2 = _stab_common(T, D, F, E, g, device)
    x = torch.randn(T, D, generator=g, device=device) * 1.0e-30
    rb = torch.zeros(E, device=device)
    return _finish(x, rw, rb, w1, w2, A, dtype)


def _mode_x_huge(shape, seed, scale, dtype, device):
    """Six decades of magnitude in one tensor: saturated router plus expert
    activations pushed to ~1e8, still finite in fp32."""
    T, D = int(shape["T"]), int(shape["D"])
    F, E, A = int(shape["F"]), int(shape["E"]), int(shape["A"])
    g = torch.Generator(device=device)
    g.manual_seed(seed & 0x7FFFFFFF)
    rw, w1, w2 = _stab_common(T, D, F, E, g, device)
    mag = torch.rand(T, D, generator=g, device=device) * 6.0 - 3.0
    x = (torch.randn(T, D, generator=g, device=device)
         * torch.pow(10.0, mag) * float(scale))
    rb = torch.zeros(E, device=device)
    return _finish(x, rw, rb, w1, w2, A, dtype)


def _mode_bias_extreme(shape, seed, scale, dtype, device):
    """router_b at +/-20: selection is bias-dominated while the combine
    weights must keep normalizing nominal raw scores."""
    T, D = int(shape["T"]), int(shape["D"])
    F, E, A = int(shape["F"]), int(shape["E"]), int(shape["A"])
    g = torch.Generator(device=device)
    g.manual_seed(seed & 0x7FFFFFFF)
    rw, w1, w2 = _stab_common(T, D, F, E, g, device)
    x = torch.randn(T, D, generator=g, device=device) * float(scale)
    rb = torch.where(torch.rand(E, generator=g, device=device) < 0.5,
                     -20.0, 20.0)
    return _finish(x, rw, rb, w1, w2, A, dtype)


STABILITY_MODES = {
    "router_saturated": _mode_router_saturated,
    "score_plateau": _mode_score_plateau,
    "x_tiny": _mode_x_tiny,
    "x_huge": _mode_x_huge,
    "bias_extreme": _mode_bias_extreme,
}

SMOKE_SHAPE = {"T": 1024, "D": 1024, "F": 704, "E": 32, "A": 4}
# Tie-forcing on purpose, with A=4: guaranteed exact score ties, so a
# scheduling-dependent tie winner or a racy float combine cannot repeat
# bitwise across the three runs.
DETERMINISM_SHAPE = {"T": 2048, "D": 1024, "F": 704, "E": 64, "A": 4,
                     "tie_pool": 8}
STABILITY_SHAPE = {"T": 1024, "D": 1024, "F": 704, "E": 32, "A": 4}


# ---------------------------------------------------------------------------
# bench.py published surface (source of truth; bench.py mirrors these by
# hand because it ships standalone in /workspace). DISJOINT from the graded
# set and smaller.
# ---------------------------------------------------------------------------

BENCH_PUBLISHED_SHAPES = {
    "p1": {"T": 4096, "D": 1024, "F": 704, "E": 32, "A": 4,
           "dtypes": ["bfloat16"]},
    "p2": {"T": 2048, "D": 2048, "F": 1408, "E": 64, "A": 8,
           "dtypes": ["bfloat16"]},
}
BENCH_CHECK_SHAPE = {"T": 512, "D": 512, "F": 256, "E": 16, "A": 4}
BENCH_CHECK_SHAPE_NPOT = {"T": 301, "D": 200, "F": 96, "E": 11, "A": 3}
BENCH_CHECK_SHAPE_TIES = {"T": 256, "D": 256, "F": 128, "E": 16, "A": 4,
                          "tie_pool": 4}


# ---------------------------------------------------------------------------
# Negative controls (Invariant 17, both halves). The from-scratch oracle
# passes G1/G4 honestly, so most arms run with the exemption env OFF;
# nc_differential_path runs WITH it (the NCDP pattern: even an
# oracle-privileged run must be zeroed by G2 when it plants a regime
# switch).
# ---------------------------------------------------------------------------

def _mut_tiebreak(src: str) -> str:
    a1 = "    key = (u << 10) | (E - 1 - e).to(tl.int64)\n"
    a2 = "        col = E - 1 - (mx & 1023).to(tl.int32)\n"
    if a1 not in src or a2 not in src:
        raise RuntimeError("INERT: tie-break anchors not found in base source")
    src = src.replace(a1, "    key = (u << 10) | e.to(tl.int64)\n", 1)
    return src.replace(a2, "        col = (mx & 1023).to(tl.int32)\n", 1)


_ATOMIC_ANCHOR = """\
        y = torch.empty(T, D, dtype=cdt, device=dev)
        _combine_fwd_kernel[(T, triton.cdiv(D, BD))](
            yp, w, pos, y, D, A, BLOCK=BD)
"""

_ATOMIC_REPLACEMENT = """\
        y32 = torch.zeros(T, D, dtype=torch.float32, device=dev)
        _combine_atomic_kernel[(TA, triton.cdiv(D, BD))](
            yp, ws, toks, y32, D, BLOCK=BD)
        y = y32.to(cdt)
"""

_ATOMIC_KERNEL = """

@triton.jit
def _combine_atomic_kernel(YP, WS, TOKS, Y, D, BLOCK: tl.constexpr):
    q = tl.program_id(0).to(tl.int64)
    pd = tl.program_id(1).to(tl.int64)
    d = pd * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    dm = d < D
    t = tl.load(TOKS + q).to(tl.int64)
    wv = tl.load(WS + q)
    row = tl.load(YP + q * D + d, mask=dm, other=0.0)
    tl.atomic_add(Y + t * D + d, wv * row, mask=dm)
"""


def _mut_atomic(src: str) -> str:
    if _ATOMIC_ANCHOR not in src:
        raise RuntimeError("INERT: ordered-combine anchor not found in base source")
    return src.replace(_ATOMIC_ANCHOR, _ATOMIC_REPLACEMENT, 1) + _ATOMIC_KERNEL


_DP_ANCHOR = """\
        _select_kernel[(T,)](logits, rb, sel, ssel, w, ssum, E,
                             A=A, BE=max(triton.next_power_of_2(E), 2))
"""

_DP_REPLACEMENT = """\
        _sm = torch.sigmoid(logits)
        if float((_sm - 0.5).abs().max()) >= 0.4999999:
            _select_kernel[(T,)](logits, rb, sel, ssel, w, ssum, E,
                                 A=A, BE=max(triton.next_power_of_2(E), 2))
        else:
            _sel64 = torch.topk(_sm + rb, A, dim=-1).indices
            sel.copy_(_sel64.to(torch.int32))
            ssel.copy_(torch.gather(_sm, 1, _sel64))
            w.fill_(1.0 / A)
            ssum.copy_(ssel.sum(dim=-1))
"""


def _mut_dp(src: str) -> str:
    if _DP_ANCHOR not in src:
        raise RuntimeError("INERT: selection-launch anchor not found in base source")
    return src.replace(_DP_ANCHOR, _DP_REPLACEMENT, 1)


# NC-FALLBACK body: correct, deterministic, zero Triton -- the same math as
# the reference, standalone.
_FALLBACK_SRC = '''\
"""Pure-torch fallback candidate (negative control): no Triton anywhere."""
import torch


def _pick(biased, A):
    E = biased.shape[-1]
    b = torch.where(biased == 0.0, torch.zeros_like(biased), biased).contiguous()
    ib = b.view(torch.int32).to(torch.int64)
    u = torch.where(ib >= 0, ib ^ 0x80000000, (~ib) & 0xFFFFFFFF)
    e_idx = torch.arange(E, device=biased.device, dtype=torch.int64)
    keys = (u << 10) | (E - 1 - e_idx)
    return torch.topk(keys, A, dim=-1, largest=True, sorted=True).indices


class _Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, router_w, router_b, w1, w2, A):
        A = int(A)
        with torch.enable_grad():
            xf = x.detach().float().requires_grad_(True)
            rwf = router_w.detach().float().requires_grad_(True)
            w1f = w1.detach().float().requires_grad_(True)
            w2f = w2.detach().float().requires_grad_(True)
            T, D = xf.shape
            E = rwf.shape[1]
            F = w1f.shape[2] // 2
            s = torch.sigmoid(xf @ rwf)
            with torch.no_grad():
                sel = _pick(s.detach() + router_b.float(), A)
            s_sel = torch.gather(s, 1, sel)
            w = s_sel / s_sel.sum(dim=-1, keepdim=True)
            w_full = s.new_zeros(T, E).scatter(1, sel, w)
            member = torch.zeros(T, E, dtype=torch.bool,
                                 device=x.device).scatter(1, sel, True)
            y = xf.new_zeros(T, D)
            for e in range(E):
                rows = member[:, e].nonzero(as_tuple=True)[0]
                if rows.numel() == 0:
                    continue
                h = xf.index_select(0, rows)
                gu = h @ w1f[e]
                fe = (torch.nn.functional.silu(gu[:, :F]) * gu[:, F:]) @ w2f[e]
                y = y.index_add(0, rows,
                                w_full.index_select(0, rows)[:, e:e + 1] * fe)
        ctx.save_for_backward(xf, rwf, w1f, w2f, y)
        ctx.dtypes = (x.dtype, router_w.dtype, w1.dtype, w2.dtype)
        return y.detach().to(x.dtype)

    @staticmethod
    def backward(ctx, dy):
        xf, rwf, w1f, w2f, y = ctx.saved_tensors
        gx, grw, gw1, gw2 = torch.autograd.grad(y, (xf, rwf, w1f, w2f),
                                                dy.float())
        dx_t, drw_t, dw1_t, dw2_t = ctx.dtypes
        return (gx.to(dx_t), grw.to(drw_t), None,
                gw1.to(dw1_t), gw2.to(dw2_t), None)


def fused_moe(x, router_w, router_b, w1, w2, A):
    return _Fn.apply(x, router_w, router_b, w1, w2, A)
'''


NEGATIVE_CONTROLS = [
    {
        "name": "nc_forbidden_call",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": ("import triton\n",
                   "import triton\n_SHORTCUT = torch._grouped_mm\n"),
        "bound_gate": "G1",
        "oracle_env": False,
        "expectation": "the forbidden-symbol scan names torch._grouped_mm and "
                       "zeroes at G1 before the candidate is ever imported",
    },
    {
        "name": "nc_unnormalized_combine",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": ("        wv = sv / ssum\n", "        wv = sv\n"),
        "bound_gate": "G2",
        "oracle_env": False,
        "expectation": "raw selected scores used as combine weights without "
                       "normalization: outputs scaled by the score sum, dense "
                       "G2 comparison fails at the smoke stage",
    },
    {
        "name": "nc_bias_in_combine",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": ("        sv = tl.sum(tl.where(hit, s, 0.0), axis=0)\n",
                   "        sv = tl.sum(tl.where(hit, b, 0.0), axis=0)\n"),
        "bound_gate": "G2",
        "oracle_env": False,
        "expectation": "the selection bias leaks into the combine weights "
                       "(biased scores normalized instead of raw scores); the "
                       "hidden bias prior is nonzero on every nominal draw, so "
                       "the dense G2 comparison fails at the smoke stage",
    },
    {
        "name": "nc_tiebreak_reversed",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": _mut_tiebreak,
        "bound_gate": "G2",
        "oracle_env": False,
        "expectation": "ties broken toward the HIGHER expert index: identical "
                       "on tie-free draws, wrong expert sets on the tie-forcing "
                       "hidden shape in both dtypes -> G2 edge-case failure",
    },
    {
        "name": "nc_atomic_combine",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": _mut_atomic,
        "bound_gate": "G2",
        "oracle_env": False,
        "expectation": "float atomic-add combine: values stay inside tolerance "
                       "(smoke passes) but the summation order races, so the "
                       "three-run bitwise determinism stage fails at G2",
    },
    {
        "name": "nc_differential_path",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": _mut_dp,
        "bound_gate": "G2",
        "oracle_env": True,
        "expectation": "the 5.4 exploit shape: detect the saturated-sigmoid "
                       "stability regime and keep exact routing only there, "
                       "with a cheap wrong selection (library tie-break, "
                       "uniform combine weights) on nominal inputs. Because "
                       "timed tensors ARE the checked tensors, the cheap path "
                       "runs on the graded draws and G2 fails at smoke even "
                       "with oracle exemptions in force",
    },
    {
        "name": "nc_torch_fallback",
        "base": "oracle",
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
