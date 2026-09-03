"""Taskdef: fused MoE + ReLU^2-gated GLU expert compute, forward + backward.

HARD member of the moe_fused_experts family (precomputed integer routing). The
hardness lever is a NONSTANDARD ACTIVATION EPILOGUE: squared-ReLU (ReLU^2, the
Primer/Squared-ReLU gate). No fused-MoE library exposes this activation as a
primitive -- vLLM / SGLang / Transformer Engine / fused_moe / grouped_gemm all
ship silu- and gelu-gated variants but not ReLU^2 -- and torch has no builtin
that composes it with a grouped GEMM epilogue. A candidate must implement the
piecewise mask-square-and-multiply epilogue itself and fuse it into the MMA
epilogue of the first expert GEMM, otherwise the two extra global-memory
round-trips (write gu, read gu, write act, read act) dominate the routed path.
The backward is likewise piecewise: d/dg[relu(g)^2] = 2*relu(g)*(g>0), and a
candidate that smooths it with an epsilon or with silu-shaped math drifts out
of tolerance on the hidden sweep.

STATUS: authored-draft. TOL/OUT_TOL/TARGET_FRACTION_OF_SOTA below are AUTHORED
PLACEHOLDERS carried from the proven moe_swiglu category, WIDENED for the
piecewise activation (its backward has zero measure at g=0 but a much larger
Lipschitz constant on the g>0 half than silu) and MUST be re-measured on the
delivery GPU.

Operator (reference.py is the definition; this is a summary):
    relu2_glu_moe(x[T,D], w1[E,D,2F], w2[E,F,D], topk_idx[T,A] i64, topk_w[T,A])
        -> y[T,D]
    y[t] = sum_a topk_w[t,a] * f(topk_idx[t,a], x[t]),  a ascending;
    f(e,h) = ((relu(h @ w1[e,:,:F]))^2 * (h @ w1[e,:,F:])) @ w2[e].
    SURFACE fwdbwd: gradients of x, w1, w2 AND topk_w are graded.

Anchor: a cuBLAS grouped-GEMM composition on expert-sorted tokens with the
ReLU^2 gate product in eager -- torch._grouped_mm for the two expert GEMM
stages, plain torch elementwise for relu-squared * up, weighted index_add
combine, fwd+bwd through native autograd (_anchor_grouped). The GEMMs are
cuBLAS and dominate device time; the activation epilogue is a second-pass
elementwise the anchor cannot fuse without a bespoke kernel. Anchor choice
mirrors moe_swiglu provenance.

Distribution invariance: every draw hides BOTH the input scale (SCALE_RANGE)
and the ROUTING distribution (zipf-like popularity, hidden strength + hidden
assignment, redrawn per invocation). Determinism uses A=8 so an atomic
scatter-add combine cannot pass.
"""

import torch

import taskdef_api  # noqa: F401

ENTRY_NAME = "relu2_glu_moe"
REF_NAME = "relu2_glu_moe_ref"
SIGNATURE = (
    "relu2_glu_moe(x[T,D], w1[E,D,2F], w2[E,F,D], topk_idx[T,A] int64, "
    "topk_w[T,A]) -> y[T,D]"
)
SURFACE = "fwdbwd"
TASK_NAME = "hephaestus/relu2_glu_moe"

# AUTHORED PLACEHOLDER (needs GPU calibration). Gradient tolerance widened vs
# swiglu_moe: ReLU^2's g>0 branch has derivative 2*g rather than silu's bounded
# derivative, so bf16 rounding on active tiles accumulates larger absolute
# error over long expert segments.
TOL = {
    "bfloat16": (18.0, 0.10),
    "float32": (5.0, 0.02),
}

# AUTHORED PLACEHOLDER (needs GPU calibration). Output-only tolerance (tight;
# catches a smoothed-gate or silu-shaped substitute).
OUT_TOL = {
    "bfloat16": (0.16, 0.10),
    "float32": (0.05, 0.01),
}


def compare(got, want, dtype_name, shape):
    """Dense comparison for the OUTPUT under OUT_TOL. Gradients keep the
    protocol path (TOL) via the default grad comparison."""
    return taskdef_api.default_compare(got, want, dtype_name, shape, OUT_TOL)


FORBIDDEN = [
    "torch._grouped_mm",
    "_grouped_mm",
    "grouped_mm",
    "fused_moe",
    "vllm",
    "sglang",
    "grouped_gemm",
    "transformer_engine",
    "cutlass",
    "flashinfer",
]

VENDOR_LIB_PACKAGE = ""
VENDOR_LIB_SUBDIRS = ()

ADOPTION_FLOOR = 0.60

# AUTHORED PLACEHOLDER (needs GPU calibration). Higher headroom than the
# silu/gelu members: the anchor cannot fuse the piecewise ReLU^2 epilogue.
TARGET_FRACTION_OF_SOTA = 0.70


# ---------------------------------------------------------------------------
# Input draws
# ---------------------------------------------------------------------------


def _routing(T, E, A, g, device, collapse=False):
    """Hidden-seeded zipf-like popularity draw; loads always imbalanced with
    no fixed hot expert."""
    if collapse:
        idx = torch.zeros(T, A, dtype=torch.int64, device=device)
        w = torch.full((T, A), 1.0 / A, device=device, dtype=torch.float32)
        return idx, w
    strength = float(torch.empty((), device=device).uniform_(0.8, 2.2, generator=g))
    perm = torch.randperm(E, generator=g, device=device)
    ranks = torch.empty(E, device=device, dtype=torch.float32)
    ranks[perm] = torch.arange(E, device=device, dtype=torch.float32)
    bias = -strength * torch.log1p(ranks)
    logits = torch.randn(T, E, generator=g, device=device) + bias[None, :]
    vals, idx = torch.topk(logits, A, dim=-1)
    return idx, torch.softmax(vals, dim=-1)


def _finish(x, w1, w2, idx, w, dtype):
    x = x.to(dtype).contiguous().requires_grad_(True)
    w1 = w1.to(dtype).contiguous().requires_grad_(True)
    w2 = w2.to(dtype).contiguous().requires_grad_(True)
    w = w.to(dtype).detach().contiguous().requires_grad_(True)
    args = {"x": x, "w1": w1, "w2": w2, "topk_idx": idx.contiguous(), "topk_w": w}
    return args, ("x", "w1", "w2", "topk_w")


def make_inputs(shape, seed, scale, dtype, device):
    T, D, F, E, A = (int(shape[k]) for k in ("T", "D", "F", "E", "A"))
    g = torch.Generator(device=device)
    g.manual_seed(seed & 0x7FFFFFFF)
    x = torch.randn(T, D, generator=g, device=device) * float(scale)
    w1 = torch.randn(E, D, 2 * F, generator=g, device=device) * (D**-0.5)
    w2 = torch.randn(E, F, D, generator=g, device=device) * (F**-0.5)
    idx, w = _routing(
        T, E, A, g, device, collapse=shape.get("route_mode") == "collapse"
    )
    return _finish(x, w1, w2, idx, w, dtype)


# ---------------------------------------------------------------------------
# Anchor: cuBLAS grouped-GEMM composition on expert-sorted tokens, eager ReLU^2
# ---------------------------------------------------------------------------


def _sorted_routing(topk_idx, topk_w, E, A):
    flat = topk_idx.reshape(-1)
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=E)
    src = order // A
    w_slot = topk_w.reshape(-1).index_select(0, order)
    return order, counts, src, w_slot


def _anchor_grouped(x, w1, w2, topk_idx, topk_w):
    """torch._grouped_mm on expert-sorted tokens; eager relu^2 * up; weighted
    index_add combine; backward via native autograd."""
    T, D = x.shape
    E = w1.shape[0]
    F = w1.shape[2] // 2
    A = topk_idx.shape[1]
    _, counts, src, w_slot = _sorted_routing(topk_idx, topk_w, E, A)
    offs = counts.cumsum(0).to(torch.int32)
    xg = x.index_select(0, src)
    h = torch._grouped_mm(xg, w1, offs=offs)
    r = torch.nn.functional.relu(h[:, :F])
    act = r * r * h[:, F:]
    yp = torch._grouped_mm(act, w2, offs=offs)
    return x.new_zeros(T, D).index_add(0, src, yp * w_slot.unsqueeze(1))


def sota(args):
    return _anchor_grouped(
        args["x"], args["w1"], args["w2"], args["topk_idx"], args["topk_w"]
    )


# ---------------------------------------------------------------------------
# Stability modes (finiteness-and-not-raising gate; drift recorded only)
# ---------------------------------------------------------------------------


def _mk(route="zipf", x_mode="normal", w_mode="normal", w_uniform=False, w_zero=False):
    def maker(shape, seed, scale, dtype, device):
        T, D, F, E, A = (int(shape[k]) for k in ("T", "D", "F", "E", "A"))
        g = torch.Generator(device=device)
        g.manual_seed(seed & 0x7FFFFFFF)
        x = torch.randn(T, D, generator=g, device=device) * float(scale)
        if x_mode == "tiny":
            x = x * 1.0e-4
        elif x_mode == "sixdecade":
            mag = torch.rand(T, D, generator=g, device=device) * 6.0 - 3.0
            x = x * torch.pow(10.0, mag)
        w1 = torch.randn(E, D, 2 * F, generator=g, device=device) * (D**-0.5)
        w2 = torch.randn(E, F, D, generator=g, device=device) * (F**-0.5)
        if w_mode == "sixdecade":
            m1 = torch.rand(E, D, 2 * F, generator=g, device=device) * 6.0 - 3.0
            m2 = torch.rand(E, F, D, generator=g, device=device) * 6.0 - 3.0
            w1 = w1 * torch.pow(10.0, m1)
            w2 = w2 * torch.pow(10.0, m2)
        idx, w = _routing(T, E, A, g, device, collapse=route == "collapse")
        if w_uniform:
            w = torch.full_like(w, 1.0 / A)
        if w_zero:
            w = torch.zeros_like(w)
        return _finish(x, w1, w2, idx, w, dtype)

    return maker


STABILITY_MODES = {
    "collapse_routing": _mk(route="collapse"),
    "uniform_weights": _mk(w_uniform=True),
    "x_tiny": _mk(x_mode="tiny"),
    "x_huge": _mk(x_mode="sixdecade"),
    "w_sixdecade": _mk(w_mode="sixdecade"),
    "zero_weights": _mk(w_zero=True),
}

SMOKE_SHAPE = {"T": 512, "D": 512, "F": 256, "E": 8, "A": 4}
DETERMINISM_SHAPE = {"T": 2048, "D": 1024, "F": 512, "E": 16, "A": 8}
STABILITY_SHAPE = {"T": 1024, "D": 1024, "F": 704, "E": 16, "A": 4}


# ---------------------------------------------------------------------------
# Negative controls (Invariant 17, both halves).
# ---------------------------------------------------------------------------

_FALLBACK_SRC = '''\
"""Pure-torch fallback candidate (negative control): no Triton anywhere."""
import torch


def relu2_glu_moe(x, w1, w2, topk_idx, topk_w):
    T, D = x.shape
    E = w1.shape[0]
    F = w1.shape[2] // 2
    A = topk_idx.shape[1]
    xf, w1f, w2f, wf = x.float(), w1.float(), w2.float(), topk_w.float()
    y = xf.new_zeros(T, D)
    for a in range(A):
        idx = topk_idx[:, a]
        contrib = xf.new_zeros(T, D)
        for e in range(E):
            rows = (idx == e).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            h = xf.index_select(0, rows)
            gu = h @ w1f[e]
            r = torch.nn.functional.relu(gu[:, :F])
            contrib = contrib.index_copy(0, rows, (r * r * gu[:, F:]) @ w2f[e])
        y = y + wf[:, a:a + 1] * contrib
    return y.to(x.dtype)
'''


NEGATIVE_CONTROLS = [
    {
        "name": "nc_forbidden_call",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": ("import triton\n", "import triton\n_SHORTCUT = torch._grouped_mm\n"),
        "bound_gate": "G1",
        "oracle_env": False,
        "expectation": "the forbidden-symbol scan names torch._grouped_mm and "
        "zeroes at G1 before the candidate is ever imported",
    },
    {
        "name": "nc_smoothed_gate",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": (
            "    r = tl.where(g > 0.0, g, 0.0)\n"
            "    tl.store(OUT + o, r * r * u, mask=m)",
            "    s = tl.sigmoid(g)\n    tl.store(OUT + o, g * s * u, mask=m)",
        ),
        "bound_gate": "G2",
        "oracle_env": False,
        "expectation": "silu substituted for relu^2 in the elementwise gate: y "
        "and every gradient drift outside OUT_TOL/TOL on the "
        "hidden sweep and G2 fails at its first stage",
    },
    {
        "name": "nc_drop_token",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": (
            "        tw = topk_w.contiguous()\n",
            "        tw = topk_w.contiguous()\n"
            "        _sl = tw.argmin(dim=1, keepdim=True)\n"
            "        tw = tw.scatter(1, _sl, 0.0)\n",
        ),
        "bound_gate": "G2",
        "oracle_env": False,
        "expectation": "the classic capacity-trim cheat: the lowest-weight slot "
        "of every token is silently dropped; y and every gradient "
        "lose that slot's contribution",
    },
    {
        "name": "nc_nondeterministic_combine",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": (
            "        _combine_fwd_kernel[(triton.cdiv(T, CT), "
            "triton.cdiv(D, CD))](\n"
            "            yp, w_slot, inv, y, T, D, A, BT=CT, BD=CD)",
            "        y = torch.zeros(T, D, device=x.device, dtype=dt)\n"
            "        y.index_add_(0, src_tok, yp * w_slot.unsqueeze(1))",
        ),
        "bound_gate": "G2",
        "oracle_env": False,
        "expectation": "atomic scatter-add combine: correct within tolerance on "
        "any single run, but eight racing contributions per "
        "output element make bf16 order change between runs and "
        "the determinism stage fails",
    },
    {
        "name": "nc_differential_path",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": (
            "    return _Relu2GluMoeOracle.apply(x, w1, w2, topk_idx, topk_w)",
            "    if float(x.abs().amax()) <= 64.0:\n"
            "        topk_w = torch.where(topk_idx == (w1.shape[0] - 1),\n"
            "                             torch.zeros_like(topk_w), "
            "topk_w)\n"
            "    return _Relu2GluMoeOracle.apply(x, w1, w2, topk_idx, "
            "topk_w)",
        ),
        "bound_gate": "G2",
        "oracle_env": True,
        "expectation": "the exploit shape: full math only when a detected "
        "input-magnitude statistic looks stress-scale, one expert "
        "skipped otherwise. Runs with FORGE_ORACLE=1 so G1/G4 "
        "exemptions are live; the cheap path runs on graded draws "
        "and G2 fails",
    },
    {
        "name": "nc_torch_fallback",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": lambda src: _FALLBACK_SRC,
        "bound_gate": "G4",
        "oracle_env": False,
        "expectation": "correct pure-torch composition, 100% of device time "
        "outside candidate Triton kernels: G2 passes and the "
        "written-kernel adoption floor zeroes at G4",
    },
]
