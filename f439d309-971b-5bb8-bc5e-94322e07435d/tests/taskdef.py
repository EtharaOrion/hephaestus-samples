"""Taskdef: SwiGLU MoE for wide-FFN, high-E shape distributions.

HARD member of the moe_fused_experts family (precomputed integer routing).
Operator math is identical to the family baseline (silu-gated GLU experts);
the hardness lever is a shape distribution that pushes TWO KernelBench-hard
axes at once: (a) variable-length grouped GEMM with E up to 256 experts and
strongly imbalanced token counts per expert (strong zipf, many
short/empty segments, few long ones), and (b) wide-vs-tall aspect where the
FFN intermediate F is 2--3x the model dim D. A candidate that specializes to
a small, balanced expert set (as swiglu_moe or geglu_moe permit) fails here:
segment-tile schedulers have to handle 200+ segments with sub-tile counts,
and the wide FFN inflates the intermediate act[T*A_eff, F] to the point that
a naive materialization is memory-bound.

STATUS: authored-draft. TOL/OUT_TOL/TARGET_FRACTION_OF_SOTA are AUTHORED
PLACEHOLDERS carried from moe_swiglu, WIDENED for the longer per-expert
segments possible at E=256 with strong zipf (one expert can hold the
majority of T*A slots), and MUST be re-measured on the delivery GPU.

Operator (reference.py is the definition):
    wide_expert_swiglu_moe(x[T,D], w1[E,D,2F], w2[E,F,D], topk_idx[T,A] i64,
        topk_w[T,A]) -> y[T,D]
    y[t] = sum_a topk_w[t,a] * f(topk_idx[t,a], x[t]);
    f(e,h) = (silu(h@w1[e,:,:F]) * (h@w1[e,:,F:])) @ w2[e].
    SURFACE fwdbwd: gradients of x, w1, w2 AND topk_w (four).

Anchor: cuBLAS grouped-GEMM composition on expert-sorted tokens (identical
to swiglu_moe's anchor); the wide-vs-tall aspect and E=256 stress the
grouped GEMM scheduler and the intermediate-materialization traffic.
"""

import torch

import taskdef_api  # noqa: F401

ENTRY_NAME = "wide_expert_swiglu_moe"
REF_NAME = "wide_expert_swiglu_moe_ref"
SIGNATURE = (
    "wide_expert_swiglu_moe(x[T,D], w1[E,D,2F], w2[E,F,D], "
    "topk_idx[T,A] int64, topk_w[T,A]) -> y[T,D]"
)
SURFACE = "fwdbwd"
TASK_NAME = "hephaestus/wide_expert_swiglu_moe"

# Gradient TOL widened vs swiglu_moe: at E=256 with strong zipf one expert can
# accumulate 60%+ of all T*A tokens, so dw1[e_hot]/dw2[e_hot] row-sums span
# tens of thousands of rows and the honest atol floor climbs.
TOL = {
    "bfloat16": (20.0, 0.10),
    "float32": (6.0, 0.02),
}

OUT_TOL = {
    "bfloat16": (0.16, 0.10),
    "float32": (0.05, 0.01),
}


def compare(got, want, dtype_name, shape):
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
TARGET_FRACTION_OF_SOTA = 0.72  # widened headroom vs swiglu_moe (0.78)


# ---------------------------------------------------------------------------
# Input draws -- STRONGER zipf than swiglu_moe (2.0..3.5 vs 0.8..2.2).
# ---------------------------------------------------------------------------


def _routing(T, E, A, g, device, collapse=False):
    """Hidden-seeded strong-zipf popularity draw. Strength 2.0..3.5 (vs
    0.8..2.2 in swiglu_moe) so loads at E=256 have one hot expert receiving
    the majority of slots, the tail dwindling to empty."""
    if collapse:
        idx = torch.zeros(T, A, dtype=torch.int64, device=device)
        w = torch.full((T, A), 1.0 / A, device=device, dtype=torch.float32)
        return idx, w
    strength = float(torch.empty((), device=device).uniform_(2.0, 3.5, generator=g))
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
# Anchor
# ---------------------------------------------------------------------------


def _sorted_routing(topk_idx, topk_w, E, A):
    flat = topk_idx.reshape(-1)
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=E)
    src = order // A
    w_slot = topk_w.reshape(-1).index_select(0, order)
    return order, counts, src, w_slot


def _anchor_grouped(x, w1, w2, topk_idx, topk_w):
    T, D = x.shape
    E = w1.shape[0]
    F = w1.shape[2] // 2
    A = topk_idx.shape[1]
    _, counts, src, w_slot = _sorted_routing(topk_idx, topk_w, E, A)
    offs = counts.cumsum(0).to(torch.int32)
    xg = x.index_select(0, src)
    h = torch._grouped_mm(xg, w1, offs=offs)
    act = torch.nn.functional.silu(h[:, :F]) * h[:, F:]
    yp = torch._grouped_mm(act, w2, offs=offs)
    return x.new_zeros(T, D).index_add(0, src, yp * w_slot.unsqueeze(1))


def sota(args):
    return _anchor_grouped(
        args["x"], args["w1"], args["w2"], args["topk_idx"], args["topk_w"]
    )


# ---------------------------------------------------------------------------
# Stability modes
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

SMOKE_SHAPE = {"T": 512, "D": 512, "F": 1024, "E": 16, "A": 4}
DETERMINISM_SHAPE = {"T": 2048, "D": 768, "F": 2048, "E": 32, "A": 8}
STABILITY_SHAPE = {"T": 1024, "D": 512, "F": 1408, "E": 64, "A": 4}


# ---------------------------------------------------------------------------
# Negative controls
# ---------------------------------------------------------------------------

_FALLBACK_SRC = '''\
"""Pure-torch fallback candidate (negative control): no Triton anywhere."""
import torch


def wide_expert_swiglu_moe(x, w1, w2, topk_idx, topk_w):
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
            fe = (torch.nn.functional.silu(gu[:, :F]) * gu[:, F:]) @ w2f[e]
            contrib = contrib.index_copy(0, rows, fe)
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
        "name": "nc_skip_cold_experts",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": (
            "                    if rows.numel() == 0:\n"
            "                        continue\n",
            "                    if rows.numel() < 4:\n"
            "                        continue\n",
        ),
        "bound_gate": "G2",
        "oracle_env": False,
        "expectation": "the temptation of the wide-zipf distribution: skip "
        "experts holding fewer than 4 tokens (mostly the tail). "
        "y and w1/w2/topk_w gradients lose those slots, G2 fails",
    },
    {
        "name": "nc_wrong_gate_up",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": (
            "    s = tl.sigmoid(g)\n    tl.store(OUT + o, g * s * u, mask=m)",
            "    s = tl.sigmoid(u)\n    tl.store(OUT + o, u * s * g, mask=m)",
        ),
        "bound_gate": "G2",
        "oracle_env": False,
        "expectation": "gate/up swap: silu(u)*g != silu(g)*u, G2 fails at its "
        "first stage",
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
        "expectation": "classic capacity-trim: lowest-weight slot dropped per "
        "token, y and grads lose that slot, G2 fails",
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
        "expectation": "atomic scatter-add combine: bit-wise order changes across "
        "the three determinism runs at A=8, G2 fails at stage 4",
    },
    {
        "name": "nc_differential_path",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": (
            "    return _WideExpertSwigluMoeOracle.apply(x, w1, w2, topk_idx, topk_w)",
            "    if float(x.abs().amax()) <= 64.0:\n"
            "        topk_w = torch.where(topk_idx == (w1.shape[0] - 1),\n"
            "                             torch.zeros_like(topk_w), "
            "topk_w)\n"
            "    return _WideExpertSwigluMoeOracle.apply(x, w1, w2, "
            "topk_idx, topk_w)",
        ),
        "bound_gate": "G2",
        "oracle_env": True,
        "expectation": "magnitude-branch exploit: full math on stress-scale only, "
        "one expert skipped otherwise. FORGE_ORACLE=1 keeps G1/G4 "
        "exemptions live, cheap path runs on graded draws, G2 fails",
    },
    {
        "name": "nc_torch_fallback",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": lambda src: _FALLBACK_SRC,
        "bound_gate": "G4",
        "oracle_env": False,
        "expectation": "correct pure-torch composition, 100% device time outside "
        "candidate Triton kernels: G2 passes, G4 zeroes",
    },
]
