"""Taskdef: shared expert + routed SwiGLU experts, forward + backward (fwdbwd).

The HARD member of the moe_fused_experts family (precomputed integer routing).
Two structurally different GEMM regimes coexist in ONE operator: a DENSE
always-on SHARED expert applied to every token (its own weights ws1/ws2, weight
1, no routing) ADDED to the irregular routed SwiGLU path. SIX gradients are
graded: dx, dws1, dws2 (shared) and dw1, dw2, dtopk_w (routed). Everything
category-specific the generic verifier consumes lives here (protocol:
core/taskdef_api.py).

STATUS: authored-draft. TOL/OUT_TOL/TARGET_FRACTION_OF_SOTA below are AUTHORED
PLACEHOLDERS carried from the proven moe_swiglu category; they MUST be
re-measured on the delivery GPU. CALIBRATION NOTE: dws1/dws2 accumulate over
ALL T tokens (up to 16384 rows -- longer than any routed segment), so their
honest atol floor is likely the LARGEST in the family; and the reachable
fraction of the anchor is the LOWEST (two regimes to fuse), so
TARGET_FRACTION_OF_SOTA should calibrate below the other members'.

Operator (reference.py is the definition; this is a summary):
    shared_plus_routed_swiglu(x[T,D], ws1[D,2F], ws2[F,D], w1[E,D,2F],
        w2[E,F,D], topk_idx[T,A] i64, topk_w[T,A]) -> y[T,D]
    y[t] = f_shared(x[t]) + sum_a topk_w[t,a] * f(topk_idx[t,a], x[t]);
    f_shared(h) = (silu(h@ws1[:, :F]) * (h@ws1[:, F:])) @ ws2   (weight 1, all t);
    f(e,h)      = (silu(h@w1[e,:,:F]) * (h@w1[e,:,F:])) @ w2[e].
    SURFACE fwdbwd: gradients of x, ws1, ws2, w1, w2 AND topk_w are graded,
    timed and determinism-checked. topk_idx is precomputed routing, no grad.

Anchor: a production composition of the SAME two regimes -- the routed path is a
cuBLAS grouped-GEMM composition on expert-sorted tokens (torch._grouped_mm,
eager silu, weighted index_add combine) and the shared path is a dense torch
SwiGLU MLP over all tokens; fwd+bwd through native autograd (_anchor below).
The GEMMs are cuBLAS and dominate device time. Anchor choice must be MEASURED
on the delivery image (see moe_swiglu provenance).

Distribution invariance: every draw hides BOTH the input scale (SCALE_RANGE)
and the ROUTING distribution (zipf-like popularity, hidden strength + hidden
assignment, redrawn per invocation). The always-on shared path means expert
loads are imbalanced but the shared GEMM is dense every time. The tensors that
are timed are the tensors that are checked; NC-DP proves the closure with the
exemption env ON. Determinism uses A=8 so an atomic scatter-add combine cannot
pass.
"""

import torch

import taskdef_api  # noqa: F401  (protocol module; defaults resolved via cfg)

ENTRY_NAME = "shared_plus_routed_swiglu"
REF_NAME = "shared_plus_routed_swiglu_ref"
SIGNATURE = ("shared_plus_routed_swiglu(x[T,D], ws1[D,2F], ws2[F,D], "
             "w1[E,D,2F], w2[E,F,D], topk_idx[T,A] int64, topk_w[T,A]) -> y[T,D]")
SURFACE = "fwdbwd"
TASK_NAME = "hephaestus/shared_plus_routed_swiglu"

# AUTHORED PLACEHOLDER (needs GPU calibration). Gradient tolerance; dws1/dws2
# accumulate over ALL T tokens, so their honest atol floor is the family's
# largest -- calibrate on the oracle over the full graded/correctness set.
TOL = {
    "bfloat16": (18.0, 0.10),
    "float32": (5.0, 0.02),
}

# AUTHORED PLACEHOLDER (needs GPU calibration). Output-only tolerance (tight;
# catches output-corrupting cheats -- a dropped shared path, a dropped slot).
OUT_TOL = {
    "bfloat16": (0.18, 0.10),
    "float32": (0.05, 0.01),
}


def compare(got, want, dtype_name, shape):
    """Dense comparison for the OUTPUT under OUT_TOL. Gradients keep the
    protocol path (TOL) via the default grad comparison."""
    return taskdef_api.default_compare(got, want, dtype_name, shape, OUT_TOL)


# G1 word-boundary scan over comment-stripped candidate source.
FORBIDDEN = [
    "torch._grouped_mm", "_grouped_mm", "grouped_mm", "fused_moe", "vllm",
    "sglang", "grouped_gemm", "transformer_engine", "cutlass", "flashinfer",
]

# Anchor is a torch/cuBLAS composition: no vendorable python source tree ->
# shingle scan skipped (recorded available:false). FORBIDDEN symbol scan runs.
VENDOR_LIB_PACKAGE = ""
VENDOR_LIB_SUBDIRS = ()

ADOPTION_FLOOR = 0.60

# AUTHORED PLACEHOLDER (needs GPU calibration). The hard member: two GEMM
# regimes to fuse, so the reachable fraction is the family's lowest -- set below
# the other members' and below the oracle's repeated worst case on this image.
TARGET_FRACTION_OF_SOTA = 0.72


# ---------------------------------------------------------------------------
# Input draws
# ---------------------------------------------------------------------------

def _routing(T, E, A, g, device, collapse=False):
    """Hidden-seeded routing draw with a zipf-like popularity bias. Both the
    bias STRENGTH and the ASSIGNMENT of popularity ranks to expert ids are
    redrawn per invocation: loads are always imbalanced, no fixed hot expert.
    Returns (topk_idx int64 [T,A], topk_w float32 [T,A] softmax-normalized)."""
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


def _finish(x, ws1, ws2, w1, w2, idx, w, dtype):
    x = x.to(dtype).contiguous().requires_grad_(True)
    ws1 = ws1.to(dtype).contiguous().requires_grad_(True)
    ws2 = ws2.to(dtype).contiguous().requires_grad_(True)
    w1 = w1.to(dtype).contiguous().requires_grad_(True)
    w2 = w2.to(dtype).contiguous().requires_grad_(True)
    w = w.to(dtype).detach().contiguous().requires_grad_(True)
    args = {"x": x, "ws1": ws1, "ws2": ws2, "w1": w1, "w2": w2,
            "topk_idx": idx.contiguous(), "topk_w": w}
    return args, ("x", "ws1", "ws2", "w1", "w2", "topk_w")


def make_inputs(shape, seed, scale, dtype, device):
    """One reproducible draw. `scale` multiplies x. Weights use 1/sqrt(fan_in).
    Optional shape key route_mode: "collapse" sends every routed slot to expert
    0 (the shared path is always dense over all tokens regardless)."""
    T, D, F, E, A = (int(shape[k]) for k in ("T", "D", "F", "E", "A"))
    g = torch.Generator(device=device)
    g.manual_seed(seed & 0x7FFFFFFF)
    x = torch.randn(T, D, generator=g, device=device) * float(scale)
    ws1 = torch.randn(D, 2 * F, generator=g, device=device) * (D ** -0.5)
    ws2 = torch.randn(F, D, generator=g, device=device) * (F ** -0.5)
    w1 = torch.randn(E, D, 2 * F, generator=g, device=device) * (D ** -0.5)
    w2 = torch.randn(E, F, D, generator=g, device=device) * (F ** -0.5)
    idx, w = _routing(T, E, A, g, device,
                      collapse=shape.get("route_mode") == "collapse")
    return _finish(x, ws1, ws2, w1, w2, idx, w, dtype)


# ---------------------------------------------------------------------------
# Anchor: grouped-GEMM routed path + dense torch shared path
# ---------------------------------------------------------------------------

def _sorted_routing(topk_idx, topk_w, E, A):
    flat = topk_idx.reshape(-1)
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=E)
    src = order // A
    w_slot = topk_w.reshape(-1).index_select(0, order)
    return order, counts, src, w_slot


def _anchor(x, ws1, ws2, w1, w2, topk_idx, topk_w):
    """Production composition of THIS operator: routed path via cuBLAS grouped
    GEMM on expert-sorted tokens; shared path a dense torch SwiGLU MLP over all
    tokens; sum; backward via native autograd."""
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
    routed = x.new_zeros(T, D).index_add(0, src, yp * w_slot.unsqueeze(1))
    gs = x @ ws1
    shared = (torch.nn.functional.silu(gs[:, :F]) * gs[:, F:]) @ ws2
    return routed + shared


def sota(args):
    """Production anchor invocation: the timing denominator ONLY (its output
    and gradients are never used for correctness)."""
    return _anchor(args["x"], args["ws1"], args["ws2"], args["w1"], args["w2"],
                   args["topk_idx"], args["topk_w"])


# ---------------------------------------------------------------------------
# Stability modes (finiteness-and-not-raising gate; drift recorded only)
# ---------------------------------------------------------------------------

def _mk(route="zipf", x_mode="normal", w_mode="normal", w_uniform=False,
        w_zero=False):
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
        ws1 = torch.randn(D, 2 * F, generator=g, device=device) * (D ** -0.5)
        ws2 = torch.randn(F, D, generator=g, device=device) * (F ** -0.5)
        w1 = torch.randn(E, D, 2 * F, generator=g, device=device) * (D ** -0.5)
        w2 = torch.randn(E, F, D, generator=g, device=device) * (F ** -0.5)
        if w_mode == "sixdecade":
            m1 = torch.rand(E, D, 2 * F, generator=g, device=device) * 6.0 - 3.0
            m2 = torch.rand(E, F, D, generator=g, device=device) * 6.0 - 3.0
            w1 = w1 * torch.pow(10.0, m1)
            w2 = w2 * torch.pow(10.0, m2)
        idx, w = _routing(T, E, A, g, device, collapse=route == "collapse")
        if w_uniform:
            w = torch.full_like(w, 1.0 / A)
        if w_zero:
            # Routed contribution vanishes; the SHARED path still fires, so y is
            # NOT zero here (unlike the pure-routed members) -- stability only
            # gates finiteness.
            w = torch.zeros_like(w)
        return _finish(x, ws1, ws2, w1, w2, idx, w, dtype)
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
# A=8 on purpose: eight weighted routed contributions plus the shared term land
# on every output element, so an atomic scatter-add combine cannot repeat
# bitwise across the three determinism runs.
DETERMINISM_SHAPE = {"T": 2048, "D": 1024, "F": 512, "E": 16, "A": 8}
STABILITY_SHAPE = {"T": 1024, "D": 1024, "F": 704, "E": 16, "A": 4}


# ---------------------------------------------------------------------------
# Negative controls (Invariant 17, both halves). Every arm must zero by EXACTLY
# its bound gate. NOTE: mutate anchors are matched to THIS folder's
# oracle_kernel.py/kernel.py; because those Triton bodies are UNVERIFIED,
# re-validate every anchor during the controls stage on GPU.
# ---------------------------------------------------------------------------

_FALLBACK_SRC = '''\
"""Pure-torch fallback candidate (negative control): no Triton anywhere."""
import torch


def shared_plus_routed_swiglu(x, ws1, ws2, w1, w2, topk_idx, topk_w):
    T, D = x.shape
    E = w1.shape[0]
    F = w1.shape[2] // 2
    A = topk_idx.shape[1]
    xf, wf = x.float(), topk_w.float()
    ws1f, ws2f = ws1.float(), ws2.float()
    w1f, w2f = w1.float(), w2.float()
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
    gs = xf @ ws1f
    shared = (torch.nn.functional.silu(gs[:, :F]) * gs[:, F:]) @ ws2f
    return (y + shared).to(x.dtype)
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
        "name": "nc_wrong_gate_up",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": ("    s = tl.sigmoid(acc_g)\n"
                   "    tl.store(ACT + rows[:, None] * F + cf[None, :], "
                   "acc_g * s * acc_u, mask=m2)",
                   "    s = tl.sigmoid(acc_u)\n"
                   "    tl.store(ACT + rows[:, None] * F + cf[None, :], "
                   "acc_u * s * acc_g, mask=m2)"),
        "bound_gate": "G2",
        "oracle_env": False,
        "expectation": "gate and up projections swapped in the routed epilogue "
                       "(silu applied to the up half): silu(u)*g != silu(g)*u, "
                       "so output and every gradient are wrong and G2 fails",
    },
    {
        "name": "nc_drop_shared",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": ("    shared = _SharedSwigluOracle.apply(x, ws1, ws2)\n"
                   "    return routed + shared",
                   "    shared = _SharedSwigluOracle.apply(x, ws1, ws2)\n"
                   "    return routed"),
        "bound_gate": "G2",
        "oracle_env": False,
        "expectation": "the always-on shared expert is dropped from the sum: y "
                       "loses the dense shared contribution on EVERY token, and "
                       "dws1/dws2 collapse to zero -- far outside tolerance, G2 "
                       "fails at its first stage",
    },
    {
        "name": "nc_drop_token",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": ("        tw = topk_w.contiguous()\n",
                   "        tw = topk_w.contiguous()\n"
                   "        _sl = tw.argmin(dim=1, keepdim=True)\n"
                   "        tw = tw.scatter(1, _sl, 0.0)\n"),
        "bound_gate": "G2",
        "oracle_env": False,
        "expectation": "the classic capacity-trim cheat in the routed path: the "
                       "lowest-weight slot of every token is silently dropped, so "
                       "y and the routed gradients lose that slot, outside tol",
    },
    {
        "name": "nc_nondeterministic_combine",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": ("        _combine_fwd_kernel[(triton.cdiv(T, CT), "
                   "triton.cdiv(D, CD))](\n"
                   "            yp, w_slot, inv, y, T, D, A, BT=CT, BD=CD)",
                   "        y = torch.zeros(T, D, device=x.device, dtype=dt)\n"
                   "        y.index_add_(0, src_tok, yp * w_slot.unsqueeze(1))"),
        "bound_gate": "G2",
        "oracle_env": False,
        "expectation": "atomic scatter-add routed combine: correct within "
                       "tolerance on any single run, but with eight contributions "
                       "racing per output element the bf16 accumulation order "
                       "changes between runs and the determinism stage fails",
    },
    {
        "name": "nc_differential_path",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": ("    routed = _RoutedSwigluOracle.apply(x, w1, w2, topk_idx, "
                   "topk_w)",
                   "    if float(x.abs().amax()) <= 64.0:\n"
                   "        topk_w = torch.where(topk_idx == (w1.shape[0] - 1),\n"
                   "                             torch.zeros_like(topk_w), "
                   "topk_w)\n"
                   "    routed = _RoutedSwigluOracle.apply(x, w1, w2, topk_idx, "
                   "topk_w)"),
        "bound_gate": "G2",
        "oracle_env": True,
        "expectation": "the exploit shape: full math only when a detected "
                       "input-magnitude statistic looks stress-scale, one routed "
                       "expert skipped otherwise. Run with FORGE_ORACLE=1 so the "
                       "G1/G4 exemptions are live and cannot absorb the blame: "
                       "because the timed tensors ARE the checked tensors, the "
                       "cheap path runs on the graded draws and G2 fails",
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
                       "passes and the written-kernel adoption floor zeroes at G4",
    },
]
