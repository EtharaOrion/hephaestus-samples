"""Taskdef: fused row-wise softmax + top-k speculative-decoding gate (SURFACE fwd).

Selection family. Everything category-specific the generic verifier consumes
lives here (protocol: seed/forge/core/taskdef_api.py).

Operator semantics (reference.py is the definition; this is a summary):
    fused_softmax_topk(logits[R, N], k) -> (probs[R, k] fp32, indices[R, k] int64)
    Canonical fp32 softmax over each row (max-subtract, exp, sum-normalize),
    then top-k of the row (equivalently, top-k of the logits since softmax
    is strictly increasing). indices are DESCENDING by value with LOWER
    ORIGINAL INDEX tie-break; -0.0 == +0.0 one tie group. probs are the
    softmax probabilities at those indices, ALWAYS fp32. Grading is EXACT
    on indices and dtype-tolerance on probs. `compare` overrides the dense
    walk to enforce this split.

Hard lever (spec.yaml justifies at length): NO single torch/vendor builtin
fuses softmax and top-k in one pass. The natural composition
torch.softmax(x, -1) then torch.topk(probs, k) reads the row THREE TIMES
(max, sum, and again for topk); a fused kernel reads it once and
overlaps the softmax reduction with the top-k threshold descent. This is
the speculative-decoding gate operator in every modern LLM serving stack,
and the current sota is a hand-written fused kernel (FlashAttention-style
one-pass online softmax + partial top-k). A candidate must reproduce that
fusion.

Anchor: torch.softmax + torch.topk composition. TIMING DENOMINATOR ONLY.
Its tie-break can differ from ours (CUB's tuned paths break value ties
differently), never consulted for correctness. reference.py's stable
argsort + canonical fp32 softmax is the definition. No vendorable source
tree: VENDOR_LIB_SUBDIRS empty; the FORBIDDEN symbol scan still runs.

Distribution invariance is made real exactly as for the rest of the family
(hidden scale + hidden distribution shape per draw; timed == checked).
process_checks drops the magnitude-branch lint (softmax max-subtract IS a
magnitude branch) and keeps no_grader_reference.

STATUS: authored-draft -- needs GPU calibration. TARGET_FRACTION_OF_SOTA
and the golden/controls numbers below are placeholders, NOT measured on
hardware.
"""

import re

import torch

import taskdef_api  # noqa: F401

ENTRY_NAME = "fused_softmax_topk"
REF_NAME = "fused_softmax_topk_ref"
SIGNATURE = (
    "fused_softmax_topk(logits[R, N], k) -> (probs[R, k] fp32, indices[R, k] int64)"
)
SURFACE = "fwd"
TASK_NAME = "hephaestus/fused_softmax_topk"

# Indices are graded EXACT via the compare override; probs are graded with
# these dtype-dependent tolerances. The bf16 tolerance covers the tail of
# probs values where the canonical fp32 softmax reduction agrees at the
# 5e-4 level with alternative reduction orders on bf16-magnitude logits.
TOL = {"float32": (1.0e-6, 1.0e-5), "bfloat16": (5.0e-4, 5.0e-3)}

FORBIDDEN = [
    "torch.topk",
    "torch.sort",
    "torch.argsort",
    "torch.msort",
    "torch.kthvalue",
    "torch.unique",
    "torch.median",
    "torch.nanmedian",
    "torch.quantile",
    "torch.nanquantile",
    ".topk",
    ".sort",
    ".argsort",
    ".msort",
    ".kthvalue",
    ".unique",
    ".median",
    ".quantile",
    "torch.softmax",
    "F.softmax",
    "torch.nn.functional.softmax",
    ".softmax",
    "log_softmax",
    "_softmax",
]

VENDOR_LIB_PACKAGE = ""
VENDOR_LIB_SUBDIRS = ()

ADOPTION_FLOOR = 0.60

# PLACEHOLDER (authored-draft, NOT GPU-measured). The composition anchor
# reads the row 3x; a properly fused candidate reads it 1x, so the
# reachable fraction is moderate. Calibration must overwrite.
TARGET_FRACTION_OF_SOTA = 0.30


def _base_draw(R, N, seed, scale, device):
    g = torch.Generator(device=device)
    g.manual_seed(seed & 0x7FFFFFFF)
    dist = int(torch.randint(0, 3, (1,), generator=g, device=device).item())
    if dist == 0:
        x = torch.randn(R, N, generator=g, device=device, dtype=torch.float32)
    elif dist == 1:
        x = torch.randn(R, N, generator=g, device=device, dtype=torch.float32) * 4.0
    else:
        x = (
            torch.rand(R, N, generator=g, device=device, dtype=torch.float32) * 8.0
            - 4.0
        )
    return x * scale, g


def make_inputs(shape, seed, scale, dtype, device):
    """One reproducible draw. NaN-free by construction. Logits scales are
    bounded so exp doesn't overflow fp32 even with the hidden scale.

    Optional shape keys (verifier-side only, never published):
      quantize: int Q -> logits snapped to Q levels (heavy per-row ties).
      specials: truthy -> planted +inf/-inf and duplicate-max blocks in
                          the first 106 columns of each row.
      peaked:   truthy -> add a large positive offset to one column,
                          producing a near-onehot softmax distribution.
    """
    R, N, k = int(shape["R"]), int(shape["N"]), int(shape["k"])
    if not (1 <= k <= N):
        raise ValueError(f"k={k} out of range for N={N}")
    x, g = _base_draw(R, N, seed, scale, device)
    if shape.get("quantize"):
        q = int(shape["quantize"])
        lev = torch.floor(torch.rand(R, N, generator=g, device=device) * q)
        x = (lev - q // 2) * float(scale)
    if shape.get("specials"):
        x = -x.abs() - 2.0 * float(scale)
        # A duplicate-max BLOCK at a finite value. +inf cannot be used here: the
        # reference's canonical policy subtracts the row max, so a +inf max gives
        # inf - inf = NaN for every row and the operator is undefined. The tie
        # stress (29 identical maxima) is preserved. -inf below stays: it is exact.
        x[:, 0:29] = 2.0 * float(scale)
        x[:, 40:60] = float("-inf")
        zeros = torch.tensor([0.0, -0.0], device=device).repeat(8)
        x[:, 70:86] = zeros
        sub = torch.tensor([1e-39, -1e-39], device=device).repeat(8)
        x[:, 90:106] = sub
    if shape.get("peaked"):
        col = int(torch.randint(0, N, (1,), generator=g, device=device).item())
        x[:, col] = x[:, col] + 20.0
    x = x.to(dtype).contiguous()
    return {"logits": x, "k": k}, ()


def sota(args):
    """Production anchor: torch.softmax + torch.topk (three-pass composition).
    Timing denominator ONLY."""
    logits, k = args["logits"], int(args["k"])
    probs_full = torch.softmax(logits.float(), dim=-1)
    v, i = torch.topk(probs_full, k, dim=-1, largest=True, sorted=True)
    return v, i


def compare(got, want, dtype_name, shape):
    """indices EXACT; probs tolerance per TOL[dtype_name]. [] iff equivalent."""
    try:
        want_p, want_i = want
        if not (isinstance(got, (tuple, list)) and len(got) == 2):
            return [f"output must be a (probs, indices) pair, got {type(got).__name__}"]
        got_p, got_i = got
        if not isinstance(got_i, torch.Tensor) or not isinstance(got_p, torch.Tensor):
            return ["output pair must hold two tensors"]
        fails = []
        if got_i.dtype != torch.int64:
            fails.append(f"indices dtype must be torch.int64, got {got_i.dtype}")
        if tuple(got_i.shape) != tuple(want_i.shape):
            fails.append(f"indices shape {tuple(got_i.shape)} != {tuple(want_i.shape)}")
        if got_p.dtype != torch.float32:
            fails.append(f"probs dtype must be torch.float32, got {got_p.dtype}")
        if tuple(got_p.shape) != tuple(want_p.shape):
            fails.append(f"probs shape {tuple(got_p.shape)} != {tuple(want_p.shape)}")
        if fails:
            return fails
        if not torch.equal(got_i, want_i):
            n = int((got_i != want_i).sum())
            rows = int((got_i != want_i).any(dim=-1).sum())
            fails.append(
                f"indices differ from the exact (value desc, index asc) "
                f"order: {n} positions across {rows} rows"
            )
        if not torch.isfinite(got_p).all():
            fails.append("non-finite values in candidate probs")
        else:
            atol, rtol = TOL[dtype_name]
            d = (got_p - want_p).abs()
            lim = atol + rtol * want_p.abs()
            bad = int((d > lim).sum())
            if bad:
                fails.append(
                    f"probs: {bad} elements outside tolerance "
                    f"(atol={atol}, rtol={rtol}); max abs {float(d.max()):.3e}"
                )
        return fails
    except Exception as e:  # noqa: BLE001
        return [f"comparison failed on malformed output: {type(e).__name__}: {e}"]


def _mode_all_equal(shape, seed, scale, dtype, device):
    R, N, k = int(shape["R"]), int(shape["N"]), int(shape["k"])
    x = torch.full((R, N), 0.5 * float(scale), device=device, dtype=torch.float32)
    return {"logits": x.to(dtype).contiguous(), "k": k}, ()


def _mode_neg_inf_blocks(shape, seed, scale, dtype, device):
    R, N, k = int(shape["R"]), int(shape["N"]), int(shape["k"])
    x, _ = _base_draw(R, N, seed, scale, device)
    x[:, ::11] = float("-inf")  # >= k finite entries remain
    return {"logits": x.to(dtype).contiguous(), "k": k}, ()


def _mode_large_range(shape, seed, scale, dtype, device):
    R, N, k = int(shape["R"]), int(shape["N"]), int(shape["k"])
    g = torch.Generator(device=device)
    g.manual_seed(seed & 0x7FFFFFFF)
    x = (
        torch.randn(R, N, generator=g, device=device, dtype=torch.float32)
        * 30.0
        * float(scale)
    )
    return {"logits": x.to(dtype).contiguous(), "k": k}, ()


def _mode_peaked(shape, seed, scale, dtype, device):
    R, N, k = int(shape["R"]), int(shape["N"]), int(shape["k"])
    g = torch.Generator(device=device)
    g.manual_seed(seed & 0x7FFFFFFF)
    x = torch.randn(R, N, generator=g, device=device, dtype=torch.float32) * float(
        scale
    )
    col = int(torch.randint(0, N, (1,), generator=g, device=device).item())
    x[:, col] = x[:, col] + 25.0
    return {"logits": x.to(dtype).contiguous(), "k": k}, ()


def _mode_dup8(shape, seed, scale, dtype, device):
    R, N, k = int(shape["R"]), int(shape["N"]), int(shape["k"])
    g = torch.Generator(device=device)
    g.manual_seed(seed & 0x7FFFFFFF)
    lev = torch.floor(torch.rand(R, N, generator=g, device=device) * 8.0)
    x = (lev - 4.0) * float(scale)
    return {"logits": x.to(dtype).contiguous(), "k": k}, ()


STABILITY_MODES = {
    "all_equal_plateau": _mode_all_equal,
    "neg_inf_blocks": _mode_neg_inf_blocks,
    "large_range_30": _mode_large_range,
    "peaked_onehot_ish": _mode_peaked,
    "eight_level_duplicates": _mode_dup8,
}

SMOKE_SHAPE = {"R": 32, "N": 4096, "k": 16}
DETERMINISM_SHAPE = {"R": 64, "N": 32768, "k": 128, "quantize": 64}
STABILITY_SHAPE = {"R": 32, "N": 65536, "k": 64}


def process_checks(src):
    return {
        "no_grader_reference": not re.search(
            r"target\.json|hidden_shapes|/bundle/tests", src
        ),
    }


ANTI_CHEAT_RUBRIC_IDS = [
    "no_result_memoization",
    "no_call_count_shortcut",
    "no_timing_context_detection",
    "kernels_actually_launch",
    "no_distribution_dispatch",
    "no_judge_manipulation",
    "timing_measured_on_shipped_bytes",
    "timing_uses_synchronised_clock",
    "timing_covers_selection_scope",
]


BENCH_PUBLISHED_SHAPES = {
    "p1": {"R": 256, "N": 32000, "k": 32, "dtypes": ["bfloat16"]},  # LLM vocab size
    "p2": {
        "R": 128,
        "N": 128256,
        "k": 64,
        "dtypes": ["bfloat16"],
    },  # Llama-3 vocab size
}
BENCH_CHECK_SHAPE = {"R": 16, "N": 8192, "k": 32}
BENCH_CHECK_SHAPE_NPOT = {"R": 5, "N": 10007, "k": 23}
BENCH_CHECK_SHAPE_TIES = {"R": 8, "N": 8192, "k": 96, "quantize": 16}


# ---------------------------------------------------------------------------
# Negative controls (Invariant 17). Each arm must zero by EXACTLY its bound
# gate; the from-scratch oracle passes G1/G4 honestly. (Validated at GPU
# calibration; authored-draft.)
# ---------------------------------------------------------------------------

_FALLBACK_SRC = '''\
"""Pure-torch fallback candidate (negative control): no Triton anywhere."""
import torch


def fused_softmax_topk(logits, k):
    k = int(k)
    R, N = logits.shape
    xf = logits.float().contiguous()
    m = xf.max(dim=-1, keepdim=True).values
    e = (xf - m).exp()
    s = e.sum(dim=-1, keepdim=True)
    probs_full = e / s
    xr = torch.where(xf == 0.0, torch.zeros_like(xf), xf)
    b = xr.view(torch.int32).to(torch.int64)
    u = torch.where(b >= 0, b ^ 0x80000000, (~b) & 0xFFFFFFFF)
    inv = ((1 << 23) - 1) - torch.arange(N, device=logits.device, dtype=torch.int64)
    keys = (u << 23) | inv
    out_i = torch.empty(R, k, dtype=torch.int64, device=logits.device)
    for j in range(k):
        mx, _ = keys.max(dim=-1)
        col = ((1 << 23) - 1) - (mx & ((1 << 23) - 1))
        out_i[:, j] = col
        keys.scatter_(1, col.unsqueeze(1), -1)
    probs = probs_full.gather(1, out_i)
    return probs, out_i
'''


NEGATIVE_CONTROLS = [
    {
        "name": "nc_forbidden_call",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": ("import triton\n", "import triton\n_SHORTCUT = torch.topk\n"),
        "bound_gate": "G1",
        "oracle_env": False,
        "expectation": "the forbidden-symbol scan names torch.topk and zeroes at G1 "
        "before the candidate is imported",
    },
    {
        "name": "nc_forbidden_softmax",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": (
            "import triton\n",
            "import triton\nimport torch.nn.functional as _F\n_SHORTCUT = _F.softmax\n",
        ),
        "bound_gate": "G1",
        "oracle_env": False,
        "expectation": "the forbidden-symbol scan names F.softmax and zeroes at G1 "
        "before the candidate is imported (softmax must be fused, not "
        "delegated)",
    },
    {
        "name": "nc_wrong_normalisation",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": (
            "s = e_row.sum(dim=-1, keepdim=True)",
            "s = torch.ones_like(e_row.sum(dim=-1, keepdim=True))",
        ),
        "bound_gate": "G2",
        "oracle_env": False,
        "expectation": "skips the softmax normalization (divides by 1 instead of the "
        "row exp-sum); returned probs are unnormalized exp values and "
        "fail the probs tolerance on every row",
    },
    {
        "name": "nc_bf16_accumulator",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": (
            "xf = logits.float().contiguous()",
            "xf = logits.contiguous()  # keep bf16 accumulator",
        ),
        "bound_gate": "G2",
        "oracle_env": False,
        "expectation": "keeps the softmax reduction in bf16 instead of the canonical "
        "fp32 upcast; on bf16 inputs the accumulator drift exceeds the "
        "probs tolerance and G2 fails",
    },
    {
        "name": "nc_tiebreak_reversed",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": (
            "return _impl(logits, int(k))",
            "p, i = _impl(torch.flip(logits, dims=[-1]), int(k))\n"
            "    return p, (logits.shape[-1] - 1 - i)",
        ),
        "bound_gate": "G2",
        "oracle_env": False,
        "expectation": "identical probs, ties broken by HIGHER original index; the "
        "duplicate-guaranteeing shapes fail the exact index comparison at G2",
    },
    {
        "name": "nc_nondeterministic_ties",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": (
            "return _impl(logits, int(k))",
            "globals()['_NCALLS'] = globals().get('_NCALLS', 0) + 1\n"
            "    if globals()['_NCALLS'] % 2 == 0:\n"
            "        p, i = _impl(torch.flip(logits, dims=[-1]), int(k))\n"
            "        return p, (logits.shape[-1] - 1 - i)\n"
            "    return _impl(logits, int(k))",
        ),
        "bound_gate": "G2",
        "oracle_env": False,
        "expectation": "simulated race: the tie winner alternates between invocations; "
        "the three-run determinism stage on the quantized draw catches it at G2",
    },
    {
        "name": "nc_differential_path",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": (
            "return _impl(logits, int(k))",
            "if float(logits.abs().max()) > 1.0e2:\n"
            "        return _impl(logits, int(k))\n"
            "    xh = logits[:, : logits.shape[-1] // 2].contiguous()\n"
            "    return _impl(xh, int(k))",
        ),
        "bound_gate": "G2",
        "oracle_env": False,
        "expectation": "the §7.1 exploit: branch on a magnitude statistic, full-scan "
        "only on 'stress-looking' data. Timed tensors ARE checked "
        "tensors, so the shrunk output shape mismatches at G2; no "
        "lint involved",
    },
    {
        "name": "nc_torch_fallback",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": lambda src: _FALLBACK_SRC,
        "bound_gate": "G4",
        "oracle_env": False,
        "expectation": "correct, deterministic pure-torch composition, 100% of device "
        "time outside candidate Triton kernels: G2 passes and the "
        "written-kernel adoption floor zeroes at G4",
    },
]
