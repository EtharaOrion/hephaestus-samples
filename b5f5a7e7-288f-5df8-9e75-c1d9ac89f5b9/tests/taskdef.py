"""Taskdef: row-wise argmax with mixed input dtype (SURFACE fwd).

Selection family. Everything category-specific the generic verifier consumes
lives here (protocol: seed/forge/core/taskdef_api.py).

Operator semantics (reference.py is the definition; this is a summary):
    argmax_mixed_dtype(x[R, N]) -> indices[R] int64
    k=1 degenerate row-wise argmax; ties broken by LOWER original index
    first; -0.0 == +0.0 one tie group. Input dtype is one of float32,
    bfloat16, float16 drawn per invocation from the hidden dtype set for
    that shape. Exactly one correct answer per input -> EXACT equality;
    `compare` overrides the dense walk.

Hard lever (spec.yaml justifies at length): TWO combined levers -- the
k=1 degenerate fast path AND mixed input dtype including fp16 (a format
the rest of the family does not test). torch.argmax runs on any dtype
but its tie-break is unspecified across CUB releases, so it is a timing
denominator only; a from-scratch candidate must produce the exact
lower-index tie-break under all three formats, which means the composite
key path has to switch bit-width per dtype: fp16 -> int16 bitcast (16 bits
for the value key), bf16 -> int16 (top 16 bits of fp32), fp32 -> int32.
The 16-bit-key formats leave a very narrow slot for the index encoding
(more index bits mean fewer value bits) so a naive port of the family's
composite-key trick fails immediately.

Anchor: CUDA torch.argmax (dim=-1). TIMING DENOMINATOR ONLY. Its
tie-break can differ from ours across CUB releases, never consulted for
correctness. reference.py's stable argsort is the definition. No
vendorable source tree: VENDOR_LIB_SUBDIRS empty; the FORBIDDEN symbol
scan still runs.

Distribution invariance is made real exactly as for the rest of the
family (hidden scale + hidden distribution shape per draw; timed ==
checked). process_checks drops the magnitude-branch lint (argmax IS a
magnitude branch) and keeps no_grader_reference.

STATUS: authored-draft -- needs GPU calibration. TARGET_FRACTION_OF_SOTA
and the golden/controls numbers below are placeholders, NOT measured on
hardware.
"""

import re

import torch

import taskdef_api  # noqa: F401

ENTRY_NAME = "argmax_mixed_dtype"
REF_NAME = "argmax_mixed_dtype_ref"
SIGNATURE = "argmax_mixed_dtype(x[R, N]) -> indices[R] int64"
SURFACE = "fwd"
TASK_NAME = "hephaestus/argmax_mixed_dtype"

# Indices only, no float compare -- tolerances present for the protocol
# but never used by `compare` (which is int-exact). fp16 entry kept so
# grade_core's per-dtype tolerance lookup for the stability drift record
# succeeds on fp16 draws.
TOL = {"float32": (0.0, 0.0), "bfloat16": (0.0, 0.0), "float16": (0.0, 0.0)}

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
    "torch.argmax",
    ".argmax",
]

VENDOR_LIB_PACKAGE = ""
VENDOR_LIB_SUBDIRS = ()

ADOPTION_FLOOR = 0.60

# PLACEHOLDER (authored-draft, NOT GPU-measured). Argmax over a large row
# is memory-bound at HBM peak on the anchor; the reachable fraction is
# small. Calibration must overwrite.
TARGET_FRACTION_OF_SOTA = 0.15


def _base_draw(R, N, seed, scale, device):
    g = torch.Generator(device=device)
    g.manual_seed(seed & 0x7FFFFFFF)
    dist = int(torch.randint(0, 3, (1,), generator=g, device=device).item())
    if dist == 0:
        x = torch.randn(R, N, generator=g, device=device, dtype=torch.float32)
    elif dist == 1:
        x = torch.randn(R, N, generator=g, device=device, dtype=torch.float32).exp()
    else:
        x = (
            torch.rand(R, N, generator=g, device=device, dtype=torch.float32) * 2.0
            - 1.0
        )
    return x * scale, g


def _dtype_from_name(name):
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[name]


def _cast_no_overflow(xf, dtype):
    """fp16 overflow clamp: fp32 draws routinely produce values outside the
    fp16 range under the hidden scale, and torch.to(fp16) silently emits
    +/-inf for those. The operator handles legitimate +/-inf correctly (they
    win / lose their row), but that would collapse the argmax to a fixed
    position and hide tie-break bugs. Clamp fp16 draws into the safe range
    while leaving fp32/bf16 alone."""
    if dtype == torch.float16:
        max_fp16 = 65504.0
        xf = torch.clamp(xf, min=-max_fp16 * 0.5, max=max_fp16 * 0.5)
    return xf.to(dtype).contiguous()


def make_inputs(shape, seed, scale, dtype, device):
    """One reproducible draw. NaN-free by construction.

    Optional shape keys (verifier-side only, never published):
      quantize: int Q  -> values snapped to Q integer levels * scale (massive
                          exact duplicates in EVERY dtype).
      specials: truthy -> +inf/-inf blocks, +/-0.0 tie group and +/- subnormals
                          planted in the first 106 columns of the row.
      near_max: truthy -> the row's true max is placed at a random index
                          exactly once; all other entries are strictly less
                          (a determinism-style probe for tie behaviour).
    """
    R, N = int(shape["R"]), int(shape["N"])
    x, g = _base_draw(R, N, seed, scale, device)
    if shape.get("quantize"):
        q = int(shape["quantize"])
        lev = torch.floor(torch.rand(R, N, generator=g, device=device) * q)
        x = (lev - q // 2) * float(scale)
    if shape.get("specials"):
        x = -x.abs() - 2.0 * float(scale)
        x[:, 0:29] = float("inf")
        x[:, 40:60] = float("-inf")
        zeros = torch.tensor([0.0, -0.0], device=device).repeat(8)
        x[:, 70:86] = zeros
        sub = torch.tensor([1e-39, -1e-39], device=device).repeat(8)
        x[:, 90:106] = sub
    if shape.get("near_max"):
        idx = torch.randint(0, N, (R,), generator=g, device=device)
        x = -x.abs()
        x.scatter_(1, idx.unsqueeze(1), 1.0)
    x = _cast_no_overflow(x, dtype)
    return {"x": x}, ()


def sota(args):
    """Production anchor: CUDA torch.argmax. Timing denominator ONLY."""
    x = args["x"]
    return torch.argmax(x, dim=-1)


def compare(got, want, dtype_name, shape):
    """EXACT int64 equality on indices. [] iff equivalent."""
    try:
        if not isinstance(got, torch.Tensor):
            return [f"output must be a tensor, got {type(got).__name__}"]
        if not isinstance(want, torch.Tensor):
            return [f"reference output must be a tensor, got {type(want).__name__}"]
        fails = []
        if got.dtype != torch.int64:
            fails.append(f"indices dtype must be torch.int64, got {got.dtype}")
        if tuple(got.shape) != tuple(want.shape):
            fails.append(f"indices shape {tuple(got.shape)} != {tuple(want.shape)}")
        if fails:
            return fails
        if not torch.equal(got, want):
            n = int((got != want).sum())
            fails.append(
                f"indices differ from the exact (value desc, index asc) "
                f"argmax: {n} rows"
            )
        return fails
    except Exception as e:  # noqa: BLE001
        return [f"comparison failed on malformed output: {type(e).__name__}: {e}"]


def _mode_all_equal(shape, seed, scale, dtype, device):
    R, N = int(shape["R"]), int(shape["N"])
    x = torch.full((R, N), 0.5 * float(scale), device=device, dtype=torch.float32)
    return {"x": _cast_no_overflow(x, dtype)}, ()


def _mode_neg_inf_blocks(shape, seed, scale, dtype, device):
    R, N = int(shape["R"]), int(shape["N"])
    x, _ = _base_draw(R, N, seed, scale, device)
    x[:, ::11] = float("-inf")
    return {"x": _cast_no_overflow(x, dtype)}, ()


def _mode_subnormals(shape, seed, scale, dtype, device):
    """A row saturated with the smallest positive subnormals of each dtype.
    fp16 subnormals ~6e-8, bf16 subnormals ~9e-41, fp32 subnormals ~1e-45.
    The `equal at subnormal precision` behaviour differs across formats;
    the tie-break rule (lower index) must resolve the ambiguity."""
    R, N = int(shape["R"]), int(shape["N"])
    g = torch.Generator(device=device)
    g.manual_seed(seed & 0x7FFFFFFF)
    magnitude = {torch.float32: 1e-40, torch.bfloat16: 1e-38, torch.float16: 6e-6}[
        dtype
    ]
    lev = torch.randint(0, 3, (R, N), generator=g, device=device, dtype=torch.int32)
    x = lev.to(torch.float32) * magnitude
    return {"x": _cast_no_overflow(x, dtype)}, ()


def _mode_reverse_ramp(shape, seed, scale, dtype, device):
    R, N = int(shape["R"]), int(shape["N"])
    ramp = torch.arange(N, device=device, dtype=torch.float32).flip(0) * float(scale)
    x = ramp.unsqueeze(0).expand(R, N).contiguous()
    return {"x": _cast_no_overflow(x, dtype)}, ()


def _mode_dup8(shape, seed, scale, dtype, device):
    R, N = int(shape["R"]), int(shape["N"])
    g = torch.Generator(device=device)
    g.manual_seed(seed & 0x7FFFFFFF)
    lev = torch.floor(torch.rand(R, N, generator=g, device=device) * 8.0)
    x = (lev - 4.0) * float(scale)
    return {"x": _cast_no_overflow(x, dtype)}, ()


STABILITY_MODES = {
    "all_equal_plateau": _mode_all_equal,
    "neg_inf_blocks": _mode_neg_inf_blocks,
    "subnormal_saturated": _mode_subnormals,
    "reverse_ramp": _mode_reverse_ramp,
    "eight_level_duplicates": _mode_dup8,
}

SMOKE_SHAPE = {"R": 32, "N": 4096}
DETERMINISM_SHAPE = {"R": 128, "N": 65536, "quantize": 128}
STABILITY_SHAPE = {"R": 32, "N": 65536}


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
    "p1": {"R": 512, "N": 65536, "dtypes": ["bfloat16"]},
    "p2": {"R": 128, "N": 524288, "dtypes": ["bfloat16"]},
}
BENCH_CHECK_SHAPE = {"R": 16, "N": 8192}
BENCH_CHECK_SHAPE_NPOT = {"R": 5, "N": 10007}
BENCH_CHECK_SHAPE_TIES = {"R": 8, "N": 8192, "quantize": 16}


# ---------------------------------------------------------------------------
# Negative controls (Invariant 17). Each arm must zero by EXACTLY its bound
# gate; the from-scratch oracle passes G1/G4 honestly. (Validated at GPU
# calibration; authored-draft.)
# ---------------------------------------------------------------------------

# Correct, deterministic, zero-Triton pure-torch argmax. Composite key
# ((float bits reordered) << 24) | (max_idx - col) with a single row max.
_FALLBACK_SRC = '''\
"""Pure-torch fallback candidate (negative control): no Triton anywhere."""
import torch


def argmax_mixed_dtype(x):
    R, N = x.shape
    xf = x.float().contiguous()
    xf = torch.where(xf == 0.0, torch.zeros_like(xf), xf)
    b = xf.view(torch.int32).to(torch.int64)
    u = torch.where(b >= 0, b ^ 0x80000000, (~b) & 0xFFFFFFFF)
    inv = ((1 << 24) - 1) - torch.arange(N, device=x.device, dtype=torch.int64)
    keys = (u << 24) | inv
    mx, _ = keys.max(dim=-1)
    return ((1 << 24) - 1) - (mx & ((1 << 24) - 1))
'''


NEGATIVE_CONTROLS = [
    {
        "name": "nc_forbidden_call",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": ("import triton\n", "import triton\n_SHORTCUT = torch.argmax\n"),
        "bound_gate": "G1",
        "oracle_env": False,
        "expectation": "the forbidden-symbol scan names torch.argmax and zeroes at G1 "
        "before the candidate is imported (argmax must be from-scratch, "
        "not delegated to the CUB-tuned builtin)",
    },
    {
        "name": "nc_tiebreak_reversed",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": (
            "return _impl(x)",
            "i = _impl(torch.flip(x, dims=[-1]))\n    return (x.shape[-1] - 1 - i)",
        ),
        "bound_gate": "G2",
        "oracle_env": False,
        "expectation": "identical value winners, ties broken by HIGHER original index; "
        "the duplicate-guaranteeing shapes fail the exact index "
        "comparison at G2",
    },
    {
        "name": "nc_fp16_upcast",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": ("xf = x.to(tl.float32)", "xf = x  # keep fp16 key precision"),
        "bound_gate": "G2",
        "oracle_env": False,
        "expectation": "keeps the composite-key derivation at input precision. On "
        "fp16 draws the resulting 16-bit key loses tie-break bits and "
        "the exact G2 comparison fails on the duplicate-saturated fp16 "
        "shape",
    },
    {
        "name": "nc_nondeterministic_ties",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": (
            "return _impl(x)",
            "globals()['_NCALLS'] = globals().get('_NCALLS', 0) + 1\n"
            "    if globals()['_NCALLS'] % 2 == 0:\n"
            "        i = _impl(torch.flip(x, dims=[-1]))\n"
            "        return (x.shape[-1] - 1 - i)\n"
            "    return _impl(x)",
        ),
        "bound_gate": "G2",
        "oracle_env": False,
        "expectation": "simulated race: the tie winner alternates between invocations; "
        "the three-run bitwise determinism stage on the quantized draw "
        "catches it at G2",
    },
    {
        "name": "nc_differential_path",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": (
            "return _impl(x)",
            "if float(x.abs().max()) > 1.0e4:\n"
            "        return _impl(x)\n"
            "    xh = x[:, : x.shape[-1] // 2].contiguous()\n"
            "    return _impl(xh)",
        ),
        "bound_gate": "G2",
        "oracle_env": False,
        "expectation": "the §7.1 exploit: branch on a magnitude statistic, full scan "
        "only on 'stress-looking' data, half-row scan otherwise. Timed "
        "tensors ARE checked tensors, so the wrong index range on the "
        "half path fails G2; no lint involved",
    },
    {
        "name": "nc_first_row_only",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": ("return out", "return out[:1].expand_as(out).contiguous()"),
        "bound_gate": "G2",
        "oracle_env": False,
        "expectation": "returns row 0's argmax broadcast to every row -- a shortcut "
        "that always trivially passes shape checks; the exact G2 "
        "comparison fails on every row past 0",
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
