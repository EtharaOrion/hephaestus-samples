"""Taskdef: row-wise descending top-k over VERY LARGE N with SMALL k (SURFACE fwd).

Selection family. Everything category-specific the generic verifier consumes
lives here (protocol: seed/forge/core/taskdef_api.py).

Operator semantics (reference.py is the definition; this is a summary):
    topk_giant_row(x[R, N], k) -> (values[R, k], indices[R, k] int64)
    Descending top-k of each row. N in [262144, 4194304] at every graded
    shape and k in [16, 128] at every graded shape. Ties broken by LOWER
    original index first; -0.0 == +0.0 one tie group; values are BIT-EXACT
    gathers of the input element (input dtype); indices int64. Exactly one
    correct answer per input -> EXACT equality; `compare` overrides the
    dense walk.

Hard lever (spec.yaml justifies at length): NO torch/vendor builtin is
purpose-tuned for k << sqrt(N) with N in the multi-megabyte range. CUB's
radix top-k is tuned for a moderate ratio; at k=16 over a 4M row the
theoretical work is dominated by the SINGLE STREAMING READ of the row and
the log(N/k) radix passes, and every wasted pass costs full HBM
bandwidth. An iterative masked-max starter pays O(k) full-row reads and
scores near zero; a competitive candidate must implement two-phase select
(block-local top-k stored, then a reduce phase across blocks) or a
histogram-based threshold descent.

Anchor: CUDA torch.topk (CUB radix select) as the TIMING DENOMINATOR only.
It handles arbitrary (N, k) and is the strongest general vendor path;
correctness is graded against reference.py's stable argsort. No vendorable
source tree: VENDOR_LIB_SUBDIRS empty; the FORBIDDEN symbol scan still runs.

Distribution invariance is made real exactly as for the rest of the family
(hidden scale + hidden distribution shape per draw; timed == checked).
process_checks drops the magnitude-branch lint (selection math IS magnitude
comparison) and keeps no_grader_reference.

STATUS: authored-draft -- needs GPU calibration. TARGET_FRACTION_OF_SOTA
and the golden/controls numbers below are placeholders, NOT measured on
hardware.
"""

import re

import torch

import taskdef_api  # noqa: F401

ENTRY_NAME = "topk_giant_row"
REF_NAME = "topk_giant_row_ref"
SIGNATURE = "topk_giant_row(x[R, N], k) -> (values[R, k], indices[R, k] int64)"
SURFACE = "fwd"
TASK_NAME = "hephaestus/topk_giant_row"

TOL = {"float32": (0.0, 0.0), "bfloat16": (0.0, 0.0)}

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
]

VENDOR_LIB_PACKAGE = ""
VENDOR_LIB_SUBDIRS = ()

ADOPTION_FLOOR = 0.60

# PLACEHOLDER (authored-draft, NOT GPU-measured). Reachable fraction is
# small here because the anchor's single-streaming-read regime is already
# close to HBM peak. Calibration must overwrite.
TARGET_FRACTION_OF_SOTA = 0.10


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


def make_inputs(shape, seed, scale, dtype, device):
    """One reproducible draw. NaN-free by construction.

    Optional shape keys (verifier-side only, never published):
      quantize: int Q  -> values snapped to Q integer levels * scale (massive
                          exact duplicates in EVERY dtype).
      specials: truthy -> +inf/-inf blocks, +/-0.0 tie group and +/- subnormals
                          planted in the first 106-column prefix of the row.
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
        x[:, 0:29] = float("inf")
        x[:, 40:60] = float("-inf")
        zeros = torch.tensor([0.0, -0.0], device=device).repeat(8)
        x[:, 70:86] = zeros
        sub = torch.tensor([1e-39, -1e-39], device=device).repeat(8)
        x[:, 90:106] = sub
    x = x.to(dtype).contiguous()
    return {"x": x, "k": k}, ()


def sota(args):
    """Production anchor: CUDA torch.topk. Timing denominator ONLY."""
    x, k = args["x"], int(args["k"])
    v, i = torch.topk(x, k, dim=-1, largest=True, sorted=True)
    return v, i


_BITVIEW = {"float32": torch.int32, "bfloat16": torch.int16}


def compare(got, want, dtype_name, shape):
    """EXACT comparison. [] iff equivalent; failure strings otherwise."""
    try:
        want_v, want_i = want
        if not (isinstance(got, (tuple, list)) and len(got) == 2):
            return [
                f"output must be a (values, indices) pair, got {type(got).__name__}"
            ]
        got_v, got_i = got
        if not isinstance(got_i, torch.Tensor) or not isinstance(got_v, torch.Tensor):
            return ["output pair must hold two tensors"]
        fails = []
        if got_i.dtype != torch.int64:
            fails.append(f"indices dtype must be torch.int64, got {got_i.dtype}")
        if tuple(got_i.shape) != tuple(want_i.shape):
            fails.append(f"indices shape {tuple(got_i.shape)} != {tuple(want_i.shape)}")
        if got_v.dtype != want_v.dtype:
            fails.append(f"values dtype {got_v.dtype} != input dtype {want_v.dtype}")
        if tuple(got_v.shape) != tuple(want_v.shape):
            fails.append(f"values shape {tuple(got_v.shape)} != {tuple(want_v.shape)}")
        if fails:
            return fails
        if not torch.equal(got_i, want_i):
            n = int((got_i != want_i).sum())
            rows = int((got_i != want_i).any(dim=-1).sum())
            fails.append(
                f"indices differ from the exact (value desc, index asc) "
                f"order: {n} positions across {rows} rows"
            )
        bv = _BITVIEW[dtype_name]
        gb = got_v.contiguous().view(bv)
        wb = want_v.contiguous().view(bv)
        if not torch.equal(gb, wb):
            n = int((gb != wb).sum())
            fails.append(
                f"values are not bit-exact gathers of the input: "
                f"{n} elements differ at the bit level"
            )
        return fails
    except Exception as e:  # noqa: BLE001
        return [f"comparison failed on malformed output: {type(e).__name__}: {e}"]


def _mode_all_equal(shape, seed, scale, dtype, device):
    R, N, k = int(shape["R"]), int(shape["N"]), int(shape["k"])
    x = torch.full((R, N), 0.5 * float(scale), device=device, dtype=torch.float32)
    return {"x": x.to(dtype).contiguous(), "k": k}, ()


def _mode_neg_inf_blocks(shape, seed, scale, dtype, device):
    R, N, k = int(shape["R"]), int(shape["N"]), int(shape["k"])
    x, _ = _base_draw(R, N, seed, scale, device)
    x[:, ::13] = float("-inf")  # >= k finite entries remain
    return {"x": x.to(dtype).contiguous(), "k": k}, ()


def _mode_six_decade(shape, seed, scale, dtype, device):
    R, N, k = int(shape["R"]), int(shape["N"]), int(shape["k"])
    g = torch.Generator(device=device)
    g.manual_seed(seed & 0x7FFFFFFF)
    mag = torch.rand(R, N, generator=g, device=device) * 6.0 - 3.0
    x = (
        torch.randn(R, N, generator=g, device=device)
        * torch.pow(10.0, mag)
        * float(scale)
    )
    return {"x": x.to(dtype).contiguous(), "k": k}, ()


def _mode_reverse_sorted(shape, seed, scale, dtype, device):
    R, N, k = int(shape["R"]), int(shape["N"]), int(shape["k"])
    ramp = torch.arange(N, device=device, dtype=torch.float32) * float(scale)
    x = ramp.flip(0).unsqueeze(0).expand(R, N).contiguous()  # already-descending row
    return {"x": x.to(dtype).contiguous(), "k": k}, ()


def _mode_dup8(shape, seed, scale, dtype, device):
    R, N, k = int(shape["R"]), int(shape["N"]), int(shape["k"])
    g = torch.Generator(device=device)
    g.manual_seed(seed & 0x7FFFFFFF)
    lev = torch.floor(torch.rand(R, N, generator=g, device=device) * 8.0)
    x = (lev - 4.0) * float(scale)
    return {"x": x.to(dtype).contiguous(), "k": k}, ()


STABILITY_MODES = {
    "all_equal_plateau": _mode_all_equal,
    "neg_inf_blocks": _mode_neg_inf_blocks,
    "six_decade_range": _mode_six_decade,
    "already_reverse_sorted": _mode_reverse_sorted,
    "eight_level_duplicates": _mode_dup8,
}

# SMOKE keeps N large (this is the giant-N task); DETERMINISM saturates ties.
SMOKE_SHAPE = {"R": 8, "N": 262144, "k": 16}
DETERMINISM_SHAPE = {"R": 32, "N": 262144, "k": 64, "quantize": 128}
STABILITY_SHAPE = {"R": 16, "N": 524288, "k": 32}


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
    "p1": {"R": 16, "N": 524288, "k": 32, "dtypes": ["bfloat16"]},
    "p2": {"R": 8, "N": 1048576, "k": 64, "dtypes": ["bfloat16"]},
}
BENCH_CHECK_SHAPE = {"R": 4, "N": 262144, "k": 16}
BENCH_CHECK_SHAPE_NPOT = {"R": 3, "N": 262147, "k": 24}
BENCH_CHECK_SHAPE_TIES = {"R": 4, "N": 262144, "k": 96, "quantize": 16}


# ---------------------------------------------------------------------------
# Negative controls (Invariant 17). Each arm must zero by EXACTLY its bound
# gate; the from-scratch oracle passes G1/G4 honestly. (Validated at GPU
# calibration; authored-draft.)
# ---------------------------------------------------------------------------

# Correct, deterministic, zero-Triton pure-torch top-k. Iterative composite-key
# max extraction; row-batched so the int64 key buffer stays bounded.
_FALLBACK_SRC = '''\
"""Pure-torch fallback candidate (negative control): no Triton anywhere."""
import torch


def topk_giant_row(x, k):
    k = int(k)
    R, N = x.shape
    out_i = torch.empty(R, k, dtype=torch.int64, device=x.device)
    rows_per = max(1, int(4.0e9 // (N * 8)))
    for r0 in range(0, R, rows_per):
        xb = x[r0:r0 + rows_per].float().contiguous()
        xb = torch.where(xb == 0.0, torch.zeros_like(xb), xb)
        b = xb.view(torch.int32).to(torch.int64)
        u = torch.where(b >= 0, b ^ 0x80000000, (~b) & 0xFFFFFFFF)
        inv = (1 << 23) - 1 - torch.arange(N, device=x.device, dtype=torch.int64)
        keys = (u << 23) | inv
        del b, u, xb
        for j in range(k):
            mx, _ = keys.max(dim=-1)
            col = ((1 << 23) - 1) - (mx & ((1 << 23) - 1))
            out_i[r0:r0 + keys.shape[0], j] = col
            keys.scatter_(1, col.unsqueeze(1), -1)
        del keys
    return x.gather(1, out_i), out_i
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
        "name": "nc_first_block_only",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": (
            "return _topk_impl(x, int(k))",
            "N = x.shape[-1]\n"
            "    xh = x[:, : N // 2].contiguous()\n"
            "    return _topk_impl(xh, int(k))",
        ),
        "bound_gate": "G2",
        "oracle_env": False,
        "expectation": "the classic giant-N exploit: scan only half the row so the "
        "read cost halves. On the graded shapes many winners live in "
        "the trimmed half; the exact G2 comparison fails on almost "
        "every row",
    },
    {
        "name": "nc_wrong_k",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": (
            "    return values, out_idx\n",
            "    out_idx = out_idx.clone()\n"
            "    out_idx[:, -1] = out_idx[:, 0]\n"
            "    return x.gather(1, out_idx), out_idx\n",
        ),
        "bound_gate": "G2",
        "oracle_env": False,
        "expectation": "duplicates the rank-0 winner into the rank-(k-1) slot; the "
        "exact G2 comparison fails wherever k >= 2",
    },
    {
        "name": "nc_tiebreak_reversed",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": (
            "return _topk_impl(x, int(k))",
            "v, i = _topk_impl(torch.flip(x, dims=[-1]), int(k))\n"
            "    return v, (x.shape[-1] - 1 - i)",
        ),
        "bound_gate": "G2",
        "oracle_env": False,
        "expectation": "identical values, ties broken by HIGHER original index; the "
        "duplicate-guaranteeing shapes fail the exact index comparison at G2",
    },
    {
        "name": "nc_nondeterministic_ties",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": (
            "return _topk_impl(x, int(k))",
            "globals()['_NCALLS'] = globals().get('_NCALLS', 0) + 1\n"
            "    if globals()['_NCALLS'] % 2 == 0:\n"
            "        v, i = _topk_impl(torch.flip(x, dims=[-1]), int(k))\n"
            "        return v, (x.shape[-1] - 1 - i)\n"
            "    return _topk_impl(x, int(k))",
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
            "return _topk_impl(x, int(k))",
            "if float(x.abs().max()) > 1.0e4:\n"
            "        return _topk_impl(x, int(k))\n"
            "    xh = x[:, : x.shape[-1] // 2].contiguous()\n"
            "    return _topk_impl(xh, int(k))",
        ),
        "bound_gate": "G2",
        "oracle_env": False,
        "expectation": "the §7.1 exploit: branch on a magnitude statistic, full-scan "
        "only on 'stress-looking' data, half-row scan otherwise. Timed "
        "tensors ARE checked tensors, so the shrunk output shape "
        "mismatches at G2; no lint involved",
    },
    {
        "name": "nc_torch_fallback",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": lambda src: _FALLBACK_SRC,
        "bound_gate": "G4",
        "oracle_env": False,
        "expectation": "correct, deterministic pure-torch composition, 100% of device time "
        "outside candidate Triton kernels: G2 passes and the written-kernel "
        "adoption floor zeroes at G4",
    },
]
