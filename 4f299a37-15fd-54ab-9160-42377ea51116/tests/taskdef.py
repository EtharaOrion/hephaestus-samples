"""Taskdef: segmented descending top-k within fixed-length row segments (fwd).

Selection family -- the family's HARD task. Everything category-specific the
generic verifier consumes lives here (protocol: seed/forge/core/taskdef_api.py).

Operator semantics (reference.py is the definition; this is a summary):
    topk_segmented(x[R, N], k, segment) -> (values[R, G, k], indices[R, G, k] int64)
    Each row is cut into G = N // segment contiguous length-S segments (S =
    segment divides N) and a DESCENDING top-k is taken inside every segment
    independently. Indices are GLOBAL row positions (in [0, N)): index =
    s*S + local, with local the position inside segment s. Ties broken by LOWER
    original index first (global and local order agree inside a segment);
    -0.0 == +0.0 one tie group; values are BIT-EXACT gathers of the input
    element (input dtype); indices int64. 1 <= k <= S. Exactly one correct
    answer per input -> EXACT equality; `compare` overrides the dense walk.

Anchor: there is NO single torch builtin. The timing denominator applies
CUDA torch.topk to the [R, G, S] segment view (batched CUB radix select over
many short rows -- a near-roofline vendor path) and globalises the indices.
It is a TIMING DENOMINATOR ONLY (its tie-break differs), never consulted for
correctness. No vendorable source tree: VENDOR_LIB_SUBDIRS empty; the FORBIDDEN
symbol scan still runs.

Distribution invariance is made real exactly as for the rest of the family
(hidden scale + hidden distribution shape per draw; timed == checked).
process_checks drops the magnitude-branch lint and keeps no_grader_reference.

STATUS: authored-draft -- needs GPU calibration. TARGET_FRACTION_OF_SOTA and
the golden/controls numbers below are placeholders, NOT measured on hardware.
"""

import re

import torch

import taskdef_api  # noqa: F401

ENTRY_NAME = "topk_segmented"
REF_NAME = "topk_segmented_ref"
SIGNATURE = "topk_segmented(x[R, N], k, segment) -> (values[R, G, k], indices[R, G, k] int64)"
SURFACE = "fwd"
TASK_NAME = "hephaestus/topk_segmented"

TOL = {"float32": (0.0, 0.0), "bfloat16": (0.0, 0.0)}

FORBIDDEN = [
    "torch.topk", "torch.sort", "torch.argsort", "torch.msort",
    "torch.kthvalue", "torch.unique", "torch.median", "torch.nanmedian",
    "torch.quantile", "torch.nanquantile",
    ".topk", ".sort", ".argsort", ".msort", ".kthvalue", ".unique",
    ".median", ".quantile",
]

VENDOR_LIB_PACKAGE = ""
VENDOR_LIB_SUBDIRS = ()

ADOPTION_FLOOR = 0.60

# PLACEHOLDER (authored-draft, NOT GPU-measured). The vendor anchor is a
# batched radix-select over many short rows and is very hard to beat from
# scratch, so the eventual measured target is expected LOW (this is the hard
# task). Calibration must overwrite with a value below the oracle's repeated
# worst case.
TARGET_FRACTION_OF_SOTA = 0.12


# ---------------------------------------------------------------------------
# Input draws
# ---------------------------------------------------------------------------

def _base_draw(R, N, seed, scale, device):
    g = torch.Generator(device=device)
    g.manual_seed(seed & 0x7FFFFFFF)
    dist = int(torch.randint(0, 3, (1,), generator=g, device=device).item())
    if dist == 0:
        x = torch.randn(R, N, generator=g, device=device, dtype=torch.float32)
    elif dist == 1:
        x = torch.randn(R, N, generator=g, device=device, dtype=torch.float32).exp()
    else:
        x = torch.rand(R, N, generator=g, device=device, dtype=torch.float32) * 2.0 - 1.0
    return x * scale, g


def make_inputs(shape, seed, scale, dtype, device):
    """One reproducible draw. NaN-free by construction. `segment` divides N and
    1 <= k <= segment.

    Optional shape keys (verifier-side only, never published):
      quantize: int Q  -> values snapped to Q integer levels * scale (massive
                          exact duplicates in EVERY dtype, WITHIN every segment).
      specials: truthy -> +inf/-inf blocks, +/-0.0 tie group and +/- subnormals
                          planted in the first segment's top region.
    """
    R, N, k = int(shape["R"]), int(shape["N"]), int(shape["k"])
    S = int(shape["segment"])
    if N % S != 0:
        raise ValueError(f"segment={S} must divide N={N}")
    if not (1 <= k <= S):
        raise ValueError(f"k={k} out of range for segment length S={S}")
    x, g = _base_draw(R, N, seed, scale, device)
    if shape.get("quantize"):
        q = int(shape["quantize"])
        lev = torch.floor(torch.rand(R, N, generator=g, device=device) * q)
        x = (lev - q // 2) * float(scale)
    if shape.get("specials"):
        x = -x.abs() - 2.0 * float(scale)          # shift base below the specials
        x[:, 0:29] = float("inf")
        x[:, 40:60] = float("-inf")                # legal; unselected while >= k finite remain
        zeros = torch.tensor([0.0, -0.0], device=device).repeat(8)
        x[:, 70:86] = zeros
        sub = torch.tensor([1e-39, -1e-39], device=device).repeat(8)
        x[:, 90:106] = sub
    x = x.to(dtype).contiguous()
    return {"x": x, "k": k, "segment": S}, ()


def sota(args):
    """Production anchor: torch.topk over the [R, G, S] segment view, indices
    globalised. Timing denominator ONLY (its tie-break differs; never
    compared)."""
    x, k, S = args["x"], int(args["k"]), int(args["segment"])
    R, N = x.shape
    G = N // S
    xr = x.view(R, G, S)
    v, i_local = torch.topk(xr, k, dim=-1, largest=True, sorted=True)
    seg_off = (torch.arange(G, device=x.device, dtype=torch.int64) * S).view(1, G, 1)
    return v.contiguous(), (i_local.to(torch.int64) + seg_off).contiguous()


# ---------------------------------------------------------------------------
# Correctness: exact per-segment index/tie-break semantics (global indices)
# ---------------------------------------------------------------------------

_BITVIEW = {"float32": torch.int32, "bfloat16": torch.int16}


def compare(got, want, dtype_name, shape):
    """EXACT comparison over the [R, G, k] outputs. [] iff equivalent."""
    try:
        want_v, want_i = want
        if not (isinstance(got, (tuple, list)) and len(got) == 2):
            return [f"output must be a (values, indices) pair, got {type(got).__name__}"]
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
            segs = int((got_i != want_i).any(dim=-1).sum())
            fails.append(f"global indices differ from the exact per-segment "
                         f"(value desc, index asc) order: {n} positions across "
                         f"{segs} segments")
        bv = _BITVIEW[dtype_name]
        gb = got_v.contiguous().view(bv)
        wb = want_v.contiguous().view(bv)
        if not torch.equal(gb, wb):
            n = int((gb != wb).sum())
            fails.append(f"values are not bit-exact gathers of the input: "
                         f"{n} elements differ at the bit level")
        return fails
    except Exception as e:  # noqa: BLE001
        return [f"comparison failed on malformed output: {type(e).__name__}: {e}"]


# ---------------------------------------------------------------------------
# Stability modes (finiteness-and-not-raising gate; drift recorded only). Each
# carries `segment` so the entry receives S.
# ---------------------------------------------------------------------------

def _mode_all_equal(shape, seed, scale, dtype, device):
    R, N, k = int(shape["R"]), int(shape["N"]), int(shape["k"])
    S = int(shape["segment"])
    x = torch.full((R, N), 0.5 * float(scale), device=device, dtype=torch.float32)
    return {"x": x.to(dtype).contiguous(), "k": k, "segment": S}, ()


def _mode_neg_inf_blocks(shape, seed, scale, dtype, device):
    R, N, k = int(shape["R"]), int(shape["N"]), int(shape["k"])
    S = int(shape["segment"])
    x, _ = _base_draw(R, N, seed, scale, device)
    x[:, ::7] = float("-inf")                        # ~N/7 poisoned; >= k finite per segment remain
    return {"x": x.to(dtype).contiguous(), "k": k, "segment": S}, ()


def _mode_six_decade(shape, seed, scale, dtype, device):
    R, N, k = int(shape["R"]), int(shape["N"]), int(shape["k"])
    S = int(shape["segment"])
    g = torch.Generator(device=device)
    g.manual_seed(seed & 0x7FFFFFFF)
    mag = torch.rand(R, N, generator=g, device=device) * 6.0 - 3.0
    x = torch.randn(R, N, generator=g, device=device) * torch.pow(10.0, mag) * float(scale)
    return {"x": x.to(dtype).contiguous(), "k": k, "segment": S}, ()


def _mode_segment_sorted(shape, seed, scale, dtype, device):
    R, N, k = int(shape["R"]), int(shape["N"]), int(shape["k"])
    S = int(shape["segment"])
    x, _ = _base_draw(R, N, seed, scale, device)
    G = N // S
    xr = x.view(R, G, S)
    xr = torch.sort(xr, dim=-1, descending=True).values   # verifier-side; candidates may not
    return {"x": xr.reshape(R, N).to(dtype).contiguous(), "k": k, "segment": S}, ()


def _mode_dup8(shape, seed, scale, dtype, device):
    R, N, k = int(shape["R"]), int(shape["N"]), int(shape["k"])
    S = int(shape["segment"])
    g = torch.Generator(device=device)
    g.manual_seed(seed & 0x7FFFFFFF)
    lev = torch.floor(torch.rand(R, N, generator=g, device=device) * 8.0)
    x = (lev - 4.0) * float(scale)
    return {"x": x.to(dtype).contiguous(), "k": k, "segment": S}, ()


STABILITY_MODES = {
    "all_equal_plateau": _mode_all_equal,
    "neg_inf_blocks": _mode_neg_inf_blocks,
    "six_decade_range": _mode_six_decade,
    "per_segment_sorted": _mode_segment_sorted,
    "eight_level_duplicates": _mode_dup8,
}

SMOKE_SHAPE = {"R": 32, "N": 4096, "k": 16, "segment": 256}
# Quantized on purpose: guaranteed massive ties within each segment.
DETERMINISM_SHAPE = {"R": 64, "N": 65536, "k": 64, "segment": 256, "quantize": 128}
STABILITY_SHAPE = {"R": 32, "N": 65536, "k": 64, "segment": 512}


def process_checks(src):
    return {
        "no_grader_reference": not re.search(r"target\.json|hidden_shapes|/bundle/tests", src),
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
    "p1": {"R": 128, "N": 65536, "k": 32, "segment": 256, "dtypes": ["bfloat16"]},
    "p2": {"R": 32, "N": 262144, "k": 64, "segment": 512, "dtypes": ["bfloat16"]},
}
BENCH_CHECK_SHAPE = {"R": 16, "N": 8192, "k": 8, "segment": 128}
BENCH_CHECK_SHAPE_NPOT = {"R": 5, "N": 10281, "k": 7, "segment": 3427}   # 10281 = 3 * 3427
BENCH_CHECK_SHAPE_TIES = {"R": 8, "N": 8192, "k": 24, "segment": 256, "quantize": 16}


# ---------------------------------------------------------------------------
# Negative controls (Invariant 17). Each arm must zero by EXACTLY its bound
# gate; the from-scratch oracle passes G1/G4 honestly. (Validated at GPU
# calibration; authored-draft.)
# ---------------------------------------------------------------------------

# Correct, deterministic, zero-Triton pure-torch segmented selection: composite
# desc key with LOCAL column, iterative per-(row,segment) max extraction, global
# indices. Uses NO forbidden symbol -> reaches G4.
_FALLBACK_SRC = '''\
"""Pure-torch fallback candidate (negative control): no Triton anywhere."""
import torch


def topk_segmented(x, k, segment):
    k, S = int(k), int(segment)
    R, N = x.shape
    G = N // S
    M = R * G
    xr = x.contiguous().view(M, S)
    xf = torch.where(xr == 0.0, torch.zeros_like(xr.float()), xr.float())
    b = xf.view(torch.int32).to(torch.int64)
    u = torch.where(b >= 0, b ^ 0x80000000, (~b) & 0xFFFFFFFF)
    inv = 2097151 - torch.arange(S, device=x.device, dtype=torch.int64)
    keys = (u << 21) | inv                                   # [M, S]
    local = torch.empty(M, k, dtype=torch.int64, device=x.device)
    for j in range(k):
        mx, _ = keys.max(dim=-1)
        col = 2097151 - (mx & 2097151)
        local[:, j] = col
        keys.scatter_(1, col.unsqueeze(1), -1)
    local = local.view(R, G, k)
    seg_off = (torch.arange(G, device=x.device, dtype=torch.int64) * S).view(1, G, 1)
    idx = local + seg_off
    vals = x.gather(1, idx.reshape(R, G * k)).reshape(R, G, k)
    return vals, idx
'''


NEGATIVE_CONTROLS = [
    {
        "name": "nc_forbidden_call",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": ("import triton\n",
                   "import triton\n_SHORTCUT = torch.topk\n"),
        "bound_gate": "G1",
        "oracle_env": False,
        "expectation": "the forbidden-symbol scan names torch.topk and zeroes at G1 "
                       "before the candidate is imported",
    },
    {
        "name": "nc_missing_segment_offset",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": ("idx = local + seg_off",
                   "idx = local"),
        "bound_gate": "G2",
        "oracle_env": False,
        "expectation": "drops the per-segment global offset, so every segment's indices "
                       "collapse to [0, S); for G > 1 the indices (and gathered values) "
                       "are wrong for every segment past the first and the exact G2 "
                       "comparison fails",
    },
    {
        "name": "nc_wrong_k",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": ("    return vals, idx\n",
                   "    idx = idx.clone()\n"
                   "    idx[:, :, -1] = idx[:, :, 0]\n"
                   "    return x.gather(1, idx.reshape(x.shape[0], -1)).reshape(vals.shape), idx\n"),
        "bound_gate": "G2",
        "oracle_env": False,
        "expectation": "duplicates the rank-0 winner into the rank-(k-1) slot of every "
                       "segment (values kept consistent with the wrong index); the exact "
                       "G2 comparison fails wherever k >= 2",
    },
    {
        "name": "nc_tiebreak_reversed",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": ("return _seg_impl(x, int(k), int(segment))",
                   "S = int(segment)\n"
                   "    R, N = x.shape\n"
                   "    xr = x.contiguous().view(R, N // S, S)\n"
                   "    xf = torch.flip(xr, dims=[-1]).reshape(R, N)\n"
                   "    v, i = _seg_impl(xf, int(k), S)\n"
                   "    seg = i // S\n"
                   "    return v, seg * S + (S - 1 - (i - seg * S))"),
        "bound_gate": "G2",
        "oracle_env": False,
        "expectation": "identical values, ties broken by HIGHER local index within each "
                       "segment; the duplicate-guaranteeing shapes fail the exact index "
                       "comparison at G2",
    },
    {
        "name": "nc_nondeterministic_ties",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": ("return _seg_impl(x, int(k), int(segment))",
                   "globals()['_NCALLS'] = globals().get('_NCALLS', 0) + 1\n"
                   "    if globals()['_NCALLS'] % 2 == 0:\n"
                   "        S = int(segment)\n"
                   "        R, N = x.shape\n"
                   "        xr = x.contiguous().view(R, N // S, S)\n"
                   "        xf = torch.flip(xr, dims=[-1]).reshape(R, N)\n"
                   "        v, i = _seg_impl(xf, int(k), S)\n"
                   "        seg = i // S\n"
                   "        return v, seg * S + (S - 1 - (i - seg * S))\n"
                   "    return _seg_impl(x, int(k), int(segment))"),
        "bound_gate": "G2",
        "oracle_env": False,
        "expectation": "simulated race: the per-segment tie winner alternates between "
                       "invocations; the three-run bitwise determinism stage on the "
                       "quantized draw catches it at G2",
    },
    {
        "name": "nc_differential_path",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": ("return _seg_impl(x, int(k), int(segment))",
                   "if float(x.abs().max()) > 1.0e4:\n"
                   "        return _seg_impl(x, int(k), int(segment))\n"
                   "    S = int(segment)\n"
                   "    xh = x[:, : (x.shape[-1] // S // 2) * S].contiguous()\n"
                   "    return _seg_impl(xh, int(k), S)"),
        "bound_gate": "G2",
        "oracle_env": False,
        "expectation": "the §7.1 exploit: branch on a magnitude statistic, full segment "
                       "grid only on 'stress-looking' data, half the segments otherwise. "
                       "Timed tensors ARE checked tensors, so the shrunk output shape "
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
