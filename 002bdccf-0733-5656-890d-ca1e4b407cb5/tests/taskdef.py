"""Taskdef: ragged descending top-k with variable-length segments (SURFACE fwd).

Selection family. Everything category-specific the generic verifier consumes
lives here (protocol: seed/forge/core/taskdef_api.py).

Operator semantics (reference.py is the definition; this is a summary):
    topk_ragged(x[NNZ], seg_offsets[R+1] int64, k)
        -> (values[R, k], indices[R, k] int64)
    x is a 1D flat tensor; seg_offsets is an int64 prefix-sum: row r covers
    the flat slice [seg_offsets[r], seg_offsets[r+1]). Row lengths are
    ARBITRARY (this is the variable-length segments hardness lever, distinct
    from topk_segmented's fixed length S). Every row L_r >= k.
    Descending top-k inside each row; ties broken by LOWER original GLOBAL
    index first (global and local order agree inside a row); -0.0 == +0.0
    one tie group; values are BIT-EXACT gathers of the input element; indices
    are GLOBAL positions in [0, NNZ); int64. EXACT equality; `compare`
    overrides the dense walk.

Hard lever (spec.yaml justifies at length): NO single torch/vendor builtin
computes ragged top-k. The natural denominator pads all rows to the maximum
length with `-inf` and calls `torch.topk` -- correct only when no valid
entry is itself `-inf`, and it wastes device memory in proportion to the
length skew (a batch with one long row and many short ones spends most of
its bandwidth on padding). A from-scratch kernel must extract each row's
top-k from a variable-length slice, and it must NOT pay the pad-to-max
bandwidth tax the anchor pays, so the reachable fraction depends on the
skew of every batch.

Anchor: pad-to-max + torch.topk. TIMING DENOMINATOR ONLY. No vendorable
source tree: VENDOR_LIB_SUBDIRS empty; the FORBIDDEN symbol scan still runs.

Distribution invariance is made real exactly as for the rest of the family
(hidden scale + hidden distribution shape per draw; timed == checked).
The ROW-LENGTH DISTRIBUTION is also hidden per invocation (drawn inside
make_inputs from a uniform range specified by shape["Lmin"]/["Lmax"]) so a
candidate cannot precompute a fixed schedule. process_checks drops the
magnitude-branch lint and keeps no_grader_reference.

STATUS: authored-draft -- needs GPU calibration. TARGET_FRACTION_OF_SOTA
and the golden/controls numbers below are placeholders, NOT measured on
hardware.
"""

import re

import torch

import taskdef_api  # noqa: F401

ENTRY_NAME = "topk_ragged"
REF_NAME = "topk_ragged_ref"
SIGNATURE = "topk_ragged(x[NNZ], seg_offsets[R+1] int64, k) -> (values[R, k], indices[R, k] int64)"
SURFACE = "fwd"
TASK_NAME = "hephaestus/topk_ragged"

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
# workload-dependent (skew of the row-length distribution); calibration
# must overwrite with a value below the oracle's repeated worst case.
TARGET_FRACTION_OF_SOTA = 0.20


def _base_draw_flat(NNZ, seed, scale, device):
    g = torch.Generator(device=device)
    g.manual_seed(seed & 0x7FFFFFFF)
    dist = int(torch.randint(0, 3, (1,), generator=g, device=device).item())
    if dist == 0:
        x = torch.randn(NNZ, generator=g, device=device, dtype=torch.float32)
    elif dist == 1:
        x = torch.randn(NNZ, generator=g, device=device, dtype=torch.float32).exp()
    else:
        x = torch.rand(NNZ, generator=g, device=device, dtype=torch.float32) * 2.0 - 1.0
    return x * scale, g


def _draw_lengths(R, Lmin, Lmax, g, device):
    """Uniform-random row lengths in [Lmin, Lmax]. Sums to NNZ (verifier-
    computed). Every row >= Lmin so the L_r >= k precondition holds as long
    as k <= Lmin (make_inputs asserts)."""
    if Lmax < Lmin:
        raise ValueError(f"Lmax={Lmax} < Lmin={Lmin}")
    lens = torch.randint(
        Lmin, Lmax + 1, (R,), generator=g, device=device, dtype=torch.int64
    )
    return lens


def make_inputs(shape, seed, scale, dtype, device):
    """One reproducible draw. NaN-free by construction.

    shape keys:
      R:    row count
      Lmin: minimum row length (>= k)
      Lmax: maximum row length
      k:    top-k

    Optional (verifier-side, never published):
      quantize: int Q -> values snapped to Q integer levels * scale.
      specials: truthy -> +/-inf, +/-0, +/-subnormals planted in row 0.
    """
    R = int(shape["R"])
    Lmin, Lmax = int(shape["Lmin"]), int(shape["Lmax"])
    k = int(shape["k"])
    if k > Lmin:
        raise ValueError(f"k={k} > Lmin={Lmin}: cannot guarantee L_r >= k")
    _, g = _base_draw_flat(1, seed, 1.0, device)  # discard dummy draw, keep g
    lens = _draw_lengths(R, Lmin, Lmax, g, device)
    seg_offsets = torch.zeros(R + 1, dtype=torch.int64, device=device)
    seg_offsets[1:] = torch.cumsum(lens, dim=0)
    NNZ = int(seg_offsets[-1].item())
    x, g2 = _base_draw_flat(NNZ, seed ^ 0x5A5A5A5A, scale, device)
    if shape.get("quantize"):
        q = int(shape["quantize"])
        lev = torch.floor(torch.rand(NNZ, generator=g2, device=device) * q)
        x = (lev - q // 2) * float(scale)
    if shape.get("specials"):
        L0 = int(lens[0].item())
        x0 = -x[:L0].abs() - 2.0 * float(scale)
        end = min(L0, 106)
        if L0 >= 29:
            x0[0:29] = float("inf")
        if L0 >= 60:
            x0[40:60] = float("-inf")
        if L0 >= 86:
            zeros = torch.tensor([0.0, -0.0], device=device).repeat(8)
            x0[70:86] = zeros
        if L0 >= 106:
            sub = torch.tensor([1e-39, -1e-39], device=device).repeat(8)
            x0[90:106] = sub
        x = torch.cat([x0[:L0], x[L0:]], dim=0)[:NNZ]
        del end
    x = x.to(dtype).contiguous()
    return {"x": x, "seg_offsets": seg_offsets.contiguous(), "k": k}, ()


def sota(args):
    """Production anchor: pad-to-max with -inf, then torch.topk. Timing
    denominator ONLY (its -inf sentinel and tie-break make it wrong for
    correctness when a valid entry is itself -inf; never compared)."""
    x, seg, k = args["x"], args["seg_offsets"], int(args["k"])
    R = int(seg.shape[0]) - 1
    lens = seg[1:] - seg[:-1]
    Lmax = int(lens.max().item())
    padded = torch.full((R, Lmax), float("-inf"), device=x.device, dtype=x.dtype)
    for r in range(R):
        lo, hi = int(seg[r].item()), int(seg[r + 1].item())
        padded[r, : hi - lo] = x[lo:hi]
    v, i_local = torch.topk(padded, k, dim=-1, largest=True, sorted=True)
    i_global = i_local.to(torch.int64) + seg[:-1].view(R, 1)
    return v.contiguous(), i_global.contiguous()


_BITVIEW = {"float32": torch.int32, "bfloat16": torch.int16}


def compare(got, want, dtype_name, shape):
    """EXACT comparison. [] iff equivalent; failure strings otherwise.
    Indices GLOBAL (int64); values BIT equal against reference gather."""
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
                f"global indices differ from the exact per-row "
                f"(value desc, index asc) order: {n} positions "
                f"across {rows} rows"
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
    args, _ = make_inputs(shape, seed, scale, dtype, device)
    args["x"] = torch.full_like(args["x"], (0.5 * float(scale)))
    return args, ()


def _mode_neg_inf_blocks(shape, seed, scale, dtype, device):
    args, _ = make_inputs(shape, seed, scale, dtype, device)
    xf = args["x"].float()
    xf[::11] = float("-inf")
    args["x"] = xf.to(dtype).contiguous()
    return args, ()


def _mode_six_decade(shape, seed, scale, dtype, device):
    args, _ = make_inputs(shape, seed, scale, dtype, device)
    NNZ = args["x"].shape[0]
    g = torch.Generator(device=device)
    g.manual_seed(seed & 0x7FFFFFFF)
    mag = torch.rand(NNZ, generator=g, device=device) * 6.0 - 3.0
    xf = (
        torch.randn(NNZ, generator=g, device=device)
        * torch.pow(10.0, mag)
        * float(scale)
    )
    args["x"] = xf.to(dtype).contiguous()
    return args, ()


def _mode_extreme_skew(shape, seed, scale, dtype, device):
    """One giant row plus many short rows: hardest length skew for a
    pad-to-max scheduler; stability mode gates on finiteness only."""
    R = int(shape["R"])
    Lmin, Lmax, k = int(shape["Lmin"]), int(shape["Lmax"]), int(shape["k"])
    _, g = _base_draw_flat(1, seed, 1.0, device)
    lens = torch.full((R,), Lmin, dtype=torch.int64, device=device)
    lens[0] = Lmax
    seg_offsets = torch.zeros(R + 1, dtype=torch.int64, device=device)
    seg_offsets[1:] = torch.cumsum(lens, dim=0)
    NNZ = int(seg_offsets[-1].item())
    x, _ = _base_draw_flat(NNZ, seed ^ 0x33333333, scale, device)
    return {
        "x": x.to(dtype).contiguous(),
        "seg_offsets": seg_offsets.contiguous(),
        "k": k,
    }, ()


def _mode_dup8(shape, seed, scale, dtype, device):
    args, _ = make_inputs(shape, seed, scale, dtype, device)
    NNZ = args["x"].shape[0]
    g = torch.Generator(device=device)
    g.manual_seed(seed & 0x7FFFFFFF)
    lev = torch.floor(torch.rand(NNZ, generator=g, device=device) * 8.0)
    args["x"] = ((lev - 4.0) * float(scale)).to(dtype).contiguous()
    return args, ()


STABILITY_MODES = {
    "all_equal_plateau": _mode_all_equal,
    "neg_inf_blocks": _mode_neg_inf_blocks,
    "six_decade_range": _mode_six_decade,
    "extreme_length_skew": _mode_extreme_skew,
    "eight_level_duplicates": _mode_dup8,
}

SMOKE_SHAPE = {"R": 32, "Lmin": 128, "Lmax": 512, "k": 8}
DETERMINISM_SHAPE = {"R": 64, "Lmin": 256, "Lmax": 1024, "k": 32, "quantize": 128}
STABILITY_SHAPE = {"R": 64, "Lmin": 512, "Lmax": 2048, "k": 32}


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
    "p1": {"R": 128, "Lmin": 256, "Lmax": 2048, "k": 32, "dtypes": ["bfloat16"]},
    "p2": {"R": 64, "Lmin": 1024, "Lmax": 8192, "k": 64, "dtypes": ["bfloat16"]},
}
BENCH_CHECK_SHAPE = {"R": 16, "Lmin": 128, "Lmax": 512, "k": 16}
BENCH_CHECK_SHAPE_NPOT = {"R": 5, "Lmin": 251, "Lmax": 2003, "k": 23}
BENCH_CHECK_SHAPE_TIES = {"R": 8, "Lmin": 256, "Lmax": 1024, "k": 32, "quantize": 16}


# ---------------------------------------------------------------------------
# Negative controls (Invariant 17). Each arm must zero by EXACTLY its bound
# gate; the from-scratch oracle passes G1/G4 honestly. (Validated at GPU
# calibration; authored-draft.)
# ---------------------------------------------------------------------------

_FALLBACK_SRC = '''\
"""Pure-torch fallback candidate (negative control): no Triton anywhere."""
import torch


def topk_ragged(x, seg_offsets, k):
    k = int(k)
    R = int(seg_offsets.shape[0]) - 1
    off = seg_offsets.tolist()
    out_i = torch.empty(R, k, dtype=torch.int64, device=x.device)
    out_v = torch.empty(R, k, dtype=x.dtype, device=x.device)
    for r in range(R):
        lo, hi = off[r], off[r + 1]
        row = x[lo:hi].float().contiguous()
        row = torch.where(row == 0.0, torch.zeros_like(row), row)
        b = row.view(torch.int32).to(torch.int64)
        u = torch.where(b >= 0, b ^ 0x80000000, (~b) & 0xFFFFFFFF)
        L = hi - lo
        inv = ((1 << 23) - 1) - torch.arange(L, device=x.device, dtype=torch.int64)
        keys = (u << 23) | inv
        for j in range(k):
            mx = keys.max()
            col = ((1 << 23) - 1) - int(mx.item()) & ((1 << 23) - 1)
            g = int(((1 << 23) - 1) - (int(mx.item()) & ((1 << 23) - 1)))
            out_i[r, j] = g + lo
            keys[g] = -1
        out_v[r] = x.gather(0, out_i[r])
    return out_v, out_i
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
        "name": "nc_missing_row_offset",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": ("out_i[r] = local_i + lo\n", "out_i[r] = local_i\n"),
        "bound_gate": "G2",
        "oracle_env": False,
        "expectation": "drops the per-row global offset, so every row's indices "
        "collapse to [0, L_r); for r >= 1 the returned indices point "
        "into the wrong row and the exact G2 comparison fails",
    },
    {
        "name": "nc_fixed_length_assumption",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": (
            "L = int(seg_offsets[r + 1].item()) - lo",
            "L = int(seg_offsets[1].item()) - int(seg_offsets[0].item())",
        ),
        "bound_gate": "G2",
        "oracle_env": False,
        "expectation": "hard-codes row 0's length for every row; on the ragged draws "
        "every row past 0 reads a wrong slice and the exact G2 "
        "comparison fails",
    },
    {
        "name": "nc_tiebreak_reversed",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": (
            "return _topk_impl(x, seg_offsets, int(k))",
            "xr = torch.flip(x, dims=[0])\n"
            "    NNZ = int(x.shape[0])\n"
            "    seg_r = NNZ - torch.flip(seg_offsets, dims=[0])\n"
            "    v, i = _topk_impl(xr, seg_r, int(k))\n"
            "    return v, (NNZ - 1 - i)",
        ),
        "bound_gate": "G2",
        "oracle_env": False,
        "expectation": "identical values, ties broken by HIGHER original global index; "
        "the duplicate-guaranteeing shapes fail the exact index "
        "comparison at G2",
    },
    {
        "name": "nc_nondeterministic_ties",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": (
            "return _topk_impl(x, seg_offsets, int(k))",
            "globals()['_NCALLS'] = globals().get('_NCALLS', 0) + 1\n"
            "    if globals()['_NCALLS'] % 2 == 0:\n"
            "        xr = torch.flip(x, dims=[0])\n"
            "        NNZ = int(x.shape[0])\n"
            "        seg_r = NNZ - torch.flip(seg_offsets, dims=[0])\n"
            "        v, i = _topk_impl(xr, seg_r, int(k))\n"
            "        return v, (NNZ - 1 - i)\n"
            "    return _topk_impl(x, seg_offsets, int(k))",
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
            "return _topk_impl(x, seg_offsets, int(k))",
            "if float(x.abs().max()) > 1.0e4:\n"
            "        return _topk_impl(x, seg_offsets, int(k))\n"
            "    seg_h = seg_offsets.clone()\n"
            "    seg_h[-1] = seg_offsets[len(seg_offsets) // 2]\n"
            "    return _topk_impl(x, seg_h[: len(seg_offsets) // 2 + 1], int(k))",
        ),
        "bound_gate": "G2",
        "oracle_env": False,
        "expectation": "the §7.1 exploit: branch on a magnitude statistic, full row "
        "set only on 'stress-looking' data, half the rows otherwise. "
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
        "expectation": "correct, deterministic pure-torch composition (per-row python "
        "loop), 100% of device time outside candidate Triton kernels: "
        "G2 passes and the written-kernel adoption floor zeroes at G4",
    },
]
