#!/usr/bin/env python3
"""Verifier for the chunked gated-delta-rule bundle.

Score model, in full:

    r_i = t_sota_i / t_candidate_i          per graded shape, clipped to [0, 1]
    R   = geomean_i(r_i)
    score = R                               zero unless every gate passes

The delivered metric is `score`: the raw geomean fraction of the production
kernel, NOT divided by any target. The target is retained only as an
informational figure and no longer gates. Adoption stays a GATE -- a
sub-threshold own-Triton share zeroes the score -- rather than a multiplier.

The denominator is flash-linear-attention's own production Triton kernel, not
eager and not torch.compile. That choice is load-bearing: a project measurement
recorded eager overstating the same kernel by 7.34x geomean on the forward
surface, and KernelBench-Verified reports a frontier model falling from 1.43x to
0.88x once its baseline is corrected. Grading against a weak denominator
measures the denominator.

Per-shape clipping exists because the anchor is a real kernel rather than a
theoretical bound. Where FLA is itself weak at some hidden shape, r_i could
otherwise exceed 1 and let one shape carry the geomean.

DISTRIBUTION INVARIANCE. An audited external run on this exact operator family
inflated a published score 4.3x by detecting that the checker validated a 2x
key-magnitude stress distribution while the benchmark timed nominal inputs, and
switching to cheaper math only on the timed path. A static authored-kernel lint
reported that submission CLEAN. Two properties close it here and both are
required: the input scale is drawn per invocation from a hidden distribution, so
no fixed magnitude threshold separates the regimes, and the timed tensors ARE
the checked tensors, in one process, under one policy.
"""

import argparse
import hashlib
import importlib.util
import io
import json
import keyword
import math
import os
import re
import statistics
import sys
import time
import tokenize
import traceback
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

FORBIDDEN = [
    "fla.ops.gated_delta_rule",
    "fla.ops.kda",
    "fla.ops.chunk_kda",
    "chunk_gated_delta_rule",
    "fused_recurrent_gated_delta_rule",
    "naive_recurrent_gated_delta_rule",
    "chunk_kda",
    "fused_recurrent_kda",
    "naive_chunk_kda",
    "naive_recurrent_kda",
]

# Every input the entry point takes a gradient for. All five are compared against
# the reference; see the G2 block for why the list is explicit rather than derived.
GRAD_NAMES = ("q", "k", "v", "g", "beta")
# bfloat16 and float16 are arXiv:2603.21331 section 4.6 exactly: FP16 1e-2,
# BF16 2e-2.
#
# float32 is NOT the paper's 1e-4, and that is a measurement rather than a
# preference. Setting 1e-4 was attempted and reverted: the ORACLE fails it, and
# the oracle IS the production kernel. Graded against reference.py at the real
# correctness-only shapes, in float32, worst absolute output deviation:
#
#   c3  T=65    B2 H4/HV12 K96  V96    6.243e-5   PASSES 1e-4
#   c1  T=127   B1 H4/HV4  K128 V128   2.392e-4   fails, 46 elements outside
#   c2  T=1000  B3 H8/HV8  K64  V64    2.984e-4   fails, 4497 elements outside
#
# and the gradients miss 1e-4 by another order of magnitude at every shape. Note
# that the deviation rises MONOTONICALLY WITH T while K, V, B and the head count
# all move around it. That is the mechanism showing itself: this operator is a
# sequential recurrence over T timesteps, every chunk-parallel formulation
# reassociates that scan, and the reassociation error accumulates along the
# sequence. It is arithmetic, not implementation quality -- the paper's table is
# calibrated on fused elementwise and matmul kernels, where a fused result differs
# from a reference by a single rounding and there is no sequence to accumulate
# along. Adopting 1e-4 would zero every candidate including the anchor, and would
# fail the bundle's own oracle control on the two longest correctness shapes while
# passing the shortest, which is the signature of a tolerance set below the
# operator's own noise floor rather than of a defect.
#
# The bound is measured instead. Over four shapes spanning T in {256, 512, 1024,
# 333}, K/V in {64, 96, 128} and both HV == H and HV == 3H, counting elements
# outside atol + rtol*|ref| across the output and all five gradients:
#
#   1e-4  ->  81053 outside      5e-4  ->  41 outside
#   1e-3  ->      0 outside      2e-3  ->   0 outside
#
# 1e-3 is the knee; the gate is twice it, which leaves the worst observed output
# deviation (2.984e-4) a factor of 6.7 of headroom. Still an order of magnitude
# tighter than the bfloat16 bound this harness has always used, and float32 is a
# verification dtype that is never timed, so it can zero a submission but can
# never move its speed score.
TOL = {
    "bfloat16": (2.0e-2, 3.125e-2),
    "float16": (1.0e-2, 3.90625e-3),
    "float32": (2.0e-3, 2.0e-3),
}
ADOPTION_FLOOR = 0.60

# G1's structural half. FORBIDDEN above catches an import and a call; it does not
# catch a COPY. A renamed vendored copy of the library's own Triton source carries
# no forbidden symbol, so it scans clean, and its kernels are genuine declared
# triton.jit kernels, so device-time attribution passes it too. Both gates built to
# stop a candidate claiming credit for the library's work look straight past the
# most direct way of doing it.
#
# Calibrated on the sixteen delivered candidates rather than guessed. Normalised
# 32-token shingle containment against the installed library source measures:
#
#   hand-written candidates      0.0000 .. 0.0662   (fourteen of sixteen)
#   the bundle's own reference    0.1171            (not a candidate; upper bound
#                                                    on honest structural overlap)
#   vendored copies               0.8431, 0.8673    (0ad2a69d iter-07, which said
#                                                    so in its own trajectory, and
#                                                    f5fb3504 iter-05, which did
#                                                    not and was never identified)
#   a byte copy of a library file 1.0000
#
# 0.40 sits an order of magnitude above the highest honest candidate and half the
# lowest vendored one. Identifiers are erased before shingling, so renaming every
# symbol does not move the number; that is the entire point.
VENDOR_NGRAM = 32
VENDOR_CONTAINMENT_CEILING = 0.40
# The operator family plus the shared chunk/scan helpers a vendored copy pulls in
# with it. Anything the library ships elsewhere is not what this gate is about.
VENDOR_LIB_SUBDIRS = ("gated_delta_rule", "kda", "common", "utils")
_TOKEN_SKIP = frozenset(
    {
        tokenize.COMMENT,
        tokenize.NL,
        tokenize.NEWLINE,
        tokenize.INDENT,
        tokenize.DEDENT,
        tokenize.ENCODING,
        tokenize.ENDMARKER,
    }
)


def declared_jit_names(path: Path) -> set:
    """Names decorated with @triton.jit in the submitted source."""
    import ast

    tree = ast.parse(path.read_text())
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for d in node.decorator_list:
                src = ast.unparse(d)
                if "triton.jit" in src or src.endswith("jit"):
                    out.add(node.name)
    return out


def _kernel_is_mine(key: str, names: set) -> bool:
    """Is this profiler entry a launch of a Triton kernel THIS module declares?

    Substring containment was the rule here, and it was exploitable. Measured
    2026-08-14 against real profiler keys: a module declaring `@triton.jit def
    gemm(...)` matched `ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x3_nn`
    -- cuBLAS -- and one declaring `reduce` matched
    `at::native::reduce_kernel<512, 1, ReduceOp<float>>`. So a candidate could
    declare one trivial kernel, route the arithmetic through torch, and have the
    library's device time counted as its own. `gemm` and `reduce` are not
    adversarial names; they are what you would call those kernels anyway. The gate
    that exists to stop delegation was defeated by naming.

    Two conditions now, both required. The entry must not look like a vendor or
    ATen symbol, and the declared name must match the key exactly or as its
    leading component -- Triton appends specialisation suffixes after an
    underscore, it does not embed the name mid-string.
    """
    import re as _re

    if "::" in key or key.startswith("void ") or "<" in key:
        return False  # C++/ATen mangling
    if _re.match(r"^(ampere|turing|volta|hopper|maxwell|pascal|sm\d+)_", key):
        return False  # cuBLAS/cuDNN generated names
    if _re.search(r"(cutlass|cublas|cudnn|nvjet)", key, _re.I):
        return False
    return any(key == n or key.startswith(n + "_") for n in names)


def kernel_adoption(torch, fn, path: Path, inp):
    """Share of graded device time spent in Triton kernels this module declares."""
    names = declared_jit_names(path)
    if not names:
        return 0.0, {
            "declared": [],
            "note": "the submitted module declares no triton.jit kernel",
        }
    from torch.profiler import ProfilerActivity, profile

    for _ in range(3):
        fwd_bwd(fn, inp, torch)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fwd_bwd(fn, inp, torch)
        torch.cuda.synchronize()
    total = mine = 0.0
    for e in prof.key_averages():
        dev = float(getattr(e, "self_device_time_total", 0) or 0)
        if dev <= 0:
            continue
        total += dev
        if _kernel_is_mine(e.key, names):
            mine += dev
    return (mine / total if total else 0.0), {
        "declared": sorted(names),
        "device_us_total": round(total, 1),
        "device_us_declared": round(mine, 1),
    }


class CandidateExited(Exception):
    """The candidate called sys.exit during import or execution."""


def load(path: Path, name: str):
    """Import the candidate with SystemExit trapped.

    Without this, a candidate that calls sys.exit(0) on import terminates the
    grader before it writes a score, and an unwritten score file is not
    distinguishable from a clean run by anything downstream. SystemExit does not
    inherit from Exception, so a bare `except Exception` never sees it.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except SystemExit as e:
        raise CandidateExited(
            f"candidate raised SystemExit({e.code}) during import"
        ) from None
    return mod


def forbidden_hits(path: Path) -> list:
    src = path.read_text()
    # comments stripped: reading FLA is allowed and encouraged, calling it is not
    code = "\n".join(l.split("#")[0] for l in src.splitlines())
    return [s for s in FORBIDDEN if re.search(r"\b" + re.escape(s) + r"\b", code)]


def normalized_tokens(src: str) -> list:
    """Python source as a rename-invariant token stream.

    Every identifier collapses to a single placeholder, every literal to a
    per-kind placeholder, and comments and formatting disappear entirely. What
    survives is control flow, call structure and operator sequence -- the part of
    a copy that renaming cannot alter. Reading the library is explicitly allowed,
    so a check that could be defeated by find-and-replace would measure nothing.
    """
    out = []
    try:
        toks = tokenize.generate_tokens(io.StringIO(src).readline)
        for t in toks:
            if t.type in _TOKEN_SKIP:
                continue
            if t.type == tokenize.STRING:
                out.append("S")
            elif t.type == tokenize.NUMBER:
                out.append("0")
            elif t.type == tokenize.NAME:
                out.append(t.string if keyword.iskeyword(t.string) else "N")
            else:
                out.append(t.string)
    except (tokenize.TokenError, IndentationError, SyntaxError, ValueError):
        # A source this cannot tokenise is one G0 is about to reject anyway.
        return out
    return out


def _shingles(toks: list, n: int) -> set:
    return {hash(tuple(toks[i : i + n])) for i in range(len(toks) - n + 1)}


def library_shingles(n: int):
    """Shingles of the installed library's own source, or None if unreadable.

    Located through the import system rather than a hardcoded path, so it follows
    the library actually installed in the image instead of one the Dockerfile
    happened to clone. `find_spec` rather than `import`: this needs the package's
    location on disk, not its module object, and resolving it must not depend on
    the package importing cleanly. Shingles never cross a file boundary.
    """
    try:
        spec = importlib.util.find_spec("fla")
        if spec is None or not spec.origin:
            return None
        ops = Path(spec.origin).resolve().parent / "ops"
    except Exception:  # noqa: BLE001
        return None
    if not ops.is_dir():
        return None
    acc, files = set(), 0
    for sub in VENDOR_LIB_SUBDIRS:
        d = ops / sub
        if not d.is_dir():
            continue
        for f in sorted(d.rglob("*.py")):
            try:
                acc |= _shingles(normalized_tokens(f.read_text(errors="replace")), n)
            except OSError:
                continue
            files += 1
    return (acc, files) if acc else None


def vendored_containment(path: Path) -> dict:
    """Share of the candidate's structure that appears verbatim in the library.

    Reported always, so the number is on the record whether or not it trips. If
    the library source cannot be read this records `available: false` and does
    NOT gate: absence of evidence is not evidence of vendoring, and the same
    mistake in the opposite direction -- missing data scored as a pass -- is what
    left the anti-cheat layer inert on the one iteration that ever scored. The
    candidate cannot manufacture that state: it runs as `agent` and the library
    tree is root-owned.
    """
    lib = library_shingles(VENDOR_NGRAM)
    if lib is None:
        return {
            "available": False,
            "gated": False,
            "reason": "the installed library source could not be read",
        }
    lib_grams, files = lib
    try:
        cand = _shingles(
            normalized_tokens(path.read_text(errors="replace")), VENDOR_NGRAM
        )
    except OSError:
        return {"available": False, "gated": False, "reason": "candidate unreadable"}
    if not cand:
        # Shorter than one shingle. Too small to be a copy of anything.
        return {
            "available": True,
            "gated": False,
            "containment": 0.0,
            "ngram": VENDOR_NGRAM,
            "ceiling": VENDOR_CONTAINMENT_CEILING,
            "candidate_shingles": 0,
            "library_files": files,
        }
    share = len(cand & lib_grams) / len(cand)
    return {
        "available": True,
        "gated": share > VENDOR_CONTAINMENT_CEILING,
        "containment": round(share, 6),
        "ngram": VENDOR_NGRAM,
        "ceiling": VENDOR_CONTAINMENT_CEILING,
        "candidate_shingles": len(cand),
        "library_files": files,
    }


def make_inputs(torch, B, T, H, HV, K, V, dtype, gen, mode="nominal"):
    """One draw. The scale is random and hidden, and this same draw is both
    checked for correctness and timed, so no branch can tell the two apart.

    `mode` selects the stability stage's adversarial variants. It defaults to the
    nominal draw, which is the ONLY one used for anything that is timed: an
    adversarial input must never reach the speed measurement, or the score would
    depend on how a candidate behaves on a distribution it is not graded for.
    """
    scale = float(torch.empty((), device="cuda").uniform_(0.35, 2.5, generator=gen))
    f = lambda *s: (
        torch.randn(*s, device="cuda", dtype=torch.float32, generator=gen) * scale
    )  # noqa: E731
    # q and k are L2-normalised along the head dimension. The delta rule is a
    # contraction only when beta * ||k||^2 stays bounded; unnormalised keys at
    # a scale of 2.5 with K=128 diverge to inf, which is the operator's own
    # behaviour and not an implementation defect. Real callers normalise, and
    # FLA carries use_qk_l2norm_in_kernel for exactly this reason. This holds in
    # every mode below: the probes stress the gate, the write strength and the
    # value magnitude, never the contraction condition itself.
    qn = torch.nn.functional.normalize(f(B, T, H, K), dim=-1)
    kn = torch.nn.functional.normalize(f(B, T, H, K), dim=-1)
    vv = f(B, T, HV, V)
    gg = -torch.rand(B, T, HV, device="cuda", dtype=torch.float32, generator=gen) * 0.5
    bb = (
        torch.rand(B, T, HV, device="cuda", dtype=torch.float32, generator=gen) * 0.9
        + 0.05
    )

    # Stage 3. Each probe drives one term of the recurrence to an end of its
    # range, which is where a kernel that folded a term into a fast path, or that
    # accumulated in the wrong precision, stops agreeing with the reference.
    if mode == "no_decay":  # exp(g) -> 1: nothing decays, the state accumulates
        gg = torch.full_like(gg, -1e-6)
    elif mode == "max_decay":  # the strongest decay the graded range allows
        gg = torch.full_like(gg, -0.5)
    elif mode == "beta_full":  # the write replaces the prediction outright
        bb = torch.ones_like(bb)
    elif mode == "v_dynamic":  # six decades of magnitude inside one tensor
        vv = vv * torch.pow(10.0, torch.empty_like(vv).uniform_(-3, 3, generator=gen))
    elif mode == "v_underflow":  # values near the bottom of the dtype's range
        vv = vv * 1e-4
    elif mode != "nominal":
        raise ValueError(f"unknown input mode {mode!r}")

    q = qn.to(dtype).requires_grad_()
    k = kn.to(dtype).requires_grad_()
    v = vv.to(dtype).requires_grad_()
    g = gg.requires_grad_()
    beta = bb.to(dtype).requires_grad_()
    return dict(q=q, k=k, v=v, g=g, beta=beta), scale


def fwd_bwd(fn, inp, torch):
    for t in inp.values():
        if t.grad is not None:
            t.grad = None
    out = fn(**inp)
    out.sum().backward()
    return out


class _Clock:
    """Timing primitives captured BEFORE the candidate is imported.

    A candidate that runs `torch.cuda.Event.elapsed_time = lambda s, o: 1.0` at
    import (or patches Event / .record / torch.cuda.synchronize) forces every
    device time to a constant and fabricates the ratio, so an arbitrarily slow
    kernel is paid full reward while the correctness gates pass honestly. The
    timing runs in the same interpreter that imports the candidate, so it cannot
    be moved out of reach; instead this binds the ORIGINAL callables in main()
    before any candidate byte executes and invokes them UNBOUND. A later
    reassignment of the class attribute rebinds the name but not the object held
    here.
    """

    __slots__ = ("_Event", "_record", "_elapsed", "sync")

    def __init__(self, torch):
        self._Event = torch.cuda.Event
        self._record = torch.cuda.Event.record
        self._elapsed = torch.cuda.Event.elapsed_time
        self.sync = torch.cuda.synchronize

    def event(self):
        return self._Event(enable_timing=True)

    def record(self, ev):
        self._record(ev)

    def elapsed(self, a, b) -> float:
        return self._elapsed(a, b)


def one_sample(clock, torch, fn, inp, flush) -> float:
    """One L2-cold device-time sample of a single forward-plus-backward."""
    flush.zero_()
    clock.sync()
    a, b = clock.event(), clock.event()
    clock.record(a)
    fwd_bwd(fn, inp, torch)
    clock.record(b)
    clock.sync()
    return clock.elapsed(a, b)


def timed_ratio(clock, torch, cand, sota, inp, trials=15, budget_s=50.0):
    """Median of PAIRED, INTERLEAVED, ORDER-BALANCED ratio samples.

    The graded quantity is a ratio, so it must be measured as a ratio. Timing the
    candidate to completion and then the anchor to completion measures them in
    two different places on the clock-and-thermal curve, and every drift between
    those places lands entirely in the ratio.

    Measured cost of getting this wrong, on the ORACLE, whose candidate IS the
    anchor and whose every per-shape ratio is therefore known to be exactly 1.0:
    one shape read 0.7920, a 26 percent gap between a kernel and itself, and the
    geometric mean came out at 0.939257 against a 0.95 target, so the oracle
    failed its own control. Across three runs of identical bytes the oracle
    measured 0.939257, 0.965076 and 0.988552.

    Two properties fix it. Samples are INTERLEAVED, so both sides see the same
    drift within a few hundred microseconds of each other and it divides out. And
    the order is ALTERNATED, so any residual first-versus-second position effect
    enters with opposite sign on odd and even trials instead of accumulating.

    Note what this does NOT fix, because it cannot be fixed by measuring better:
    r_i = min(ratio, 1) is an asymmetric estimator. Upward noise is truncated at
    1.0 while downward noise is kept in full, so E[geomean] < 1 strictly whenever
    the candidate is near the anchor, and it falls further the more shapes are
    graded. In the failing run 3 of 8 shapes were clipped, discarding exactly the
    excursions that would have balanced the 5 below. The clip is deliberate - the
    score is a saturating FRACTION of a production kernel and beating it earns no
    extra credit - so the residual bias is absorbed by setting the target below
    the oracle's repeated worst case rather than at its mean.
    """
    flush = torch.empty(128 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    for _ in range(3):
        fwd_bwd(cand, inp, torch)
        fwd_bwd(sota, inp, torch)
    clock.sync()

    t0 = time.time()
    fwd_bwd(cand, inp, torch)
    clock.sync()
    one = time.time() - t0
    if one * 2 * trials > budget_s:
        trials = max(3, min(trials, int(budget_s / (2 * one)) or 3))

    for _ in range(3):
        fwd_bwd(cand, inp, torch)
        fwd_bwd(sota, inp, torch)
    clock.sync()

    ratios, cands, sotas = [], [], []
    for i in range(trials):
        if i % 2 == 0:
            tc = one_sample(clock, torch, cand, inp, flush)
            ts = one_sample(clock, torch, sota, inp, flush)
        else:
            ts = one_sample(clock, torch, sota, inp, flush)
            tc = one_sample(clock, torch, cand, inp, flush)
        cands.append(tc)
        sotas.append(ts)
        ratios.append(ts / tc)
    return (
        statistics.median(ratios),
        statistics.median(cands),
        statistics.median(sotas),
        trials,
    )


def environment_record(torch) -> dict:
    """What produced this number. None of it was recorded before 2026-08-12, so a
    score could not be reproduced or compared across hosts."""
    env = {}
    try:
        env["gpu_name"] = torch.cuda.get_device_name(0)
        env["gpu_count"] = torch.cuda.device_count()
        cap = torch.cuda.get_device_capability(0)
        env["gpu_capability"] = f"sm_{cap[0]}{cap[1]}"
    except Exception:  # noqa: BLE001
        env["gpu_name"] = "unavailable"
    env["torch_version"] = str(getattr(torch, "__version__", ""))
    env["cuda_version"] = str(
        getattr(getattr(torch, "version", None), "cuda", "") or ""
    )
    try:
        import triton

        env["triton_version"] = str(getattr(triton, "__version__", ""))
    except Exception:  # noqa: BLE001
        env["triton_version"] = ""
    return env


def metrics_record(
    per_shape: dict, R: float, target: float, adopt, starter_fraction
) -> dict:
    """The headline figures, so nobody reconstructs them from per_shape by hand.

    `speedup_vs_starter` is the number that says what the agent actually achieved
    and that nothing reported until now. It is a ratio of two MEASURED quantities:
    this run's geomean, and the starter's own fraction, which the NC1 control
    measures against the same graded set on the same hardware.

    `score_log_scaled` is recorded and GATES NOTHING. Kernel work compounds
    multiplicatively while the shipped reward is linear in the ratio, so every
    attempt ever measured occupies about a tenth of the [0,1] range. The
    compiler-RL literature log-transforms speedup for exactly this reason. Emitting
    it alongside the real score costs nothing and means a later decision about
    the score model can be made on a recorded series rather than on an argument.
    """
    cand = [v["candidate_ms"] for v in per_shape.values() if v.get("candidate_ms")]
    sota = [v["sota_ms"] for v in per_shape.values() if v.get("sota_ms")]
    m = {
        "fraction_of_sota": round(R, 6),
        "speedup_vs_production": round(R, 6),
        "slowdown_vs_production": round(1.0 / R, 4) if R > 0 else None,
        "configurations_timed": len(per_shape),
        "clipped_shapes": sum(1 for v in per_shape.values() if v.get("clipped")),
        "score_formula": "geomean_i(min(t_prod/t_cand,1)); 0 unless every gate passes",
        "target_fraction_of_sota": target,
        "target_fraction_of_sota_note": "informational only; the score no longer "
        "divides by or gates on the target",
    }
    if starter_fraction and R > 0:
        m["speedup_vs_starter"] = round(R / starter_fraction, 4)
        m["starter_fraction_of_sota"] = starter_fraction
        try:
            lo, hi = math.log(R / starter_fraction), math.log(target / starter_fraction)
            m["score_log_scaled"] = round(max(0.0, min(1.0, lo / hi)), 6)
            m["score_log_scaled_note"] = "diagnostic only; the reward is `score`"
        except ValueError:
            pass
    if cand:
        m["candidate_ms_total"] = round(sum(cand), 4)
        m["candidate_ms_geomean"] = round(statistics.geometric_mean(cand), 6)
    if sota:
        m["production_ms_total"] = round(sum(sota), 4)
        m["production_ms_geomean"] = round(statistics.geometric_mean(sota), 6)
    if per_shape:
        best = max(per_shape.items(), key=lambda kv: kv[1].get("fraction_of_sota", 0))
        worst = min(per_shape.items(), key=lambda kv: kv[1].get("fraction_of_sota", 1))
        m["best_shape"] = {
            "shape": best[0],
            "fraction": best[1].get("fraction_of_sota"),
        }
        m["worst_shape"] = {
            "shape": worst[0],
            "fraction": worst[1].get("fraction_of_sota"),
        }
    if adopt is not None:
        m["written_kernel_adoption"] = round(adopt, 6)
    return m


def close(torch, got, want, dtype_name):
    atol, rtol = TOL[dtype_name]
    if got is None or want is None:
        return False, "missing tensor"
    if not torch.isfinite(got).all():
        return False, "non-finite values in candidate output"
    d = (got.float() - want.float()).abs()
    lim = atol + rtol * want.float().abs()
    bad = int((d > lim).sum())
    return bad == 0, f"{bad} elements outside tolerance, max abs {float(d.max()):.3e}"


# ---- the five-stage correctness harness ---------------------------------
#
# arXiv:2603.21331 section 4 specifies five stages: smoke, a shape sweep across
# 10+ configurations and three dtypes, numerical stability under adversarial
# inputs, determinism over three runs, and non-power-of-two edge cases. This
# bundle previously collapsed all of it into one loop, which is the largest
# single divergence from the paper it is built against: there was no smoke stage,
# no fp32 verification set, no adversarial probe at all, and no determinism
# check. The stages are named in the score so the reader can see which one a
# candidate died on rather than a flat list of failures.
#
# Stage order is chosen for FAST FAILURE. Smoke, determinism and stability are
# seconds each and need no timing; the sweep carries the whole paired-timing
# cost. A candidate that is broken, non-deterministic or unstable now finds out
# in seconds instead of after a full timed sweep.
SMOKE_SHAPE = {"B": 1, "T": 64, "H": 2, "HV": 2, "K": 32, "V": 32}
STABILITY_MODES = ("no_decay", "max_decay", "beta_full", "v_dynamic", "v_underflow")
STABILITY_SHAPE = {"B": 2, "T": 512, "H": 4, "HV": 4, "K": 64, "V": 64}
DETERMINISM_SHAPE = {"B": 2, "T": 512, "H": 4, "HV": 4, "K": 64, "V": 64}
DETERMINISM_RUNS = 3


def _grads_of(inp):
    return {
        n: (inp[n].grad.clone() if inp[n].grad is not None else None)
        for n in GRAD_NAMES
    }


def _run_candidate(torch, fn, inp):
    """Call the candidate with the fault flag latched and the fault forced to surface.

    An illegal access inside a candidate kernel is asynchronous; without the sync
    it raises at the next synchronising call, which is usually outside whatever
    handler is watching, and the fault gets attributed to the harness.
    """
    global CANDIDATE_ENTERED
    CANDIDATE_ENTERED = True
    out = fwd_bwd(fn, inp, torch)
    grads = _grads_of(inp)
    torch.cuda.synchronize()
    return out, grads


def _compare(torch, got, got_grads, want, want_grads, dt_name, label):
    """One (output + five gradients) comparison. Returns a list of failure strings."""
    fails = []
    ok_o, det_o = close(torch, got, want, dt_name)
    if not ok_o:
        fails.append(f"{label}: out[{det_o}]")
    for n in GRAD_NAMES:
        cg, rg = got_grads.get(n), want_grads.get(n)
        if cg is None or rg is None:
            fails.append(f"{label}: d{n}[missing gradient]")
            continue
        ok_g, det_g = close(torch, cg, rg, dt_name)
        if not ok_g:
            fails.append(f"{label}: d{n}[{det_g}]")
    return fails


def _failure_summary(fails: list) -> str:
    """Describe a correctness failure by CONFIGURATION and by TENSOR, separately.

    `fails` holds one string per failed tensor, and each is prefixed with the
    configuration it came from ("smoke/bfloat16: dq[...]"). Reporting len(fails)
    as a count of configurations conflates the two, and the two mean opposite
    things: five tensors failing on one shape says the backward is comprehensively
    broken, five shapes each failing one tensor says there is an edge case. On the
    fifty-iteration run of 4c3d4af6 the dominant failure was every gradient
    non-finite on the single smoke shape, and it was reported as "5
    configuration(s)" -- understating it, in the very string the refinement loop
    hands back to the agent as `G2 FAILED (5 cfg)`.
    """
    cfgs = []
    for f in fails:
        c = str(f).split(":", 1)[0].strip()
        if c not in cfgs:
            cfgs.append(c)
    n_c, n_t = len(cfgs), len(fails)
    shown = ", ".join(cfgs[:3]) + (f" and {n_c - 3} more" if n_c > 3 else "")
    return (
        f"{n_c} configuration{'' if n_c == 1 else 's'} ({shown}), "
        f"{n_t} failed tensor{'' if n_t == 1 else 's'}"
    )


def stage_smoke(torch, cand, ref, seed_base, gen):
    """Stage 1. One small input, run before anything expensive.

    Catches an import that works but a kernel that does not: compile errors,
    shape mismatches and gross numerical bugs, in about a second, instead of
    after a full timed sweep.
    """
    s, dt_name = SMOKE_SHAPE, "bfloat16"
    gen.manual_seed(seed_base + 9001)
    inp, _ = make_inputs(
        torch,
        s["B"],
        s["T"],
        s["H"],
        s["HV"],
        s["K"],
        s["V"],
        getattr(torch, dt_name),
        gen,
    )
    try:
        got, got_g = _run_candidate(torch, cand, inp)
    except Exception as e:  # noqa: BLE001
        return [f"smoke/{dt_name}: candidate raised {type(e).__name__}: {e}"], {}
    want = fwd_bwd(ref, inp, torch)
    fails = _compare(
        torch, got, got_g, want, _grads_of(inp), dt_name, f"smoke/{dt_name}"
    )
    return fails, {"shape": dict(s), "dtype": dt_name}


def stage_determinism(torch, cand, seed_base, gen):
    """Stage 4. Same input, three runs, bitwise identical output and gradients.

    Catches races in parallel reductions and non-deterministic atomics. This is a
    property of the CANDIDATE against ITSELF -- the reference is not involved --
    so it cannot be affected by tolerance. A kernel that is not reproducible is
    not measurable: the speed number would be attached to one of several answers.
    """
    s, dt_name = DETERMINISM_SHAPE, "bfloat16"
    gen.manual_seed(seed_base + 9002)
    inp, _ = make_inputs(
        torch,
        s["B"],
        s["T"],
        s["H"],
        s["HV"],
        s["K"],
        s["V"],
        getattr(torch, dt_name),
        gen,
    )
    runs = []
    try:
        # Warm up first, and discard it. Triton autotune benchmarks several
        # configurations on the first launch, so an un-warmed first call can
        # return the output of a different configuration than the calls after it
        # and read as non-determinism when the kernel is perfectly reproducible.
        _run_candidate(torch, cand, inp)
        for _ in range(DETERMINISM_RUNS):
            out, grads = _run_candidate(torch, cand, inp)
            runs.append((out.clone(), grads))
    except Exception as e:  # noqa: BLE001
        return [f"determinism/{dt_name}: candidate raised {type(e).__name__}: {e}"], {}

    fails, first_out, first_g = [], runs[0][0], runs[0][1]
    for i, (out, grads) in enumerate(runs[1:], start=2):
        if not torch.equal(first_out, out):
            d = (first_out.float() - out.float()).abs().max()
            fails.append(
                f"determinism/{dt_name}: run {i} output differs from run 1, "
                f"max abs {float(d):.3e}"
            )
        for n in GRAD_NAMES:
            a, b = first_g.get(n), grads.get(n)
            if a is None or b is None or not torch.equal(a, b):
                fails.append(f"determinism/{dt_name}: run {i} d{n} differs from run 1")
    return fails, {"runs": DETERMINISM_RUNS, "shape": dict(s), "dtype": dt_name}


def stage_stability(torch, cand, ref, seed_base, gen):
    """Stage 3. Adversarial inputs, each driving one term to an end of its range.

    WHAT THIS GATES ON, and why it is not tolerance. The paper's own stability
    stage checks NaN/Inf, and measurement here says that is the only defensible
    criterion: run against the delivered oracle -- which IS the production kernel
    -- the `v_dynamic` probe puts 3285 elements of dq outside bf16 tolerance and
    `no_decay` puts 102 elements of dg outside it. Those are not defects. Six
    decades of value magnitude in bf16, and a gate so close to 1 that the state
    accumulates over the whole sequence, make a chunked reordering of the
    arithmetic legitimately disagree with a sequential fp32 scan. A stage the
    production kernel cannot pass would zero every honest candidate and would
    have to be waived for the oracle, which is how a gate stops measuring
    anything.

    So the obligation is STABILITY, which is what the stage is named for: do not
    crash, and do not produce a non-finite value where the reference is finite. A
    candidate that returns NaN under an extreme gate has a real defect no
    tolerance argument excuses. The tolerance comparison is still computed and
    recorded per probe, as a diagnostic the reader can see, and never gates.

    Nothing here is ever timed.
    """
    s, dt_name = STABILITY_SHAPE, "bfloat16"
    fails, detail = [], {}
    for i, mode in enumerate(STABILITY_MODES):
        gen.manual_seed(seed_base + 9100 + i * 7)
        inp, _ = make_inputs(
            torch,
            s["B"],
            s["T"],
            s["H"],
            s["HV"],
            s["K"],
            s["V"],
            getattr(torch, dt_name),
            gen,
            mode=mode,
        )
        want = fwd_bwd(ref, inp, torch)
        want_g = _grads_of(inp)
        if not torch.isfinite(want).all():
            detail[mode] = {
                "verdict": "skipped",
                "note": "the reference is not finite on this probe",
            }
            continue
        try:
            got, got_g = _run_candidate(torch, cand, inp)
        except Exception as e:  # noqa: BLE001
            fails.append(f"stability/{mode}: candidate raised {type(e).__name__}: {e}")
            detail[mode] = {
                "verdict": "raised",
                "note": f"{type(e).__name__}: {e}"[:200],
            }
            continue

        # The gating half: finite where the reference is finite.
        nonfinite = []
        if not torch.isfinite(got).all():
            nonfinite.append("out")
        for n in GRAD_NAMES:
            cg, rg = got_g.get(n), want_g.get(n)
            if cg is None:
                nonfinite.append(f"d{n}[missing]")
            elif (
                rg is not None
                and torch.isfinite(rg).all()
                and not torch.isfinite(cg).all()
            ):
                nonfinite.append(f"d{n}")
        if nonfinite:
            fails.append(
                f"stability/{mode}: non-finite values in {', '.join(nonfinite)} "
                f"where the reference is finite"
            )

        # The recorded half: how far it drifted, gating nothing.
        drift = len(_compare(torch, got, got_g, want, want_g, dt_name, mode))
        detail[mode] = {
            "verdict": "non-finite" if nonfinite else "stable",
            "tensors_outside_tolerance": drift,
            "note": "tolerance is recorded here, never gated; see the docstring",
        }
    return fails, {
        "probes": detail,
        "dtype": dt_name,
        "shape": dict(s),
        "gates_on": "finiteness and not raising, never tolerance",
    }


_SLOT_RE = re.compile(r"TASK-CANARY-SLOT-(\d+)\s+([0-9a-f]{64})")
_PLACEHOLDER = "0" * 64
CANDIDATE_ENTERED = False

# The fatal handler at the bottom cannot see main()'s locals, so a crash discarded
# everything already established: which bytes were graded, and which gates had
# already passed. Measured 2026-08-12: every crashed iteration recorded an empty
# candidate_sha256 and no checks at all, which left the fault unattributable from
# the artifact alone. This aliases the live result dict so the handler can carry
# that evidence into the score it writes.
PARTIAL: dict = {}


def _canary_tokens(text: str) -> list[str]:
    found = sorted(_SLOT_RE.findall(text), key=lambda m: int(m[0]))
    toks = [t for _, t in found]
    return [] if (not toks or any(t == _PLACEHOLDER for t in toks)) else toks


def _is_authored_oracle(path: Path) -> bool:
    """Whether the graded bytes ARE the private oracle this bundle froze.

    The oracle carries this bundle's canary tokens, derived from its canonical
    content hash and planted only into artifacts Harbor mounts for the oracle
    phase. An agent never sees that tree and cannot derive the tokens, so a
    candidate cannot carry them; solve.sh installs the oracle byte for byte, so
    the oracle always does. This replaces an environment variable, which exempts
    whoever sets it and therefore could not be shipped in task.toml, with a
    property of the graded bytes themselves.
    """
    try:
        mine = _canary_tokens(
            Path(__file__).read_text(encoding="utf-8", errors="replace")
        )
        if not mine or not path.is_file():
            return False
        return (
            _canary_tokens(path.read_text(encoding="utf-8", errors="replace")) == mine
        )
    except OSError:
        return False


def main() -> int:
    # CANDIDATE_ENTERED is no longer touched here: every candidate call goes
    # through _run_candidate, which owns the latch.
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", type=Path, default=Path("/workspace/kernel.py"))
    ap.add_argument("--reference", type=Path, default=Path("/workspace/reference.py"))
    ap.add_argument("--out", type=Path, default=Path("/logs/verifier/score.json"))
    args = ap.parse_args()

    # A candidate can terminate this process in ways no `except` can see. A
    # measured case: NC5 delegated the T=2048 scan to torch.compile and the
    # container died with rc 139, SIGSEGV, so the bottom-of-file handler never
    # ran and NO score file was written at all. An unwritten score is the worst
    # failure mode this project has, because absence of a result is missing data
    # and missing data has already been mistaken for evidence of difficulty once
    # (a dead harness scoring 0/25 selected SHIP). So the zero is written FIRST,
    # before any candidate code is imported, and every return path overwrites it.
    # verifier_incomplete stays true only if control never reached emit().
    write_score(
        {
            "score": 0.0,
            "process": 0.0,
            "verifier_incomplete": True,
            "reason": "verifier did not complete: the candidate terminated the "
            "grading process before any verdict was reached",
        },
        args.out,
    )

    # judge_unavailable is initialised FALSE rather than written only when it goes
    # true. A field that exists only in the failure case cannot be asserted on the
    # healthy path: the compiled test resolves it and fails for every submission the
    # judge actually reached, which reads as a broken harness when nothing is wrong.
    # emit_reward.py already defaulted it, so score.json and reward.json disagreed.
    result = {
        "score": 0.0,
        "process": 0.0,
        "reason": "",
        "checks": {},
        "verifier_incomplete": False,
        "judge_unavailable": False,
    }
    globals()["PARTIAL"] = result

    # Preserve WHAT WAS GRADED, and fingerprint it.
    #
    # Harbor tears the agent container down after the verifier phase and its
    # artifact export came back empty on the smoke runs, so from outside the
    # trial there is no way to tell a solver that rewrote the kernel from one
    # that never touched it. The pilot's liveness gate depends on exactly that
    # distinction: an attempt where nothing was edited is missing data, and
    # missing data must never aggregate as evidence that the task is hard. The
    # digest is compared host-side against the frozen starter, so the gate needs
    # no cooperation from the agent container.
    #
    # /logs/verifier is the one directory Harbor downloads, which is why the copy
    # lands beside score.json rather than in artifacts/.
    if args.candidate.is_file():
        src = args.candidate.read_bytes()
        result["candidate_sha256"] = hashlib.sha256(src).hexdigest()
        result["candidate_bytes"] = len(src)
        try:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            (args.out.parent / "candidate_kernel.py").write_bytes(src)
        except OSError:
            pass
    else:
        result["candidate_sha256"] = ""
    cfg = json.loads((HERE / "hidden_shapes.json").read_text())
    tgt = json.loads((HERE / "target.json").read_text())
    TARGET = tgt["target_fraction_of_sota"]

    # Drained and closed FIRST, before any third-party import. HERE is on
    # sys.path, so a shadow module resolved ahead of torch or fla would execute
    # during those imports; draining here means fd 3 is already gone by then and
    # no import-time code can read the secret to forge a completion tag.
    _LATE["auth"] = _drain_auth_fd()

    import torch
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    # Captured before load() runs a single candidate byte, so a candidate that
    # patches torch.cuda.Event or synchronize at import cannot reach the
    # primitives the graded ratio is measured with.
    _clock = _Clock(torch)
    _fixtures_before = _fixture_digests()

    # emit() finishes the record on every early-return path and needs these.
    _LATE.update({"args": args, "torch": torch})

    # Recorded HERE, not beside the metrics, because a run that dies at G2 has an
    # environment too and that is exactly when you want it. Until 2026-08-13 this
    # was written only on the path that reached timing, so on the fifty-iteration
    # run of 4c3d4af6 the 36 iterations that failed early carried no GPU, torch,
    # triton or CUDA version at all -- the ones a reader is most likely to want to
    # attribute. It costs nothing and cannot fail: every field is a version string.
    result["environment"] = environment_record(torch)

    by_canary = _is_authored_oracle(args.candidate)
    by_env = os.environ.get("ORACLE_MODE") == "1"
    oracle_mode = by_canary or by_env
    result["oracle_mode"] = oracle_mode
    result["oracle_route"] = "canary" if by_canary else ("env" if by_env else "")

    # ---- G1 forbidden symbols, by name AND by structure -------------------
    # One gate, two halves. The name half rejects calling the library; the
    # structural half rejects shipping a renamed copy of it. Both are the same
    # obligation -- do not claim credit for the library's implementation -- and a
    # submission that fails either has not written a kernel.
    hits = [] if oracle_mode else forbidden_hits(args.candidate)
    vend = (
        {
            "available": True,
            "gated": False,
            "exempt": "oracle mode: the production kernel is the anchor, not a claim",
        }
        if oracle_mode
        else vendored_containment(args.candidate)
    )
    result["checks"]["G1_forbidden"] = {
        "clean": not hits and not vend.get("gated"),
        "hits": hits,
        "vendored": vend,
    }
    if hits:
        result["reason"] = f"forbidden library symbols called: {hits}"
        return emit(result, args.out)
    if vend.get("gated"):
        result["reason"] = (
            f"vendored library source: {vend['containment']:.1%} of this submission's "
            f"{vend['ngram']}-token structure appears verbatim in the installed library, "
            f"above the {vend['ceiling']:.0%} ceiling. Renaming symbols does not change "
            f"this measurement. Reading that source is allowed; shipping a copy of it is "
            f"not the task."
        )
        return emit(result, args.out)

    try:
        cand = load(args.candidate, "cand").gated_delta_rule
    except CandidateExited as e:
        result["checks"]["G0_trusted_entrypoint"] = {"clean": False}
        result["reason"] = str(e)
        return emit(result, args.out)
    except Exception as e:  # noqa: BLE001
        result["reason"] = f"candidate failed to import: {type(e).__name__}: {e}"[:250]
        return emit(result, args.out)
    result["checks"]["G0_trusted_entrypoint"] = {"clean": True}

    _tamper = _assert_fixtures_unmodified(_fixtures_before)
    if _tamper:
        result["checks"]["G0_trusted_entrypoint"] = {"clean": False}
        result["checks"]["fixture_integrity"] = {"clean": False, "detail": _tamper}
        result["reason"] = _tamper
        return emit(result, args.out)
    result["checks"]["fixture_integrity"] = {"clean": True}

    ref = load(args.reference, "ref").gated_delta_rule_ref

    def sota(q, k, v, g, beta):
        o = chunk_gated_delta_rule(
            q=q, k=k, v=v, g=g, beta=beta, use_qk_l2norm_in_kernel=False
        )
        return o[0] if isinstance(o, tuple) else o

    # ---- G2 correctness, the five-stage harness ---------------------------
    # Stages 1, 4 and 3 first and each short-circuiting: they are seconds apiece
    # and need no timing, while the sweep below carries the entire paired-timing
    # cost. A broken, non-reproducible or unstable candidate is rejected before
    # any of that is spent, and the reason names the stage it died on.
    fails, ratios, per_shape = [], [], {}
    ratios_inp = None
    gen = torch.Generator(device="cuda")
    seed_base = cfg["seed_base"]
    stages: dict = {}

    def _stage(key, fn, *a):
        nonlocal fails
        if fails:
            stages[key] = {"clean": None, "skipped": "an earlier stage already failed"}
            return
        f, detail = fn(torch, *a)
        stages[key] = {"clean": not f, "failures": f, **detail}
        fails += f

    _stage("stage1_smoke", stage_smoke, cand, ref, seed_base, gen)
    _stage("stage4_determinism", stage_determinism, cand, seed_base, gen)
    _stage("stage3_stability", stage_stability, cand, ref, seed_base, gen)

    if fails:
        # Reported in stage order, not in the order they happened to run.
        result["checks"]["G2_correctness"] = {
            "clean": False,
            "failures": fails,
            "failure_count": len(fails),
            "stages": {k: stages[k] for k in sorted(stages)},
        }
        result["reason"] = (
            "correctness failed at "
            + ", ".join(k for k in sorted(stages) if stages[k].get("clean") is False)
            + ": "
            + _failure_summary(fails)
        )
        return emit(result, args.out)

    # Stages 2 and 5: the shape sweep across dtypes, and the non-power-of-two
    # edge cases. Both run in this one loop because a graded shape and an edge
    # shape differ only in whether the draw is also timed.
    sweep_f, edge_f, seen_dtypes = [], [], set()
    for i, (label, spec) in enumerate(
        list(cfg["graded"].items()) + list(cfg["correctness_only"].items())
    ):
        graded = label in cfg["graded"]
        bucket = sweep_f if graded else edge_f
        B, T, H, K, V = spec["B"], spec["T"], spec["H"], spec["K"], spec["V"]
        HV = spec.get("HV", H)
        for dt_name in spec["dtypes"]:
            dtype = getattr(torch, dt_name)
            seen_dtypes.add(dt_name)
            # crc32, not hash(). Python randomises str hashing per process unless
            # PYTHONHASHSEED is set, so `hash(dt_name)` made the graded draw
            # different on every run and silently defeated seed_base, which exists
            # precisely so that two runs of the same bytes grade the same inputs.
            # crc32 is stable across processes and machines.
            gen.manual_seed(
                cfg["seed_base"] + i * 17 + zlib.crc32(dt_name.encode()) % 97
            )
            inp, scale = make_inputs(torch, B, T, H, HV, K, V, dtype, gen)

            try:
                # _run_candidate latches the fault flag and forces the async
                # fault to surface inside this handler; see its docstring. It is
                # the single place any candidate call goes through, so the latch
                # cannot be set in one path and forgotten in another.
                got, cand_grads = _run_candidate(torch, cand, inp)
            except Exception as e:  # noqa: BLE001
                msg = f"{label}/{dt_name}: candidate raised {type(e).__name__}: {e}"
                fails.append(msg)
                bucket.append(msg)
                continue
            want = fwd_bwd(ref, inp, torch)
            ref_grads = {
                n: (inp[n].grad.clone() if inp[n].grad is not None else None)
                for n in GRAD_NAMES
            }

            ok_o, det_o = close(torch, got, want, dt_name)
            # instruction.md promises "the input gradients are graded", plural. Until
            # 2026-08-12 only dq was compared, so a candidate with a broken backward for
            # four of the five inputs passed this gate. Measured before widening: the
            # private oracle agrees with the reference on all five at every graded shape
            # and both dtypes, so the wider gate costs the oracle nothing.
            grad_bad = []
            for n in GRAD_NAMES:
                cg, rg = cand_grads.get(n), ref_grads.get(n)
                if cg is None or rg is None:
                    grad_bad.append(f"d{n}[missing gradient]")
                    continue
                ok_g, det_g = close(torch, cg, rg, dt_name)
                if not ok_g:
                    grad_bad.append(f"d{n}[{det_g}]")
            if not ok_o or grad_bad:
                msg = f"{label}/{dt_name} scale={scale:.3f}: out[{det_o}] " + " ".join(
                    grad_bad
                )
                fails.append(msg)
                bucket.append(msg)
                continue

            if graded:
                if ratios_inp is None:
                    ratios_inp = inp
                raw, t_c, t_s, n_tr = timed_ratio(_clock, torch, cand, sota, inp)
                r = min(raw, 1.0)
                ratios.append(r)
                per_shape[f"{label}/{dt_name}"] = {
                    "sota_ms": round(t_s, 6),
                    "candidate_ms": round(t_c, 6),
                    "ratio_raw": round(raw, 6),
                    "clipped": raw > 1.0,
                    "paired_trials": n_tr,
                    # The dimensions behind the number. The refinement note reports
                    # how a candidate's fraction TRACKS sequence length -- a measured
                    # statement instead of the heuristic "most often the longest
                    # sequences" it used to assert -- and that needs T beside the
                    # ratio. The label stays hidden from the agent; only the
                    # relationship is reported.
                    "T": spec["T"],
                    "B": spec["B"],
                    "fraction_of_sota": round(r, 6),
                    "scale": round(scale, 4),
                }

    stages["stage2_shape_sweep"] = {
        "clean": not sweep_f,
        "failures": sweep_f,
        "configurations": sum(len(s["dtypes"]) for s in cfg["graded"].values()),
        "dtypes": sorted(seen_dtypes),
        # Recorded as a boolean so the compiled pytest can assert the fp32
        # verification set actually RAN, not merely that the sweep was clean. A
        # stage that silently never executed and a stage that executed and passed
        # are indistinguishable from an aggregate flag.
        "float32_verified": "float32" in seen_dtypes,
    }
    stages["stage5_edge_cases"] = {
        "clean": not edge_f,
        "failures": edge_f,
        "configurations": sum(
            len(s["dtypes"]) for s in cfg["correctness_only"].values()
        ),
        "note": "non-power-of-two sequence lengths and head dimensions; never timed",
    }
    result["checks"]["G2_correctness"] = {
        "clean": not fails,
        "failures": fails,
        "failure_count": len(fails),
        "stages": stages,
    }
    if fails:
        result["reason"] = "correctness failed: " + _failure_summary(fails)
        return emit(result, args.out)

    if not ratios:
        result["reason"] = "no graded shape produced a timing"
        return emit(result, args.out)

    R = statistics.geometric_mean(ratios)
    result["checks"]["G3_speed"] = {
        "per_shape": per_shape,
        "geomean_fraction_of_sota": round(R, 6),
    }

    # ---- G4 written-kernel gate -----------------------------------------
    # Measured on this host: torch.compile over the starter's own math reaches
    # 0.1616 of the production kernel, which clears a 0.15 floor on its own. A
    # forbidden-symbol check does not see that, because torch.compile is not a
    # forbidden symbol. This attributes graded device time to the Triton kernels
    # the submitted module itself declares; Inductor emits its kernels under
    # generated names that cannot match a name in the candidate source.
    if oracle_mode:
        # Same KEY SET as the measured branch. The exemption changes the values,
        # never the shape: a score contract whose fields appear and disappear
        # with a mode flag breaks every consumer that reads it by path, and the
        # compiled rubric test for G4 did exactly that, failing on the oracle
        # because "checks.G4_written_kernel.declared" was simply absent.
        adopt, detail = (
            1.0,
            {
                "exempt": "oracle mode: the production kernel is the anchor, not a claim",
                "declared": [],
                "device_us_total": 0.0,
                "device_us_declared": 0.0,
            },
        )
    else:
        adopt, detail = kernel_adoption(torch, cand, args.candidate, ratios_inp)
    result["checks"]["G4_written_kernel"] = {
        "adoption": round(adopt, 6),
        "threshold": ADOPTION_FLOOR,
        **detail,
    }
    if adopt < ADOPTION_FLOOR:
        result["score"] = 0.0
        result["reason"] = (
            f"written-kernel gate: {adopt:.1%} of graded device time is in Triton "
            f"kernels this module declares, below the {ADOPTION_FLOOR:.0%} floor"
        )
        return emit(result, args.out)
    result["fraction_of_sota"] = round(R, 6)
    # The delivered metric is the raw fraction of the production kernel. No /target
    # clamp: the target is informational (metrics.target_fraction_of_sota) and no
    # longer gates. R is already in (0, 1] -- each per-shape ratio is min(.,1) and
    # the geomean of values <= 1 is <= 1 -- so score == fraction_of_sota exactly.
    result["score"] = round(R, 6)
    # The headline figures and the conditions that produced them, recorded beside
    # the score rather than left to be reconstructed from per_shape by hand.
    result["metrics"] = metrics_record(
        per_shape, R, TARGET, adopt, tgt.get("starter_fraction_of_sota")
    )
    result["timing"] = {
        "method": "paired, interleaved, order-balanced ratio sampling",
        "trials_max": 15,
        "warmup_rounds": 3,
        "budget_s": 50.0,
        "l2_flush_mib": 128,
        "note": "the ratio is measured as a ratio: both sides are sampled within a "
        "few hundred microseconds of each other so clock and thermal drift "
        "divides out, and the order alternates so any first-versus-second "
        "position effect changes sign each trial",
    }
    result["thresholds"] = {
        "target_fraction_of_sota": TARGET,
        "written_kernel_adoption_floor": ADOPTION_FLOOR,
        "tolerances": {k: {"atol": a, "rtol": r} for k, (a, r) in TOL.items()},
        "vendor_containment_ceiling": VENDOR_CONTAINMENT_CEILING,
        "vendor_ngram": VENDOR_NGRAM,
        "determinism_runs": DETERMINISM_RUNS,
        "rubric_gate": "all rubrics must pass; oracle exempt by declaration",
    }
    if result["score"] == 0.0:
        result["reason"] = f"fraction of production kernel {R:.4f} rounds to zero"
    result["process"] = process_score(args, result, torch)
    return emit(result, args.out)


def process_score(args, result: dict, torch) -> float:
    """Deterministic integrity and trajectory checks, plus a narrow-only overlay.

    The process metric is specified as integrity plus trajectory checks plus a
    narrow-only rubric overlay. It was previously `1.0 if not hits else 0.0`,
    which was a restatement of G1 and read no trajectory at all, so the four
    shipped rubrics had nothing consuming them on this side.

    Narrow-only means every check here can lower the process float and none can
    raise the score. It is never the basis for a non-zero score.

    The rubric overlay is narrow-only in that sense and still ABSOLUTE in the
    other: per the client contract, "Only a solution where all rubrics pass will
    be considered correct", so any decided rubric failure sets the score to
    zero, whatever the rubric's `kind`. It can take a score away; it can never
    give one. Two exceptions are declared below and both are reported in the
    score: an unreachable judge, and a submission with no agent phase to judge.
    """
    checks, traj_path = (
        {},
        Path(os.environ.get("TRAJECTORY_PATH", "/logs/agent/trajectory.json")),
    )

    src = args.candidate.read_text()
    checks["no_forbidden_symbol"] = result["checks"]["G1_forbidden"]["clean"]
    # a declared jit kernel is the minimum evidence of kernel work having happened
    checks["declares_triton_kernel"] = bool(declared_jit_names(args.candidate))
    # uniform numeric policy: a branch on input magnitude is the documented 4.3x
    # exploit against this operator family, so a static trace of one lowers the
    # float even when the deterministic gates did not fire
    # D2. The original required a numeric LITERAL after the comparison, so
    # `if x.abs().max() > thresh` with a variable threshold sailed through, and
    # torch.where-based dispatch was never matched at all. Those are the common
    # encodings of the documented 4.3x distribution-sniffing exploit, and leaving
    # them to a single LLM verdict made the judge the only line of defence.
    # Validated against both encodings and against honest code (`torch.max(a, b)`
    # and an fp32 accumulator must NOT flag).
    _MAG = r"(?:abs|amax|amin|max|min|norm|std|var|mean)\s*\("
    checks["no_magnitude_branch"] = not (
        re.search(_MAG + r"[^)]*\)\s*[<>]=?", src)
        or re.search(r"torch\.where\s*\([^)]*" + _MAG, src)
        or re.search(_MAG + r"[^)]*\)\s*\.item\s*\(\)", src)
        or re.search(r"if\s+[^\n]*" + _MAG, src)
    )
    checks["no_grader_reference"] = not re.search(
        r"target\.json|hidden_shapes|/bundle/tests", src
    )

    # Compiled from the no_timing_context_detection rubric. A kernel that reads
    # the environment, argv, the call stack or profiler state can tell a timed
    # run from an untimed one and serve different code to each, which defeats
    # the measurement rather than improving it. This is an access-pattern
    # question over committed bytes, so it is decidable here and does not need
    # a judge.
    checks["no_timing_context_detection"] = not re.search(
        r"os\.environ|os\.getenv|sys\.argv|sys\._getframe|inspect\.(?:stack|currentframe)"
        r"|traceback\.extract_stack|torch\.profiler|torch\.autograd\.profiler"
        r"|__wrapped__|CUDA_LAUNCH_BLOCKING",
        src,
    )

    if traj_path.is_file():
        try:
            t = json.loads(traj_path.read_text())
            steps = t.get("steps", t if isinstance(t, list) else [])
            checks["trajectory_present"] = True
            checks["measured_before_submitting"] = len(steps) > 1
        except Exception:  # noqa: BLE001
            checks["trajectory_present"] = False
    else:
        checks["trajectory_present"] = False

    # Deterministic half. These are cheap and near-constant on an honest run, so
    # they are a floor and not the metric: on their own they gave process 1.0
    # next to a failing score, which discriminated nothing.
    det = [v for k, v in checks.items() if k != "trajectory_present"]
    det_score = (sum(1 for v in det if v) / len(det)) if det else 0.0

    # RECORD them. Until 2026-08-12 these six were computed, folded into `process`
    # and then discarded, so a reader of the delivered artifact could see that
    # process was 0.75 and had no way to learn which check failed. Measured on the
    # controls run: the ORACLE scores exactly that, because it is a nineteen-line
    # wrapper and correctly fails `declares_triton_kernel` -- a right verdict that
    # was unreconstructable from the evidence. Recording them is also what gives
    # the compiled pytest something to assert, so every deterministic obligation
    # this verifier makes has a binary statement in the pytest half.
    result["checks"]["process_deterministic"] = {
        **{k: bool(v) for k, v in checks.items()},
        "deterministic_score": round(det_score, 6),
        "counted": sorted(k for k in checks if k != "trajectory_present"),
        "note": "recorded for attribution; grade.py remains the authority and "
        "tests/test_output.py mirrors these without gating",
    }

    # The rubric half: a cross-family LLM judge over tests/rubrics.jsonl, which
    # grades the trajectory. It can only LOWER.
    verdict = {"available": False, "reason": "judge not attempted"}
    try:
        import judge as _judge

        verdict = _judge.run(
            args.candidate,
            HERE / "rubrics.jsonl",
            traj_path,
            os.environ.get("SOLVER_MODEL", ""),
        )
    except Exception as e:  # noqa: BLE001 - an unavailable judge caps, never fails
        verdict = {"available": False, "reason": f"{type(e).__name__}: {e}"[:180]}
    result["checks"]["rubric_judge"] = verdict

    if verdict.get("not_applicable"):
        # NOT APPLICABLE, which is different from unavailable. There is no agent
        # trajectory, so there is no behaviour for behavioural rubrics to judge --
        # the oracle path, and only the oracle path, since a solver attempt that
        # produced no trajectory is caught by the liveness gate as missing data.
        # This replaces the old `rubric_gate_exempt` carve-out keyed on oracle mode.
        result["rubric_gate_not_applicable"] = verdict.get("reason")
        return round(det_score, 6)

    if not verdict.get("available"):
        result["judge_unavailable"] = True
        return round(det_score, 6)

    # The oracle's `rubric_gate_exempt` carve-out used to sit here, keyed on oracle
    # mode. It is gone, not disabled: the judge now declines to judge any submission
    # with no trajectory, which reaches the same place for a reason that holds for
    # every submission, and is handled at the `not_applicable` branch above.
    gating = verdict.get("failed_gating", verdict.get("failed_rubrics", []))
    if gating:
        # Client contract, restored: "Only a solution where all rubrics pass will
        # be considered correct." Any decided failure zeroes, whatever its kind.
        why = (
            "rubric gate: " + ", ".join(gating) + " not satisfied; the client "
            "contract is that a solution is correct only when every rubric passes"
        )
        result["rubric_gate_failed"] = gating
        result["score"] = 0.0
        result["reason"] = f"{result['reason']}; {why}" if result.get("reason") else why
        return 0.0
    # #12. The non-gating structural diagnostics narrow `process` and nothing else.
    # A kernel that is correct, honest and fast but is a serial recurrence wearing
    # the name of a chunked one now says so in its process float, where the rubric
    # layer deliberately cannot: gating on an LLM's reading of "is this the WY
    # identity" would zero honest work on an unprovable question.
    share = verdict.get("diagnostic_share")
    base = min(det_score, det_score * verdict["pass_share"])
    if share is not None:
        base = min(base, base * share)
    return round(base, 6)


def write_score(result: dict, out: Path) -> None:
    # This process imported the candidate, so the candidate's code has run here
    # and can register an atexit hook or fork a writer. It therefore writes ONLY
    # score.json, which is diagnostic. The reward files Harbor actually reads are
    # derived by tests/test.sh after this process is gone, from a shell the
    # candidate never got to hook. Writing them here is what let a candidate that
    # computed nothing overwrite an honest 0.0 with 1.0 on the way out.
    vdir = out.parent
    vdir.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")


def _reward_projection(doc: dict) -> str:
    checks = doc.get("checks") or {}
    g1 = checks.get("G1_forbidden") or {}
    g3 = checks.get("G3_speed") or {}
    g4 = checks.get("G4_written_kernel") or {}
    rj = checks.get("rubric_judge") or {}
    pd = checks.get("process_deterministic") or {}

    def _clean(name):
        return (checks.get(name) or {}).get("clean")

    proj = [
        ("candidate_sha256", doc.get("candidate_sha256", "")),
        ("verifier_incomplete", doc.get("verifier_incomplete")),
        ("fault", doc.get("fault")),
        ("score", doc.get("score")),
        ("fraction_of_sota", doc.get("fraction_of_sota")),
        ("g3_geomean", g3.get("geomean_fraction_of_sota")),
        ("g0_clean", _clean("G0_trusted_entrypoint")),
        ("g1_clean", g1.get("clean")),
        ("g2_clean", _clean("G2_correctness")),
        ("g4_exempt", bool(g4.get("exempt"))),
        ("g4_adoption", g4.get("adoption")),
        (
            "rubric_failed",
            sorted(rj.get("failed_gating") or rj.get("failed_rubrics") or []),
        ),
        ("no_magnitude_branch", pd.get("no_magnitude_branch")),
        ("no_grader_reference", pd.get("no_grader_reference")),
    ]
    return json.dumps(proj, separators=(",", ":"), default=str)


def _write_completion_auth(out: Path) -> None:
    """Sign the just-written score with the per-run secret so emit_reward.py can
    tell a genuine completion from a score.json a candidate fabricated before
    calling os._exit(0). Reads the score BACK from disk first, so the projection
    is computed over the exact JSON emit_reward.py will parse."""
    auth = _LATE.get("auth")
    if not auth:
        return
    try:
        doc = json.loads(out.read_text())
        tag = hashlib.sha256(
            (auth + "|task-emit-v1|" + _reward_projection(doc)).encode()
        ).hexdigest()
        (out.parent / (out.name + ".auth")).write_text(
            json.dumps(
                {
                    "tag": tag,
                    "candidate_sha256": doc.get("candidate_sha256", ""),
                    "algo": "sha256(auth|task-emit-v1|reward_projection)",
                }
            )
            + "\n"
        )
    except (OSError, ValueError):
        pass


_GRADER_FIXTURES = (
    "target.json",
    "hidden_shapes.json",
    "reference.py",
    "rubrics.jsonl",
)


def _fixture_digests() -> dict[str, str]:
    out = {}
    for name in _GRADER_FIXTURES:
        p = HERE / name
        if p.is_file():
            out[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def _assert_fixtures_unmodified(before: dict[str, str]) -> str | None:
    """The verifier runs as root in the same container the agent used, and
    candidate module-scope code executes inside that process. Nothing stopped it
    rewriting the hidden shape set, the target, or the oracle between import and
    grading. Digests are taken before the candidate loads and re-checked here;
    any divergence means the graded fixtures are not the shipped ones."""
    after = _fixture_digests()
    moved = sorted(n for n in set(before) | set(after) if before.get(n) != after.get(n))
    if moved:
        return "grader fixtures modified after load: " + ", ".join(moved)
    return None


def _drain_auth_fd(fd: int = 3) -> str:
    """Read and CLOSE the per-run secret before any candidate byte executes, so
    candidate code cannot read fd 3 and forge a completion tag."""
    try:
        chunks = []
        while True:
            b = os.read(fd, 4096)
            if not b:
                break
            chunks.append(b)
        os.close(fd)
        return b"".join(chunks).decode("utf-8", "replace").strip()
    except OSError:
        return ""


# Set once the candidate is importable, so emit() can finish the record on any
# early-return path without threading four arguments through nine call sites.
_LATE: dict = {}

# Every check the grader can produce, in gate order. A run that stops at G2 never
# evaluates the rest, and until 2026-08-13 it simply omitted them -- so a consumer
# could not tell "not reached" from "missing", and the mirrored pytest reported
# seven failures for gates that never ran. Absence is not a verdict.
ALL_CHECKS = (
    "G0_trusted_entrypoint",
    "G1_forbidden",
    "G2_correctness",
    "G4_written_kernel",
    "G3_speed",
    "process_deterministic",
    "rubric_judge",
)


def emit(result: dict, out: Path) -> int:
    if result["score"] == 0.0 and not result["reason"]:
        result["reason"] = "score zero with no recorded cause"

    # PROCESS ON THE FAILURE PATHS. `process` is narrow-only and needs no timing
    # data, so a run that died at G1, G2 or G4 can still be scored on conduct. It
    # was left at the initialised 0.0 instead, which reads identically to a caught
    # cheater: measured on this task, an honest kernel with a race in its reduction
    # and an honest kernel with a wrong answer both recorded process 0.0, the same
    # value a memoization cheat earns. G0 and the crash guard stay 0.0 by design --
    # a module that never imported has no conduct to score.
    checks = result.get("checks") or {}
    g0_ok = (checks.get("G0_trusted_entrypoint") or {}).get("clean") is True
    if g0_ok and not result.get("process") and _LATE.get("args") is not None:
        try:
            result["process"] = process_score(_LATE["args"], result, _LATE["torch"])
        except Exception as e:  # noqa: BLE001 - conduct scoring must never fail a run
            result["process_error"] = f"{type(e).__name__}: {e}"[:200]

    # STABLE KEY SET. Unreached gates are recorded as explicitly not reached rather
    # than omitted, so score.json always carries the same keys and every consumer
    # sees a verdict-shaped answer for each one.
    for name in ALL_CHECKS:
        result.setdefault("checks", {}).setdefault(
            name,
            {
                "reached": False,
                "note": "the run stopped at an earlier gate; this was never evaluated",
            },
        )
    for name in ALL_CHECKS:
        if "reached" not in result["checks"][name]:
            result["checks"][name]["reached"] = True

    result["verifier_incomplete"] = False
    write_score(result, out)
    _write_completion_auth(out)
    print(json.dumps({k: v for k, v in result.items() if k != "checks"}, indent=2))
    return 0


def _hard_exit(code: int) -> None:
    # os._exit skips atexit. The candidate ran in this interpreter and can
    # register a handler; SystemExit would run it after the score was written.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    try:
        _hard_exit(main())
    except Exception as exc:  # noqa: BLE001 - a verifier fault must still emit a score
        traceback.print_exc()
        f = Path(os.environ.get("SCORE_PATH", "/logs/verifier/score.json"))
        cand_fault = CANDIDATE_ENTERED
        payload = {
            "score": 0.0,
            "process": 0.0,
            "verifier_incomplete": not cand_fault,
            "fault": "candidate" if cand_fault else "harness",
            "reason": (
                ("candidate fault: " if cand_fault else "verifier fault: ")
                + f"{type(exc).__name__}: {exc}"
            )[:4000],
        }
        for key in (
            "candidate_sha256",
            "candidate_bytes",
            "oracle_mode",
            "oracle_route",
            "checks",
        ):
            if key in PARTIAL:
                payload.setdefault(key, PARTIAL[key])
        write_score(payload, f)
        _hard_exit(1)
