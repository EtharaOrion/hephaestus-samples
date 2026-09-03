#!/usr/bin/env python3
"""Generic verifier engine for GPU-kernel RL task bundles.

This is the KDA bundle's tests/grade.py with every category-specific value
lifted into a sibling `taskdef.py` module (protocol: core/taskdef_api.py).
The machinery and its ORDER are preserved exactly, because each piece was
paid for by a measured failure:

    zero-score-first write        an unwritten score is indistinguishable from
                                  a clean run; the zero lands before any
                                  candidate code is imported
    candidate fingerprint         sha256 + byte count + a copy beside score.json
    canary / FORGE_ORACLE detect  oracle exemptions are a property of the graded
                                  bytes (or an operator-owned env var), never of
                                  task.toml
    G1  forbidden symbols         name half: calling the library
        + shingle vendoring       structure half: shipping a renamed copy of it;
                                  32-token rename-normalized shingles, skipped
                                  (recorded, never gated) when the category has
                                  no vendor library configured
    G0  guarded import            SystemExit trapped; sys.exit(0) on import must
                                  not read as success
    G2  five stages, fast-fail    1 smoke -> 4 determinism (3 bitwise runs, incl.
                                  grads on fwdbwd) -> 3 stability (finiteness
                                  only) -> 2+5 hidden sweep + NPOT edges (output
                                  plus ALL declared gradients per shape x dtype)
    G3  paired timing             interleaved, order-balanced per-trial ratios,
                                  L2 flush, hidden per-invocation scale, timed
                                  tensors ARE the checked tensors;
                                  r_i = min(t_sota/t_cand, 1), R = geomean,
                                  score = R (fraction_of_sota), 0 unless every gate passes
    G4  written-kernel adoption   torch.profiler self device time attributed to
                                  @triton.jit kernels the candidate module
                                  itself declares (exact or name_ prefix match,
                                  vendor-symbol rejection)
    process score                 deterministic checks + cross-family LLM rubric
                                  judge; any DECIDED rubric failure zeroes the
                                  score; an unreachable judge degrades to
                                  judge_unavailable and never zeroes
    emit()                        stable score.json key set; unreached gates
                                  carry {"reached": false}; then os._exit so no
                                  candidate atexit hook outlives the verdict

Seeding convention (zlib.crc32, never hash()): sweep draw i uses
seed_base + i*17 + crc32(dtype)%97; smoke +9001; determinism +9002;
stability mode i +9100+7*i.

Env contract: CANDIDATE_TREE, SCORE_PATH, FORGE_ORACLE, FORGE_BRIDGE (consumed
by the optional judge module), SOLVER_MODEL, TRAJECTORY_PATH.
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

import taskdef_api  # noqa: E402  (protocol defaults + shared compare helpers)

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


# ---------------------------------------------------------------------------
# G4 helpers: declared-kernel discovery and profiler attribution
# ---------------------------------------------------------------------------


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

    Substring containment was the rule here once and it was exploitable: a
    module declaring `gemm` matched cuBLAS's generated names and one declaring
    `reduce` matched ATen's reduce_kernel, so a candidate could route the
    arithmetic through torch and have the library's device time counted as its
    own. Two conditions now, both required: the entry must not look like a
    vendor or ATen symbol, and the declared name must match the key exactly or
    as its leading component (Triton appends specialisation suffixes after an
    underscore, it does not embed the name mid-string).
    """
    if "::" in key or key.startswith("void ") or "<" in key:
        return False  # C++/ATen mangling
    if re.match(r"^(ampere|turing|volta|hopper|maxwell|pascal|sm\d+)_", key):
        return False  # cuBLAS/cuDNN generated names
    if re.search(r"(cutlass|cublas|cudnn|nvjet)", key, re.I):
        return False
    return any(key == n or key.startswith(n + "_") for n in names)


def kernel_adoption(torch, runner, fn, path: Path, args, grad_names):
    """Share of graded device time spent in Triton kernels this module declares."""
    names = declared_jit_names(path)
    if not names:
        return 0.0, {
            "declared": [],
            "note": "the submitted module declares no triton.jit kernel",
        }
    from torch.profiler import ProfilerActivity, profile

    for _ in range(3):
        runner(fn, args, grad_names)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        runner(fn, args, grad_names)
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


# ---------------------------------------------------------------------------
# G0: guarded import
# ---------------------------------------------------------------------------


class CandidateExited(Exception):
    """The candidate called sys.exit during import or execution."""


def load(path: Path, name: str):
    """Import the candidate with SystemExit trapped.

    Without this, a candidate that calls sys.exit(0) on import terminates the
    grader before it writes a score, and an unwritten score file is not
    distinguishable from a clean run by anything downstream. SystemExit does
    not inherit from Exception, so a bare `except Exception` never sees it.
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


# ---------------------------------------------------------------------------
# G1: forbidden symbols by name, vendored structure by shingle
# ---------------------------------------------------------------------------


def forbidden_hits(path: Path, forbidden) -> list:
    src = path.read_text()
    # comments stripped: reading the library is allowed and encouraged, calling it is not
    code = "\n".join(l.split("#")[0] for l in src.splitlines())
    return [s for s in forbidden if re.search(r"\b" + re.escape(s) + r"\b", code)]


def normalized_tokens(src: str) -> list:
    """Python source as a rename-invariant token stream.

    Every identifier collapses to a single placeholder, every literal to a
    per-kind placeholder, and comments and formatting disappear entirely. What
    survives is control flow, call structure and operator sequence -- the part
    of a copy that renaming cannot alter.
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


def library_shingles(package: str, subdirs, n: int):
    """Shingles of the installed vendor library's own source, or None.

    Located through the import system rather than a hardcoded path, so it
    follows the library actually installed in the image. `find_spec` rather
    than `import`: this needs the package's location on disk, not its module
    object. Shingles never cross a file boundary. Subdirs are joined onto the
    package root (e.g. "ops/gated_delta_rule").
    """
    try:
        spec = importlib.util.find_spec(package)
        if spec is None or not spec.origin:
            return None
        root = Path(spec.origin).resolve().parent
    except Exception:  # noqa: BLE001
        return None
    acc, files = set(), 0
    for sub in subdirs:
        d = root.joinpath(*str(sub).split("/"))
        if not d.is_dir():
            continue
        for f in sorted(d.rglob("*.py")):
            try:
                acc |= _shingles(normalized_tokens(f.read_text(errors="replace")), n)
            except OSError:
                continue
            files += 1
    return (acc, files) if acc else None


def vendored_containment(path: Path, td) -> dict:
    """Share of the candidate's structure that appears verbatim in the library.

    Reported always, so the number is on the record whether or not it trips.
    A category with no vendor library configured (empty VENDOR_LIB_SUBDIRS or
    empty VENDOR_LIB_PACKAGE) records available:false and never gates -- the
    FORBIDDEN symbol scan still runs. Absence of evidence is not evidence of
    vendoring, and missing data scored as a pass is the defect this bundle
    family exists to avoid.
    """
    package = taskdef_api.cfg(td, "VENDOR_LIB_PACKAGE")
    subdirs = tuple(taskdef_api.cfg(td, "VENDOR_LIB_SUBDIRS") or ())
    ngram = taskdef_api.cfg(td, "VENDOR_NGRAM")
    ceiling = taskdef_api.cfg(td, "VENDOR_CONTAINMENT_CEILING")
    if not package or not subdirs:
        return {
            "available": False,
            "gated": False,
            "reason": "no vendor library configured for this category; "
            "the forbidden-symbol scan still ran",
        }
    lib = library_shingles(package, subdirs, ngram)
    if lib is None:
        return {
            "available": False,
            "gated": False,
            "reason": "the installed library source could not be read",
        }
    lib_grams, files = lib
    try:
        cand = _shingles(normalized_tokens(path.read_text(errors="replace")), ngram)
    except OSError:
        return {"available": False, "gated": False, "reason": "candidate unreadable"}
    if not cand:
        # Shorter than one shingle. Too small to be a copy of anything.
        return {
            "available": True,
            "gated": False,
            "containment": 0.0,
            "ngram": ngram,
            "ceiling": ceiling,
            "candidate_shingles": 0,
            "library_files": files,
        }
    share = len(cand & lib_grams) / len(cand)
    return {
        "available": True,
        "gated": share > ceiling,
        "containment": round(share, 6),
        "ngram": ngram,
        "ceiling": ceiling,
        "candidate_shingles": len(cand),
        "library_files": files,
    }


# ---------------------------------------------------------------------------
# Surface runner: the ONE way any graded callable is invoked
# ---------------------------------------------------------------------------


class Runner:
    """Runs a callable over an args dict on the category's surface.

    fwdbwd: zero grads, forward, loss(out).backward(). fwd: forward only.
    The same runner is used for correctness, determinism, stability, timing
    and adoption, which is what makes the timed tensors the checked tensors.
    """

    def __init__(self, torch, surface: str, loss):
        self.torch, self.surface, self.loss = torch, surface, loss

    def __call__(self, fn, args: dict, grad_names):
        torch = self.torch
        for n in grad_names:
            t = args.get(n)
            if isinstance(t, torch.Tensor) and t.grad is not None:
                t.grad = None
        out = fn(args)
        if self.surface == "fwdbwd" and grad_names:
            self.loss(out).backward()
        return out


def _grads_of(args: dict, grad_names) -> dict:
    return {
        n: (
            args[n].grad.clone() if getattr(args[n], "grad", None) is not None else None
        )
        for n in grad_names
    }


CANDIDATE_ENTERED = False


def _run_candidate(torch, runner, fn, args, grad_names):
    """Call the candidate with the fault flag latched and the fault forced to surface.

    An illegal access inside a candidate kernel is asynchronous; without the
    sync it raises at the next synchronising call, which is usually outside
    whatever handler is watching, and the fault gets attributed to the harness.
    """
    global CANDIDATE_ENTERED
    CANDIDATE_ENTERED = True
    out = runner(fn, args, grad_names)
    grads = _grads_of(args, grad_names)
    torch.cuda.synchronize()
    return out, grads


# ---------------------------------------------------------------------------
# Input draws: hidden scale + taskdef.make_inputs
# ---------------------------------------------------------------------------


def draw_inputs(torch, td, shape: dict, seed: int, dtype, maker=None):
    """One reproducible draw: hidden scale from the category distribution, then
    the taskdef's own maker. The scale generator is CPU-side and seeded from
    the same integer, so the draw is deterministic per (seed) without
    perturbing whatever generator the taskdef seeds internally."""
    g = torch.Generator()
    g.manual_seed((seed & 0x7FFFFFFF) ^ 0x5CA1AB1E)
    fn = getattr(td, "draw_scale", None)
    if fn is not None:
        scale = float(fn(g))
    else:
        lo, hi = taskdef_api.cfg(td, "SCALE_RANGE")
        scale = float(torch.empty(()).uniform_(lo, hi, generator=g))
    spec = {k: v for k, v in shape.items() if k != "dtypes"}
    make = maker if maker is not None else td.make_inputs
    args, grad_names = make(spec, seed, scale, dtype, "cuda")
    if td.SURFACE == "fwd" and grad_names:
        raise RuntimeError(
            "taskdef protocol violation: SURFACE 'fwd' returned "
            f"grad_names {tuple(grad_names)!r}"
        )
    return args, tuple(grad_names), scale


def _compare_out(td, got, want, dt_name: str, shape: dict) -> list:
    fn = getattr(td, "compare", None)
    if fn is None:
        return taskdef_api.default_compare(got, want, dt_name, shape, td.TOL)
    return fn(got, want, dt_name, shape)


def _compare_grads(td, got_g: dict, want_g: dict, grad_names, dt_name: str) -> list:
    """Gradients are dense float tensors on every surface that has them, so they
    always use the dense tolerance whatever `compare` the output uses."""
    atol, rtol = td.TOL[dt_name]
    fails = []
    for n in grad_names:
        cg, rg = got_g.get(n), want_g.get(n)
        if cg is None or rg is None:
            fails.append(f"d{n}[missing gradient]")
            continue
        ok, det = taskdef_api.close_dense(cg, rg, atol, rtol)
        if not ok:
            fails.append(f"d{n}[{det}]")
    return fails


def _float_tensors(out) -> list:
    return [t for t in taskdef_api.flatten_output(out) if t.is_floating_point()]


def _all_finite(torch, out) -> bool:
    """Finiteness over the float tensors of an output structure. Integer
    outputs (top-k indices, routing tables) have no notion of finiteness."""
    return all(bool(torch.isfinite(t).all()) for t in _float_tensors(out))


# ---------------------------------------------------------------------------
# G3: paired, interleaved, order-balanced ratio timing
# ---------------------------------------------------------------------------


class _Clock:
    """Timing primitives captured BEFORE the candidate is ever imported.

    Exploit probe (d): a candidate that runs, at import,
    `torch.cuda.Event.elapsed_time = lambda s, o: 1.0` (or patches
    torch.cuda.Event / .record / torch.cuda.synchronize) forces every measured
    device time to a constant and fabricates the ratio -- an 84x-too-slow kernel
    is paid full reward while G2 and G4 pass honestly. The timing runs in the
    same interpreter that imports the candidate, so it cannot be moved out of
    reach; instead this binds the ORIGINAL callables here, in main(), before any
    candidate byte executes, and invokes them UNBOUND. A later reassignment of
    the class attribute rebinds the name but not the object this still holds
    (verified: the captured function returns the real 96 ms while the patched
    class attribute returns the planted 1.0). The candidate cannot pre-empt the
    capture: grade_core imports torch itself and captures before load().
    """

    __slots__ = ("_Event", "_record", "_elapsed", "sync", "perf")

    def __init__(self, torch, time_mod):
        self._Event = torch.cuda.Event
        self._record = torch.cuda.Event.record  # unbound: patch-proof
        self._elapsed = torch.cuda.Event.elapsed_time  # unbound: patch-proof
        self.sync = torch.cuda.synchronize
        self.perf = time_mod.perf_counter

    def event(self):
        return self._Event(enable_timing=True)

    def record(self, ev):
        self._record(ev)

    def elapsed(self, a, b) -> float:
        return self._elapsed(a, b)


def one_sample(clock, runner, fn, args, grad_names, flush):
    """One L2-cold sample: (device_ms, wall_ms). Device time via the captured
    original CUDA-event primitives; wall time via the captured perf_counter as an
    independent, RECORDED (never gated -- launch-bound kernels legitimately show
    wall >> device) cross-measurement so a reader can see the two agree."""
    flush.zero_()
    clock.sync()
    a, b = clock.event(), clock.event()
    w0 = clock.perf()
    clock.record(a)
    runner(fn, args, grad_names)
    clock.record(b)
    clock.sync()
    w1 = clock.perf()
    return clock.elapsed(a, b), (w1 - w0) * 1000.0


def timed_ratio(
    torch,
    runner,
    cand,
    sota,
    args,
    grad_names,
    trials=15,
    budget_s=50.0,
    flush_mib=128,
    clock=None,
):
    """Median of PAIRED, INTERLEAVED, ORDER-BALANCED ratio samples.

    The graded quantity is a ratio, so it is measured as a ratio: samples are
    interleaved so both sides see the same clock/thermal drift within a few
    hundred microseconds and it divides out, and the order alternates so any
    residual first-versus-second position effect enters with opposite sign on
    odd and even trials instead of accumulating. Measured cost of doing this
    wrong, on the KDA oracle whose every true per-shape ratio is exactly 1.0:
    one shape read 0.7920 and the oracle failed its own control.

    What this cannot fix: r_i = min(ratio, 1) is an asymmetric estimator, so
    E[geomean] < 1 whenever the candidate is near the anchor. The clip is
    deliberate -- the score is a saturating FRACTION of a production kernel --
    and the residual bias is absorbed by setting the target below the oracle's
    repeated worst case rather than at its mean.
    """
    if clock is None:
        clock = _Clock(torch, time)  # measure.py path: no candidate in the loop
    flush = torch.empty(
        flush_mib * 1024 * 1024 // 4, dtype=torch.float32, device="cuda"
    )
    for _ in range(3):
        runner(cand, args, grad_names)
        runner(sota, args, grad_names)
    clock.sync()

    t0 = clock.perf()
    runner(cand, args, grad_names)
    clock.sync()
    one = clock.perf() - t0
    if one * 2 * trials > budget_s:
        trials = max(3, min(trials, int(budget_s / (2 * one)) or 3))

    for _ in range(3):
        runner(cand, args, grad_names)
        runner(sota, args, grad_names)
    clock.sync()

    ratios, cands, sotas, cwalls = [], [], [], []
    for i in range(trials):
        if i % 2 == 0:
            tc, wc = one_sample(clock, runner, cand, args, grad_names, flush)
            ts, _ = one_sample(clock, runner, sota, args, grad_names, flush)
        else:
            ts, _ = one_sample(clock, runner, sota, args, grad_names, flush)
            tc, wc = one_sample(clock, runner, cand, args, grad_names, flush)
        cands.append(tc)
        sotas.append(ts)
        cwalls.append(wc)
        ratios.append(ts / tc)
    return (
        statistics.median(ratios),
        statistics.median(cands),
        statistics.median(sotas),
        trials,
        statistics.median(cwalls),
    )


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


def environment_record(torch) -> dict:
    """What produced this number, recorded on every path including early death."""
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

    `speedup_vs_starter` is a ratio of two MEASURED quantities: this run's
    geomean and the starter's own fraction (measured by the starter control on
    the same graded set and hardware). `outcome_log_scaled` is recorded and
    GATES NOTHING: kernel work compounds multiplicatively while the shipped
    reward is linear in the ratio, so the log-transformed series is kept for a
    later score-model decision to be made on a recorded series.
    """
    cand = [v["candidate_ms"] for v in per_shape.values() if v.get("candidate_ms")]
    sota = [v["sota_ms"] for v in per_shape.values() if v.get("sota_ms")]
    m = {
        "fraction_of_sota": round(R, 6),
        "speedup_vs_production": round(R, 6),
        "slowdown_vs_production": round(1.0 / R, 4) if R > 0 else None,
        "configurations_timed": len(per_shape),
        "clipped_shapes": sum(1 for v in per_shape.values() if v.get("clipped")),
        "score_formula": "fraction_of_sota if all_gates_pass else 0",
        # Informational only: the target no longer normalizes or gates the
        # delivered score (which is the raw fraction_of_sota). Kept so a reader
        # can still see the frontier-defeat reference the measure stage derived.
        "target_fraction_of_sota": target,
    }
    if starter_fraction and R > 0:
        m["speedup_vs_starter"] = round(R / starter_fraction, 4)
        m["starter_fraction_of_sota"] = starter_fraction
        try:
            lo, hi = math.log(R / starter_fraction), math.log(target / starter_fraction)
            m["outcome_log_scaled"] = round(max(0.0, min(1.0, lo / hi)), 6)
            m["outcome_log_scaled_note"] = "diagnostic only; the reward is `score`"
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


# ---------------------------------------------------------------------------
# The five-stage correctness harness (order fixed: 1 -> 4 -> 3 -> 2+5)
# ---------------------------------------------------------------------------


def _failure_summary(fails: list) -> str:
    """Describe a correctness failure by CONFIGURATION and by TENSOR, separately:
    five tensors failing on one shape says the backward is comprehensively
    broken, five shapes each failing one tensor says there is an edge case."""
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


def stage_smoke(torch, td, runner, cand, ref, seed_base):
    """Stage 1. One small input, run before anything expensive. Catches an
    import that works but a kernel that does not, in about a second."""
    s, dt_name = td.SMOKE_SHAPE, taskdef_api.cfg(td, "SMOKE_DTYPE")
    inp, gn, _ = draw_inputs(torch, td, s, seed_base + 9001, getattr(torch, dt_name))
    try:
        got, got_g = _run_candidate(torch, runner, cand, inp, gn)
    except Exception as e:  # noqa: BLE001
        return [f"smoke/{dt_name}: candidate raised {type(e).__name__}: {e}"], {}
    want = runner(ref, inp, gn)
    want_g = _grads_of(inp, gn)
    fails = [
        f"smoke/{dt_name}: {f}"
        for f in _compare_out(td, got, want, dt_name, s)
        + _compare_grads(td, got_g, want_g, gn, dt_name)
    ]
    return fails, {"shape": dict(s), "dtype": dt_name}


def stage_determinism(torch, td, runner, cand, seed_base):
    """Stage 4. Same input, three runs, bitwise identical output (and, on a
    fwdbwd surface, gradients). A property of the CANDIDATE against ITSELF --
    the reference is not involved -- so it cannot be affected by tolerance. A
    kernel that is not reproducible is not measurable."""
    s, dt_name = td.DETERMINISM_SHAPE, taskdef_api.cfg(td, "DETERMINISM_DTYPE")
    n_runs = taskdef_api.cfg(td, "DETERMINISM_RUNS")
    inp, gn, _ = draw_inputs(torch, td, s, seed_base + 9002, getattr(torch, dt_name))
    runs = []
    try:
        # Warm up first, and discard it: Triton autotune benchmarks several
        # configurations on the first launch, so an un-warmed first call can
        # return the output of a different configuration than the calls after
        # it and read as non-determinism on a perfectly reproducible kernel.
        _run_candidate(torch, runner, cand, inp, gn)
        for _ in range(n_runs):
            out, grads = _run_candidate(torch, runner, cand, inp, gn)
            runs.append(([t.clone() for t in taskdef_api.flatten_output(out)], grads))
    except Exception as e:  # noqa: BLE001
        return [f"determinism/{dt_name}: candidate raised {type(e).__name__}: {e}"], {}

    fails, (first_out, first_g) = [], runs[0]
    for i, (out, grads) in enumerate(runs[1:], start=2):
        if len(out) != len(first_out):
            fails.append(
                f"determinism/{dt_name}: run {i} returned {len(out)} tensors, "
                f"run 1 returned {len(first_out)}"
            )
            continue
        for j, (a, b) in enumerate(zip(first_out, out)):
            if not torch.equal(a, b):
                tag = f"output[{j}]" if len(first_out) > 1 else "output"
                if a.is_floating_point():
                    d = (a.float() - b.float()).abs().max()
                    fails.append(
                        f"determinism/{dt_name}: run {i} {tag} differs from "
                        f"run 1, max abs {float(d):.3e}"
                    )
                else:
                    fails.append(
                        f"determinism/{dt_name}: run {i} {tag} differs from run 1"
                    )
        for n in gn:
            a, b = first_g.get(n), grads.get(n)
            if a is None or b is None or not torch.equal(a, b):
                fails.append(f"determinism/{dt_name}: run {i} d{n} differs from run 1")
    return fails, {"runs": n_runs, "shape": dict(s), "dtype": dt_name}


def stage_stability(torch, td, runner, cand, ref, seed_base):
    """Stage 3. Adversarial inputs from the taskdef's mode makers, each driving
    one term of the operator to an end of its range.

    WHAT THIS GATES ON, and why it is not tolerance: measured on the KDA
    oracle, adversarial probes legitimately put thousands of elements outside
    tolerance -- a chunked reordering of the arithmetic disagrees with a
    sequential fp32 scan on six decades of magnitude, and a stage the
    production kernel cannot pass would zero every honest candidate. So the
    obligation is STABILITY: do not crash, and do not produce a non-finite
    value where the reference is finite. The tolerance comparison is still
    computed and recorded per probe, as a diagnostic, and never gates.
    Integer output tensors have no finiteness and are exempt by construction.

    Nothing here is ever timed.
    """
    s, dt_name = td.STABILITY_SHAPE, taskdef_api.cfg(td, "STABILITY_DTYPE")
    modes = td.STABILITY_MODES
    fails, detail = [], {}
    for i, (mode, maker) in enumerate(modes.items()):
        seed = seed_base + 9100 + i * 7
        inp, gn, _ = draw_inputs(
            torch, td, s, seed, getattr(torch, dt_name), maker=maker
        )
        want = runner(ref, inp, gn)
        want_g = _grads_of(inp, gn)
        if not _all_finite(torch, want):
            detail[mode] = {
                "verdict": "skipped",
                "note": "the reference is not finite on this probe",
            }
            continue
        try:
            got, got_g = _run_candidate(torch, runner, cand, inp, gn)
        except Exception as e:  # noqa: BLE001
            fails.append(f"stability/{mode}: candidate raised {type(e).__name__}: {e}")
            detail[mode] = {
                "verdict": "raised",
                "note": f"{type(e).__name__}: {e}"[:200],
            }
            continue

        # The gating half: finite where the reference is finite.
        nonfinite = []
        if not _all_finite(torch, got):
            nonfinite.append("out")
        for n in gn:
            cg, rg = got_g.get(n), want_g.get(n)
            if cg is None:
                nonfinite.append(f"d{n}[missing]")
            elif (
                rg is not None
                and bool(torch.isfinite(rg).all())
                and not bool(torch.isfinite(cg).all())
            ):
                nonfinite.append(f"d{n}")
        if nonfinite:
            fails.append(
                f"stability/{mode}: non-finite values in {', '.join(nonfinite)} "
                f"where the reference is finite"
            )

        # The recorded half: how far it drifted, gating nothing.
        drift = len(
            _compare_out(td, got, want, dt_name, s)
            + _compare_grads(td, got_g, want_g, gn, dt_name)
        )
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


# ---------------------------------------------------------------------------
# Oracle detection: canary tokens in the graded bytes, or the operator env var
# ---------------------------------------------------------------------------

_SLOT_RE = re.compile(r"FORGE-CANARY-SLOT-(\d+)\s+([0-9a-f]{64})")
_PLACEHOLDER = "0" * 64

# The fatal handler at the bottom cannot see main()'s locals, so a crash used to
# discard everything already established: which bytes were graded, and which
# gates had already passed. This aliases the live result dict so the handler can
# carry that evidence into the score it writes.
PARTIAL: dict = {}


def _canary_tokens(text: str) -> list:
    found = sorted(_SLOT_RE.findall(text), key=lambda m: int(m[0]))
    toks = [t for _, t in found]
    return [] if (not toks or any(t == _PLACEHOLDER for t in toks)) else toks


def _my_canary_tokens() -> list:
    """Tokens planted into this verifier at freeze. The freeze step may plant
    them into the bundle's grade.py shim or into grade_core.py itself; both are
    checked so either planting site works."""
    for p in (HERE / "grade.py", Path(__file__)):
        try:
            if p.is_file():
                toks = _canary_tokens(p.read_text(encoding="utf-8", errors="replace"))
                if toks:
                    return toks
        except OSError:
            continue
    return []


def _is_authored_oracle(path: Path) -> bool:
    """Whether the graded bytes ARE the private oracle this bundle froze.

    The oracle carries this bundle's canary tokens, derived from its canonical
    content hash and planted only into artifacts Harbor mounts for the oracle
    phase. An agent never sees that tree and cannot derive the tokens; solve.sh
    installs the oracle byte for byte, so the oracle always does.
    """
    try:
        mine = _my_canary_tokens()
        if not mine or not path.is_file():
            return False
        return (
            _canary_tokens(path.read_text(encoding="utf-8", errors="replace")) == mine
        )
    except OSError:
        return False


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _import_taskdef():
    """The category module, imported by literal name from this directory."""
    import taskdef  # noqa: PLC0415

    errs = taskdef_api.validate_taskdef(taskdef)
    if errs:
        raise RuntimeError("taskdef protocol violation: " + "; ".join(errs))
    return taskdef


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--candidate",
        type=Path,
        default=Path(os.environ.get("CANDIDATE_TREE", "/workspace")) / "kernel.py",
    )
    ap.add_argument("--reference", type=Path, default=HERE / "reference.py")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(os.environ.get("SCORE_PATH", "/logs/verifier/score.json")),
    )
    ap.add_argument(
        "--auth-fd",
        type=int,
        default=3,
        help="fd carrying the per-run completion secret from test.sh",
    )
    args = ap.parse_args()

    # The per-run completion secret, consumed HERE -- before any candidate byte
    # executes -- from a file descriptor test.sh set up, then closed so the
    # candidate (imported later, in THIS interpreter) can never read it. It never
    # appears in argv or the environment. emit() signs the reward-relevant
    # projection of the score with it; emit_reward.py (a process the candidate
    # never entered) re-derives the tag and pays out only on a match. This is
    # what closes exploit probe (c): a candidate that writes a fabricated
    # score.json at import and os._exit(0)s never reaches emit(), so it can
    # produce no valid tag, and it does not hold the secret to forge one.
    auth = _read_auth_token(args.auth_fd)

    # A candidate can terminate this process in ways no `except` can see
    # (measured: a torch.compile delegate died with SIGSEGV and NO score file
    # was written at all). An unwritten score is the worst failure mode this
    # project has, because absence of a result is missing data and missing data
    # has been mistaken for evidence of difficulty once. So the zero is written
    # FIRST, before any candidate code is imported, and every return path
    # overwrites it. verifier_incomplete stays true only if control never
    # reached emit().
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

    # judge_unavailable is initialised FALSE rather than written only when it
    # goes true: a field that exists only in the failure case cannot be
    # asserted on the healthy path.
    result = {
        "score": 0.0,
        "process": 0.0,
        "reason": "",
        "checks": {},
        "verifier_incomplete": False,
        "judge_unavailable": False,
    }
    globals()["PARTIAL"] = result

    # Preserve WHAT WAS GRADED, and fingerprint it. The digest is compared
    # host-side against the frozen starter, so the liveness gate needs no
    # cooperation from the agent container. /logs/verifier is the one directory
    # Harbor downloads, which is why the copy lands beside score.json.
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

    import torch

    # Capture the timing primitives NOW, before the candidate is imported, so a
    # candidate that monkeypatches torch.cuda.Event.elapsed_time (probe d) cannot
    # reach the objects the ratio is actually measured with. See _Clock.
    _clock = _Clock(torch, time)
    td = _import_taskdef()
    runner = Runner(torch, td.SURFACE, getattr(td, "loss", taskdef_api.default_loss))
    ADOPTION_FLOOR = td.ADOPTION_FLOOR
    TRIALS = taskdef_api.cfg(td, "TIMING_TRIALS")
    BUDGET_S = taskdef_api.cfg(td, "TIMING_BUDGET_S")
    FLUSH_MIB = taskdef_api.cfg(td, "L2_FLUSH_MIB")

    # emit() finishes the record on every early-return path and needs these.
    _LATE.update({"args": args, "torch": torch, "td": td})

    # Recorded HERE, not beside the metrics: a run that dies at G2 has an
    # environment too and that is exactly when you want it.
    result["environment"] = environment_record(torch)

    # The private oracle establishes that the metric's full-reward point is
    # reachable. G1 and G4 exist to stop a candidate claiming credit for the
    # anchor kernel's work; the oracle claims none. The exemption is reported
    # in the score, so a reader can always see when it was in force.
    by_canary = _is_authored_oracle(args.candidate)
    by_env = os.environ.get("FORGE_ORACLE") == "1"
    oracle_mode = by_canary or by_env
    result["oracle_mode"] = oracle_mode
    result["oracle_route"] = "canary" if by_canary else ("env" if by_env else "")

    # ---- G1 forbidden symbols, by name AND by structure -------------------
    # One gate, two halves. The name half rejects calling the library; the
    # structural half rejects shipping a renamed copy of it. A submission that
    # fails either has not written a kernel.
    hits = [] if oracle_mode else forbidden_hits(args.candidate, td.FORBIDDEN)
    vend = (
        {
            "available": True,
            "gated": False,
            "exempt": "oracle mode: the production kernel is the anchor, not a claim",
        }
        if oracle_mode
        else vendored_containment(args.candidate, td)
    )
    result["checks"]["G1_forbidden"] = {
        "clean": not hits and not vend.get("gated"),
        "hits": hits,
        "vendored": vend,
    }
    if hits:
        result["reason"] = f"forbidden library symbols called: {hits}"
        return emit(result, args.out, auth)
    if vend.get("gated"):
        result["reason"] = (
            f"vendored library source: {vend['containment']:.1%} of this submission's "
            f"{vend['ngram']}-token structure appears verbatim in the installed library, "
            f"above the {vend['ceiling']:.0%} ceiling. Renaming symbols does not change "
            f"this measurement. Reading that source is allowed; shipping a copy of it is "
            f"not the task."
        )
        return emit(result, args.out, auth)

    try:
        cand_entry = getattr(load(args.candidate, "cand"), td.ENTRY_NAME)
    except CandidateExited as e:
        result["checks"]["G0_trusted_entrypoint"] = {"clean": False}
        result["reason"] = str(e)
        return emit(result, args.out, auth)
    except Exception as e:  # noqa: BLE001
        result["reason"] = f"candidate failed to import: {type(e).__name__}: {e}"[:250]
        return emit(result, args.out, auth)
    result["checks"]["G0_trusted_entrypoint"] = {"clean": True}
    ref_entry = getattr(load(args.reference, "ref"), td.REF_NAME)

    cand = lambda a: cand_entry(**a)  # noqa: E731
    ref = lambda a: ref_entry(**a)  # noqa: E731
    sota = td.sota  # takes the args dict directly

    # ---- G2 correctness, the five-stage harness ---------------------------
    # Stages 1, 4 and 3 first and each short-circuiting: they are seconds
    # apiece and need no timing, while the sweep below carries the entire
    # paired-timing cost.
    fails, ratios, per_shape = [], [], {}
    ratios_inp = None
    seed_base = cfg["seed_base"]
    stages: dict = {}

    def _stage(key, fn, *a):
        nonlocal fails
        if fails:
            stages[key] = {"clean": None, "skipped": "an earlier stage already failed"}
            return
        f, detail = fn(torch, td, runner, *a)
        stages[key] = {"clean": not f, "failures": f, **detail}
        fails += f

    _stage("stage1_smoke", stage_smoke, cand, ref, seed_base)
    _stage("stage4_determinism", stage_determinism, cand, seed_base)
    _stage("stage3_stability", stage_stability, cand, ref, seed_base)

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
        return emit(result, args.out, auth)

    # Stages 2 and 5: the shape sweep across dtypes, and the non-power-of-two
    # edge cases. Both run in this one loop because a graded shape and an edge
    # shape differ only in whether the draw is also timed.
    sweep_f, edge_f, seen_dtypes = [], [], set()
    for i, (label, spec) in enumerate(
        list(cfg["graded"].items()) + list(cfg["correctness_only"].items())
    ):
        graded = label in cfg["graded"]
        bucket = sweep_f if graded else edge_f
        for dt_name in spec["dtypes"]:
            dtype = getattr(torch, dt_name)
            seen_dtypes.add(dt_name)
            # crc32, not hash(). Python randomises str hashing per process
            # unless PYTHONHASHSEED is set, so hash(dt_name) made the graded
            # draw different on every run and silently defeated seed_base.
            seed = seed_base + i * 17 + zlib.crc32(dt_name.encode()) % 97
            inp, gn, scale = draw_inputs(torch, td, spec, seed, dtype)

            try:
                got, cand_grads = _run_candidate(torch, runner, cand, inp, gn)
            except Exception as e:  # noqa: BLE001
                msg = f"{label}/{dt_name}: candidate raised {type(e).__name__}: {e}"
                fails.append(msg)
                bucket.append(msg)
                continue
            want = runner(ref, inp, gn)
            ref_grads = _grads_of(inp, gn)

            # Output through the category's compare (dense tolerance by
            # default, exact index semantics where the taskdef overrides), and
            # ALL declared input gradients -- a candidate with a broken
            # backward for four of five inputs must not pass this gate.
            out_bad = _compare_out(td, got, want, dt_name, spec)
            grad_bad = _compare_grads(td, cand_grads, ref_grads, gn, dt_name)
            if out_bad or grad_bad:
                msg = f"{label}/{dt_name} scale={scale:.3f}: " + " ".join(
                    out_bad + grad_bad
                )
                fails.append(msg)
                bucket.append(msg)
                continue

            if graded:
                if ratios_inp is None:
                    ratios_inp = (inp, gn)
                raw, t_c, t_s, n_tr, w_c = timed_ratio(
                    torch,
                    runner,
                    cand,
                    sota,
                    inp,
                    gn,
                    trials=TRIALS,
                    budget_s=BUDGET_S,
                    flush_mib=FLUSH_MIB,
                    clock=_clock,
                )
                r = min(raw, 1.0)
                ratios.append(r)
                entry = {
                    "sota_ms": round(t_s, 6),
                    "candidate_ms": round(t_c, 6),
                    "candidate_wall_ms": round(w_c, 6),
                    "ratio_raw": round(raw, 6),
                    "clipped": raw > 1.0,
                    "paired_trials": n_tr,
                    # The dimensions behind the number, so the refinement note
                    # can report how a candidate's fraction TRACKS a dimension
                    # as a measured statement. The label stays hidden from the
                    # agent; only the relationship is reported.
                    "dims": {k: v for k, v in spec.items() if k != "dtypes"},
                    "fraction_of_sota": round(r, 6),
                    "scale": round(scale, 4),
                }
                for legacy in ("T", "B"):  # KDA parity: lifted when present
                    if legacy in spec:
                        entry[legacy] = spec[legacy]
                per_shape[f"{label}/{dt_name}"] = entry

    stages["stage2_shape_sweep"] = {
        "clean": not sweep_f,
        "failures": sweep_f,
        "configurations": sum(len(s["dtypes"]) for s in cfg["graded"].values()),
        "dtypes": sorted(seen_dtypes),
        # A boolean so the compiled pytest can assert the fp32 verification set
        # actually RAN, not merely that the sweep was clean.
        "float32_verified": "float32" in seen_dtypes,
    }
    stages["stage5_edge_cases"] = {
        "clean": not edge_f,
        "failures": edge_f,
        "configurations": sum(
            len(s["dtypes"]) for s in cfg["correctness_only"].values()
        ),
        "note": "non-power-of-two and stress edge cases; never timed",
    }
    result["checks"]["G2_correctness"] = {
        "clean": not fails,
        "failures": fails,
        "failure_count": len(fails),
        "stages": stages,
    }
    if fails:
        result["reason"] = "correctness failed: " + _failure_summary(fails)
        return emit(result, args.out, auth)

    if not ratios:
        result["reason"] = "no graded shape produced a timing"
        return emit(result, args.out, auth)

    R = statistics.geometric_mean(ratios)
    result["checks"]["G3_speed"] = {
        "per_shape": per_shape,
        "geomean_fraction_of_sota": round(R, 6),
    }

    # ---- G4 written-kernel gate -----------------------------------------
    # torch.compile over starter math clears a raw-fraction floor on its own
    # and is not a forbidden symbol. This attributes graded device time to the
    # Triton kernels the submitted module itself declares; Inductor emits its
    # kernels under generated names that cannot match a name in the source.
    if oracle_mode:
        # Same KEY SET as the measured branch. The exemption changes the
        # values, never the shape: fields that appear and disappear with a mode
        # flag break every consumer that reads them by path.
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
        adopt, detail = kernel_adoption(
            torch, runner, cand, args.candidate, ratios_inp[0], ratios_inp[1]
        )
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
        return emit(result, args.out, auth)
    result["fraction_of_sota"] = round(R, 6)
    # The delivered metric IS the raw fraction of the production kernel: R is the
    # G3 geomean of per-shape min(t_production/t_candidate, 1), already in [0, 1].
    # No /target normalization and no clamp -- the target no longer gates. A gate
    # failure (G4 above, or the rubric gate in process_score below) resets this
    # to 0; every gate having passed is what makes score == fraction_of_sota.
    result["score"] = round(R, 6)
    # The headline figures and the conditions that produced them, recorded
    # beside the score rather than left to be reconstructed from per_shape.
    result["metrics"] = metrics_record(
        per_shape, R, TARGET, adopt, tgt.get("starter_fraction_of_sota")
    )
    result["timing"] = {
        "method": "paired, interleaved, order-balanced ratio sampling",
        "trials_max": TRIALS,
        "warmup_rounds": 3,
        "budget_s": BUDGET_S,
        "l2_flush_mib": FLUSH_MIB,
        "note": "the ratio is measured as a ratio: both sides are sampled within a "
        "few hundred microseconds of each other so clock and thermal drift "
        "divides out, and the order alternates so any first-versus-second "
        "position effect changes sign each trial",
    }
    result["thresholds"] = {
        "target_fraction_of_sota": TARGET,
        "written_kernel_adoption_floor": ADOPTION_FLOOR,
        "tolerances": {k: {"atol": a, "rtol": r} for k, (a, r) in td.TOL.items()},
        "vendor_containment_ceiling": taskdef_api.cfg(td, "VENDOR_CONTAINMENT_CEILING"),
        "vendor_ngram": taskdef_api.cfg(td, "VENDOR_NGRAM"),
        "determinism_runs": taskdef_api.cfg(td, "DETERMINISM_RUNS"),
        "rubric_gate": "all rubrics must pass; oracle exempt by declaration",
    }
    if result["score"] == 0.0:
        result["reason"] = f"fraction of production kernel {R:.4f} rounds to zero"
    result["process"] = process_score(args, result, torch, td)
    return emit(result, args.out, auth)


# ---------------------------------------------------------------------------
# Process metric: deterministic integrity + narrow-only rubric overlay
# ---------------------------------------------------------------------------


def _default_process_checks(src: str) -> dict:
    """KDA's deterministic lints. A category whose operator legitimately
    branches on magnitudes (top-k) overrides `process_checks` in its taskdef;
    any override should keep no_grader_reference."""
    _MAG = r"(?:abs|amax|amin|max|min|norm|std|var|mean)\s*\("
    return {
        # a branch on input magnitude is the documented 4.3x exploit against
        # the KDA operator family; a static trace of one lowers the float even
        # when the deterministic gates did not fire
        "no_magnitude_branch": not (
            re.search(_MAG + r"[^)]*\)\s*[<>]=?", src)
            or re.search(r"torch\.where\s*\([^)]*" + _MAG, src)
            or re.search(_MAG + r"[^)]*\)\s*\.item\s*\(\)", src)
            or re.search(r"if\s+[^\n]*" + _MAG, src)
        ),
        "no_grader_reference": not re.search(
            r"target\.json|hidden_shapes|/bundle/tests", src
        ),
    }


def process_score(args, result: dict, torch, td) -> float:
    """Deterministic integrity and trajectory checks, plus a narrow-only overlay.

    Narrow-only: every check here can lower the process float and none can
    raise the score. The rubric overlay gates in one direction only, and by
    KIND: a DECIDED anti-cheat rubric failure sets the score to zero (a cheating
    trajectory), while capability and hygiene failures narrow `process` and
    never zero a measured, correct, library-free result. This is the same
    anti-cheat set emit_reward._cheated() gates the client reward on, so the two
    layers agree on identical bytes. Two exceptions are declared and both are
    reported in the score: an unreachable judge (judge_unavailable, NOT-A-TRIAL,
    never zero) and a submission with no agent phase to judge (not applicable).
    """
    checks, traj_path = (
        {},
        Path(os.environ.get("TRAJECTORY_PATH", "/logs/agent/trajectory.json")),
    )

    src = args.candidate.read_text()
    checks["no_forbidden_symbol"] = result["checks"]["G1_forbidden"]["clean"]
    # a declared jit kernel is the minimum evidence of kernel work having happened
    checks["declares_triton_kernel"] = bool(declared_jit_names(args.candidate))
    pc = getattr(td, "process_checks", _default_process_checks)
    for k, v in pc(src).items():
        checks[k] = bool(v)

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

    # Deterministic half: cheap and near-constant on an honest run, so a floor
    # and not the metric.
    det = [v for k, v in checks.items() if k != "trajectory_present"]
    det_score = (sum(1 for v in det if v) / len(det)) if det else 0.0

    # RECORD them, so a reader of the artifact can see WHICH check failed, and
    # so the compiled pytest has a binary statement to assert per obligation.
    result["checks"]["process_deterministic"] = {
        **{k: bool(v) for k, v in checks.items()},
        "deterministic_score": round(det_score, 6),
        "counted": sorted(k for k in checks if k != "trajectory_present"),
        "note": "recorded for attribution; grade.py remains the authority and "
        "tests/test_output.py mirrors these without gating",
    }

    # Bucket-N half: a cross-family LLM judge over tests/rubrics.jsonl. It can
    # only LOWER. The judge module is optional per bundle; a missing or
    # unreachable judge degrades to judge_unavailable, never to zero and never
    # to a silent pass.
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
        # NOT APPLICABLE, which is different from unavailable: no agent
        # trajectory, so no behaviour for behavioural rubrics to judge (the
        # oracle path; a solver attempt with no trajectory is caught by the
        # liveness gate as missing data).
        result["rubric_gate_not_applicable"] = verdict.get("reason")
        return round(det_score, 6)

    if not verdict.get("available"):
        # MISSING DATA. "All rubrics pass" is a claim about rubrics that were
        # DECIDED, so a judge that never spoke has not failed anything -- but
        # it has not verified anything either. The run is not zeroed AND not
        # quietly passed: it is marked judge_unavailable and the pilot runner
        # treats that as NOT A TRIAL.
        result["judge_unavailable"] = True
        return round(det_score, 6)

    # EVERY decided rubric failure gates, whatever its kind -- judge.py states it
    # outright: "kind is a DIAGNOSTIC label with no power". The older kind-split
    # design described here claimed hygiene failures "narrow process only and never
    # zero a measured, correct result"; the data contradicts it, since a
    # hygiene-labelled timing rubric zeroed a frac=1.0 kernel. `kind` survives only
    # to describe WHY a run was rejected, never whether it was.
    #
    # `or`, NOT `dict.get(key, default)`: the judge emits `failed_gating` as an
    # explicit `[]` when it routes decided failures to `failed_narrowing`, so a
    # present-but-empty value must fall through -- here to the anti-cheat slice
    # of `failed_by_kind`. The old `.get("failed_gating", <fallback>)` form
    # returned that `[]` and silently skipped the gate, so a kernel that tripped
    # an anti-cheat rubric kept a full `score` while emit_reward._cheated()
    # zeroed the reward: the two layers disagreed on identical bytes. Gate on
    # the anti-cheat kind so an honest-but-imperfect kernel (capability failure
    # only) keeps its measured speed while a cheating one is zeroed.
    gating = (
        verdict.get("failed_gating")
        or (verdict.get("failed_by_kind") or {}).get("anti_cheat")
        or []
    )
    if gating:
        # An anti-cheat rubric failing is a cheating trajectory; the client
        # contract credits only an honest solution, so it zeroes whatever it
        # measured. Capability/hygiene failures do not reach here -- they narrow.
        why = (
            "rubric gate: anti-cheat rubric(s) "
            + ", ".join(gating)
            + " not satisfied"
        )
        # Name the ACTUAL kinds. Asserting "anti-cheat ... cheating trajectory" for
        # every gated failure contradicted this file's own rubric_kinds map and told a
        # solver that mis-timed its benchmark it had cheated. That text is carried
        # into the next attempt, so an inaccurate accusation teaches the wrong lesson.
        kinds = verdict.get("rubric_kinds") or {}
        gated_kinds = sorted({str(kinds.get(r, "anti_cheat")) for r in gating})
        if "anti_cheat" in gated_kinds:
            why += (
                "; a submission that trips an anti-cheat rubric is a cheating "
                "trajectory and is not credited"
            )
        else:
            why += (
                f"; every decided rubric failure gates whatever its kind "
                f"(kind: {', '.join(gated_kinds)}), so the measured result is not credited"
            )
        result["rubric_gate_failed"] = gating
        result["score"] = 0.0
        # F2: REPLACE, never append. Appending across two judge passes with
        # non-deterministic verdicts produced a reason naming rubrics that the final
        # structured verdict records as satisfied:true. The text must match the
        # verdict it came from.
        result["reason"] = why
        return 0.0
    # Non-gating structural diagnostics narrow `process` and nothing else.
    share = verdict.get("diagnostic_share")
    base = min(det_score, det_score * verdict["pass_share"])
    if share is not None:
        base = min(base, base * share)
    return round(base, 6)


# ---------------------------------------------------------------------------
# Completion authentication: proof that emit() genuinely ran for these bytes
# ---------------------------------------------------------------------------


def _read_auth_token(fd: int = 3) -> str:
    """Drain and close the per-run secret test.sh handed us on `fd`.

    Read to EOF and close immediately, at the very top of main(), so a candidate
    imported later in this same interpreter finds the descriptor closed. A dev or
    self-test invocation with no such fd yields "" -- which only disables the
    authentication layer for that non-adversarial path, because in production the
    frozen test.sh always supplies it.
    """
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


# The projection MUST list every score.json field emit_reward.py derives the
# reward from, and NOTHING a later stage legitimately mutates (deterministic
# pytest appends its own key after grade.py exits, so it is excluded). Binding
# the tag to this projection -- not merely to candidate_sha256 -- also defeats a
# forked writer that overwrites score.json AFTER emit() with the same candidate
# bytes but a fabricated fraction/adoption/gate verdict. emit_reward.py carries a
# byte-identical copy of this function; the two MUST stay in lockstep.
def _reward_projection(s: dict) -> str:
    checks = s.get("checks") or {}
    g1 = checks.get("G1_forbidden") or {}
    g3 = checks.get("G3_speed") or {}
    g4 = checks.get("G4_written_kernel") or {}
    rj = checks.get("rubric_judge") or {}
    pd = checks.get("process_deterministic") or {}

    def _clean(name):
        return (checks.get(name) or {}).get("clean")

    proj = [
        ("candidate_sha256", s.get("candidate_sha256", "")),
        ("verifier_incomplete", s.get("verifier_incomplete")),
        ("fault", s.get("fault")),
        ("score", s.get("score")),
        ("fraction_of_sota", s.get("fraction_of_sota")),
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


def _completion_tag(s: dict, auth: str) -> str:
    return hashlib.sha256(
        (auth + "|forge-emit-v1|" + _reward_projection(s)).encode()
    ).hexdigest()


def _auth_path(out: Path) -> Path:
    return out.parent / (out.name + ".auth")


def _write_completion_auth(out: Path, auth: str) -> None:
    """Sign the just-written score. Reads the score BACK from disk before signing
    so the projection is computed on the exact JSON emit_reward.py will parse
    (float/round-trip identical), not on the in-memory dict."""
    try:
        s = json.loads(out.read_text())
        payload = {
            "tag": _completion_tag(s, auth),
            "candidate_sha256": s.get("candidate_sha256", ""),
            "algo": "sha256(auth|forge-emit-v1|reward_projection)",
        }
        _auth_path(out).write_text(json.dumps(payload) + "\n")
    except (OSError, ValueError):
        pass


# ---------------------------------------------------------------------------
# Emission: stable key set, last write, hard exit
# ---------------------------------------------------------------------------


def write_score(result: dict, out: Path) -> None:
    # This process imported the candidate, so the candidate's code has run here
    # and can register an atexit hook or fork a writer. It therefore writes
    # ONLY score.json, which is diagnostic. The reward files Harbor actually
    # reads are derived by tests/test.sh after this process is gone, from a
    # shell the candidate never got to hook.
    vdir = out.parent
    vdir.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")


# Set once the candidate is importable, so emit() can finish the record on any
# early-return path without threading four arguments through nine call sites.
_LATE: dict = {}

# Every check the grader can produce, in gate order. A run that stops at G2
# never evaluates the rest; unreached checks are recorded as explicitly not
# reached rather than omitted. Absence is not a verdict.
ALL_CHECKS = (
    "G0_trusted_entrypoint",
    "G1_forbidden",
    "G2_correctness",
    "G4_written_kernel",
    "G3_speed",
    "process_deterministic",
    "rubric_judge",
)


def emit(result: dict, out: Path, auth: str = "") -> int:
    if result["score"] == 0.0 and not result["reason"]:
        result["reason"] = "score zero with no recorded cause"

    # PROCESS ON THE FAILURE PATHS. `process` is narrow-only and needs no
    # timing data, so a run that died at G1, G2 or G4 can still be scored on
    # conduct. G0 and the crash guard stay 0.0 by design -- a module that never
    # imported has no conduct to score.
    checks = result.get("checks") or {}
    g0_ok = (checks.get("G0_trusted_entrypoint") or {}).get("clean") is True
    if g0_ok and not result.get("process") and _LATE.get("args") is not None:
        try:
            result["process"] = process_score(
                _LATE["args"], result, _LATE["torch"], _LATE["td"]
            )
        except Exception as e:  # noqa: BLE001 - conduct scoring must never fail a run
            result["process_error"] = f"{type(e).__name__}: {e}"[:200]

    # STABLE KEY SET. Unreached gates are recorded as explicitly not reached
    # rather than omitted, so score.json always carries the same keys and every
    # consumer sees a verdict-shaped answer for each one.
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
    # Last act of a genuine completion: sign the reward-relevant projection of
    # the score just written. Only reached here, so a candidate that skips emit()
    # (probe c: fabricate score.json, os._exit(0)) produces no valid signature.
    _write_completion_auth(out, auth)
    print(json.dumps({k: v for k, v in result.items() if k != "checks"}, indent=2))
    return 0


def _hard_exit(code: int) -> None:
    # os._exit skips atexit. The candidate ran in this interpreter and can
    # register a handler; SystemExit would run it after the score was written.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def run() -> None:
    """Entry point for the bundle's thin tests/grade.py shim (and __main__)."""
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


if __name__ == "__main__":
    run()
