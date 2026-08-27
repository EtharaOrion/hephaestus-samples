#!/usr/bin/env python3
"""TASKDEF PROTOCOL v1 — the contract every kernel-category module implements.

A *taskdef* is one Python module, `taskdef.py`, living in a category directory
(`seed/forge/categories/<name>/`). The generic verifier `grade_core.py` imports
it by the literal name `taskdef` from its own directory (the scaffolder copies
it beside grade_core.py into the bundle's `tests/`), and every category-specific
value the verifier uses comes from that module. grade_core carries NO category
knowledge of its own; the KDA bundle's grade.py is the machinery being
generalized here, and its gate order, seeding convention, zero-score-first emit
contract and stable score.json key set are preserved exactly.

The protocol is designed to cover four structurally different categories:

  topk        forward-only; entry returns (values, indices); correctness is
              EXACT index-set / tie-break semantics, not dense tolerance;
              no gradients anywhere.                    -> SURFACE="fwd",
              GRAD-free make_inputs, `compare` OVERRIDDEN.
  moe_swiglu  forward+backward differentiable dense op whose args include
              integer routing tensors that carry no grad. -> SURFACE="fwdbwd",
              make_inputs returns grad_names covering only the float inputs.
  moe_routed  forward+backward; routing computed inside the op itself.
              -> SURFACE="fwdbwd"; default dense compare usually suffices.
  gdn_decode  forward-only recurrent decode step with an initial-state input
              and a TUPLE output (out, new_state).       -> SURFACE="fwd";
              default compare already walks tuples.

========================  REQUIRED ATTRIBUTES  ========================

ENTRY_NAME : str
    Name of the callable `/workspace/kernel.py` must export. Called as
    `entry(**args)` where `args` is the dict make_inputs returned.

REF_NAME : str
    Name of the callable the frozen `tests/reference.py` exports. Same calling
    convention. Convention: f"{ENTRY_NAME}_ref".

SIGNATURE : str
    Human-readable signature of the entry point, for docs/instruction prose.
    e.g. "gated_delta_rule(q[B,T,H,K], k, v, g, beta) -> [B,T,HV,V]".

SURFACE : str, one of "fwd" | "fwdbwd"
    "fwdbwd": every candidate/reference/anchor call runs forward plus
    `loss(out).backward()`; gradients of every name in grad_names are graded,
    timed and determinism-checked. "fwd": no backward anywhere — correctness,
    determinism, stability, timing and adoption all run forward only.

make_inputs(shape: dict, seed: int, scale: float, dtype, device: str)
        -> (args: dict[str, object], grad_names: tuple[str, ...])
    One reproducible draw. `shape` is one entry of hidden_shapes.json (its
    "dtypes" key removed). `seed` fully determines the draw: seed your own
    `torch.Generator(device)` with it (never Python `hash()` anywhere).
    `scale` is the hidden per-invocation magnitude grade_core drew from
    SCALE_RANGE — multiply your nominal distribution by it; the same draw is
    both checked and timed, which is the distribution-invariance property
    (frontier-defeat-analysis §7.1). `args` may contain non-tensor values and
    integer tensors (routing); only names listed in grad_names get
    `.requires_grad_()` treatment/gradient grading, and grad_names MUST be
    empty when SURFACE == "fwd".

sota(args: dict) -> output
    The production-anchor invocation (the denominator of every graded ratio).
    Receives the SAME args dict the candidate got. Must be side-effect
    compatible with repeated calls. For fwdbwd surfaces its output must
    support the same `loss(...)` backward.

TOL : dict[str, tuple[float, float]]
    dtype-name -> (atol, rtol) for the default dense comparison and for ALL
    gradient comparisons. MEASURED per category (calibrate on the oracle;
    KDA's fp32 knee had to be re-derived), never copied from a paper table.

FORBIDDEN : list[str]
    Library symbols whose word-boundary appearance in comment-stripped
    candidate source zeroes at G1. May be empty for a category with no anchor
    library importable from the candidate's environment.

VENDOR_LIB_PACKAGE : str
    Import name of the anchor library whose source the shingle scan compares
    against (e.g. "fla"). "" disables the structural scan.

VENDOR_LIB_SUBDIRS : tuple[str, ...]
    Subdirectories (relative to the installed package root, "/"-joined) whose
    .py files feed the 32-token rename-normalized shingle set. EMPTY TUPLE =>
    grade_core skips the shingle scan entirely (recorded as available:false,
    never gated) while still running the FORBIDDEN symbol scan.

STABILITY_MODES : dict[str, callable] (ordered; may be empty)
    name -> input-maker with the same signature as make_inputs. Stage 3 runs
    each maker once at STABILITY_SHAPE and gates ONLY on finiteness-and-not-
    raising where the reference is finite (tolerance drift is recorded, never
    gated). Integer output tensors are exempt from finiteness by construction.

SMOKE_SHAPE / DETERMINISM_SHAPE / STABILITY_SHAPE : dict
    Category-shaped dicts (same keys as hidden_shapes.json entries, minus
    "dtypes") for stages 1, 4 and 3.

ADOPTION_FLOOR : float
    G4 minimum share of graded device time spent inside @triton.jit kernels
    the candidate module itself declares. Per-category headroom measurement
    (golden stage) should inform it; KDA ships 0.60.

NEGATIVE_CONTROLS : list[dict]
    Mutation specs consumed by controls_runner.py. Each:
      {
        "name":        str,
        "base":        "oracle" | "starter"      (which source gets mutated),
        "target_file": str                       (bundle-root-relative, e.g.
                                                  "kernel.py"; the mutated copy
                                                  becomes workspace/kernel.py),
        "mutate":      callable(src: str) -> str
                       OR (anchor: str, replacement: str)  — anchor ABSENT in
                       the base source raises INERT and FAILS the control,
        "bound_gate":  "G0"|"G1"|"G2"|"G3"|"G4"|"process",
        "oracle_env":  bool (default False) — run this arm with FORGE_ORACLE=1
                       so G1/G4 exemptions cannot fire before the bound gate
                       (the NCDP pattern: plant the defect ON the oracle),
        "expectation": str  — prose of what must happen,
      }
    Both-halves rule (Invariant 17): a zero produced by any gate OTHER than
    bound_gate FAILS the control.

========================  OPTIONAL ATTRIBUTES  ========================
(defaults in parentheses; grade_core resolves via getattr)

compare(candidate_out, ref_out, dtype_name: str, shape: dict) -> list[str]
    (default: `taskdef_api.default_compare` bound with TOL). Returns failure
    strings, [] when equivalent. Override for non-dense semantics — topk MUST
    override with exact index-set/tie-break comparison (e.g. values compared
    exactly per rank, indices as sets within tied-value groups). The default
    walks tensors, tuples/lists (by position) and dicts (by key): float
    tensors dense-tolerance per TOL, integer/bool tensors exact equality.

loss(out) -> scalar tensor   (default: `taskdef_api.default_loss`)
    Reduction used for the backward on fwdbwd surfaces: sum over every
    floating tensor found in the output structure.

SCALE_RANGE : (lo, hi)                        ((0.35, 2.5))
draw_scale(gen: torch.Generator) -> float     (uniform over SCALE_RANGE)
SMOKE_DTYPE / DETERMINISM_DTYPE / STABILITY_DTYPE : str   ("bfloat16")
DETERMINISM_RUNS : int                        (3)
TIMING_TRIALS : int                           (15)
TIMING_BUDGET_S : float                       (50.0)   per graded shape set
L2_FLUSH_MIB : int                            (128)
VENDOR_NGRAM : int                            (32)
VENDOR_CONTAINMENT_CEILING : float            (0.40)  — recalibrate per category
TARGET_FRACTION_OF_SOTA : float               (0.95)  — measure.py seed value;
    the authoritative runtime value is tests/target.json, set below the
    oracle's repeated worst case.
FRONTIER_DEFEAT_FLOOR_S : float               (0.15)  — consumed as given,
    never lowered.
ANTI_CHEAT_RUBRIC_IDS : list[str]             (KDA's nine-id set) — must match
    ids in tests/rubrics.jsonl; scaffold writes them to
    tests/anti_cheat_rubrics.json for emit_reward.py.
process_checks(src: str) -> dict[str, bool]   (KDA's deterministic regex set:
    no_magnitude_branch, no_grader_reference). Override when the operator's
    own math legitimately branches on magnitudes — topk — so the
    magnitude-branch lint does not false-flag honest kernels; keep
    no_grader_reference in any override.
ORACLE_ENV_FOR_GOLDEN : bool                  (False) — set True only when the
    category oracle wraps the anchor library itself (KDA-style 19-line
    wrapper) and cannot pass G1/G4 honestly.
TASK_NAME : str        (f"hephaestus/{category_dir_name}") for task.toml.

========================  FILES BESIDE taskdef.py  ====================

hidden_shapes.json : {"seed_base": int,
                      "graded": {label: {<dims...>, "dtypes": [...]}},
                      "correctness_only": {label: {...}}}
    Seeding convention (grade_core enforces, zlib.crc32 NEVER hash()):
      sweep draw i (over graded+correctness_only concatenated, in file order):
          seed_base + i*17 + zlib.crc32(dtype_name)%97
      smoke: seed_base+9001   determinism: seed_base+9002
      stability mode i: seed_base+9100+7*i
target.json    : {"target_fraction_of_sota", "starter_fraction_of_sota",
                  "frontier_defeat_floor_S"} (measure.py derives it)
reference.py   : exports REF_NAME; plain-PyTorch oracle math, fp32 where the
                 recurrence demands it. The verifier uses its own frozen copy.
kernel.py      : the starter the agent edits. Must be CORRECT (G2-clean).
oracle_kernel.py : the private full-reward solution controls_runner grades.
instruction.md / bench.py / rubrics.jsonl / diagnostics.jsonl / task.toml :
                 per-category authored surfaces (see seed/bundle-wip for the
                 exemplar; rubrics lines are exactly {"id", "rubric"}).
"""

from __future__ import annotations

# Default set carried from the KDA bundle's emit_reward.py; a category
# overriding ANTI_CHEAT_RUBRIC_IDS replaces it wholesale.
DEFAULT_ANTI_CHEAT_RUBRIC_IDS = [
    "no_result_memoization",
    "no_call_count_shortcut",
    "no_timing_context_detection",
    "kernels_actually_launch",
    "no_value_dependent_dispatch",
    "no_judge_manipulation",
    "timing_measured_on_shipped_bytes",
    "timing_uses_synchronised_clock",
    "timing_covers_forward_and_backward",
]

REQUIRED_ATTRS = (
    "ENTRY_NAME", "REF_NAME", "SIGNATURE", "SURFACE", "make_inputs", "sota",
    "TOL", "FORBIDDEN", "VENDOR_LIB_PACKAGE", "VENDOR_LIB_SUBDIRS",
    "STABILITY_MODES", "SMOKE_SHAPE", "DETERMINISM_SHAPE", "STABILITY_SHAPE",
    "ADOPTION_FLOOR", "NEGATIVE_CONTROLS",
)

DEFAULTS = {
    "SCALE_RANGE": (0.35, 2.5),
    "SMOKE_DTYPE": "bfloat16",
    "DETERMINISM_DTYPE": "bfloat16",
    "STABILITY_DTYPE": "bfloat16",
    "DETERMINISM_RUNS": 3,
    "TIMING_TRIALS": 15,
    "TIMING_BUDGET_S": 50.0,
    "L2_FLUSH_MIB": 128,
    "VENDOR_NGRAM": 32,
    "VENDOR_CONTAINMENT_CEILING": 0.40,
    "TARGET_FRACTION_OF_SOTA": 0.95,
    "FRONTIER_DEFEAT_FLOOR_S": 0.15,
    "ANTI_CHEAT_RUBRIC_IDS": DEFAULT_ANTI_CHEAT_RUBRIC_IDS,
    "ORACLE_ENV_FOR_GOLDEN": False,
}


def validate_taskdef(mod) -> list:
    """Return a list of protocol violations (empty when the module conforms)."""
    errs = [f"missing required attribute {a!r}" for a in REQUIRED_ATTRS
            if not hasattr(mod, a)]
    surface = getattr(mod, "SURFACE", None)
    if surface not in ("fwd", "fwdbwd"):
        errs.append(f"SURFACE must be 'fwd' or 'fwdbwd', got {surface!r}")
    tol = getattr(mod, "TOL", {})
    if isinstance(tol, dict):
        for k, v in tol.items():
            if not (isinstance(v, (tuple, list)) and len(v) == 2):
                errs.append(f"TOL[{k!r}] must be (atol, rtol)")
    else:
        errs.append("TOL must be a dict dtype_name -> (atol, rtol)")
    for nc in getattr(mod, "NEGATIVE_CONTROLS", []) or []:
        for key in ("name", "target_file", "mutate", "bound_gate", "expectation"):
            if key not in nc:
                errs.append(f"negative control {nc.get('name', '?')!r} missing {key!r}")
        if nc.get("base", "oracle") not in ("oracle", "starter"):
            errs.append(f"negative control {nc.get('name', '?')!r}: base must be oracle|starter")
    return errs


def cfg(mod, name: str):
    """Resolve an optional taskdef attribute against the protocol defaults."""
    if name in DEFAULTS:
        return getattr(mod, name, DEFAULTS[name])
    return getattr(mod, name)


# ---------------------------------------------------------------------------
# Default comparison / loss / traversal helpers. Pure functions of tensors so
# every category (and grade_core itself) shares one implementation.
# ---------------------------------------------------------------------------

def flatten_output(out) -> list:
    """Depth-first list of tensors inside a tensor / tuple / list / dict output.

    None entries disappear; dicts walk in sorted-key order so two structurally
    equal outputs flatten identically. Anything that is not a tensor and not a
    container is ignored (scalars carried beside tensors are not graded).
    """
    import torch
    acc = []
    if out is None:
        return acc
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (tuple, list)):
        for o in out:
            acc.extend(flatten_output(o))
        return acc
    if isinstance(out, dict):
        for k in sorted(out):
            acc.extend(flatten_output(out[k]))
        return acc
    return acc


def close_dense(got, want, atol: float, rtol: float):
    """KDA's `close`: finite, then |got-want| <= atol + rtol*|want| everywhere."""
    import torch
    if got is None or want is None:
        return False, "missing tensor"
    if not torch.isfinite(got).all():
        return False, "non-finite values in candidate output"
    d = (got.float() - want.float()).abs()
    lim = atol + rtol * want.float().abs()
    bad = int((d > lim).sum())
    return bad == 0, f"{bad} elements outside tolerance, max abs {float(d.max()):.3e}"


def default_compare(got, want, dtype_name: str, shape: dict, tol: dict) -> list:
    """Structure-walking dense comparison; the overridable G2 semantics seam.

    Float tensors: tolerance per `tol[dtype_name]`. Integer/bool tensors: exact
    equality (routing tables, index outputs). Structure mismatch is itself a
    failure. Categories with permutation/tie-break semantics (topk) override
    `compare` in their taskdef instead of bending this one.
    """
    import torch
    ct, rt = flatten_output(got), flatten_output(want)
    if len(ct) != len(rt):
        return [f"output structure mismatch: candidate has {len(ct)} tensors, "
                f"reference has {len(rt)}"]
    atol, rtol = tol[dtype_name]
    fails = []
    for j, (c, r) in enumerate(zip(ct, rt)):
        tag = f"out[{j}]" if len(ct) > 1 else "out"
        if c is None or r is None:
            fails.append(f"{tag}[missing tensor]")
            continue
        if c.shape != r.shape:
            fails.append(f"{tag}[shape {tuple(c.shape)} != {tuple(r.shape)}]")
            continue
        if r.is_floating_point():
            ok, det = close_dense(c, r, atol, rtol)
            if not ok:
                fails.append(f"{tag}[{det}]")
        else:
            if c.dtype != r.dtype or not torch.equal(c, r):
                n = int((c != r).sum()) if c.shape == r.shape and c.dtype == r.dtype else -1
                fails.append(f"{tag}[integer tensor differs"
                             + (f", {n} elements" if n >= 0 else "") + "]")
    return fails


def default_loss(out):
    """Scalar reduction for the fwdbwd backward: sum of every float tensor."""
    total = None
    for t in flatten_output(out):
        if t.is_floating_point():
            s = t.sum()
            total = s if total is None else total + s
    if total is None:
        raise RuntimeError("default_loss: output contains no floating tensor to reduce")
    return total
