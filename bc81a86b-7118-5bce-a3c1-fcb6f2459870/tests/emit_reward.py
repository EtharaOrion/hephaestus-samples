#!/usr/bin/env python3
"""Derive the reward files Harbor reads, from a process the candidate never entered.

grade.py imports the candidate, so anything it writes can be overwritten by an
atexit hook or a forked writer the candidate registered. This runs afterwards,
from tests/test.sh, and is the last writer. It also reads its own write back, so
a candidate that forked a delayed racer is recorded rather than credited.
"""

import hashlib
import json
import os
import pathlib
import sys

score_path, vdir = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])


def _read_auth_token(fd: int = 3) -> str:
    """The per-run secret test.sh handed us on fd 3. This process never imported
    the candidate, so the secret cannot have been observed by candidate code.
    That is the whole basis for trusting the completion tag."""
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


try:
    s = json.loads(score_path.read_text())
except (ValueError, OSError) as exc:
    s = {
        "score": 0.0,
        "process": 0.0,
        "reason": f"score file unreadable: {exc}"[:200],
    }
    score_path.write_text(json.dumps(s, indent=2) + "\n")


def num(key: str) -> float:
    try:
        return float(s.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _rubrics_fraction() -> float:
    rj = (s.get("checks") or {}).get("rubric_judge") or {}
    if not rj.get("available"):
        return 0.0
    ps = rj.get("pass_share")
    if isinstance(ps, (int, float)) and not isinstance(ps, bool):
        return float(ps)
    verdicts = [
        v
        for v in (rj.get("verdicts") or {}).values()
        if isinstance(v, dict) and v.get("satisfied") is not None
    ]
    if not verdicts:
        return 0.0
    return sum(1 for v in verdicts if v.get("satisfied")) / len(verdicts)


def _pytest_fraction() -> float:
    dp = s.get("deterministic_pytest")
    if not isinstance(dp, dict):
        dp = {}
    total = (dp.get("passed") or 0) + (dp.get("failed") or 0) + (dp.get("errors") or 0)
    return ((dp.get("passed") or 0) / total) if total else 0.0


def _gate_clean(prefix: str) -> bool:
    for k, v in (s.get("checks") or {}).items():
        if k.startswith(prefix + "_") and isinstance(v, dict):
            if "clean" in v:
                return v.get("clean") is True
            if prefix == "G4":
                a, thr = v.get("adoption"), v.get("threshold", 0.6)
                return v.get("exempt") is True or (a is not None and a >= thr)
    return False


# score = fraction_of_sota when the run is VALID and not cheating, else 0. Validity
# is G0/G1/G2 correctness AND the G4 written-kernel adoption gate: a wrong answer, a
# library call, or a sub-threshold own-Triton share each hard-zeros the reward, as
# does a cheating trajectory (anti-cheat rubrics + deterministic integrity checks).
# Adoption is a GATE, not a multiplier -- it no longer scales the reward down, it
# either clears its floor or the reward is exactly 0. Above the gates the reward is
# the raw geomean fraction of the production kernel, matching score.json's `score`.
ANTI_CHEAT_RUBRICS = frozenset(
    {
        "no_result_memoization",
        "no_call_count_shortcut",
        "no_timing_context_detection",
        "kernels_actually_launch",
        "no_value_dependent_dispatch",
        "no_judge_manipulation",
        "timing_measured_on_shipped_bytes",
        "timing_uses_synchronised_clock",
        "timing_covers_forward_and_backward",
    }
)


def _agent_completed() -> bool:
    if s.get("verifier_incomplete") is True:
        return False
    if s.get("fault") in ("candidate", "harness"):
        return False
    return bool(s.get("candidate_sha256"))


def _cheated() -> bool:
    rj = (s.get("checks") or {}).get("rubric_judge") or {}
    if (
        set(rj.get("failed_gating") or rj.get("failed_rubrics") or [])
        & ANTI_CHEAT_RUBRICS
    ):
        return True
    pd = (s.get("checks") or {}).get("process_deterministic") or {}
    return (
        pd.get("no_magnitude_branch") is False or pd.get("no_grader_reference") is False
    )


def _rubric_gate_failed() -> bool:
    """ANY decided rubric failure zeros the reward, matching grade.py's gate and
    task.toml's declared contract, 'zero unless every gate passes'. Restricting
    this to ANTI_CHEAT_RUBRICS let a capability failure such as fused_chunk_carry
    pay out a non-zero reward.json while score.json recorded zero for the same
    rollout, and Harbor reads reward.json. An unavailable judge is missing data,
    not failure, and is handled by the existing judge_unavailable path."""
    rj = (s.get("checks") or {}).get("rubric_judge") or {}
    if not rj.get("available"):
        return False
    return bool(rj.get("failed_gating") or rj.get("failed_rubrics"))


def _completion_authenticated() -> bool:
    """grade.py's emit() signed the score's reward-relevant projection with the
    per-run secret and left the tag beside score.json. A fabricated score.json --
    written by candidate code that never reached emit() and never held the secret
    -- carries no matching tag and pays out zero. A post-emit overwrite is
    rejected on the same basis: the projection it was signed over changed."""
    auth = _read_auth_token()
    ap = score_path.parent / (score_path.name + ".auth")
    try:
        va = json.loads(ap.read_text())
    except (OSError, ValueError):
        return False
    if not isinstance(va, dict):
        return False
    expect = hashlib.sha256(
        (auth + "|task-emit-v1|" + _reward_projection(s)).encode()
    ).hexdigest()
    return va.get("tag") == expect


def _fraction() -> float:
    f = s.get("fraction_of_sota")
    if isinstance(f, (int, float)) and not isinstance(f, bool):
        return float(f)
    g3 = (s.get("checks") or {}).get("G3_speed") or {}
    return float(g3.get("geomean_fraction_of_sota") or 0.0)


def _adoption() -> float:
    g4 = (s.get("checks") or {}).get("G4_written_kernel") or {}
    if g4.get("exempt"):
        return 1.0
    a = g4.get("adoption")
    return float(a) if isinstance(a, (int, float)) and not isinstance(a, bool) else 0.0


_authentic = _completion_authenticated()
_valid = (
    _authentic
    and _agent_completed()
    and _gate_clean("G0")
    and _gate_clean("G1")
    and _gate_clean("G2")
    and _gate_clean("G4")
    and not _rubric_gate_failed()
)
score = _fraction() if (_valid and not _cheated()) else 0.0
if not _authentic:
    s["completion_unauthenticated"] = True
    s["reason"] = (
        "reward withheld: grade.py emit() left no valid completion signature "
        "beside score.json, so these bytes are not a genuine verifier completion"
    )
    score_path.write_text(json.dumps(s, indent=2) + "\n")
reward = {
    "score": round(score, 6),
    "fraction_of_sota": round(_fraction(), 6),
    "adoption": round(_adoption(), 6),
    "rubrics": round(_rubrics_fraction(), 6),
    "pytest": round(_pytest_fraction(), 6),
}
# Mirror the headline metrics so a consumer reading only reward.json sees what
# was achieved, not just the scalar. Copied rather than recomputed: this process
# never entered the candidate, and recomputing would mean trusting numbers the
# candidate could have influenced. Absent keys are simply omitted.
_m = s.get("metrics")
if not isinstance(_m, dict):
    _m = {}
_spill = {}
for _k in (
    "speedup_vs_production",
    "slowdown_vs_production",
    "speedup_vs_starter",
    "starter_fraction_of_sota",
    "score_log_scaled",
    "written_kernel_adoption",
    "configurations_timed",
    "candidate_ms_geomean",
    "production_ms_geomean",
    "best_shape",
    "worst_shape",
    "target_fraction_of_sota",
):
    if _k not in _m:
        continue
    # `best_shape` and `worst_shape` are dicts -- {"shape": ..., "fraction": ...} --
    # so they are routed to the sidecar rather than copied. They were being copied
    # verbatim, which meant every iteration that got far enough to be TIMED would
    # have carried two dicts into reward.json; only iterations that died before
    # timing escaped it, which is why the first bundle to fail this way failed on
    # the environment block instead. Filter at the source: the guard below is an
    # assertion, not the mechanism.
    if isinstance(_m[_k], (int, float)) and not isinstance(_m[_k], bool):
        reward[_k] = _m[_k]
    else:
        _spill[_k] = _m[_k]
# reward.json IS NUMERIC-ONLY, and the environment block is why this rule exists.
#
# Harbor parses every key under `rewards` as a reward metric and validates each
# value as a number. A nested dict does not degrade gracefully -- it aborts the
# trial. Measured on bundle 03b0ff1f iteration 1: harbor recorded
#
#   2 validation errors for VerifierResult
#   rewards.environment.float  Input should be a valid number ... input_type=dict
#
# and `verifier_result` came back EMPTY. The grade itself survived only because
# this harness reads score.json directly; anyone running the bundle through
# harbor -- which is the delivery target -- would have seen every iteration
# report no reward at all. The fla bundle hit this same wall and its contract
# test has said so since: numeric-only here, detail in a sibling file harbor
# does not parse.
_env = s.get("environment") if isinstance(s.get("environment"), dict) else {}
_diag = {}
if _env:
    _diag["environment"] = {
        k: _env[k]
        for k in ("gpu_name", "torch_version", "triton_version", "cuda_version")
        if k in _env
    }
_diag["judge_unavailable"] = bool(s.get("judge_unavailable"))
_diag.update(_spill)
(vdir / "reward_diagnostics.json").write_text(json.dumps(_diag, indent=2) + "\n")

# bool is a subclass of int and validates cleanly, so this one may stay numeric.
reward["judge_unavailable"] = int(bool(s.get("judge_unavailable")))

_bad = {
    k: type(v).__name__
    for k, v in reward.items()
    if not isinstance(v, (int, float)) or isinstance(v, bool)
}
if _bad:
    # Fail loudly here rather than let harbor reject the whole trial later, where
    # the message names pydantic and not the key that caused it.
    raise SystemExit(f"reward.json must be numeric-only; non-numeric keys: {_bad}")

(vdir / "reward.txt").write_text(f"{score}\n")
(vdir / "reward.json").write_text(json.dumps(reward, indent=2) + "\n")

back = (vdir / "reward.txt").read_text().strip()
if back != str(score):
    s["reward_tampering_detected"] = True
    s["reason"] = (
        f"reward tampering: reward.txt held {back!r} after the verifier wrote "
        f"{score!r}, so a candidate-spawned writer raced the grader"
    )
    score_path.write_text(json.dumps(s, indent=2) + "\n")
    (vdir / "reward.txt").write_text("0.0\n")
    (vdir / "reward.json").write_text(
        json.dumps(
            {"score": 0.0, "rubrics": 0.0, "pytest": 0.0, "fraction_of_sota": 0.0},
            indent=2,
        )
        + "\n"
    )
    print("REWARD TAMPERING DETECTED, zeroed")
