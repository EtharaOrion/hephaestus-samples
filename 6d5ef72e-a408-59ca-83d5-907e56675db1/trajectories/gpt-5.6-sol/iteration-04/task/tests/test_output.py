"""GENERATED SECTION. DO NOT HAND-EDIT. Source of truth: solution/grounding.yaml

Compiled rubric tests, one per `mode: compiled` item in solution/rubrics.jsonl.

FORGE Phase 2 item 7e proves completeness by identifier SET EQUALITY between the
compiled items and the tests here. These are generated from the rubric rather
than hand-written, because a hand-written set silently falls out of sync the
moment the rubric changes: the previous version had 4 tests against 13 items and
still reported itself complete.

They carry the implied relation only, no criterion prose and no reference text,
per item 10f.
"""

import json
import os
from pathlib import Path

import pytest

SCORE = Path(os.environ.get("SCORE_PATH", "/logs/verifier/score.json"))
TARGET = json.loads((Path(__file__).resolve().parent / "target.json").read_text())


def _s() -> dict:
    if not SCORE.is_file():
        pytest.fail("no score artifact")
    return json.loads(SCORE.read_text())


def _gate(prefix: str) -> dict:
    checks = _s().get("checks", {})
    for k, v in checks.items():
        if k.startswith(prefix + "_"):
            return v
    pytest.fail(f"no check recorded for gate {prefix}")


def _resolve(path: str):
    """Walk a dotted field path into score.json, failing if any segment is absent.

    This is what makes a compiled item over score.json non-vacuous: the item's
    criterion quotes field paths, and the generated test asserts those exact
    paths resolve. An item naming a field the verifier never writes therefore
    fails here rather than passing silently, which is how a previous rubric came
    to carry compiled criteria about an ordering record, per-trial NaN reports
    and a matched-symbol count, none of which score.json contains.
    """
    node = _s()
    for seg in path.split("."):
        if not isinstance(node, dict) or seg not in node:
            pytest.fail(f"score.json has no field {path!r} (missing at {seg!r})")
        node = node[seg]
    return node


def test_G1vendored() -> None:
    assert _gate("G1")["clean"] is True
    assert _gate("G1")["hits"] == [], "forbidden symbols recorded"
    _resolve("checks.G1_forbidden.vendored.available")
    _resolve("checks.G1_forbidden.vendored.gated")
    assert _resolve("checks.G1_forbidden.vendored.available") is True, "checks.G1_forbidden.vendored.available is not true"
    assert _resolve("checks.G1_forbidden.vendored.gated") is not True, "checks.G1_forbidden.vendored.gated is true"


def test_G2stages() -> None:
    g2 = _gate("G2")
    assert g2["clean"] is True
    assert g2["failure_count"] == 0, "failure_count is non-zero"
    assert not g2["failures"], "failures list is non-empty"
    _resolve("checks.G2_correctness.stages.stage1_smoke.clean")
    _resolve("checks.G2_correctness.stages.stage2_shape_sweep.clean")
    _resolve("checks.G2_correctness.stages.stage3_stability.clean")
    _resolve("checks.G2_correctness.stages.stage4_determinism.clean")
    _resolve("checks.G2_correctness.stages.stage5_edge_cases.clean")
    assert _resolve("checks.G2_correctness.stages.stage1_smoke.clean") is True, "checks.G2_correctness.stages.stage1_smoke.clean is not true"
    assert _resolve("checks.G2_correctness.stages.stage2_shape_sweep.clean") is True, "checks.G2_correctness.stages.stage2_shape_sweep.clean is not true"
    assert _resolve("checks.G2_correctness.stages.stage3_stability.clean") is True, "checks.G2_correctness.stages.stage3_stability.clean is not true"
    assert _resolve("checks.G2_correctness.stages.stage4_determinism.clean") is True, "checks.G2_correctness.stages.stage4_determinism.clean is not true"
    assert _resolve("checks.G2_correctness.stages.stage5_edge_cases.clean") is True, "checks.G2_correctness.stages.stage5_edge_cases.clean is not true"


def test_DTYPEcoverage() -> None:
    _resolve("checks.G2_correctness.stages.stage2_shape_sweep.float32_verified")
    assert _resolve("checks.G2_correctness.stages.stage2_shape_sweep.float32_verified") is True, "checks.G2_correctness.stages.stage2_shape_sweep.float32_verified is not true"


def test_PROCdeterministic() -> None:
    _resolve("checks.process_deterministic.no_forbidden_symbol")
    _resolve("checks.process_deterministic.no_magnitude_branch")
    _resolve("checks.process_deterministic.no_grader_reference")
    _resolve("checks.process_deterministic.declares_triton_kernel")
    assert _resolve("checks.process_deterministic.no_forbidden_symbol") is True, "checks.process_deterministic.no_forbidden_symbol is not true"
    assert _resolve("checks.process_deterministic.no_magnitude_branch") is True, "checks.process_deterministic.no_magnitude_branch is not true"
    assert _resolve("checks.process_deterministic.no_grader_reference") is True, "checks.process_deterministic.no_grader_reference is not true"


def test_G0() -> None:
    assert _gate("G0")["clean"] is True
    _resolve("checks.G0_trusted_entrypoint.clean")


def test_HARNESSintegrity() -> None:
    _resolve("verifier_incomplete")
    assert _resolve("verifier_incomplete") is not True, "verifier_incomplete is true"
    assert _resolve("judge_unavailable") is not True, "judge_unavailable is true"


def test_G1() -> None:
    assert _gate("G1")["clean"] is True
    assert _gate("G1")["hits"] == [], "forbidden symbols recorded"
    _resolve("checks.G1_forbidden.clean")
    _resolve("checks.G1_forbidden.hits")


def test_G2() -> None:
    g2 = _gate("G2")
    assert g2["clean"] is True
    assert g2["failure_count"] == 0, "failure_count is non-zero"
    assert not g2["failures"], "failures list is non-empty"
    _resolve("checks.G2_correctness.clean")
    _resolve("checks.G2_correctness.failure_count")
    _resolve("checks.G2_correctness.failures")


def test_G3a() -> None:
    import math
    g3 = _gate("G3")
    geo = g3["geomean_fraction_of_sota"]
    assert isinstance(geo, (int, float)) and 0 < geo <= 1, \
        f"geomean {geo!r} is not a fraction in (0, 1]"
    per = [v["fraction_of_sota"] for v in (g3.get("per_shape") or {}).values()
           if v.get("fraction_of_sota")]
    assert per, "no per-shape record to derive the geometric mean from"
    want = math.exp(sum(math.log(x) for x in per) / len(per))
    assert abs(want - geo) <= 1e-6 * max(1.0, want), \
        f"geomean {geo} is not the geometric mean of its own per-shape record ({want})"
    _resolve("checks.G3_speed.geomean_fraction_of_sota")


def test_G3b() -> None:
    per = _gate("G3")["per_shape"]
    assert per, "no per-shape record to back the geometric mean"
    for _shape, _rec in per.items():
        _f = _rec.get("fraction_of_sota")
        assert isinstance(_f, (int, float)) and _f > 0, \
            f"{_shape}: fraction missing, null or non-positive"
    _resolve("checks.G3_speed.per_shape")
    _resolve("checks.G3_speed.geomean_fraction_of_sota")


def test_G4a() -> None:
    assert _gate("G4")["adoption"] >= _gate("G4")["threshold"]
    _resolve("checks.G4_written_kernel.adoption")
    _resolve("checks.G4_written_kernel.threshold")


def test_G4b() -> None:
    g4 = _gate("G4")
    assert g4["declared"], "no triton.jit kernel declared by the candidate"
    assert 0 < g4["device_us_declared"] <= g4["device_us_total"], \
        "declared device time is zero or exceeds the total"
    _resolve("checks.G4_written_kernel.declared")
    _resolve("checks.G4_written_kernel.device_us_declared")
    _resolve("checks.G4_written_kernel.device_us_total")


