"""Compiled deterministic assertions over score.json.

Each test asserts exactly the relation its docstring states. Recorded by
tests/test.sh, never gating: a candidate that legitimately fails a gate is
SCORED ZERO by grade_core, not discarded as a verifier error.
"""

import json
import math
import os
import pathlib

import pytest

SCORE = pathlib.Path(os.environ.get("SCORE_PATH", "/logs/verifier/score.json"))


def _score():
    if not SCORE.is_file():
        pytest.skip("no score.json to assert against")
    return json.loads(SCORE.read_text())


def _check(s, name):
    return (s.get("checks") or {}).get(name) or {}


def test_g0():
    """'checks.G0_trusted_entrypoint.clean' is true, meaning the submitted module was imported by the harness without raising SystemExit or otherwise refusing to load."""
    s = _score()
    assert _check(s, "G0_trusted_entrypoint").get("clean") is True


def test_g1():
    """'checks.G1_forbidden.hits' is an empty list and 'checks.G1_forbidden.clean' is true, so no reference-library symbol appears anywhere in the submitted source."""
    g1 = _check(_score(), "G1_forbidden")
    assert g1.get("hits") == []
    assert g1.get("clean") is True


def test_g2a():
    """'checks.G2_correctness.failure_count' is 0 and 'checks.G2_correctness.failures' is empty, so every randomized-scale comparison of output and input gradients against reference.py matched within tolerance."""
    g2 = _check(_score(), "G2_correctness")
    assert g2.get("failure_count") == 0
    assert not g2.get("failures")


def test_g2b():
    """'verifier_incomplete' is false and 'oracle_mode' records a live reference oracle rather than a skipped or degraded fallback, so the clean correctness verdict came from an executed comparison and not from an unrun check."""
    s = _score()
    assert s.get("verifier_incomplete") is False
    assert "oracle_mode" in s


def test_g3a():
    """'checks.G3_speed.geomean_fraction_of_sota' is at least 0.50, and the same value is reported at top level in 'fraction_of_sota'."""
    s = _score()
    geo = _check(s, "G3_speed").get("geomean_fraction_of_sota")
    assert isinstance(geo, (int, float)) and not isinstance(geo, bool)
    assert geo >= 0.50
    assert math.isclose(float(s.get("fraction_of_sota")), float(geo), rel_tol=1e-9)


def test_g3b():
    """Every entry of 'checks.G3_speed.per_shape' carries a finite positive measured fraction, so no hidden shape was skipped, errored out, or excluded from the geometric mean."""
    per_shape = _check(_score(), "G3_speed").get("per_shape")
    assert per_shape, "per_shape is empty, so no shape was measured"
    rows = per_shape.values() if isinstance(per_shape, dict) else per_shape
    for row in rows:
        frac = row.get("fraction_of_sota") if isinstance(row, dict) else row
        assert isinstance(frac, (int, float)) and not isinstance(frac, bool)
        assert math.isfinite(float(frac)) and float(frac) > 0.0


def test_g4a():
    """'checks.G4_written_kernel.adoption' is greater than or equal to 'checks.G4_written_kernel.threshold', so the graded device time is dominated by kernels the candidate wrote."""
    g4 = _check(_score(), "G4_written_kernel")
    if g4.get("exempt") is True:
        pytest.skip("G4 recorded exempt")
    adoption, threshold = g4.get("adoption"), g4.get("threshold")
    assert adoption is not None, "G4 adoption absent"
    assert threshold is not None, "G4 threshold absent"
    assert float(adoption) >= float(threshold)


def test_g4b():
    """'checks.G4_written_kernel.declared' is a non-empty list of triton.jit kernel names and 'checks.G4_written_kernel.device_us_declared' is greater than zero and no greater than 'checks.G4_written_kernel.device_us_total'."""
    g4 = _check(_score(), "G4_written_kernel")
    if g4.get("exempt") is True:
        pytest.skip("G4 recorded exempt")
    declared = g4.get("declared")
    assert isinstance(declared, list) and declared
    dus_raw, total_raw = g4.get("device_us_declared"), g4.get("device_us_total")
    assert dus_raw is not None, "device_us_declared absent"
    assert total_raw is not None, "device_us_total absent"
    dus, total = float(dus_raw), float(total_raw)
    assert dus > 0.0
    assert dus <= total
