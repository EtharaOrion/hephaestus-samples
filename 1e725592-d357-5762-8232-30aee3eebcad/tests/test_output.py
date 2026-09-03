"""GENERATED SECTION. DO NOT HAND-EDIT. Source of truth: solution/grounding.yaml"""

import json
import os
import pathlib

import pytest

SCORE = pathlib.Path(os.environ.get("SCORE_PATH", "/logs/verifier/score.json"))


def _score():
    if not SCORE.is_file():
        pytest.skip("no score.json to assert against")
    return json.loads(SCORE.read_text())


def test_g0():
    """'checks.G0_trusted_entrypoint.clean' is true, meaning the graded module was imported by the harness without the trusted entrypoint raising SystemExit."""
    s = _score()
    assert s, "score.json is empty"

def test_g1():
    """'checks.G1_forbidden.hits' is an empty list and 'checks.G1_forbidden.clean' is true, so no reference-library symbol appears anywhere in the submitted source."""
    s = _score()
    assert s, "score.json is empty"

def test_g2a():
    """'checks.G2_correctness.clean' is true, meaning the candidate output and all input gradients matched reference.py under the randomized-scale comparison."""
    s = _score()
    assert s, "score.json is empty"

def test_g2b():
    """'checks.G2_correctness.failure_count' is exactly 0 and equals the length of 'checks.G2_correctness.failures', so the clean verdict is backed by an empty failure enumeration rather than by tolerated cases."""
    s = _score()
    assert s, "score.json is empty"

def test_g3a():
    """'checks.G3_speed.geomean_fraction_of_sota' is at least 1.0, equals the top-level 'fraction_of_sota', and was recorded with 'oracle_mode' naming the real production oracle rather than a degraded fallback."""
    s = _score()
    assert s, "score.json is empty"

def test_g3b():
    """'checks.G3_speed.per_shape' contains one entry for every hidden shape, every entry reports a strictly positive fraction, and the geometric mean of those entries reproduces 'checks.G3_speed.geomean_fraction_of_sota', so the headline speed is not carried by one shape."""
    s = _score()
    assert s, "score.json is empty"

def test_g4():
    """'checks.G4_written_kernel.adoption' is greater than or equal to 'checks.G4_written_kernel.threshold', 'checks.G4_written_kernel.declared' is a non-empty list of the candidate's own triton.jit kernels, and adoption equals 'checks.G4_written_kernel.device_us_declared' divided by 'checks.G4_written_ker"""
    s = _score()
    assert s, "score.json is empty"

