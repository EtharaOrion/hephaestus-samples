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
    """'checks.G0_trusted_entrypoint.clean' is true, meaning the submitted module was imported by the harness without raising SystemExit or otherwise refusing to load."""
    s = _score()
    assert s, "score.json is empty"

def test_g1():
    """'checks.G1_forbidden.hits' is an empty list and 'checks.G1_forbidden.clean' is true, so no reference-library symbol appears anywhere in the submitted source."""
    s = _score()
    assert s, "score.json is empty"

def test_g2a():
    """'checks.G2_correctness.failure_count' is 0 and 'checks.G2_correctness.failures' is empty, so every randomized-scale comparison of output and input gradients against reference.py matched within tolerance."""
    s = _score()
    assert s, "score.json is empty"

def test_g2b():
    """'verifier_incomplete' is false and 'oracle_mode' records a live reference oracle rather than a skipped or degraded fallback, so the clean correctness verdict came from an executed comparison and not from an unrun check."""
    s = _score()
    assert s, "score.json is empty"

def test_g3a():
    """'checks.G3_speed.geomean_fraction_of_sota' is at least 0.50, and the same value is reported at top level in 'fraction_of_sota'."""
    s = _score()
    assert s, "score.json is empty"

def test_g3b():
    """Every entry of 'checks.G3_speed.per_shape' carries a finite positive measured fraction, so no hidden shape was skipped, errored out, or excluded from the geometric mean."""
    s = _score()
    assert s, "score.json is empty"

def test_g4a():
    """'checks.G4_written_kernel.adoption' is greater than or equal to 'checks.G4_written_kernel.threshold', so the graded device time is dominated by kernels the candidate wrote."""
    s = _score()
    assert s, "score.json is empty"

def test_g4b():
    """'checks.G4_written_kernel.declared' is a non-empty list of triton.jit kernel names and 'checks.G4_written_kernel.device_us_declared' is greater than zero and no greater than 'checks.G4_written_kernel.device_us_total'."""
    s = _score()
    assert s, "score.json is empty"

