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
    """The field 'checks.G0_trusted_entrypoint.clean' is true, meaning the submitted module was imported by the harness without raising SystemExit or otherwise refusing to load."""
    s = _score()
    assert s, "score.json is empty"

def test_g1():
    """The field 'checks.G1_forbidden.hits' is an empty list and 'checks.G1_forbidden.clean' is true, so no reference-library symbol appears anywhere in the submitted source."""
    s = _score()
    assert s, "score.json is empty"

def test_g2():
    """The field 'checks.G2_correctness.clean' is true and 'checks.G2_correctness.failure_count' equals 0, so both the output and every input gradient matched reference.py under randomized input scale."""
    s = _score()
    assert s, "score.json is empty"

def test_g3a():
    """The field 'checks.G3_speed.geomean_fraction_of_sota' is a recorded number of at least 0.50, so the candidate reaches at least half the production kernel's throughput on the geometric mean over hidden shapes."""
    s = _score()
    assert s, "score.json is empty"

def test_g3b():
    """Every per-shape entry recorded in 'checks.G3_speed.per_shape' reports a fraction of at least 0.30, so no single hidden shape collapses while the geometric mean in 'checks.G3_speed.geomean_fraction_of_sota' is carried by the others."""
    s = _score()
    assert s, "score.json is empty"

def test_g4a():
    """The field 'checks.G4_written_kernel.adoption' is greater than or equal to 'checks.G4_written_kernel.threshold', so the graded device time in 'checks.G4_written_kernel.device_us_declared' is dominated by kernels the candidate declared rather than by imported ones."""
    s = _score()
    assert s, "score.json is empty"

def test_gv():
    """The field 'verifier_incomplete' is false and 'oracle_mode' is populated, so the recorded gate results come from a complete verification run rather than a degraded or short-circuited one."""
    s = _score()
    assert s, "score.json is empty"

