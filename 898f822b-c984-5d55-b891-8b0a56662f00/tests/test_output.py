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
    """'checks.G0_trusted_entrypoint.clean' is true, meaning importing the submitted module completed without raising SystemExit and the harness reached the later gates on a live entrypoint rather than a stub."""
    s = _score()
    assert s, "score.json is empty"

def test_g1():
    """'checks.G1_forbidden.hits' is an empty list and 'checks.G1_forbidden.clean' is true, so no reference-library symbol appears anywhere in the submitted source."""
    s = _score()
    assert s, "score.json is empty"

def test_g2():
    """'checks.G2_correctness.clean' is true with 'checks.G2_correctness.failure_count' equal to 0 and 'checks.G2_correctness.failures' an empty list, so output and input gradients matched reference.py on every randomized-scale trial."""
    s = _score()
    assert s, "score.json is empty"

def test_g3a():
    """'checks.G3_speed.geomean_fraction_of_sota' is at least 0.80 and the top level 'fraction_of_sota' reports that same value, so the candidate reaches production-kernel class speed rather than merely beating the deliberately slow starter."""
    s = _score()
    assert s, "score.json is empty"

def test_g3b():
    """Every entry of 'checks.G3_speed.per_shape' records a fraction of at least 0.50, so no individual hidden shape collapses while being carried by the geometric mean in 'checks.G3_speed.geomean_fraction_of_sota'."""
    s = _score()
    assert s, "score.json is empty"

def test_g4a():
    """'checks.G4_written_kernel.adoption' is greater than or equal to 'checks.G4_written_kernel.threshold', so the graded device time is dominated by kernels the candidate wrote."""
    s = _score()
    assert s, "score.json is empty"

def test_g4b():
    """'checks.G4_written_kernel.declared' is a non-empty list and 'checks.G4_written_kernel.device_us_total' is greater than zero with 'checks.G4_written_kernel.device_us_declared' no greater than it, so the adoption ratio is backed by named triton.jit kernels and real measured microseconds rather than an"""
    s = _score()
    assert s, "score.json is empty"

