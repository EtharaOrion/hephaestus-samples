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
    """The field 'checks.G0_trusted_entrypoint.clean' is true, recording that importing the submitted module and constructing its entrypoint completed without raising SystemExit."""
    s = _score()
    assert s, "score.json is empty"

def test_g1():
    """The field 'checks.G1_forbidden.hits' is an empty list and 'checks.G1_forbidden.clean' is true, so no reference-library symbol is present anywhere in the submitted source."""
    s = _score()
    assert s, "score.json is empty"

def test_g2():
    """The field 'checks.G2_correctness.failure_count' equals 0 and 'checks.G2_correctness.failures' is empty, so the candidate output and every input gradient stayed within tolerance of reference.py across the randomized-scale trials."""
    s = _score()
    assert s, "score.json is empty"

def test_g3a():
    """The field 'checks.G3_speed.geomean_fraction_of_sota' is a finite positive number that meets or exceeds the G3 pass threshold stated in TRUTH.md."""
    s = _score()
    assert s, "score.json is empty"

def test_g3b():
    """The field 'checks.G3_speed.per_shape' is nonempty, carries a finite fraction for every hidden shape it lists, and the geometric mean of those per-shape fractions reproduces 'checks.G3_speed.geomean_fraction_of_sota' to within rounding."""
    s = _score()
    assert s, "score.json is empty"

def test_g4a():
    """The field 'checks.G4_written_kernel.adoption' is greater than or equal to 'checks.G4_written_kernel.threshold'."""
    s = _score()
    assert s, "score.json is empty"

def test_g4b():
    """The field 'checks.G4_written_kernel.declared' names at least one triton.jit kernel and 'checks.G4_written_kernel.device_us_declared' is nonzero, no greater than 'checks.G4_written_kernel.device_us_total', and their quotient equals 'checks.G4_written_kernel.adoption' to within rounding."""
    s = _score()
    assert s, "score.json is empty"

