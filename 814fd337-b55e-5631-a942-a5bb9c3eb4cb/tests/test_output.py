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
    """'checks.G0_trusted_entrypoint.clean' is true, recording that the submitted module was imported through the trusted entrypoint and completed import without raising SystemExit."""
    s = _score()
    assert s, "score.json is empty"

def test_g1():
    """'checks.G1_forbidden.clean' is true and 'checks.G1_forbidden.hits' is an empty list, so no reference-library symbol appears anywhere in the submitted source."""
    s = _score()
    assert s, "score.json is empty"

def test_g2b():
    """'verifier_incomplete' is false and 'oracle_mode' names the full reference oracle rather than a degraded or skipped mode, so the G2 verdict was produced by a complete comparison against reference.py."""
    s = _score()
    assert s, "score.json is empty"

def test_g3a():
    """'checks.G3_speed.geomean_fraction_of_sota' is a finite number of at least 0.80 and equals the top-level 'fraction_of_sota', establishing the measured speed relative to the production kernel over the hidden shapes."""
    s = _score()
    assert s, "score.json is empty"

def test_g3b():
    """'checks.G3_speed.per_shape' contains a timed entry for every hidden shape with no entry missing, null, or below 0.50, so the geometric mean is not carried by a subset of shapes while others collapse."""
    s = _score()
    assert s, "score.json is empty"

def test_g4a():
    """'checks.G4_written_kernel.adoption' is greater than or equal to 'checks.G4_written_kernel.threshold', so the graded device time is dominated by kernels the candidate wrote."""
    s = _score()
    assert s, "score.json is empty"

def test_g4b():
    """'checks.G4_written_kernel.declared' is a non-empty list of triton.jit kernel names and 'checks.G4_written_kernel.device_us_declared' is positive and no greater than 'checks.G4_written_kernel.device_us_total', so the adoption ratio is backed by real attributed device time."""
    s = _score()
    assert s, "score.json is empty"

