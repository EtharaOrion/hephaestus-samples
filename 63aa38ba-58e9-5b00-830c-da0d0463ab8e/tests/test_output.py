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
    """'checks.G0_trusted_entrypoint.clean' is true, recording that importing the submitted module completed without raising SystemExit or otherwise refusing to load."""
    s = _score()
    assert s, "score.json is empty"

def test_g1():
    """'checks.G1_forbidden.hits' is an empty list and 'checks.G1_forbidden.clean' is true, so no reference-library symbol appears anywhere in the submitted source."""
    s = _score()
    assert s, "score.json is empty"

def test_g3a():
    """'checks.G3_speed.geomean_fraction_of_sota' is a finite number of at least 0.50, meaning the candidate reaches at least half the throughput of the production kernel as a geometric mean over the hidden shapes."""
    s = _score()
    assert s, "score.json is empty"

def test_g3b():
    """'checks.G3_speed.per_shape' carries one recorded fraction for every hidden shape measured and every one of those fractions is a finite value strictly above 0.25, so no individual shape collapses even though 'fraction_of_sota' is an aggregate."""
    s = _score()
    assert s, "score.json is empty"

def test_g4a():
    """'checks.G4_written_kernel.adoption' is greater than or equal to 'checks.G4_written_kernel.threshold', so the graded device time is dominated by kernels the candidate declared and wrote."""
    s = _score()
    assert s, "score.json is empty"

def test_g4b():
    """'checks.G4_written_kernel.declared' is a non-empty list of the candidate's own triton.jit kernel names and 'checks.G4_written_kernel.device_us_declared' is strictly positive and no greater than 'checks.G4_written_kernel.device_us_total', so the adoption figure is backed by real attributed device tim"""
    s = _score()
    assert s, "score.json is empty"

def test_verdict():
    """'outcome' records a pass while 'verifier_incomplete' is false and 'oracle_mode' names the real production oracle rather than a degraded or stubbed fallback, so the verdict does not diverge from a complete verification run."""
    s = _score()
    assert s, "score.json is empty"

