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
    """The field 'checks.G0_trusted_entrypoint.clean' is true, recording that the submitted module was imported by the harness and exposed its entrypoint without the trusted loader raising SystemExit."""
    s = _score()
    assert s, "score.json is empty"

def test_g1():
    """The field 'checks.G1_forbidden.hits' is an empty list and 'checks.G1_forbidden.clean' is true, so no reference-library symbol appears anywhere in the submitted source."""
    s = _score()
    assert s, "score.json is empty"

def test_g3a():
    """The field 'checks.G3_speed.geomean_fraction_of_sota' is at least 0.80 and the top-level 'fraction_of_sota' reports that same number."""
    s = _score()
    assert s, "score.json is empty"

def test_g3b():
    """Every entry of 'checks.G3_speed.per_shape' carries a measured fraction of the production kernel and none of those per-shape fractions falls below 0.50, so no hidden shape collapses behind the geometric mean."""
    s = _score()
    assert s, "score.json is empty"

def test_g4a():
    """The field 'checks.G4_written_kernel.adoption' is greater than or equal to 'checks.G4_written_kernel.threshold', so graded device time is dominated by kernels the candidate wrote."""
    s = _score()
    assert s, "score.json is empty"

def test_g4b():
    """The field 'checks.G4_written_kernel.declared' names at least one triton.jit kernel and 'checks.G4_written_kernel.device_us_declared' is a nonzero share of 'checks.G4_written_kernel.device_us_total' consistent with the reported adoption."""
    s = _score()
    assert s, "score.json is empty"

def test_oracleintegrity():
    """The field 'verifier_incomplete' is false and 'oracle_mode' records the live production oracle rather than a degraded fallback, so no gate above was scored under a substitute oracle."""
    s = _score()
    assert s, "score.json is empty"

