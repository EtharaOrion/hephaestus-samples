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
    """'checks.G0_trusted_entrypoint.clean' is true, meaning the submitted module was imported by the harness and completed import without raising SystemExit or otherwise refusing to load."""
    s = _score()
    assert s, "score.json is empty"

def test_g1():
    """'checks.G1_forbidden.clean' is true and 'checks.G1_forbidden.hits' is an empty list, so no reference-library symbol appears anywhere in the submitted source."""
    s = _score()
    assert s, "score.json is empty"

def test_g3a():
    """'checks.G3_speed.geomean_fraction_of_sota' is a positive number at or above the recorded G3 pass threshold and agrees with the top-level 'fraction_of_sota', so the graded speed is a measured win and not an unmeasured or defaulted value."""
    s = _score()
    assert s, "score.json is empty"

def test_g3b():
    """'checks.G3_speed.per_shape' carries one timed entry for each published hidden shape and no entry is missing, null, or zero, so the geometric mean in 'checks.G3_speed.geomean_fraction_of_sota' is not carried by a subset of shapes while others silently failed to time."""
    s = _score()
    assert s, "score.json is empty"

def test_g4a():
    """'checks.G4_written_kernel.adoption' is greater than or equal to 'checks.G4_written_kernel.threshold', so the graded device time is dominated by kernels the candidate wrote rather than by library calls."""
    s = _score()
    assert s, "score.json is empty"

def test_g4b():
    """'checks.G4_written_kernel.declared' is a non-empty list and 'checks.G4_written_kernel.device_us_declared' is greater than zero and no greater than 'checks.G4_written_kernel.device_us_total', so declared triton.jit kernels actually executed and accounted for real device time."""
    s = _score()
    assert s, "score.json is empty"

def test_g5meta():
    """'verifier_incomplete' is false and 'judge_unavailable' is false and 'oracle_mode' is populated, with 'candidate_sha256' and 'candidate_bytes' both present, so every gate verdict above was produced by a complete run over an identified artifact."""
    s = _score()
    assert s, "score.json is empty"

