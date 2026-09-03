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
    """'checks.G0_trusted_entrypoint.clean' is true, recording that importing the submitted module raised no SystemExit and the graded entrypoint was reached without the harness aborting."""
    s = _score()
    assert s, "score.json is empty"

def test_g1():
    """'checks.G1_forbidden.hits' is an empty list and 'checks.G1_forbidden.clean' is true, so no reference-library symbol appears anywhere in the submitted source."""
    s = _score()
    assert s, "score.json is empty"

def test_g3a():
    """'checks.G3_speed.geomean_fraction_of_sota' is at least 0.80, the speed bar the graded run must clear against the production kernel over the hidden shapes."""
    s = _score()
    assert s, "score.json is empty"

def test_g3b():
    """Every entry of 'checks.G3_speed.per_shape' carries a finite fraction strictly greater than zero and the top-level 'fraction_of_sota' equals 'checks.G3_speed.geomean_fraction_of_sota', so the headline speed number is backed by a complete per-shape record rather than a partial sweep."""
    s = _score()
    assert s, "score.json is empty"

def test_g4a():
    """'checks.G4_written_kernel.adoption' is greater than or equal to 'checks.G4_written_kernel.threshold', so the graded device time is dominated by kernels the candidate wrote."""
    s = _score()
    assert s, "score.json is empty"

def test_g4b():
    """'checks.G4_written_kernel.declared' is a non-empty list and 'checks.G4_written_kernel.device_us_declared' is no greater than 'checks.G4_written_kernel.device_us_total' and their ratio reproduces 'checks.G4_written_kernel.adoption', so the adoption figure is derived from actually declared triton.jit """
    s = _score()
    assert s, "score.json is empty"

def test_vercomplete():
    """'verifier_incomplete' is false, 'judge_unavailable' is false with 'checks.rubric_judge.available' true, and 'oracle_mode' records the real private oracle rather than a degraded or stubbed mode, so 'outcome' is a full verdict and not a partial one."""
    s = _score()
    assert s, "score.json is empty"

