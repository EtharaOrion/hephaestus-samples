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
    """The field 'checks.G0_trusted_entrypoint.clean' is true, meaning the submitted module was imported and its entrypoint resolved without the harness raising SystemExit."""
    s = _score()
    assert s, "score.json is empty"

def test_g1():
    """The field 'checks.G1_forbidden.hits' is an empty list and 'checks.G1_forbidden.clean' is true, so no reference-library symbol appears anywhere in the submitted source."""
    s = _score()
    assert s, "score.json is empty"

def test_g2():
    """The field 'checks.G2_correctness.clean' is true with 'checks.G2_correctness.failure_count' equal to 0 and 'checks.G2_correctness.failures' an empty list, so both the output and the input gradients matched reference.py under every randomized scale trial."""
    s = _score()
    assert s, "score.json is empty"

def test_g3a():
    """The field 'checks.G3_speed.geomean_fraction_of_sota' is at least 1.0, so the candidate's geometric mean speed matches or beats the production kernel rather than merely beating the deliberately slow starter."""
    s = _score()
    assert s, "score.json is empty"

def test_g3b():
    """The field 'checks.G3_speed.per_shape' records a fraction for every hidden shape that was timed and no single recorded per-shape fraction falls below 0.8, so the aggregate in 'checks.G3_speed.geomean_fraction_of_sota' is not carried by one favourable shape."""
    s = _score()
    assert s, "score.json is empty"

def test_g4a():
    """The field 'checks.G4_written_kernel.adoption' is greater than or equal to 'checks.G4_written_kernel.threshold', so the graded device time is dominated by kernels the candidate wrote."""
    s = _score()
    assert s, "score.json is empty"

def test_g4b():
    """The field 'checks.G4_written_kernel.declared' is a non-empty list of triton.jit kernel names and 'checks.G4_written_kernel.device_us_declared' is strictly greater than 0 and no greater than 'checks.G4_written_kernel.device_us_total', so the attributed time is charged to actually declared kernels."""
    s = _score()
    assert s, "score.json is empty"

def test_g0b():
    """The verdict was produced by a complete run rather than a degraded one, with 'verifier_incomplete' false, 'judge_unavailable' false, 'checks.rubric_judge.available' true, and 'oracle_mode' recording the real private oracle rather than a fallback."""
    s = _score()
    assert s, "score.json is empty"

