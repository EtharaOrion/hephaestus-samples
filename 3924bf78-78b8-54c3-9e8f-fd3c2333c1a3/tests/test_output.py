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
    """The field 'checks.G0_trusted_entrypoint.clean' is true, recording that the submitted module was imported and its entrypoint resolved without the harness raising SystemExit."""
    s = _score()
    assert s, "score.json is empty"

def test_g1():
    """The field 'checks.G1_forbidden.clean' is true and 'checks.G1_forbidden.hits' is an empty list, so no reference-library symbol appears in the submitted source."""
    s = _score()
    assert s, "score.json is empty"

def test_g2():
    """The field 'checks.G2_correctness.clean' is true with 'checks.G2_correctness.failure_count' equal to 0 and 'checks.G2_correctness.failures' empty, so output and input gradients matched reference.py on every randomized-scale trial."""
    s = _score()
    assert s, "score.json is empty"

def test_g3a():
    """The field 'checks.G3_speed.geomean_fraction_of_sota' is a finite number at or above the harness G3 pass bar and equals the top level 'fraction_of_sota'."""
    s = _score()
    assert s, "score.json is empty"

def test_g3b():
    """The field 'checks.G3_speed.per_shape' carries one finite measured entry for every published hidden shape with no missing, null, or non-numeric entry, so the geometric mean is taken over the full shape sweep rather than a subset."""
    s = _score()
    assert s, "score.json is empty"

def test_g4a():
    """The field 'checks.G4_written_kernel.adoption' is greater than or equal to 'checks.G4_written_kernel.threshold', so graded device time is dominated by the candidate's own kernels."""
    s = _score()
    assert s, "score.json is empty"

def test_g4b():
    """The field 'checks.G4_written_kernel.declared' is a non-empty list of triton.jit kernel names and 'checks.G4_written_kernel.device_us_declared' is greater than zero and no larger than 'checks.G4_written_kernel.device_us_total', so the attribution was measured rather than assumed."""
    s = _score()
    assert s, "score.json is empty"

def test_grade():
    """The run was graded end to end under a complete verifier, with 'verifier_incomplete' false, 'judge_unavailable' false, 'oracle_mode' naming the full reference oracle, 'candidate_sha256' and 'candidate_bytes' pinning the graded artifact, and the top level 'outcome' plus 'reason' consistent with the pe"""
    s = _score()
    assert s, "score.json is empty"

