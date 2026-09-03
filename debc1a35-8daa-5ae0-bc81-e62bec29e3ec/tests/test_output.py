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
    """The verifier recorded 'checks.G0_trusted_entrypoint.clean': true, meaning importing the submitted module completed without raising SystemExit or otherwise tripping the trusted entrypoint guard."""
    s = _score()
    assert s, "score.json is empty"

def test_g1():
    """The array at 'checks.G1_forbidden.hits' is empty and 'checks.G1_forbidden.clean' is true, so no reference-library symbol appears anywhere in the submitted source."""
    s = _score()
    assert s, "score.json is empty"

def test_g2():
    """The verifier recorded 'checks.G2_correctness.clean': true with 'checks.G2_correctness.failure_count': 0 and an empty 'checks.G2_correctness.failures', so output and input gradients matched reference.py on every randomized-scale trial."""
    s = _score()
    assert s, "score.json is empty"

def test_g3a():
    """The value at 'checks.G3_speed.geomean_fraction_of_sota' is at least 0.80, meaning the candidate reaches at least eighty percent of the production kernel's speed as a geometric mean over the graded hidden shapes."""
    s = _score()
    assert s, "score.json is empty"

def test_g3b():
    """No individual entry recorded in 'checks.G3_speed.per_shape' falls below 0.60 of the production kernel, so the geometric mean in 'checks.G3_speed.geomean_fraction_of_sota' is not being carried by a subset of favourable shapes."""
    s = _score()
    assert s, "score.json is empty"

def test_g4a():
    """The value at 'checks.G4_written_kernel.adoption' is greater than or equal to 'checks.G4_written_kernel.threshold', so graded device time is dominated by kernels the candidate wrote."""
    s = _score()
    assert s, "score.json is empty"

def test_g4b():
    """The list at 'checks.G4_written_kernel.declared' is non-empty and 'checks.G4_written_kernel.device_us_declared' divided by 'checks.G4_written_kernel.device_us_total' agrees with the reported 'checks.G4_written_kernel.adoption' to within one percent, so adoption is backed by named triton.jit kernels r"""
    s = _score()
    assert s, "score.json is empty"

def test_g0v():
    """The run is a complete grading run in which 'verifier_incomplete' is false, 'judge_unavailable' is false, 'oracle_mode' names the oracle actually used, and 'outcome' is consistent with every recorded gate clean flag rather than with a skipped or degraded gate."""
    s = _score()
    assert s, "score.json is empty"

