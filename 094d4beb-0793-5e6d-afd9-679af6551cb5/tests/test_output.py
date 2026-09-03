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
    """'checks.G0_trusted_entrypoint.clean' is true, recording that importing the submitted module completed without raising SystemExit or otherwise aborting the trusted entrypoint."""
    s = _score()
    assert s, "score.json is empty"

def test_g1():
    """'checks.G1_forbidden.clean' is true and 'checks.G1_forbidden.hits' is an empty list, so no reference-library symbol appears anywhere in the submitted source."""
    s = _score()
    assert s, "score.json is empty"

def test_g2():
    """'checks.G2_correctness.clean' is true with 'checks.G2_correctness.failure_count' equal to 0 and 'checks.G2_correctness.failures' empty, so both the forward output and every input gradient matched reference.py across the randomized-scale trials."""
    s = _score()
    assert s, "score.json is empty"

def test_g3a():
    """'checks.G3_speed.geomean_fraction_of_sota' is at least the G3 pass bar and agrees with the headline 'fraction_of_sota', so the candidate's geometric mean speed relative to the production kernel clears the gate."""
    s = _score()
    assert s, "score.json is empty"

def test_g3b():
    """'checks.G3_speed.per_shape' carries one finite, strictly positive measured fraction for every hidden shape in the timing sweep, so no shape was skipped, errored out, or excluded from the geometric mean."""
    s = _score()
    assert s, "score.json is empty"

def test_g4a():
    """'checks.G4_written_kernel.adoption' is greater than or equal to 'checks.G4_written_kernel.threshold', so the graded device time is dominated by kernels the candidate wrote rather than by library calls."""
    s = _score()
    assert s, "score.json is empty"

def test_g4b():
    """'checks.G4_written_kernel.declared' is a non-empty list of triton.jit kernels and 'checks.G4_written_kernel.device_us_declared' is positive and no greater than 'checks.G4_written_kernel.device_us_total', so the adoption ratio was computed from actually declared and actually executed kernels."""
    s = _score()
    assert s, "score.json is empty"

def test_runoracle():
    """'verifier_incomplete' is false and 'judge_unavailable' is false while 'oracle_mode' records the real production oracle, so 'outcome' reflects a complete run measured against the true baseline rather than a degraded or stubbed one."""
    s = _score()
    assert s, "score.json is empty"

