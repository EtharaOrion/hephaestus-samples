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
    """score.json records 'checks.G0_trusted_entrypoint.clean' as true, meaning the harness imported the submitted module through the trusted entrypoint without a SystemExit being raised."""
    s = _score()
    assert s, "score.json is empty"

def test_g1():
    """'checks.G1_forbidden.hits' is an empty list and 'checks.G1_forbidden.clean' is true, so no reference-library symbol occurs anywhere in the submitted source."""
    s = _score()
    assert s, "score.json is empty"

def test_g2():
    """'checks.G2_correctness.clean' is true with 'checks.G2_correctness.failure_count' equal to 0 and 'checks.G2_correctness.failures' empty, so both the output and the input gradients matched reference.py at every randomized scale exercised."""
    s = _score()
    assert s, "score.json is empty"

def test_g3a():
    """'checks.G3_speed.geomean_fraction_of_sota' is at or above the parity bar of 1.0 and agrees with the top level 'fraction_of_sota' value reported for the same run."""
    s = _score()
    assert s, "score.json is empty"

def test_g3b():
    """'checks.G3_speed.per_shape' carries one finite recorded fraction for every benchmarked hidden shape, with no entry missing, null, or non finite, so the geometric mean is not computed over a truncated shape set."""
    s = _score()
    assert s, "score.json is empty"

def test_g4a():
    """'checks.G4_written_kernel.adoption' is greater than or equal to 'checks.G4_written_kernel.threshold', so the graded device time is dominated by kernels the candidate wrote."""
    s = _score()
    assert s, "score.json is empty"

def test_g4b():
    """'checks.G4_written_kernel.declared' is a non-empty list of triton.jit kernel names and 'checks.G4_written_kernel.device_us_declared' is greater than zero and no larger than 'checks.G4_written_kernel.device_us_total'."""
    s = _score()
    assert s, "score.json is empty"

def test_g3c():
    """'oracle_mode' names the real production oracle rather than a stub or fallback and 'verifier_incomplete' is false, so no gate was scored under a degraded verifier."""
    s = _score()
    assert s, "score.json is empty"

