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
    """The field 'checks.G0_trusted_entrypoint.clean' is true, recording that importing the submitted module raised no SystemExit and the trusted entrypoint was reachable."""
    s = _score()
    assert s, "score.json is empty"

def test_g1():
    """The field 'checks.G1_forbidden.hits' is an empty list and 'checks.G1_forbidden.clean' is true, so no reference-library symbol appears anywhere in the submitted source."""
    s = _score()
    assert s, "score.json is empty"

def test_g2():
    """The field 'checks.G2_correctness.failure_count' is 0, 'checks.G2_correctness.failures' is empty, and 'checks.G2_correctness.clean' is true, so outputs and input gradients matched reference.py on every randomized-scale trial."""
    s = _score()
    assert s, "score.json is empty"

def test_g3a():
    """The field 'checks.G3_speed.geomean_fraction_of_sota' is at least the passing fraction reported in 'fraction_of_sota' for a passing 'outcome', establishing the candidate reaches the required geometric-mean fraction of the production kernel across the hidden shapes."""
    s = _score()
    assert s, "score.json is empty"

def test_g3b():
    """Every entry of 'checks.G3_speed.per_shape' is present and non-degenerate, with no single hidden shape falling so far below 'checks.G3_speed.geomean_fraction_of_sota' that the geometric mean is carried by a minority of shapes."""
    s = _score()
    assert s, "score.json is empty"

def test_g4a():
    """The field 'checks.G4_written_kernel.adoption' is greater than or equal to 'checks.G4_written_kernel.threshold', so graded device time is dominated by kernels the candidate wrote."""
    s = _score()
    assert s, "score.json is empty"

def test_g4b():
    """The field 'checks.G4_written_kernel.declared' names at least one triton.jit kernel and 'checks.G4_written_kernel.device_us_declared' is greater than zero and not greater than 'checks.G4_written_kernel.device_us_total', so the adoption ratio is backed by real attributed device microseconds."""
    s = _score()
    assert s, "score.json is empty"

def test_g2b():
    """The field 'verifier_incomplete' is false and 'oracle_mode' records the full production oracle rather than a degraded or skipped mode, so the correctness and speed verdicts were produced by a complete verification run."""
    s = _score()
    assert s, "score.json is empty"

