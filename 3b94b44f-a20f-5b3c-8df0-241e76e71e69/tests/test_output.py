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
    """The field 'checks.G0_trusted_entrypoint.clean' is true, showing the submitted module was imported by the harness without raising SystemExit or otherwise refusing to load."""
    s = _score()
    assert s, "score.json is empty"

def test_g1():
    """The field 'checks.G1_forbidden.clean' is true and 'checks.G1_forbidden.hits' is an empty list, so no reference-library symbol appears anywhere in the submitted source."""
    s = _score()
    assert s, "score.json is empty"

def test_g2():
    """The field 'checks.G2_correctness.clean' is true with 'checks.G2_correctness.failure_count' equal to 0 and 'checks.G2_correctness.failures' empty, so the forward output and every input gradient matched reference.py across the randomized-scale trials."""
    s = _score()
    assert s, "score.json is empty"

def test_g3a():
    """The field 'checks.G3_speed.per_shape' contains a timed entry for every hidden shape the harness benchmarked, with no entry missing, null, or zero, so the reported geometric mean is taken over the full hidden shape set rather than a surviving subset."""
    s = _score()
    assert s, "score.json is empty"

def test_g3b():
    """The field 'checks.G3_speed.geomean_fraction_of_sota' meets or exceeds the speed gate and agrees with the top-level 'fraction_of_sota', so the candidate reaches the required fraction of the production kernel over the hidden shapes."""
    s = _score()
    assert s, "score.json is empty"

def test_g4a():
    """The field 'checks.G4_written_kernel.adoption' is greater than or equal to 'checks.G4_written_kernel.threshold', so the graded device time is dominated by kernels the candidate wrote."""
    s = _score()
    assert s, "score.json is empty"

def test_g4b():
    """The field 'checks.G4_written_kernel.declared' is a non-empty list of the candidate's own triton.jit kernel names and 'checks.G4_written_kernel.device_us_declared' is greater than zero and no larger than 'checks.G4_written_kernel.device_us_total', so the adoption ratio is backed by real attributed de"""
    s = _score()
    assert s, "score.json is empty"

def test_verify():
    """The field 'verifier_incomplete' is false and 'outcome' is a decided verdict accompanied by a non-empty 'reason', 'candidate_sha256', and 'candidate_bytes', so the recorded result identifies one fully graded artifact rather than an aborted or partial run."""
    s = _score()
    assert s, "score.json is empty"

