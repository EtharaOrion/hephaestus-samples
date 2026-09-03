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
    """The field 'checks.G0_trusted_entrypoint.clean' is true, recording that importing the submitted module executed to completion without raising SystemExit or otherwise refusing the trusted entrypoint."""
    s = _score()
    assert s, "score.json is empty"

def test_g1():
    """The field 'checks.G1_forbidden.hits' is an empty list and 'checks.G1_forbidden.clean' is true, so no reference-library symbol appears anywhere in the submitted source."""
    s = _score()
    assert s, "score.json is empty"

def test_g2():
    """The field 'checks.G2_correctness.failure_count' is 0 with 'checks.G2_correctness.failures' empty and 'checks.G2_correctness.clean' true, meaning the output matched reference.py under randomized scale."""
    s = _score()
    assert s, "score.json is empty"

def test_veroracle():
    """The field 'verifier_incomplete' is false and 'oracle_mode' names the real production oracle rather than a degraded or stubbed fallback, so the recorded verdict was produced by a complete run."""
    s = _score()
    assert s, "score.json is empty"

def test_g3a():
    """The field 'checks.G3_speed.geomean_fraction_of_sota' is at least 0.80, so across the hidden shapes the candidate reaches at least four fifths of the production kernel's throughput."""
    s = _score()
    assert s, "score.json is empty"

def test_g3b():
    """The field 'checks.G3_speed.per_shape' is non-empty and no individual shape entry records a fraction below 0.50, so the geometric mean is not carried by one favourable shape while another collapses."""
    s = _score()
    assert s, "score.json is empty"

def test_g4a():
    """The field 'checks.G4_written_kernel.adoption' is greater than or equal to 'checks.G4_written_kernel.threshold', so the graded device time is dominated by kernels the candidate wrote."""
    s = _score()
    assert s, "score.json is empty"

def test_g4b():
    """The field 'checks.G4_written_kernel.declared' lists at least one triton.jit kernel name and 'checks.G4_written_kernel.device_us_declared' is greater than zero and no larger than 'checks.G4_written_kernel.device_us_total', so the adoption ratio is backed by real attributed device microseconds."""
    s = _score()
    assert s, "score.json is empty"

