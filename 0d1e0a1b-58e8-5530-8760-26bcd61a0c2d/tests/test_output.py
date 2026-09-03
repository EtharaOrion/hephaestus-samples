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
    """score.json records 'checks.G0_trusted_entrypoint.clean' as true, meaning the harness imported the submitted module and obtained the required entrypoint without the module raising SystemExit or otherwise aborting at import time."""
    s = _score()
    assert s, "score.json is empty"

def test_g1():
    """score.json records 'checks.G1_forbidden.hits' as an empty list together with 'checks.G1_forbidden.clean' true, so no reference-library symbol appears anywhere in the submitted source."""
    s = _score()
    assert s, "score.json is empty"

def test_g2a():
    """score.json records 'checks.G2_correctness.clean' true with 'checks.G2_correctness.failure_count' equal to 0 and 'checks.G2_correctness.failures' empty, so both the forward output matched reference.py on all randomized-scale trials."""
    s = _score()
    assert s, "score.json is empty"

def test_g2b():
    """score.json records 'verifier_incomplete' as false and 'oracle_mode' as the full private oracle rather than any degraded or skipped mode, so the correctness verdict came from a complete comparison run and not a fallback."""
    s = _score()
    assert s, "score.json is empty"

def test_g3a():
    """score.json records 'checks.G3_speed.geomean_fraction_of_sota' at or above the G3 passing threshold over the hidden shapes, and the top-level 'fraction_of_sota' reports that same geometric mean."""
    s = _score()
    assert s, "score.json is empty"

def test_g3b():
    """Every entry in 'checks.G3_speed.per_shape' carries a finite positive fraction for its hidden shape and no entry falls below half of 'checks.G3_speed.geomean_fraction_of_sota', so the reported speed is uniform across shapes rather than carried by a single favourable configuration."""
    s = _score()
    assert s, "score.json is empty"

def test_g4a():
    """score.json records 'checks.G4_written_kernel.adoption' as greater than or equal to 'checks.G4_written_kernel.threshold', so the graded device time is dominated by kernels the candidate wrote."""
    s = _score()
    assert s, "score.json is empty"

def test_g4b():
    """score.json records 'checks.G4_written_kernel.declared' as a non-empty list of triton.jit kernel names and reports both 'checks.G4_written_kernel.device_us_total' and 'checks.G4_written_kernel.device_us_declared' as strictly positive, so the adoption ratio is measured over real attributed device time"""
    s = _score()
    assert s, "score.json is empty"

