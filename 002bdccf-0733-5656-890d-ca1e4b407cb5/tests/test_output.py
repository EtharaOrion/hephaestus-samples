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
    """score.json records 'checks.G0_trusted_entrypoint.clean' as true, meaning the submitted module was imported by the trusted entrypoint without raising SystemExit."""
    s = _score()
    assert s, "score.json is empty"

def test_g1():
    """score.json records 'checks.G1_forbidden.clean' as true and 'checks.G1_forbidden.hits' as an empty list, so no reference-library symbol appears anywhere in the submitted source."""
    s = _score()
    assert s, "score.json is empty"

def test_g2():
    """score.json records 'checks.G2_correctness.clean' as true with 'checks.G2_correctness.failure_count' equal to 0 and 'checks.G2_correctness.failures' empty, so both the forward output matched reference.py under randomized scale."""
    s = _score()
    assert s, "score.json is empty"

def test_g3a():
    """score.json records 'checks.G3_speed.geomean_fraction_of_sota' as a finite number of at least 0.50 and the top-level 'fraction_of_sota' carries that same value."""
    s = _score()
    assert s, "score.json is empty"

def test_g3b():
    """'checks.G3_speed.per_shape' holds a finite strictly positive fraction for every hidden shape measured, and no single shape's fraction falls below one half of 'checks.G3_speed.geomean_fraction_of_sota', so the geometric mean is not carried by one outlier shape."""
    s = _score()
    assert s, "score.json is empty"

def test_g4a():
    """score.json records 'checks.G4_written_kernel.adoption' as greater than or equal to 'checks.G4_written_kernel.threshold', so the graded device time is dominated by kernels the candidate declared."""
    s = _score()
    assert s, "score.json is empty"

def test_g4b():
    """'checks.G4_written_kernel.declared' is a non-empty list and 'checks.G4_written_kernel.device_us_declared' divided by a strictly positive 'checks.G4_written_kernel.device_us_total' reproduces 'checks.G4_written_kernel.adoption' to within 1e-6, so the adoption figure is backed by the recorded timings """
    s = _score()
    assert s, "score.json is empty"

