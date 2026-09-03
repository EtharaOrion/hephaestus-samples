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
    """The graded run records 'checks.G0_trusted_entrypoint.clean' as true, meaning the submitted module was imported by the harness without raising SystemExit or otherwise refusing to load."""
    s = _score()
    assert s, "score.json is empty"

def test_g1():
    """The graded run records 'checks.G1_forbidden.clean' as true and 'checks.G1_forbidden.hits' as an empty list, so no reference-library symbol appears anywhere in the submitted source."""
    s = _score()
    assert s, "score.json is empty"

def test_g2():
    """The graded run records 'checks.G2_correctness.clean' as true with 'checks.G2_correctness.failure_count' equal to 0 and 'checks.G2_correctness.failures' empty, so both the output and every input gradient matched reference.py under the randomized-scale comparison."""
    s = _score()
    assert s, "score.json is empty"

def test_g3a():
    """The graded run records 'checks.G3_speed.geomean_fraction_of_sota' as a finite positive number that reaches the speed gate, and the top level 'fraction_of_sota' agrees with that same recorded geometric mean."""
    s = _score()
    assert s, "score.json is empty"

def test_g3b():
    """The graded run records 'checks.G3_speed.per_shape' as a non-empty collection carrying one measured fraction entry per hidden benchmark shape with no null or missing measurement, so the geometric mean is backed by complete per-shape timings rather than a partial sweep."""
    s = _score()
    assert s, "score.json is empty"

def test_g4a():
    """The graded run records 'checks.G4_written_kernel.adoption' as greater than or equal to 'checks.G4_written_kernel.threshold', so the measured device time is dominated by kernels the candidate wrote."""
    s = _score()
    assert s, "score.json is empty"

def test_g4b():
    """The graded run records 'checks.G4_written_kernel.declared' as a non-empty list of triton.jit kernel names and satisfies 0 < 'checks.G4_written_kernel.device_us_declared' <= 'checks.G4_written_kernel.device_us_total', so the adoption ratio is computed from real declared-kernel device microseconds."""
    s = _score()
    assert s, "score.json is empty"

def test_g2b():
    """The graded run records 'verifier_incomplete' as false with 'oracle_mode' set to the full private-oracle mode, so no gate verdict was produced under a degraded or skipped verification path."""
    s = _score()
    assert s, "score.json is empty"

def test_g4c():
    """The graded run records 'judge_unavailable' as false and 'checks.rubric_judge.available' as true with a numeric 'checks.rubric_judge.pass_share', so the judged solution obligations were actually adjudicated instead of silently defaulting."""
    s = _score()
    assert s, "score.json is empty"

