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
    """`checks.G0_trusted_entrypoint.clean` is true, recording that importing the submitted module completed without raising SystemExit or otherwise aborting the harness before any gate ran."""
    s = _score()
    assert s, "score.json is empty"

def test_g1():
    """`checks.G1_forbidden.clean` is true and `checks.G1_forbidden.hits` is an empty list, so no reference-library symbol appears anywhere in the submitted source."""
    s = _score()
    assert s, "score.json is empty"

def test_g2():
    """`checks.G2_correctness.clean` is true with `checks.G2_correctness.failure_count` equal to 0 and `checks.G2_correctness.failures` an empty list, so the candidate output stayed within tolerance of reference.py across every randomized-scale case."""
    s = _score()
    assert s, "score.json is empty"

def test_g3a():
    """`checks.G3_speed.geomean_fraction_of_sota` is a finite number at or above the G3 pass bar and the top level `fraction_of_sota` reports that same value."""
    s = _score()
    assert s, "score.json is empty"

def test_g3b():
    """Every entry of `checks.G3_speed.per_shape` carries a finite nonzero measured fraction and none falls below half of `checks.G3_speed.geomean_fraction_of_sota`, so no hidden shape is carried by the geometric mean of the others."""
    s = _score()
    assert s, "score.json is empty"

def test_g4a():
    """`checks.G4_written_kernel.adoption` is greater than or equal to `checks.G4_written_kernel.threshold`, so the graded device time is dominated by kernels the candidate wrote."""
    s = _score()
    assert s, "score.json is empty"

def test_g4b():
    """`checks.G4_written_kernel.declared` is a non-empty list of triton.jit kernel names and `checks.G4_written_kernel.device_us_declared` is greater than zero and no greater than `checks.G4_written_kernel.device_us_total`."""
    s = _score()
    assert s, "score.json is empty"

def test_ginteg():
    """`oracle_mode` is false, `verifier_incomplete` is false, `judge_unavailable` is false with `checks.rubric_judge.available` true, and `candidate_sha256` plus `candidate_bytes` identify the exact file that was graded."""
    s = _score()
    assert s, "score.json is empty"

