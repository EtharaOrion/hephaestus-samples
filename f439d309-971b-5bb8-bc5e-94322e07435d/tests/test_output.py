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
    """The field 'checks.G0_trusted_entrypoint.clean' is true, meaning importing the submitted module completed without raising SystemExit or otherwise tripping the trusted entrypoint guard."""
    s = _score()
    assert s, "score.json is empty"

def test_g1():
    """The field 'checks.G1_forbidden.hits' is an empty list and 'checks.G1_forbidden.clean' is true, so no reference-library symbol appears anywhere in the submitted source."""
    s = _score()
    assert s, "score.json is empty"

def test_g2():
    """The field 'checks.G2_correctness.clean' is true with 'checks.G2_correctness.failure_count' equal to 0 and 'checks.G2_correctness.failures' empty, so both outputs and input gradients matched reference.py under every randomized scale trial."""
    s = _score()
    assert s, "score.json is empty"

def test_g3a():
    """The field 'checks.G3_speed.geomean_fraction_of_sota' reaches the passing speed bar recorded for the task and agrees with the top level 'fraction_of_sota' reported in the same record."""
    s = _score()
    assert s, "score.json is empty"

def test_g3b():
    """Every entry of 'checks.G3_speed.per_shape' carries a measured fraction of the production kernel, with no single hidden shape collapsing far below 'checks.G3_speed.geomean_fraction_of_sota', so the speed result is not carried by one favourable shape."""
    s = _score()
    assert s, "score.json is empty"

def test_g4():
    """The field 'checks.G4_written_kernel.adoption' is greater than or equal to 'checks.G4_written_kernel.threshold' while 'checks.G4_written_kernel.declared' names at least one triton.jit kernel and 'checks.G4_written_kernel.device_us_declared' is at most 'checks.G4_written_kernel.device_us_total', so gr"""
    s = _score()
    assert s, "score.json is empty"

def test_verdict():
    """The record is a complete graded verdict rather than a degraded one, with 'verifier_incomplete' false, 'oracle_mode' recording a real oracle run, and 'outcome' plus 'process' and 'reason' populated against a submission identified by non-empty 'candidate_sha256' and 'candidate_bytes'."""
    s = _score()
    assert s, "score.json is empty"

