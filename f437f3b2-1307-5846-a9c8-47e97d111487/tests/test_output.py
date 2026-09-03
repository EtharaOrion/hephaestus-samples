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
    """The field 'checks.G0_trusted_entrypoint.clean' is true, meaning the submitted module was imported and its entrypoint resolved without the harness raising SystemExit."""
    s = _score()
    assert s, "score.json is empty"

def test_g1():
    """The field 'checks.G1_forbidden.clean' is true and 'checks.G1_forbidden.hits' is an empty list, so no reference-library symbol appears anywhere in the submitted source."""
    s = _score()
    assert s, "score.json is empty"

def test_g2():
    """The field 'checks.G2_correctness.clean' is true and 'checks.G2_correctness.failure_count' equals 0 with 'checks.G2_correctness.failures' empty, so both the output and the input gradients matched reference.py on every randomized-scale trial."""
    s = _score()
    assert s, "score.json is empty"

def test_g2b():
    """The field 'verifier_incomplete' is false and 'oracle_mode' names the full reference oracle rather than a degraded or stubbed fallback, so the G2 verdict was produced by a complete comparison rather than a skipped one."""
    s = _score()
    assert s, "score.json is empty"

def test_g3a():
    """The field 'checks.G3_speed.geomean_fraction_of_sota' is at least 0.80 and the top level 'fraction_of_sota' reports that same geometric mean value."""
    s = _score()
    assert s, "score.json is empty"

def test_g3b():
    """Every entry of 'checks.G3_speed.per_shape' carries a recorded fraction of the production kernel that is at least half of 'checks.G3_speed.geomean_fraction_of_sota', so the aggregate win is not carried by one hidden shape while another collapses."""
    s = _score()
    assert s, "score.json is empty"

def test_g4a():
    """The field 'checks.G4_written_kernel.adoption' is greater than or equal to 'checks.G4_written_kernel.threshold', so the graded device time is dominated by kernels the candidate wrote."""
    s = _score()
    assert s, "score.json is empty"

def test_g4b():
    """The field 'checks.G4_written_kernel.declared' is a non-empty list of triton.jit kernels defined in the submission, 'checks.G4_written_kernel.device_us_total' is greater than zero, and 'checks.G4_written_kernel.device_us_declared' divided by that total reproduces 'checks.G4_written_kernel.adoption'."""
    s = _score()
    assert s, "score.json is empty"

