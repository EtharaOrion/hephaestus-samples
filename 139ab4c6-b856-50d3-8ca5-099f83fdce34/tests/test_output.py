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
    """The field 'checks.G0_trusted_entrypoint.clean' is true, recording that the submitted module was imported by the trusted entrypoint and the import ran to completion without raising SystemExit."""
    s = _score()
    assert s, "score.json is empty"

def test_g1():
    """The field 'checks.G1_forbidden.hits' is an empty list and 'checks.G1_forbidden.clean' is true, so no reference-library symbol occurs anywhere in the submitted source."""
    s = _score()
    assert s, "score.json is empty"

def test_g2a():
    """The field 'checks.G2_correctness.failure_count' is 0 with 'checks.G2_correctness.failures' empty and 'checks.G2_correctness.clean' true, so every randomized-scale trial matched reference.py on the output and on all input gradients."""
    s = _score()
    assert s, "score.json is empty"

def test_g2b():
    """The field 'verifier_incomplete' is false and 'oracle_mode' names the full reference oracle rather than a degraded or skipped comparison mode, so the correctness verdict rests on a complete run."""
    s = _score()
    assert s, "score.json is empty"

def test_g3a():
    """The field 'checks.G3_speed.geomean_fraction_of_sota' equals the reported 'fraction_of_sota' and reaches at least the geometric-mean pass threshold that TRUTH.md fixes for the production kernel over the hidden shapes."""
    s = _score()
    assert s, "score.json is empty"

def test_g3b():
    """Every entry of 'checks.G3_speed.per_shape' carries a finite measured fraction and the smallest of them stays within the per-shape spread TRUTH.md permits below 'checks.G3_speed.geomean_fraction_of_sota', so the mean is not carried by a subset of shapes while another shape collapses."""
    s = _score()
    assert s, "score.json is empty"

def test_g4a():
    """The field 'checks.G4_written_kernel.adoption' is greater than or equal to 'checks.G4_written_kernel.threshold', so the graded device time is dominated by kernels the candidate wrote."""
    s = _score()
    assert s, "score.json is empty"

def test_g4b():
    """The field 'checks.G4_written_kernel.declared' is a non-empty list of triton.jit kernels and 'checks.G4_written_kernel.device_us_declared' is strictly positive and no greater than 'checks.G4_written_kernel.device_us_total'."""
    s = _score()
    assert s, "score.json is empty"

