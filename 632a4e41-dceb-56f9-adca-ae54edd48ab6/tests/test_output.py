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
    """The field 'checks.G0_trusted_entrypoint.clean' is true, meaning the submitted module was imported by the harness without raising SystemExit or otherwise aborting the trusted entrypoint."""
    s = _score()
    assert s, "score.json is empty"

def test_g1():
    """The array 'checks.G1_forbidden.hits' is empty and 'checks.G1_forbidden.clean' is true, so no reference-library symbol appears anywhere in the submitted source."""
    s = _score()
    assert s, "score.json is empty"

def test_g3a():
    """The field 'checks.G3_speed.geomean_fraction_of_sota' is at least 1.0 and agrees with the top-level 'fraction_of_sota' reported for the run."""
    s = _score()
    assert s, "score.json is empty"

def test_g3b():
    """No entry in 'checks.G3_speed.per_shape' falls below half of 'checks.G3_speed.geomean_fraction_of_sota', so the speed result is uniform across hidden shapes rather than carried by a subset of them."""
    s = _score()
    assert s, "score.json is empty"

def test_g4a():
    """The field 'checks.G4_written_kernel.adoption' is greater than or equal to 'checks.G4_written_kernel.threshold', so graded device time is dominated by kernels the candidate declared and wrote."""
    s = _score()
    assert s, "score.json is empty"

def test_g4b():
    """The list 'checks.G4_written_kernel.declared' is non-empty and 'checks.G4_written_kernel.device_us_declared' is greater than zero and not greater than 'checks.G4_written_kernel.device_us_total', so the adoption figure is backed by real declared-kernel device time."""
    s = _score()
    assert s, "score.json is empty"

