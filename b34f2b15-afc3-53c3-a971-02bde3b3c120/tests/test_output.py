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
    """`checks.G0_trusted_entrypoint.clean` is true, recording that importing the submitted module completed without raising SystemExit."""
    s = _score()
    assert s, "score.json is empty"

def test_g1():
    """`checks.G1_forbidden.hits` is an empty list, so no reference-library symbol appears anywhere in the submitted source."""
    s = _score()
    assert s, "score.json is empty"

def test_g2b():
    """`verifier_incomplete` is false and `oracle_mode` names a live oracle comparison rather than a degraded or skipped fallback, so the correctness verdict rests on a complete run."""
    s = _score()
    assert s, "score.json is empty"

def test_g3a():
    """`checks.G3_speed.geomean_fraction_of_sota` is at least 0.8 and the top-level `fraction_of_sota` reports that same value, placing the candidate within the accepted band of the production kernel over the hidden shapes."""
    s = _score()
    assert s, "score.json is empty"

def test_g3b():
    """`checks.G3_speed.per_shape` is non-empty and every shape entry it contains reports a fraction of at least 0.5, so no single hidden shape collapses while the rest carry `checks.G3_speed.geomean_fraction_of_sota`."""
    s = _score()
    assert s, "score.json is empty"

def test_g4a():
    """`checks.G4_written_kernel.adoption` is greater than or equal to `checks.G4_written_kernel.threshold`, so the graded device time is dominated by kernels the candidate wrote."""
    s = _score()
    assert s, "score.json is empty"

def test_g4b():
    """`checks.G4_written_kernel.declared` lists at least one triton.jit kernel name and `checks.G4_written_kernel.device_us_declared` is greater than zero out of a positive `checks.G4_written_kernel.device_us_total`, showing the declared kernels actually executed on device."""
    s = _score()
    assert s, "score.json is empty"

