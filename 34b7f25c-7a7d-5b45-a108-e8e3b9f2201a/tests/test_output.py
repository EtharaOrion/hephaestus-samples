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
    """The field 'checks.G0_trusted_entrypoint.clean' is true, recording that the submitted module was imported by the harness without the trusted entrypoint raising SystemExit."""
    s = _score()
    assert s, "score.json is empty"

def test_g1():
    """The field 'checks.G1_forbidden.hits' is an empty list and 'checks.G1_forbidden.clean' is true, so no reference-library symbol was found anywhere in the submitted source."""
    s = _score()
    assert s, "score.json is empty"

def test_g2():
    """The field 'checks.G2_correctness.failure_count' equals 0 with 'checks.G2_correctness.failures' empty and 'checks.G2_correctness.clean' true, meaning every randomized-scale output and input-gradient comparison against reference.py passed."""
    s = _score()
    assert s, "score.json is empty"

def test_g3a():
    """The field 'checks.G3_speed.geomean_fraction_of_sota' meets or exceeds the speed bar the gate enforces and equals the top level 'fraction_of_sota' reported for the run."""
    s = _score()
    assert s, "score.json is empty"

def test_g3b():
    """The field 'checks.G3_speed.per_shape' carries one finite positive fraction entry for every published hidden shape, with no shape omitted, null, or zero, so the geometric mean is not computed over a truncated shape set."""
    s = _score()
    assert s, "score.json is empty"

def test_g4a():
    """The field 'checks.G4_written_kernel.adoption' is greater than or equal to 'checks.G4_written_kernel.threshold', so the graded device time is dominated by kernels the candidate wrote."""
    s = _score()
    assert s, "score.json is empty"

def test_g4b():
    """The field 'checks.G4_written_kernel.declared' is a non-empty list of triton.jit kernel names and 'checks.G4_written_kernel.device_us_declared' is positive and no greater than 'checks.G4_written_kernel.device_us_total', so the adoption ratio is backed by actually measured declared-kernel time."""
    s = _score()
    assert s, "score.json is empty"

def test_verifintegrity():
    """The recorded 'outcome' and 'process' are not contradicted by the run metadata, with 'verifier_incomplete' false, 'oracle_mode' naming the real reference oracle rather than a degraded fallback, and 'candidate_bytes' positive alongside a non-empty 'candidate_sha256'."""
    s = _score()
    assert s, "score.json is empty"

