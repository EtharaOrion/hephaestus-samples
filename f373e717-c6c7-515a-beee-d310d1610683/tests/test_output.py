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
    """The submitted module loads as a trusted entrypoint, recorded as 'checks.G0_trusted_entrypoint.clean' being true, meaning import completed without raising SystemExit or otherwise aborting collection."""
    s = _score()
    assert s, "score.json is empty"

def test_g1():
    """No reference-library symbol appears in the submitted source, recorded as 'checks.G1_forbidden.hits' being an empty list together with 'checks.G1_forbidden.clean' being true."""
    s = _score()
    assert s, "score.json is empty"

def test_g2():
    """Forward output and all input gradients match reference.py under the randomized-scale sweep, recorded as 'checks.G2_correctness.failure_count' equal to 0 with 'checks.G2_correctness.failures' empty and 'checks.G2_correctness.clean' true."""
    s = _score()
    assert s, "score.json is empty"

def test_g3a():
    """The candidate reaches the speed bar against the production kernel, recorded as 'checks.G3_speed.geomean_fraction_of_sota' meeting or exceeding the gate threshold and agreeing with the top level 'fraction_of_sota' value."""
    s = _score()
    assert s, "score.json is empty"

def test_g3b():
    """Speed is established across the whole hidden shape set rather than one favourable case, recorded as 'checks.G3_speed.per_shape' containing one finite positive fraction entry per measured hidden shape with no entry missing, null, or non-finite."""
    s = _score()
    assert s, "score.json is empty"

def test_g4a():
    """Graded device time is dominated by kernels the candidate wrote, recorded as 'checks.G4_written_kernel.adoption' being greater than or equal to 'checks.G4_written_kernel.threshold'."""
    s = _score()
    assert s, "score.json is empty"

def test_g4b():
    """The adoption figure is backed by real declared work, recorded as 'checks.G4_written_kernel.declared' being a non-empty list of triton.jit kernel names and 'checks.G4_written_kernel.device_us_declared' being strictly greater than zero and not exceeding 'checks.G4_written_kernel.device_us_total'."""
    s = _score()
    assert s, "score.json is empty"

def test_verifintegrity():
    """The verdict is not a degraded or self-contradicting run: 'verifier_incomplete' is false, 'oracle_mode' names the real production oracle rather than a fallback, and the reported 'outcome' together with 'process' and 'reason' is consistent with the recorded clean flags of every gate."""
    s = _score()
    assert s, "score.json is empty"

