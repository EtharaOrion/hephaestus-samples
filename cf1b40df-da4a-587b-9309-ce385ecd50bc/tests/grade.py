#!/usr/bin/env python3
"""Thin verifier shim. The machinery lives in grade_core.py beside this file;
every category-specific value lives in taskdef.py beside it (protocol:
taskdef_api.py). This file exists so tests/ is self-contained and so the
freeze step has a stable place to plant canary tokens.
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

if __name__ == "__main__":
    # Zero-first even here: a grade_core that fails to IMPORT must still leave
    # a score behind, or an unwritten score reads like a clean run downstream.
    out = Path(os.environ.get("SCORE_PATH", "/logs/verifier/score.json"))
    argv = sys.argv[1:]
    for a, v in zip(argv, argv[1:]):
        if a == "--out":
            out = Path(v)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "score": 0.0, "process": 0.0, "verifier_incomplete": True,
            "reason": "verifier did not start: grade_core failed to import"}) + "\n")
    except OSError:
        pass
    import grade_core
    grade_core.run()
# FORGE-CANARY-BEGIN forge-canary/v1
# FORGE-CANARY-SLOT-0 b173dc269c4b528e50f1815d7b6ee405819dac3b0022ede31c6e35a20ff21149
# FORGE-CANARY-SLOT-1 df6844fd145ff8951e4f4e42b79e0fad590894f957d98730c58f0d3a6be2b8bd
# FORGE-CANARY-SLOT-2 e0cae78f54d8feaa8bfa13a59ac9973cfdc0a2ff21bbdde6670a57e046a4911b
# FORGE-CANARY-SLOT-3 edc38a4f5fb6503f8d211c7cd378c974325417c475aa6aa7329c0e106497bf0a
# FORGE-CANARY-END
