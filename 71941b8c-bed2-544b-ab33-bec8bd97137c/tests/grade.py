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
# FORGE-CANARY-SLOT-0 e1babe41f81202dabfe8359afd8b5de29f822b2471ef5b83e1a6ec13bfadd1c8
# FORGE-CANARY-SLOT-1 0226c24712e845c431dad141943e1c24d4e96ff960427d36ebcae525b45da5f5
# FORGE-CANARY-SLOT-2 f66c96128e13464a51db92505b855c2d3e45b9331e8d30843e7c68fa0fe97f49
# FORGE-CANARY-SLOT-3 3143df2c938f203c96533fb9165a5c0ac6d8607bda73a72c3ecaaf9af4f11605
# FORGE-CANARY-END
