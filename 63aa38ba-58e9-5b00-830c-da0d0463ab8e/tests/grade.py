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
# FORGE-CANARY-SLOT-0 835fa0061c214984a01a76f3a06dbedd4d863b282964d54b8dc3e00f4aeee3f1
# FORGE-CANARY-SLOT-1 b2e9295f205f11f4cbf7fb5808ed375d1bdfd797becab32e76a87104eb1f1dee
# FORGE-CANARY-SLOT-2 77935e574cff55c4ecc99d63886b3fccb324d9daa7b48c6454fb1f7d2d8b9903
# FORGE-CANARY-SLOT-3 cf6170bfc4ad891d6d0fb6a82a5b5669a0887be24c088ab861e09de947293fab
# FORGE-CANARY-END
