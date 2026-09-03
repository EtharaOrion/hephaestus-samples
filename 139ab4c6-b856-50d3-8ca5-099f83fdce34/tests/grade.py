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
# FORGE-CANARY-SLOT-0 5f502cffc6145ac72b5802e99e10768d5d4f786f8cc0834efd90a8a95b6c378e
# FORGE-CANARY-SLOT-1 f047a424b68f371e8611ef3f76c3eb166ae396a4daa9552e37a198ba573a4eba
# FORGE-CANARY-SLOT-2 738e3eba492a7096fff592ac9fd2c597e035e8adb88a0eb447fbac3d0f72894a
# FORGE-CANARY-SLOT-3 7d279c030e0d4a58f2bda4d383ff36c0a7d65c1eeb5d49ee6678dc49cf414c77
# FORGE-CANARY-END
