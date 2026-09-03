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
# FORGE-CANARY-SLOT-0 2020c1c5f3297c1c30599fd51a06c953764929922a2bd474bf0a5e627cbf8ca8
# FORGE-CANARY-SLOT-1 bfc12e636519eb56dfe1afba9612b77ddb42bdf99604dfc205fd53ba8b20a812
# FORGE-CANARY-SLOT-2 d08683cffec923750b75f3b4e1f88b74cce99be4bc0907474dee958915faf3c2
# FORGE-CANARY-SLOT-3 3fa9e5ac9e26368e5572621167a3d49453ab0411713300a41fdc249337c296ee
# FORGE-CANARY-END
