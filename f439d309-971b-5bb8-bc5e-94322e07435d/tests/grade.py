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
# FORGE-CANARY-SLOT-0 ee63dc6e01c821ba229fe54a1b17b536b4e12f77393482fb462da30ada6c7758
# FORGE-CANARY-SLOT-1 6cc28039e2c7ac13f582035bcb25468f68dca60367d553d735b2c4eb3d6f5954
# FORGE-CANARY-SLOT-2 2f82a766307e31a9f5ea6337a11f15c1db255764d7ede76ca7b45f9b5e461f75
# FORGE-CANARY-SLOT-3 4c8cd138e01315ef2dec00edbeedc16e701d4e642b5d364680b922189959c5dc
# FORGE-CANARY-END
