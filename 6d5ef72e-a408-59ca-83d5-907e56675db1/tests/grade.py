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
# TASK-CANARY-BEGIN task-canary/v1
# TASK-CANARY-SLOT-0 2921b6b99291528104b6591d4736cb49310534cb4012e28700f39551f8ea2ecf
# TASK-CANARY-SLOT-1 c1ffb6eb556e495c5c0f9bdc96a5c9c64dde01d18d652a00a5faa326d2dac06c
# TASK-CANARY-SLOT-2 8c87eb101b2bcccd8b2e10f3e5bee8f2e622aaf422fdca17f38302dfae4c9f80
# TASK-CANARY-SLOT-3 20aa8debcdab83347b37bbf06084f3eb28f80a072ed9031346e35c69bc120cfb
# TASK-CANARY-END
