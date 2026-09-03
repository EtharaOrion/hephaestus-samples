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
# FORGE-CANARY-SLOT-0 ddc04cc54b21b837ac5b0f07e0ace9920ce0f3a7163e1ed357b5c660ca2f8ef2
# FORGE-CANARY-SLOT-1 6913886b93a188cfab568dd96c3f71af289aa8d364d7b9c026c64facdfeea081
# FORGE-CANARY-SLOT-2 1a48efddf7e826c02c9a973b42b72c2edbd1a5da7375d6d69eccb4dd028744f7
# FORGE-CANARY-SLOT-3 e008db3b35ab885bdd02db1cb573c074c22bd70c5261bd9c40eb517316ca451c
# FORGE-CANARY-END
