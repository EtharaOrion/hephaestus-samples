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
# FORGE-CANARY-SLOT-0 2a1deafa70f6b6810bf015b22a201f9107dd5267ca2a152d02cc11af9d42e3a6
# FORGE-CANARY-SLOT-1 e21c30338eee80f964bf3276724a33ee4dfffac9413e43f84a415ab03f47a93d
# FORGE-CANARY-SLOT-2 0ed09920d356ba04ce8c4b4b4e818ce9825f5b39759088fd264df7dcc1ab9d94
# FORGE-CANARY-SLOT-3 8bf6068c93eaa8460bef77ff72b6a8f9b34cfc2382549e56f2d3755680ce8653
# FORGE-CANARY-END
