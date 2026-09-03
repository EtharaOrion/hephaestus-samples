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
# FORGE-CANARY-SLOT-0 f6ab49de11d2b8c644e451dbbb006d8b7894acb6f0fe89d1db6403266091d78b
# FORGE-CANARY-SLOT-1 2b0c7224a9eef2572ceced4db57eaa02dfbf7cfcbce5d1d396f9a736f5b94e8d
# FORGE-CANARY-SLOT-2 dc193ff01b034159ce65ff851c3390033cdfe7d8de6fadf586f91fb6f8ad27dd
# FORGE-CANARY-SLOT-3 56f261abf5278f2651488981c78e88b4f0191965de952be4c2ff1efdc83c843e
# FORGE-CANARY-END
