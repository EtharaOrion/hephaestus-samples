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
# FORGE-CANARY-SLOT-0 ecf5206580e37be68bc54a69b3ad246bad51767f4cc933c8777403fa864ef951
# FORGE-CANARY-SLOT-1 b6e9982ed2ead637d94fb4c2a78ffdc6d891542b2ebfaeeaf60aabe932ddb3c6
# FORGE-CANARY-SLOT-2 5211f21413770f59bf72e5dbfb4cd79bc7d06b47f1ab9d1ca507ed3dc4e73d62
# FORGE-CANARY-SLOT-3 671305a7b95a9d8f290e21eefb64b67ad48046e4587d802513cc93e2d3cc4b15
# FORGE-CANARY-END
