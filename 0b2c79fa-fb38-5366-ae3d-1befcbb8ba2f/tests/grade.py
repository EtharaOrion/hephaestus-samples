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
# FORGE-CANARY-SLOT-0 d538446f85bc39875bb628f00cf17a7a97a5d159517f50bd43ce3c06b2d8acb8
# FORGE-CANARY-SLOT-1 7df7803065da4f96c1800b8cd6e504d2305e6814fc02e8d6ebff6c2f12d828c5
# FORGE-CANARY-SLOT-2 6f74b1b3d0e00698544d032260b10ba32d241718cc4a431f33cd2d390e5e46d7
# FORGE-CANARY-SLOT-3 b3898928d8c8063037f30ba821403dbc0213ca27a8b3aec5485b3f40508e9da9
# FORGE-CANARY-END
