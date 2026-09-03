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
# FORGE-CANARY-SLOT-0 d496256d2ec9a1098ab38268bc44b8f2e34ff00736d440a3adbd8786124693dc
# FORGE-CANARY-SLOT-1 fc4911e36fce37ba42c05c98fcb3404e2f65cf4995a1f71c6ac7edf2e32f5e02
# FORGE-CANARY-SLOT-2 de521e9094845359d40b828a7fecff28f64b18a5be4d71f60cce414d011ba557
# FORGE-CANARY-SLOT-3 48cd51a7a3604374d50cf1faa7856d02176588925907c8730f704754addab66e
# FORGE-CANARY-END
