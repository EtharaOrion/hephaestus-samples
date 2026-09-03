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
# FORGE-CANARY-SLOT-0 de3338688d6986c00757312e50fa2a89281204810a3d37ec23107c88f14465e0
# FORGE-CANARY-SLOT-1 372ff620111834e8dac9c9d39937f65f2dad23ef677ee5e6eb30c1c9485a6b1a
# FORGE-CANARY-SLOT-2 3efd1b54554e6b7636893e2e72469eb8719b5697473a877c24c17aa30d6993d1
# FORGE-CANARY-SLOT-3 bae59c1d9ba44461565342f496a7573d1dfa27bdd303c745b47f816275797915
# FORGE-CANARY-END
