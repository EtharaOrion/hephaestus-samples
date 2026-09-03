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
# FORGE-CANARY-SLOT-0 4e7e762d8b3dad1dcfb246d2edc3390b8a553be5b596e14b1c00221d229e5cae
# FORGE-CANARY-SLOT-1 f08bb6ec1361f90cd08417c20f95f8ec81e31dc6bcf31a3558867cdecfbd918b
# FORGE-CANARY-SLOT-2 8c5423ffa0358f6242ad22232d8bca7a153d783182ad9b217f7d2e58727be8fc
# FORGE-CANARY-SLOT-3 7fa891e696b67473713e10a3920721b855ce5dff2494d7bc9dff7fbe0536f295
# FORGE-CANARY-END
