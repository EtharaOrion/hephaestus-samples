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
# FORGE-CANARY-SLOT-0 6c929f40021012d3fc5347a991e0c6fb274ffbfc9f7f24122fc4a34b5534855c
# FORGE-CANARY-SLOT-1 4ff9f70f7f0cf0489a84a05379a2b13c29097216dfa01de2674a3043e0417b6d
# FORGE-CANARY-SLOT-2 bc32f0dbf9a92764528e598342a9ecd2c09c651984b8c9737ba8c4a4e0aba4fc
# FORGE-CANARY-SLOT-3 ed11e1f65bf47fbebf39521f6b503c8cf022e8fb15ff3fe355377d4750f1f106
# FORGE-CANARY-END
