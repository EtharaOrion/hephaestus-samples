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
# FORGE-CANARY-SLOT-0 0dbcb37aa49e750934ca346e9728b2478ca3ac23a92fa990c606eaceeb387dc0
# FORGE-CANARY-SLOT-1 d74d65d8b8c6b275caac0adc871695e11373e8ef8a88eb37a9bb0ce1334a8523
# FORGE-CANARY-SLOT-2 b6dd38b39decebc93bb079d81367a0d7c5d9d1f5c9857133a93ba76b414a9333
# FORGE-CANARY-SLOT-3 d84ed212e57d92108a7c6f0019ccc29d9c4568f3c39780a2404ad876defbe034
# FORGE-CANARY-END
