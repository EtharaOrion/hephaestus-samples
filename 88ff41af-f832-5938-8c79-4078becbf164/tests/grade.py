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
# FORGE-CANARY-SLOT-0 e91c89c5b933e54ead0235cd801ea84738f4c76ff3e70d93ad31d8ec4d4da3b0
# FORGE-CANARY-SLOT-1 7850bfa701dbab19a3499ffa9296d72b49359e91fb8cdc0eed9bed7f701ad10d
# FORGE-CANARY-SLOT-2 541f7cad29753300209c4f5d955e762fa54c4f331a1e999d1bb17073d8ffcfe9
# FORGE-CANARY-SLOT-3 d47c7437729060ca2301d0ea7a60dbc4ad9e0c8c02aafad5654d677d0f6d3194
# FORGE-CANARY-END
