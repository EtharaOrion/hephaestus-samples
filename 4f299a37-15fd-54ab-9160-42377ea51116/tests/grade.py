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
# FORGE-CANARY-SLOT-0 4f246aa291e433a8e030563862493822331fb1392d77d71f86ad3e7c8f1cfe27
# FORGE-CANARY-SLOT-1 996489b2cafa7a5b8fe7e241faa945ee167ffbc6692ef43396b04ef87eb6d764
# FORGE-CANARY-SLOT-2 869d801445f7bc24159c0b502b1cd254bd2da38ae8b4574cf1751281f4ba7413
# FORGE-CANARY-SLOT-3 7803ba2d4ce892e0da20b10fa5754e1928e0caad133f596c2147ea29f1f338bd
# FORGE-CANARY-END
