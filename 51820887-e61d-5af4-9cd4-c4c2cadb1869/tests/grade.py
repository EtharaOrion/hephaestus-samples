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
# FORGE-CANARY-SLOT-0 a67c7147dadb026ca22380fe0ae29b13aae8013ae3d9ade671aa63d3f3914cd7
# FORGE-CANARY-SLOT-1 36d74bacec9865f8221688e19ad0abadb6fdde1389675a1d4a627add6157d4d0
# FORGE-CANARY-SLOT-2 31183c6bdaa23318a7d211bd3c1b4dbf19e583763b2a433fc87d64c4edf95ff5
# FORGE-CANARY-SLOT-3 bff5b333d2ac75c2d25ea6280dd8e7bf0268981cc2191487f9467eaf2b50256c
# FORGE-CANARY-END
