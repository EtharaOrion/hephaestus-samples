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
# FORGE-CANARY-SLOT-0 3ba11a0060fdbdbd191746ee52a4bef4e373ae78ef266af3192272c554d43c70
# FORGE-CANARY-SLOT-1 dad2583c5cdffe95684adf9ca848bd9c00fe687400e563d18eebf627d6a77325
# FORGE-CANARY-SLOT-2 2b174defda600ed37f94172c8a2ea1a97e1059c6b89f84081fbd0587c9593690
# FORGE-CANARY-SLOT-3 799409199f4e0d6101cc6bd0efc16a073d0d4b5a9e4ac512184e3ff35e067310
# FORGE-CANARY-END
