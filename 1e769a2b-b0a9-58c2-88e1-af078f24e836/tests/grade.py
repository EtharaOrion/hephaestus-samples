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
# FORGE-CANARY-SLOT-0 50b60f1e1b17fe30bb86ac7fb228cf7a7de098cb78a5cb1e72b144c1f74837c5
# FORGE-CANARY-SLOT-1 f3b6b52316a6518c57cfc9315823fc4a1c8a4ff8745408ea559892f394cf082c
# FORGE-CANARY-SLOT-2 fd2b75436b6b0b26fc14d2399d3961191945b62bf6fd4f37ff72f83f2bff7ea5
# FORGE-CANARY-SLOT-3 5f0f41d1ebd670dff3088a7c195b8a4eb6596de3385c4a1722d33e4587b15af1
# FORGE-CANARY-END
