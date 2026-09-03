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
# FORGE-CANARY-SLOT-0 8f5ee58ab8b39857e50c03beeea7a3fe78a11f30ac5c7a077d9739b51feab9f9
# FORGE-CANARY-SLOT-1 b31fe674e853ba5d9b6250c4b26c42bf20a59df2460e5acd0d203a8041ddc10f
# FORGE-CANARY-SLOT-2 016816dab60a3cf6855248a0153321e1ebb58cda5888aa5d4018d9c767b5cacf
# FORGE-CANARY-SLOT-3 4887a50f15741a1af5a39225429b1dd6b56d4101d2a794a1650d88c257cc9b15
# FORGE-CANARY-END
