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
# FORGE-CANARY-SLOT-0 beb3630eb2386931f3095668f3f06d7b4e2f0627b5e45a26f7797d6475c30a0a
# FORGE-CANARY-SLOT-1 95ad8a25a8c85262317d4d57b0fbd2237a8bb3ac27393641c70b056941aac641
# FORGE-CANARY-SLOT-2 1462a0a75eeda29612d75c913ed08043ce8515d77567866dcba50c9dca58e168
# FORGE-CANARY-SLOT-3 1dedf61af7593b69390069711c6dedc20133b4403d829b53e6875b0a1e3af5b3
# FORGE-CANARY-END
