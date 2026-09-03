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
# FORGE-CANARY-SLOT-0 044eda056a577a65c7220299a73d3149fe07aa5f3b3fcf7a93b8490da37d53ba
# FORGE-CANARY-SLOT-1 48fe8ab47273a9b68530219458d2dabc65ac7bbd7bdb6fff6c47caec9cb76d56
# FORGE-CANARY-SLOT-2 1444827bb5ef43fcd0c4d39153007a9805bb7482d970cce6982b74b42823a7c7
# FORGE-CANARY-SLOT-3 db7c789d5a08da4d80b752723905259a7ff80f7168556ba2cda874a620216e39
# FORGE-CANARY-END
