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
# FORGE-CANARY-SLOT-0 6a93dd99f533b5c6800885bdc91abae3577c184ec6ac7d5d1538ba393cb1586b
# FORGE-CANARY-SLOT-1 51e9e8fc2683454d7f16402bbe50e1981b07ea084445591440ab51fa1846cdb2
# FORGE-CANARY-SLOT-2 8f204779bb9fc59e65b62fc9125a4a4eeec5d12fbc9215ea326b216233640f7c
# FORGE-CANARY-SLOT-3 cb4d5b3ad630abb8290b0578178e274afdcff3fc29cb5a83ec2322d6cf4fb6df
# FORGE-CANARY-END
