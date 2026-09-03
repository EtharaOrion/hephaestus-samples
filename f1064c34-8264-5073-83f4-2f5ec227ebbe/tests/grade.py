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
# FORGE-CANARY-SLOT-0 bd75c14bbcecda2649446976c929276c87f84cb606f835b626dc13d237a7b3c1
# FORGE-CANARY-SLOT-1 7b198b9eceb17cf635390975a768934822f0dcde9db54e42af880cf21656f5a9
# FORGE-CANARY-SLOT-2 6f1b5d971b8cc6be8974bc414d7c5ab09158c1bfb08989b156ebe83e3c2b39f3
# FORGE-CANARY-SLOT-3 f349e824bc001431bdbc4b7f6830926c0b61185fb870fa19d266dcf1ab6f88b8
# FORGE-CANARY-END
