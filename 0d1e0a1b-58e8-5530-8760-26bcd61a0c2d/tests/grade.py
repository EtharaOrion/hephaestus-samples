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
# FORGE-CANARY-SLOT-0 a8050f31c24844e711ed303ba9424c9974246d7a21b5f270201c1f0c1287336c
# FORGE-CANARY-SLOT-1 e92e77de27b10fa804627db485af273d2d754dae99a2f31e9676c731431db03d
# FORGE-CANARY-SLOT-2 45f8c3dfa575e54e1ac967d3e94cc9965dbc1086023f4f3c310a15a51d08a6b7
# FORGE-CANARY-SLOT-3 e3613ebedb13d998f3af92885ac6b98ce6692de3cd3fc916d31117a71df07768
# FORGE-CANARY-END
