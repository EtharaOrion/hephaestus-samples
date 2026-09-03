#!/usr/bin/env python3
"""Regenerate every derived artifact in this bundle from solution/grounding.yaml.

    recompute.py            rewrite the generated artifacts in place
    recompute.py --check    rebuild into a shadow tree and diff against committed bytes

GENERATED SECTION. DO NOT HAND-EDIT. Source of truth: solution/grounding.yaml
"""
from __future__ import annotations

import argparse
import filecmp
import shutil
import sys
import tempfile
from pathlib import Path

BUNDLE = Path(__file__).resolve().parent.parent
FORGE_CORE = BUNDLE.parents[3] / "seed" / "forge" / "core"
sys.path.insert(0, str(FORGE_CORE))

import render_solution  # noqa: E402
import render_truth  # noqa: E402

CATEGORY = BUNDLE.parent
GENERATED = ("solution/TRUTH.md", "solution/rubrics.json", "solution/solve.sh", "tests/test_output.py", "environment/Dockerfile", "environment/docker-compose.yaml", "solution/contamination.yaml")


def build(dst_bundle: Path) -> None:
    render_truth_out = render_truth.render(CATEGORY)
    (dst_bundle / "solution").mkdir(parents=True, exist_ok=True)
    (dst_bundle / "tests").mkdir(parents=True, exist_ok=True)
    (dst_bundle / "solution" / "TRUTH.md").write_text(render_truth_out)
    render_solution.emit(CATEGORY, dst_bundle)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if not args.check:
        build(BUNDLE)
        print(f"regenerated bundle at {BUNDLE}")
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        shadow = Path(tmp) / "bundle"
        (shadow / "solution").mkdir(parents=True)
        shutil.copy2(BUNDLE / "solution" / "grounding.yaml", shadow / "solution" / "grounding.yaml")
        build(shadow)
        drift = [rel for rel in GENERATED
                 if not (shadow / rel).is_file()
                 or not (BUNDLE / rel).is_file()
                 or not filecmp.cmp(shadow / rel, BUNDLE / rel, shallow=False)]
        for rel in drift:
            print(f"differs: {rel}")
        print(f"generated files checked: {len(GENERATED)}")
        print(f"DRIFT: {len(drift)}")
        return 1 if drift else 0


if __name__ == "__main__":
    raise SystemExit(main())
# FORGE-CANARY-BEGIN forge-canary/v1
# FORGE-CANARY-SLOT-0 bd75c14bbcecda2649446976c929276c87f84cb606f835b626dc13d237a7b3c1
# FORGE-CANARY-SLOT-1 7b198b9eceb17cf635390975a768934822f0dcde9db54e42af880cf21656f5a9
# FORGE-CANARY-SLOT-2 6f1b5d971b8cc6be8974bc414d7c5ab09158c1bfb08989b156ebe83e3c2b39f3
# FORGE-CANARY-SLOT-3 f349e824bc001431bdbc4b7f6830926c0b61185fb870fa19d266dcf1ab6f88b8
# FORGE-CANARY-END
