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
# FORGE-CANARY-SLOT-0 6c929f40021012d3fc5347a991e0c6fb274ffbfc9f7f24122fc4a34b5534855c
# FORGE-CANARY-SLOT-1 4ff9f70f7f0cf0489a84a05379a2b13c29097216dfa01de2674a3043e0417b6d
# FORGE-CANARY-SLOT-2 bc32f0dbf9a92764528e598342a9ecd2c09c651984b8c9737ba8c4a4e0aba4fc
# FORGE-CANARY-SLOT-3 ed11e1f65bf47fbebf39521f6b503c8cf022e8fb15ff3fe355377d4750f1f106
# FORGE-CANARY-END
