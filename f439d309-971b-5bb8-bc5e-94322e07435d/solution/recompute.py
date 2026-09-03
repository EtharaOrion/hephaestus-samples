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
# FORGE-CANARY-SLOT-0 ee63dc6e01c821ba229fe54a1b17b536b4e12f77393482fb462da30ada6c7758
# FORGE-CANARY-SLOT-1 6cc28039e2c7ac13f582035bcb25468f68dca60367d553d735b2c4eb3d6f5954
# FORGE-CANARY-SLOT-2 2f82a766307e31a9f5ea6337a11f15c1db255764d7ede76ca7b45f9b5e461f75
# FORGE-CANARY-SLOT-3 4c8cd138e01315ef2dec00edbeedc16e701d4e642b5d364680b922189959c5dc
# FORGE-CANARY-END
