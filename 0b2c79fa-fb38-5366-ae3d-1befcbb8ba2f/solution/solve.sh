#!/bin/bash
# GENERATED SECTION. DO NOT HAND-EDIT. Source of truth: solution/grounding.yaml
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="${1:-${CANDIDATE_TREE:-/workspace}}"
cp -f "$HERE/oracle_kernel.py" "$WS/kernel.py"
echo "oracle installed at $WS/kernel.py"
# FORGE-CANARY-BEGIN forge-canary/v1
# FORGE-CANARY-SLOT-0 d538446f85bc39875bb628f00cf17a7a97a5d159517f50bd43ce3c06b2d8acb8
# FORGE-CANARY-SLOT-1 7df7803065da4f96c1800b8cd6e504d2305e6814fc02e8d6ebff6c2f12d828c5
# FORGE-CANARY-SLOT-2 6f74b1b3d0e00698544d032260b10ba32d241718cc4a431f33cd2d390e5e46d7
# FORGE-CANARY-SLOT-3 b3898928d8c8063037f30ba821403dbc0213ca27a8b3aec5485b3f40508e9da9
# FORGE-CANARY-END
