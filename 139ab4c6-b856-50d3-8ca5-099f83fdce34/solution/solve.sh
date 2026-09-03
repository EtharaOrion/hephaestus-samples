#!/bin/bash
# GENERATED SECTION. DO NOT HAND-EDIT. Source of truth: solution/grounding.yaml
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="${1:-${CANDIDATE_TREE:-/workspace}}"
cp -f "$HERE/oracle_kernel.py" "$WS/kernel.py"
echo "oracle installed at $WS/kernel.py"
# FORGE-CANARY-BEGIN forge-canary/v1
# FORGE-CANARY-SLOT-0 5f502cffc6145ac72b5802e99e10768d5d4f786f8cc0834efd90a8a95b6c378e
# FORGE-CANARY-SLOT-1 f047a424b68f371e8611ef3f76c3eb166ae396a4daa9552e37a198ba573a4eba
# FORGE-CANARY-SLOT-2 738e3eba492a7096fff592ac9fd2c597e035e8adb88a0eb447fbac3d0f72894a
# FORGE-CANARY-SLOT-3 7d279c030e0d4a58f2bda4d383ff36c0a7d65c1eeb5d49ee6678dc49cf414c77
# FORGE-CANARY-END
