#!/bin/bash
# GENERATED SECTION. DO NOT HAND-EDIT. Source of truth: solution/grounding.yaml
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="${1:-${CANDIDATE_TREE:-/workspace}}"
cp -f "$HERE/oracle_kernel.py" "$WS/kernel.py"
echo "oracle installed at $WS/kernel.py"
# FORGE-CANARY-BEGIN forge-canary/v1
# FORGE-CANARY-SLOT-0 2020c1c5f3297c1c30599fd51a06c953764929922a2bd474bf0a5e627cbf8ca8
# FORGE-CANARY-SLOT-1 bfc12e636519eb56dfe1afba9612b77ddb42bdf99604dfc205fd53ba8b20a812
# FORGE-CANARY-SLOT-2 d08683cffec923750b75f3b4e1f88b74cce99be4bc0907474dee958915faf3c2
# FORGE-CANARY-SLOT-3 3fa9e5ac9e26368e5572621167a3d49453ab0411713300a41fdc249337c296ee
# FORGE-CANARY-END
