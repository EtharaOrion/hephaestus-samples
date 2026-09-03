#!/bin/bash
# GENERATED SECTION. DO NOT HAND-EDIT. Source of truth: solution/grounding.yaml
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="${1:-${CANDIDATE_TREE:-/workspace}}"
cp -f "$HERE/oracle_kernel.py" "$WS/kernel.py"
echo "oracle installed at $WS/kernel.py"
# FORGE-CANARY-BEGIN forge-canary/v1
# FORGE-CANARY-SLOT-0 ee63dc6e01c821ba229fe54a1b17b536b4e12f77393482fb462da30ada6c7758
# FORGE-CANARY-SLOT-1 6cc28039e2c7ac13f582035bcb25468f68dca60367d553d735b2c4eb3d6f5954
# FORGE-CANARY-SLOT-2 2f82a766307e31a9f5ea6337a11f15c1db255764d7ede76ca7b45f9b5e461f75
# FORGE-CANARY-SLOT-3 4c8cd138e01315ef2dec00edbeedc16e701d4e642b5d364680b922189959c5dc
# FORGE-CANARY-END
