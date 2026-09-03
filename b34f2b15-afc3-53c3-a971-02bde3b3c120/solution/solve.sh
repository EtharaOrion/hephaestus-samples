#!/bin/bash
# GENERATED SECTION. DO NOT HAND-EDIT. Source of truth: solution/grounding.yaml
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="${1:-${CANDIDATE_TREE:-/workspace}}"
cp -f "$HERE/oracle_kernel.py" "$WS/kernel.py"
echo "oracle installed at $WS/kernel.py"
# FORGE-CANARY-BEGIN forge-canary/v1
# FORGE-CANARY-SLOT-0 8f5ee58ab8b39857e50c03beeea7a3fe78a11f30ac5c7a077d9739b51feab9f9
# FORGE-CANARY-SLOT-1 b31fe674e853ba5d9b6250c4b26c42bf20a59df2460e5acd0d203a8041ddc10f
# FORGE-CANARY-SLOT-2 016816dab60a3cf6855248a0153321e1ebb58cda5888aa5d4018d9c767b5cacf
# FORGE-CANARY-SLOT-3 4887a50f15741a1af5a39225429b1dd6b56d4101d2a794a1650d88c257cc9b15
# FORGE-CANARY-END
