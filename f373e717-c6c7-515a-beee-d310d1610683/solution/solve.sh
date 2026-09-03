#!/bin/bash
# GENERATED SECTION. DO NOT HAND-EDIT. Source of truth: solution/grounding.yaml
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="${1:-${CANDIDATE_TREE:-/workspace}}"
cp -f "$HERE/oracle_kernel.py" "$WS/kernel.py"
echo "oracle installed at $WS/kernel.py"
# FORGE-CANARY-BEGIN forge-canary/v1
# FORGE-CANARY-SLOT-0 beb3630eb2386931f3095668f3f06d7b4e2f0627b5e45a26f7797d6475c30a0a
# FORGE-CANARY-SLOT-1 95ad8a25a8c85262317d4d57b0fbd2237a8bb3ac27393641c70b056941aac641
# FORGE-CANARY-SLOT-2 1462a0a75eeda29612d75c913ed08043ce8515d77567866dcba50c9dca58e168
# FORGE-CANARY-SLOT-3 1dedf61af7593b69390069711c6dedc20133b4403d829b53e6875b0a1e3af5b3
# FORGE-CANARY-END
