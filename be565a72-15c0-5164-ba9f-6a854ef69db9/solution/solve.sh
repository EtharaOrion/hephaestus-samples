#!/bin/bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="${1:-${CANDIDATE_TREE:-/workspace}}"
cp -f "$HERE/oracle_kernel.py" "$WS/kernel.py"
echo "oracle installed at $WS/kernel.py"
# FORGE-CANARY-BEGIN forge-canary/v1
# FORGE-CANARY-SLOT-0 8215a22e4225cfc30e275ec814e7ddb124a14cd403662c0e21e1589362f4731a
# FORGE-CANARY-SLOT-1 f5410e9508bb5f2fef3c8124cea98af61d159d201a1609420842427ac38b4278
# FORGE-CANARY-SLOT-2 cdd60bafda970634ffb7c64018f6a7d1506f9afb2c3f710b5609f3c3b0444e4b
# FORGE-CANARY-SLOT-3 07b9e669b7aba77eae0eda15405545bc94fa27b78418c7644e3e61266f38da52
# FORGE-CANARY-END
