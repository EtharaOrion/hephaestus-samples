#!/bin/bash
# GENERATED SECTION. DO NOT HAND-EDIT. Source of truth: solution/grounding.yaml
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="${1:-${CANDIDATE_TREE:-/workspace}}"
cp -f "$HERE/oracle_kernel.py" "$WS/kernel.py"
echo "oracle installed at $WS/kernel.py"
# FORGE-CANARY-BEGIN forge-canary/v1
# FORGE-CANARY-SLOT-0 4f246aa291e433a8e030563862493822331fb1392d77d71f86ad3e7c8f1cfe27
# FORGE-CANARY-SLOT-1 996489b2cafa7a5b8fe7e241faa945ee167ffbc6692ef43396b04ef87eb6d764
# FORGE-CANARY-SLOT-2 869d801445f7bc24159c0b502b1cd254bd2da38ae8b4574cf1751281f4ba7413
# FORGE-CANARY-SLOT-3 7803ba2d4ce892e0da20b10fa5754e1928e0caad133f596c2147ea29f1f338bd
# FORGE-CANARY-END
