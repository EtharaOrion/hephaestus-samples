#!/bin/bash
# GENERATED SECTION. DO NOT HAND-EDIT. Source of truth: solution/grounding.yaml
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="${1:-${CANDIDATE_TREE:-/workspace}}"
cp -f "$HERE/oracle_kernel.py" "$WS/kernel.py"
echo "oracle installed at $WS/kernel.py"
# FORGE-CANARY-BEGIN forge-canary/v1
# FORGE-CANARY-SLOT-0 e1babe41f81202dabfe8359afd8b5de29f822b2471ef5b83e1a6ec13bfadd1c8
# FORGE-CANARY-SLOT-1 0226c24712e845c431dad141943e1c24d4e96ff960427d36ebcae525b45da5f5
# FORGE-CANARY-SLOT-2 f66c96128e13464a51db92505b855c2d3e45b9331e8d30843e7c68fa0fe97f49
# FORGE-CANARY-SLOT-3 3143df2c938f203c96533fb9165a5c0ac6d8607bda73a72c3ecaaf9af4f11605
# FORGE-CANARY-END
