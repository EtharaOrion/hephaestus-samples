#!/bin/bash
# GENERATED SECTION. DO NOT HAND-EDIT. Source of truth: solution/grounding.yaml
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="${1:-${CANDIDATE_TREE:-/workspace}}"
cp -f "$HERE/oracle_kernel.py" "$WS/kernel.py"
echo "oracle installed at $WS/kernel.py"
# FORGE-CANARY-BEGIN forge-canary/v1
# FORGE-CANARY-SLOT-0 e91c89c5b933e54ead0235cd801ea84738f4c76ff3e70d93ad31d8ec4d4da3b0
# FORGE-CANARY-SLOT-1 7850bfa701dbab19a3499ffa9296d72b49359e91fb8cdc0eed9bed7f701ad10d
# FORGE-CANARY-SLOT-2 541f7cad29753300209c4f5d955e762fa54c4f331a1e999d1bb17073d8ffcfe9
# FORGE-CANARY-SLOT-3 d47c7437729060ca2301d0ea7a60dbc4ad9e0c8c02aafad5654d677d0f6d3194
# FORGE-CANARY-END
