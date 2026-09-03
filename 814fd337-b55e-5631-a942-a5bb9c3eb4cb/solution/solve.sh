#!/bin/bash
# GENERATED SECTION. DO NOT HAND-EDIT. Source of truth: solution/grounding.yaml
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="${1:-${CANDIDATE_TREE:-/workspace}}"
cp -f "$HERE/oracle_kernel.py" "$WS/kernel.py"
echo "oracle installed at $WS/kernel.py"
# FORGE-CANARY-BEGIN forge-canary/v1
# FORGE-CANARY-SLOT-0 613d1dd8b0c1d7e4655d28fa89c2ca836cfd992893236cc41d5ff73ed1dd6276
# FORGE-CANARY-SLOT-1 ddedab146e135cbfe7af9106f7c07ab4ade9ddcf30e449923b69e83b565b3b64
# FORGE-CANARY-SLOT-2 7bc132e9a66829b35dafd53d4e676a93033fe764f867322eeba98aca31219906
# FORGE-CANARY-SLOT-3 f35bb1bcf447de1931c459f34599ea1c809dc379dda80dc035dae5c3a86de873
# FORGE-CANARY-END
