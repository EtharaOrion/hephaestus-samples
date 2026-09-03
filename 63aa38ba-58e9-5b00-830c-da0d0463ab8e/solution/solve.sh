#!/bin/bash
# GENERATED SECTION. DO NOT HAND-EDIT. Source of truth: solution/grounding.yaml
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="${1:-${CANDIDATE_TREE:-/workspace}}"
cp -f "$HERE/oracle_kernel.py" "$WS/kernel.py"
echo "oracle installed at $WS/kernel.py"
# FORGE-CANARY-BEGIN forge-canary/v1
# FORGE-CANARY-SLOT-0 835fa0061c214984a01a76f3a06dbedd4d863b282964d54b8dc3e00f4aeee3f1
# FORGE-CANARY-SLOT-1 b2e9295f205f11f4cbf7fb5808ed375d1bdfd797becab32e76a87104eb1f1dee
# FORGE-CANARY-SLOT-2 77935e574cff55c4ecc99d63886b3fccb324d9daa7b48c6454fb1f7d2d8b9903
# FORGE-CANARY-SLOT-3 cf6170bfc4ad891d6d0fb6a82a5b5669a0887be24c088ab861e09de947293fab
# FORGE-CANARY-END
