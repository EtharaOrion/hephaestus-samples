#!/bin/bash
# GENERATED SECTION. DO NOT HAND-EDIT. Source of truth: solution/grounding.yaml
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="${1:-${CANDIDATE_TREE:-/workspace}}"
cp -f "$HERE/oracle_kernel.py" "$WS/kernel.py"
echo "oracle installed at $WS/kernel.py"
# FORGE-CANARY-BEGIN forge-canary/v1
# FORGE-CANARY-SLOT-0 6c929f40021012d3fc5347a991e0c6fb274ffbfc9f7f24122fc4a34b5534855c
# FORGE-CANARY-SLOT-1 4ff9f70f7f0cf0489a84a05379a2b13c29097216dfa01de2674a3043e0417b6d
# FORGE-CANARY-SLOT-2 bc32f0dbf9a92764528e598342a9ecd2c09c651984b8c9737ba8c4a4e0aba4fc
# FORGE-CANARY-SLOT-3 ed11e1f65bf47fbebf39521f6b503c8cf022e8fb15ff3fe355377d4750f1f106
# FORGE-CANARY-END
