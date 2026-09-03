#!/bin/bash
# GENERATED SECTION. DO NOT HAND-EDIT. Source of truth: solution/grounding.yaml
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="${1:-${CANDIDATE_TREE:-/workspace}}"
cp -f "$HERE/oracle_kernel.py" "$WS/kernel.py"
echo "oracle installed at $WS/kernel.py"
# FORGE-CANARY-BEGIN forge-canary/v1
# FORGE-CANARY-SLOT-0 19386f5093ddc3f25489ad1e573171ada161012e470c3b9d8625cb5eac9c2606
# FORGE-CANARY-SLOT-1 3b085cfcafe8fff741c881194a38510b32d3e1bf8cd1dcea5c82181ef75922b8
# FORGE-CANARY-SLOT-2 b7d6aeace93d74b5bf64cb41bada31abd24dacd9928922865071163705d181cf
# FORGE-CANARY-SLOT-3 17b3553eebd0b23f27d50867357b7ddcc0cb34454ce062932fbc72fe31303d1f
# FORGE-CANARY-END
