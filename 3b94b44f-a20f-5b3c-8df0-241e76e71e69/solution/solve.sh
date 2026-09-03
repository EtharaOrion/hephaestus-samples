#!/bin/bash
# GENERATED SECTION. DO NOT HAND-EDIT. Source of truth: solution/grounding.yaml
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="${1:-${CANDIDATE_TREE:-/workspace}}"
cp -f "$HERE/oracle_kernel.py" "$WS/kernel.py"
echo "oracle installed at $WS/kernel.py"
# FORGE-CANARY-BEGIN forge-canary/v1
# FORGE-CANARY-SLOT-0 3ba11a0060fdbdbd191746ee52a4bef4e373ae78ef266af3192272c554d43c70
# FORGE-CANARY-SLOT-1 dad2583c5cdffe95684adf9ca848bd9c00fe687400e563d18eebf627d6a77325
# FORGE-CANARY-SLOT-2 2b174defda600ed37f94172c8a2ea1a97e1059c6b89f84081fbd0587c9593690
# FORGE-CANARY-SLOT-3 799409199f4e0d6101cc6bd0efc16a073d0d4b5a9e4ac512184e3ff35e067310
# FORGE-CANARY-END
