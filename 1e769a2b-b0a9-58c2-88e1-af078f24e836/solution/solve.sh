#!/bin/bash
# GENERATED SECTION. DO NOT HAND-EDIT. Source of truth: solution/grounding.yaml
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="${1:-${CANDIDATE_TREE:-/workspace}}"
cp -f "$HERE/oracle_kernel.py" "$WS/kernel.py"
echo "oracle installed at $WS/kernel.py"
# FORGE-CANARY-BEGIN forge-canary/v1
# FORGE-CANARY-SLOT-0 50b60f1e1b17fe30bb86ac7fb228cf7a7de098cb78a5cb1e72b144c1f74837c5
# FORGE-CANARY-SLOT-1 f3b6b52316a6518c57cfc9315823fc4a1c8a4ff8745408ea559892f394cf082c
# FORGE-CANARY-SLOT-2 fd2b75436b6b0b26fc14d2399d3961191945b62bf6fd4f37ff72f83f2bff7ea5
# FORGE-CANARY-SLOT-3 5f0f41d1ebd670dff3088a7c195b8a4eb6596de3385c4a1722d33e4587b15af1
# FORGE-CANARY-END
