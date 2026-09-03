#!/bin/bash
# GENERATED SECTION. DO NOT HAND-EDIT. Source of truth: solution/grounding.yaml
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="${1:-${CANDIDATE_TREE:-/workspace}}"
cp -f "$HERE/oracle_kernel.py" "$WS/kernel.py"
echo "oracle installed at $WS/kernel.py"
# FORGE-CANARY-BEGIN forge-canary/v1
# FORGE-CANARY-SLOT-0 0dbcb37aa49e750934ca346e9728b2478ca3ac23a92fa990c606eaceeb387dc0
# FORGE-CANARY-SLOT-1 d74d65d8b8c6b275caac0adc871695e11373e8ef8a88eb37a9bb0ce1334a8523
# FORGE-CANARY-SLOT-2 b6dd38b39decebc93bb079d81367a0d7c5d9d1f5c9857133a93ba76b414a9333
# FORGE-CANARY-SLOT-3 d84ed212e57d92108a7c6f0019ccc29d9c4568f3c39780a2404ad876defbe034
# FORGE-CANARY-END
