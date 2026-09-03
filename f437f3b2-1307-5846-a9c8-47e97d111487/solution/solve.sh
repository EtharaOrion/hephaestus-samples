#!/bin/bash
# GENERATED SECTION. DO NOT HAND-EDIT. Source of truth: solution/grounding.yaml
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="${1:-${CANDIDATE_TREE:-/workspace}}"
cp -f "$HERE/oracle_kernel.py" "$WS/kernel.py"
echo "oracle installed at $WS/kernel.py"
# FORGE-CANARY-BEGIN forge-canary/v1
# FORGE-CANARY-SLOT-0 4e7e762d8b3dad1dcfb246d2edc3390b8a553be5b596e14b1c00221d229e5cae
# FORGE-CANARY-SLOT-1 f08bb6ec1361f90cd08417c20f95f8ec81e31dc6bcf31a3558867cdecfbd918b
# FORGE-CANARY-SLOT-2 8c5423ffa0358f6242ad22232d8bca7a153d783182ad9b217f7d2e58727be8fc
# FORGE-CANARY-SLOT-3 7fa891e696b67473713e10a3920721b855ce5dff2494d7bc9dff7fbe0536f295
# FORGE-CANARY-END
