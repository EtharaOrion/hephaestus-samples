#!/bin/bash
# GENERATED SECTION. DO NOT HAND-EDIT. Source of truth: solution/grounding.yaml
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="${1:-${CANDIDATE_TREE:-/workspace}}"
cp -f "$HERE/oracle_kernel.py" "$WS/kernel.py"
echo "oracle installed at $WS/kernel.py"
# FORGE-CANARY-BEGIN forge-canary/v1
# FORGE-CANARY-SLOT-0 a67c7147dadb026ca22380fe0ae29b13aae8013ae3d9ade671aa63d3f3914cd7
# FORGE-CANARY-SLOT-1 36d74bacec9865f8221688e19ad0abadb6fdde1389675a1d4a627add6157d4d0
# FORGE-CANARY-SLOT-2 31183c6bdaa23318a7d211bd3c1b4dbf19e583763b2a433fc87d64c4edf95ff5
# FORGE-CANARY-SLOT-3 bff5b333d2ac75c2d25ea6280dd8e7bf0268981cc2191487f9467eaf2b50256c
# FORGE-CANARY-END
