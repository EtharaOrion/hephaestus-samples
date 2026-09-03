#!/bin/bash
# GENERATED SECTION. DO NOT HAND-EDIT. Source of truth: solution/grounding.yaml
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="${1:-${CANDIDATE_TREE:-/workspace}}"
cp -f "$HERE/oracle_kernel.py" "$WS/kernel.py"
echo "oracle installed at $WS/kernel.py"
# FORGE-CANARY-BEGIN forge-canary/v1
# FORGE-CANARY-SLOT-0 291cde5eb2f646bdd8b7a65b2d9e020499aa1f717aca1e853a0fc60c70f0b3ad
# FORGE-CANARY-SLOT-1 1f880b229b7aba1f0d26116887035113fd68d9b50e9638140bfa96b898206960
# FORGE-CANARY-SLOT-2 4dc73cace413b93e20fd114987e457a196f0a4ad4b43222c1ca235edcfc5e602
# FORGE-CANARY-SLOT-3 7d17190aceccf602a2b3f4a1af66e4c5c86a0ae61f854c04e87b211b036eb791
# FORGE-CANARY-END
