#!/bin/bash
# GENERATED SECTION. DO NOT HAND-EDIT. Source of truth: solution/grounding.yaml
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="${1:-${CANDIDATE_TREE:-/workspace}}"
cp -f "$HERE/oracle_kernel.py" "$WS/kernel.py"
echo "oracle installed at $WS/kernel.py"
# FORGE-CANARY-BEGIN forge-canary/v1
# FORGE-CANARY-SLOT-0 b173dc269c4b528e50f1815d7b6ee405819dac3b0022ede31c6e35a20ff21149
# FORGE-CANARY-SLOT-1 df6844fd145ff8951e4f4e42b79e0fad590894f957d98730c58f0d3a6be2b8bd
# FORGE-CANARY-SLOT-2 e0cae78f54d8feaa8bfa13a59ac9973cfdc0a2ff21bbdde6670a57e046a4911b
# FORGE-CANARY-SLOT-3 edc38a4f5fb6503f8d211c7cd378c974325417c475aa6aa7329c0e106497bf0a
# FORGE-CANARY-END
