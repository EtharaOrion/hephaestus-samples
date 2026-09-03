#!/bin/bash
# GENERATED SECTION. DO NOT HAND-EDIT. Source of truth: solution/grounding.yaml
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="${1:-${CANDIDATE_TREE:-/workspace}}"
cp -f "$HERE/oracle_kernel.py" "$WS/kernel.py"
echo "oracle installed at $WS/kernel.py"
# FORGE-CANARY-BEGIN forge-canary/v1
# FORGE-CANARY-SLOT-0 2a1deafa70f6b6810bf015b22a201f9107dd5267ca2a152d02cc11af9d42e3a6
# FORGE-CANARY-SLOT-1 e21c30338eee80f964bf3276724a33ee4dfffac9413e43f84a415ab03f47a93d
# FORGE-CANARY-SLOT-2 0ed09920d356ba04ce8c4b4b4e818ce9825f5b39759088fd264df7dcc1ab9d94
# FORGE-CANARY-SLOT-3 8bf6068c93eaa8460bef77ff72b6a8f9b34cfc2382549e56f2d3755680ce8653
# FORGE-CANARY-END
