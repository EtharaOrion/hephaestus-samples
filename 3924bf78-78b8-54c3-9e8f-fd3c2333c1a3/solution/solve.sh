#!/bin/bash
# GENERATED SECTION. DO NOT HAND-EDIT. Source of truth: solution/grounding.yaml
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="${1:-${CANDIDATE_TREE:-/workspace}}"
cp -f "$HERE/oracle_kernel.py" "$WS/kernel.py"
echo "oracle installed at $WS/kernel.py"
# FORGE-CANARY-BEGIN forge-canary/v1
# FORGE-CANARY-SLOT-0 044eda056a577a65c7220299a73d3149fe07aa5f3b3fcf7a93b8490da37d53ba
# FORGE-CANARY-SLOT-1 48fe8ab47273a9b68530219458d2dabc65ac7bbd7bdb6fff6c47caec9cb76d56
# FORGE-CANARY-SLOT-2 1444827bb5ef43fcd0c4d39153007a9805bb7482d970cce6982b74b42823a7c7
# FORGE-CANARY-SLOT-3 db7c789d5a08da4d80b752723905259a7ff80f7168556ba2cda874a620216e39
# FORGE-CANARY-END
