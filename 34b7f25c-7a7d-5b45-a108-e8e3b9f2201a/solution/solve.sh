#!/bin/bash
# GENERATED SECTION. DO NOT HAND-EDIT. Source of truth: solution/grounding.yaml
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="${1:-${CANDIDATE_TREE:-/workspace}}"
cp -f "$HERE/oracle_kernel.py" "$WS/kernel.py"
echo "oracle installed at $WS/kernel.py"
# FORGE-CANARY-BEGIN forge-canary/v1
# FORGE-CANARY-SLOT-0 d496256d2ec9a1098ab38268bc44b8f2e34ff00736d440a3adbd8786124693dc
# FORGE-CANARY-SLOT-1 fc4911e36fce37ba42c05c98fcb3404e2f65cf4995a1f71c6ac7edf2e32f5e02
# FORGE-CANARY-SLOT-2 de521e9094845359d40b828a7fecff28f64b18a5be4d71f60cce414d011ba557
# FORGE-CANARY-SLOT-3 48cd51a7a3604374d50cf1faa7856d02176588925907c8730f704754addab66e
# FORGE-CANARY-END
