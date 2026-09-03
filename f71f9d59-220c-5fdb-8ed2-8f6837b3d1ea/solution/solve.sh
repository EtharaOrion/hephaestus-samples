#!/bin/bash
# GENERATED SECTION. DO NOT HAND-EDIT. Source of truth: solution/grounding.yaml
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="${1:-${CANDIDATE_TREE:-/workspace}}"
cp -f "$HERE/oracle_kernel.py" "$WS/kernel.py"
echo "oracle installed at $WS/kernel.py"
# FORGE-CANARY-BEGIN forge-canary/v1
# FORGE-CANARY-SLOT-0 f6ab49de11d2b8c644e451dbbb006d8b7894acb6f0fe89d1db6403266091d78b
# FORGE-CANARY-SLOT-1 2b0c7224a9eef2572ceced4db57eaa02dfbf7cfcbce5d1d396f9a736f5b94e8d
# FORGE-CANARY-SLOT-2 dc193ff01b034159ce65ff851c3390033cdfe7d8de6fadf586f91fb6f8ad27dd
# FORGE-CANARY-SLOT-3 56f261abf5278f2651488981c78e88b4f0191965de952be4c2ff1efdc83c843e
# FORGE-CANARY-END
