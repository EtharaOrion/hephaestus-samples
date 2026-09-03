#!/bin/bash
# GENERATED SECTION. DO NOT HAND-EDIT. Source of truth: solution/grounding.yaml
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="${1:-${CANDIDATE_TREE:-/workspace}}"
cp -f "$HERE/oracle_kernel.py" "$WS/kernel.py"
echo "oracle installed at $WS/kernel.py"
# FORGE-CANARY-BEGIN forge-canary/v1
# FORGE-CANARY-SLOT-0 ecf5206580e37be68bc54a69b3ad246bad51767f4cc933c8777403fa864ef951
# FORGE-CANARY-SLOT-1 b6e9982ed2ead637d94fb4c2a78ffdc6d891542b2ebfaeeaf60aabe932ddb3c6
# FORGE-CANARY-SLOT-2 5211f21413770f59bf72e5dbfb4cd79bc7d06b47f1ab9d1ca507ed3dc4e73d62
# FORGE-CANARY-SLOT-3 671305a7b95a9d8f290e21eefb64b67ad48046e4587d802513cc93e2d3cc4b15
# FORGE-CANARY-END
