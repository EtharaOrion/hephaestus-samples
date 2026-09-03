#!/bin/bash
# GENERATED SECTION. DO NOT HAND-EDIT. Source of truth: solution/grounding.yaml
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="${1:-${CANDIDATE_TREE:-/workspace}}"
cp -f "$HERE/oracle_kernel.py" "$WS/kernel.py"
echo "oracle installed at $WS/kernel.py"
# FORGE-CANARY-BEGIN forge-canary/v1
# FORGE-CANARY-SLOT-0 ddc04cc54b21b837ac5b0f07e0ace9920ce0f3a7163e1ed357b5c660ca2f8ef2
# FORGE-CANARY-SLOT-1 6913886b93a188cfab568dd96c3f71af289aa8d364d7b9c026c64facdfeea081
# FORGE-CANARY-SLOT-2 1a48efddf7e826c02c9a973b42b72c2edbd1a5da7375d6d69eccb4dd028744f7
# FORGE-CANARY-SLOT-3 e008db3b35ab885bdd02db1cb573c074c22bd70c5261bd9c40eb517316ca451c
# FORGE-CANARY-END
