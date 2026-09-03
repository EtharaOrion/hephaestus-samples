#!/bin/bash
# GENERATED SECTION. DO NOT HAND-EDIT. Source of truth: solution/grounding.yaml
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="${1:-${CANDIDATE_TREE:-/workspace}}"
cp -f "$HERE/oracle_kernel.py" "$WS/kernel.py"
echo "oracle installed at $WS/kernel.py"
# FORGE-CANARY-BEGIN forge-canary/v1
# FORGE-CANARY-SLOT-0 6a93dd99f533b5c6800885bdc91abae3577c184ec6ac7d5d1538ba393cb1586b
# FORGE-CANARY-SLOT-1 51e9e8fc2683454d7f16402bbe50e1981b07ea084445591440ab51fa1846cdb2
# FORGE-CANARY-SLOT-2 8f204779bb9fc59e65b62fc9125a4a4eeec5d12fbc9215ea326b216233640f7c
# FORGE-CANARY-SLOT-3 cb4d5b3ad630abb8290b0578178e274afdcff3fc29cb5a83ec2322d6cf4fb6df
# FORGE-CANARY-END
