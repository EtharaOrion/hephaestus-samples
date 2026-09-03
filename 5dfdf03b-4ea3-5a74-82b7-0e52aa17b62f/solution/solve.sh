#!/bin/bash
# GENERATED SECTION. DO NOT HAND-EDIT. Source of truth: solution/grounding.yaml
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="${1:-${CANDIDATE_TREE:-/workspace}}"
cp -f "$HERE/oracle_kernel.py" "$WS/kernel.py"
echo "oracle installed at $WS/kernel.py"
# FORGE-CANARY-BEGIN forge-canary/v1
# FORGE-CANARY-SLOT-0 de3338688d6986c00757312e50fa2a89281204810a3d37ec23107c88f14465e0
# FORGE-CANARY-SLOT-1 372ff620111834e8dac9c9d39937f65f2dad23ef677ee5e6eb30c1c9485a6b1a
# FORGE-CANARY-SLOT-2 3efd1b54554e6b7636893e2e72469eb8719b5697473a877c24c17aa30d6993d1
# FORGE-CANARY-SLOT-3 bae59c1d9ba44461565342f496a7573d1dfa27bdd303c745b47f816275797915
# FORGE-CANARY-END
