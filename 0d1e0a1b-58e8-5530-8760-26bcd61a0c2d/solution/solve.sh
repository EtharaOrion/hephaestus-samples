#!/bin/bash
# GENERATED SECTION. DO NOT HAND-EDIT. Source of truth: solution/grounding.yaml
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="${1:-${CANDIDATE_TREE:-/workspace}}"
cp -f "$HERE/oracle_kernel.py" "$WS/kernel.py"
echo "oracle installed at $WS/kernel.py"
# FORGE-CANARY-BEGIN forge-canary/v1
# FORGE-CANARY-SLOT-0 a8050f31c24844e711ed303ba9424c9974246d7a21b5f270201c1f0c1287336c
# FORGE-CANARY-SLOT-1 e92e77de27b10fa804627db485af273d2d754dae99a2f31e9676c731431db03d
# FORGE-CANARY-SLOT-2 45f8c3dfa575e54e1ac967d3e94cc9965dbc1086023f4f3c310a15a51d08a6b7
# FORGE-CANARY-SLOT-3 e3613ebedb13d998f3af92885ac6b98ce6692de3cd3fc916d31117a71df07768
# FORGE-CANARY-END
