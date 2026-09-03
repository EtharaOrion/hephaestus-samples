#!/bin/bash
# GENERATED SECTION. DO NOT HAND-EDIT. Source of truth: solution/grounding.yaml
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="${1:-${CANDIDATE_TREE:-/workspace}}"
cp -f "$HERE/oracle_kernel.py" "$WS/kernel.py"
echo "oracle installed at $WS/kernel.py"
# FORGE-CANARY-BEGIN forge-canary/v1
# FORGE-CANARY-SLOT-0 bd75c14bbcecda2649446976c929276c87f84cb606f835b626dc13d237a7b3c1
# FORGE-CANARY-SLOT-1 7b198b9eceb17cf635390975a768934822f0dcde9db54e42af880cf21656f5a9
# FORGE-CANARY-SLOT-2 6f1b5d971b8cc6be8974bc414d7c5ab09158c1bfb08989b156ebe83e3c2b39f3
# FORGE-CANARY-SLOT-3 f349e824bc001431bdbc4b7f6830926c0b61185fb870fa19d266dcf1ab6f88b8
# FORGE-CANARY-END
