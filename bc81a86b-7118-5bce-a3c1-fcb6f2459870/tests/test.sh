set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="${CANDIDATE_TREE:-/workspace}"
SCORE_PATH="${SCORE_PATH:-/logs/verifier/score.json}"
VDIR="$(dirname "$SCORE_PATH")"
mkdir -p "$VDIR"

# Per-run secret. grade.py drains and closes fd 3 BEFORE it imports the candidate,
# and signs the reward projection with it in emit(); emit_reward.py -- a process the
# candidate never entered -- re-derives the tag. A candidate that fabricates
# score.json and calls os._exit(0) never held the secret, so its tag cannot match
# and the reward is withheld.
AUTH="$(head -c 32 /dev/urandom 2>/dev/null | od -An -tx1 2>/dev/null | tr -d ' \n')"
AUTH="${AUTH:-$(date +%s%N)$$RANDOM}"

"${VERIFIER_PYTHON:-python3}" "$HERE/grade.py" --candidate "$WS/kernel.py" --reference "$HERE/reference.py" --out "$SCORE_PATH" 3<<<"$AUTH"
rc=$?

if [ ! -s "$SCORE_PATH" ]; then
  printf '{"score": 0.0, "process": 0.0, "reason": "verifier produced no score file"}\n' > "$SCORE_PATH"
fi

# THE DETERMINISTIC HALF IS PYTEST. grade.py measures; tests/test_output.py asserts.
# Those assertions are compiled from the rubric items whose mode is `compiled`, one
# test per item, so the deterministic obligations are executable and binary. The
# non-deterministic half stays in tests/rubrics.jsonl, judged by an LLM, because
# "did the agent report a number it never shipped" is not decidable by assertion.
#
# The result is RECORDED and deliberately does NOT gate. pytest's exit code never
# reaches this script's, and never touches the reward: a candidate that legitimately
# fails a gate must be SCORED ZERO, not discarded as a verifier error. Letting pytest
# set the exit code would turn every honest solver failure into missing data, which is
# the exact defect this bundle was re-frozen to remove.
export SCORE_PATH
PYTEST_OUT="$VDIR/pytest_deterministic.txt"
"${VERIFIER_PYTHON:-python3}" -m pytest "$HERE/test_output.py" -q --tb=line -p no:cacheprovider \
    > "$PYTEST_OUT" 2>&1
prc=$?
"${VERIFIER_PYTHON:-python3}" - "$SCORE_PATH" "$PYTEST_OUT" "$prc" <<'PYEOF'
import json, pathlib, re, sys
score, out, prc = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), int(sys.argv[3])
try:
    s = json.loads(score.read_text())
except Exception:
    raise SystemExit(0)
txt = out.read_text() if out.is_file() else ""
passed = int(m.group(1)) if (m := re.search(r"(\d+) passed", txt)) else 0
failed = int(m.group(1)) if (m := re.search(r"(\d+) failed", txt)) else 0
errors = int(m.group(1)) if (m := re.search(r"(\d+) error", txt)) else 0
s["deterministic_pytest"] = {
    "passed": passed, "failed": failed, "errors": errors, "returncode": prc,
    "failed_tests": sorted(set(re.findall(r"\b(test_[A-Za-z0-9_]+)\b", txt)))[:20]
        if (failed or errors) else [],
    "note": "recorded, never gating: the reward comes from the score.json score field alone",
}
score.write_text(json.dumps(s, indent=2) + "\n")
PYEOF

# Derive the reward from the diagnostic score, then read it back. A candidate
# that forked a delayed writer to race this is caught by the comparison rather
# than silently credited.
"${VERIFIER_PYTHON:-python3}" "$HERE/emit_reward.py" "$SCORE_PATH" "$VDIR" 3<<<"$AUTH"

exit $rc
