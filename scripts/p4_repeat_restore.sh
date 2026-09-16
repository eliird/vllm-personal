#!/usr/bin/env bash
# Phase 4 correctness hardening: restore the same snapshot N times and verify a
# correct response each time (catches nondeterministic corruption/soak issues).
#
# Env: MODEL PORT TAG MODE IMG N
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

N="${N:-3}"
PASS=0
FAIL=0

# Kill a restored worker without matching this script's own command line.
kill_worker() {
  pkill -f '[v]llm serve' 2>/dev/null || true
  pkill -f '[V]LLM::EngineCore' 2>/dev/null || true
  sleep 2
}

for i in $(seq 1 "$N"); do
  kill_worker
  echo "=== restore #$i ==="
  out=$(MODEL="$MODEL" PORT="$PORT" TAG="${TAG}_rep$i" MODE="$MODE" IMG="$IMG" \
    bash scripts/p4_restore_vllm.sh 2>&1)
  echo "$out" | grep -E "ready=|_seconds=|PASS|FAIL"
  if echo "$out" | grep -q "^PASS"; then PASS=$((PASS + 1)); else FAIL=$((FAIL + 1)); fi
done
kill_worker
echo "repeated_restores: pass=$PASS fail=$FAIL out_of=$N"
[ "$FAIL" -eq 0 ]
