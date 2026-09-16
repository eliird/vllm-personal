#!/usr/bin/env bash
# Phase 4: restore a vLLM worker checkpointed by snapshot/scripts/p4_snapshot_vllm.sh and
# verify one correct inference. Measures restore -> ready and restore -> first
# response.
#
# Env: MODEL  PORT=8000  TAG=small  MODE=plugin|manual  IMG=/tmp/p4_snap_<tag>
set -uo pipefail
SNAP="$(cd "$(dirname "$0")/.." && pwd)"   # snapshot/

MODEL="${MODEL:?set MODEL}"
PORT="${PORT:-8000}"
TAG="${TAG:-small}"
MODE="${MODE:-plugin}"
IMG="${IMG:-/tmp/p4_snap_$TAG}"
PROMPT="${PROMPT:-The capital of France is}"
EXPECT="${EXPECT:-Paris}"
SLEEP="${SLEEP:-0}"

LOGDIR="${LOGDIR:-$SNAP/logs}"; mkdir -p "$LOGDIR"
if [ "$(id -u)" -eq 0 ]; then SUDO_CRIU=""; else SUDO_CRIU="sudo"; fi
LIBDIR_ARGS=()
if [ "$MODE" = "manual" ]; then LIBDIR_ARGS=(--libdir "$IMG/plugins"); fi

T0=$(date +%s.%N)
$SUDO_CRIU criu restore "${LIBDIR_ARGS[@]}" --restore-detached \
  --images-dir "$IMG" --tcp-established 2>&1 | tee "$LOGDIR/p4_${TAG}_restore.log"
T_RET=$(date +%s.%N)

if [ "$MODE" = "manual" ]; then
  sleep 1
  for p in $(pgrep -f 'VLLM::EngineCore' 2>/dev/null); do
    $SUDO_CRIU cuda-checkpoint --toggle --pid "$p" 2>&1 || true
  done
fi

if [ "$SLEEP" != "0" ]; then
  echo "=== wake_up (reload weights host->device, realloc KV) ==="
  curl -s -X POST "http://127.0.0.1:$PORT/wake_up" -o /dev/null -w "wake_http=%{http_code}\n"
  T_WAKE=$(date +%s.%N)
  sleep 2
fi

ready=0
for _ in $(seq 1 1200); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then ready=1; break; fi
  sleep 0.5
done
T_READY=$(date +%s.%N)
echo "ready=$ready"
echo "restore_call_seconds=$(awk -v a="$T0" -v b="$T_RET" 'BEGIN{printf "%.3f",b-a}')"
echo "restore_to_ready_seconds=$(awk -v a="$T0" -v b="$T_READY" 'BEGIN{printf "%.3f",b-a}')"

if [ "$ready" != "1" ]; then
  echo "FAIL: server not healthy after restore"
  pstree -ap 2>/dev/null | grep -i vllm || true
  exit 1
fi

curl -s "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"prompt\":\"$PROMPT\",\"max_tokens\":8,\"temperature\":0}" \
  | tee "$LOGDIR/p4_${TAG}_post_restore_response.json"
echo
T_FIRST=$(date +%s.%N)
echo "restore_to_first_response_seconds=$(awk -v a="$T0" -v b="$T_FIRST" 'BEGIN{printf "%.3f",b-a}')"

R_CALL=$(awk -v a="$T0" -v b="$T_RET" 'BEGIN{printf "%.3f",b-a}')
R_READY=$(awk -v a="$T0" -v b="$T_READY" 'BEGIN{printf "%.3f",b-a}')
R_FIRST=$(awk -v a="$T0" -v b="$T_FIRST" 'BEGIN{printf "%.3f",b-a}')
W_SEC="null"
if [ "${SLEEP:-0}" != "0" ] && [ -n "${T_WAKE:-}" ]; then
  W_SEC=$(awk -v a="$T0" -v b="$T_WAKE" 'BEGIN{printf "%.3f",b-a}')
fi
if grep -q "$EXPECT" "$LOGDIR/p4_${TAG}_post_restore_response.json"; then ok=true; else ok=false; fi
cat > "$LOGDIR/p4_${TAG}_restore_times.json" <<JSON
{"tag":"$TAG","model":"$MODEL","mode":"$MODE","sleep":${SLEEP:-0},"restore_call_seconds":$R_CALL,"wake_seconds":$W_SEC,"restore_to_ready_seconds":$R_READY,"restore_to_first_response_seconds":$R_FIRST,"correct":$ok}
JSON

if grep -q "$EXPECT" "$LOGDIR/p4_${TAG}_post_restore_response.json"; then
  echo "PASS: response contains '$EXPECT'"
else
  echo "FAIL: response does not contain '$EXPECT'"
  exit 1
fi
