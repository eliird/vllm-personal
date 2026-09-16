#!/usr/bin/env bash
# Phase 4: start a warm vLLM worker, quiesce, and checkpoint it with CRIU +
# cuda-checkpoint. Leaves images for snapshot/scripts/p4_restore_vllm.sh.
#
# Env:
#   MODEL   PORT=8000   TAG=small   MODE=plugin|manual
#   IMG=/tmp/p4_snap   EXTRA_ARGS="..."   MAX_MODEL_LEN=4096   GPU_MEM_UTIL=0.90
set -uo pipefail
SNAP="$(cd "$(dirname "$0")/.." && pwd)"   # snapshot/
ROOT="$(cd "$SNAP/.." && pwd)"             # repo root (holds .venv/)

MODEL="${MODEL:?set MODEL}"
PORT="${PORT:-8000}"
TAG="${TAG:-small}"
MODE="${MODE:-plugin}"
IMG="${IMG:-/tmp/p4_snap_$TAG}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
PROMPT="${PROMPT:-The capital of France is}"

LOGDIR="${LOGDIR:-$SNAP/logs}"; mkdir -p "$LOGDIR"
VLOG="$LOGDIR/p4_${TAG}_vllm.log"
rm -rf "$IMG"; mkdir -p "$IMG/plugins"
: > "$VLOG"

if [ "$(id -u)" -eq 0 ]; then SUDO_CRIU=""; else SUDO_CRIU="sudo"; fi

LIBDIR_ARGS=()
if [ "$MODE" = "manual" ]; then LIBDIR_ARGS=(--libdir "$IMG/plugins"); fi

FIFO=""
T0=$(date +%s.%N)
echo "launch: vllm serve $MODEL --port $PORT (mode=$MODE)" | tee -a "$VLOG"
# Log to a regular file (not a FIFO): CRIU must be able to reopen stdout on
# restore, and a deleted FIFO breaks restore.
# shellcheck disable=SC2086
setsid env UV_USE_IO_URING=0 PYTHONUNBUFFERED=1 "$ROOT/.venv/bin/vllm" serve "$MODEL" --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_MEM_UTIL" \
  $EXTRA_ARGS >> "$VLOG" 2>&1 &
SRV=$!
cleanup() {
  kill -TERM -- "-$SRV" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 1800); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 "$SRV" 2>/dev/null || { echo "server exited early"; tail -20 "$VLOG"; exit 1; }
  sleep 0.5
done
T_READY=$(date +%s.%N)
echo "ready: T0->ready=$(awk -v a="$T0" -v b="$T_READY" 'BEGIN{printf "%.3f",b-a}')s"

# Warm up one request.
curl -s "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"prompt\":\"$PROMPT\",\"max_tokens\":8,\"temperature\":0}" \
  > "$LOGDIR/p4_${TAG}_warmup.json"
cat "$LOGDIR/p4_${TAG}_warmup.json"

ROOT_PID=$SRV
echo "root_pid=$ROOT_PID"
pstree -ap "$ROOT_PID" > "$LOGDIR/p4_${TAG}_topology_before.log" 2>&1 || true
ls -l "/proc/$ROOT_PID/fd" > "$LOGDIR/p4_${TAG}_fds_before.log" 2>&1 || true
find /proc/"$ROOT_PID"/task/*/children -maxdepth 0 2>/dev/null | head >/dev/null || true
pgrep -P "$ROOT_PID" 2>/dev/null | while read -r c; do echo "child=$c $(cat /proc/$c/comm 2>/dev/null)"; done | tee "$LOGDIR/p4_${TAG}_children.log"

# Quiesce: no new requests.
sleep 2

echo "=== criu dump (mode=$MODE) ===" | tee -a "$VLOG"
DUMP_T0=$(date +%s.%N)
if [ "$MODE" = "manual" ]; then
  # Suspend CUDA in every descendant that owns a context (plugin disabled).
  for p in $(pgrep -P "$ROOT_PID" 2>/dev/null); do
    $SUDO_CRIU cuda-checkpoint --toggle --pid "$p" 2>&1 || true
  done
fi
$SUDO_CRIU criu dump "${LIBDIR_ARGS[@]}" --shell-job --tree "$ROOT_PID" \
  --images-dir "$IMG" --tcp-established --link-remap 2>&1 | tee "$LOGDIR/p4_${TAG}_dump.log"
DUMP_T1=$(date +%s.%N)
echo "dump_seconds=$(awk -v a="$DUMP_T0" -v b="$DUMP_T1" 'BEGIN{printf "%.3f",b-a}')"
echo "snapshot_size=$(du -sh "$IMG" | cut -f1)"
ps -p "$ROOT_PID" >/dev/null 2>&1 && echo "WARN: root pid still alive" || echo "root process terminated by dump"

cleanup
trap - EXIT
echo "snapshot left in $IMG"
