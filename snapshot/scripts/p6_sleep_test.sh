#!/usr/bin/env bash
# Phase 6: test vLLM sleep/wake alone (no CRIU): warm -> sleep(level=1) ->
# confirm GPU memory is released -> wake_up -> correct inference.
#
# Env: MODEL PORT TAG MAX_MODEL_LEN GPU_MEM_UTIL EXTRA_ARGS
set -uo pipefail
SNAP="$(cd "$(dirname "$0")/.." && pwd)"
# Resolve the vLLM environment: a repo-local .venv, else $VLLM_HOME/.venv.
if [ -z "${VLLM_HOME:-}" ]; then
  if [ -x "$SNAP/.venv/bin/vllm" ]; then VLLM_HOME="$SNAP"
  else VLLM_HOME="$(cd "$SNAP/.." && pwd)"; fi
fi
VLLM_BIN="${VLLM_BIN:-$VLLM_HOME/.venv/bin/vllm}"
VLLM_PY="${VLLM_PY:-$VLLM_HOME/.venv/bin/python}"

MODEL="${MODEL:?set MODEL}"
PORT="${PORT:-8500}"
TAG="${TAG:-sleep}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
PROMPT="${PROMPT:-The capital of France is}"
EXPECT="${EXPECT:-Paris}"
LOGDIR="${LOGDIR:-$SNAP/logs}"; mkdir -p "$LOGDIR"
VLOG="$LOGDIR/p6_${TAG}_vllm.log"; : > "$VLOG"

gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr -d ' MiB'; }
ask() {
  curl -s "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"prompt\":\"$PROMPT\",\"max_tokens\":8,\"temperature\":0}"
}

setsid env UV_USE_IO_URING=0 VLLM_SERVER_DEV_MODE=1 PYTHONUNBUFFERED=1 "$VLLM_BIN" serve "$MODEL" \
  --port "$PORT" --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --enable-sleep-mode $EXTRA_ARGS >> "$VLOG" 2>&1 &
SRV=$!
cleanup() { kill -TERM -- "-$SRV" 2>/dev/null || true; }
trap cleanup EXIT

for _ in $(seq 1 900); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 "$SRV" 2>/dev/null || { echo "server exited early"; tail -20 "$VLOG"; exit 1; }
  sleep 0.5
done
echo "ready, gpu=$(gpu_used) MiB"
echo "warm response: $(ask | head -c 120)"
echo "gpu_after_warm=$(gpu_used) MiB"

echo "=== sleep(level=1) ==="
curl -s -X POST "http://127.0.0.1:$PORT/sleep?level=1" -o /dev/null -w "sleep_http=%{http_code}\n"
sleep 2
echo "is_sleeping=$(curl -s http://127.0.0.1:$PORT/is_sleeping)"
echo "gpu_after_sleep=$(gpu_used) MiB"

echo "=== wake_up ==="
curl -s -X POST "http://127.0.0.1:$PORT/wake_up" -o /dev/null -w "wake_http=%{http_code}\n"
sleep 2
echo "is_sleeping=$(curl -s http://127.0.0.1:$PORT/is_sleeping)"
echo "gpu_after_wake=$(gpu_used) MiB"
resp=$(ask)
echo "post_wake response: $(echo "$resp" | head -c 160)"
echo "$resp" | grep -q "$EXPECT" && echo "PASS: response contains '$EXPECT'" || { echo "FAIL"; exit 1; }
