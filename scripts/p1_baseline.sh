#!/usr/bin/env bash
# Phase 1 startup-decomposition harness.
#
# Launches vLLM, timestamps the server log, waits for /health, sends a fixed
# prompt, and records resources. Supports a cold run (clears compile caches) and
# a warm run (reuses them) via CLEAR_CACHE.
#
# Env:
#   MODEL=<hf id or path>   PORT=8000   TAG=cold|warm
#   CLEAR_CACHE=1           EXTRA_ARGS="..."   MAX_MODEL_LEN=2048
#   PRELOAD=1               GPU_MEM_UTIL=0.95  PROMPT="..."
#
# PRELOAD=1 (default) warms the HuggingFace cache before T0 so that a cold run
# clears only the vLLM/inductor compile caches. Download time would otherwise
# be miscounted as weight load.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

MODEL="${MODEL:?set MODEL}"
PORT="${PORT:-8000}"
TAG="${TAG:-cold}"
CLEAR_CACHE="${CLEAR_CACHE:-0}"
PRELOAD="${PRELOAD:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.95}"
PROMPT="${PROMPT:-The capital of France is}"

LOGDIR="${LOGDIR:-logs}"
mkdir -p "$LOGDIR"
VLOG="$LOGDIR/p1_${TAG}_vllm.log"
RLOG="$LOGDIR/p1_${TAG}_resources.log"
FLOG="$LOGDIR/p1_${TAG}_first_response.json"
: > "$VLOG"

if [ "$CLEAR_CACHE" = "1" ]; then
  echo "clearing compile/autotune caches..."
  rm -rf "$HOME/.cache/vllm" "$HOME/.cache/torch/inductor" "$HOME/.cache/triton" \
         "$HOME/.cache/flashinfer" "$HOME/.cache/flashinfer_autotune" 2>/dev/null || true
fi

# Warm the HF cache before T0 so download time is excluded from startup and a
# "cold" run means compile-cache-cold, not first-ever-fetch. Skipped for local
# model paths.
if [ "$PRELOAD" = "1" ] && [ ! -d "$MODEL" ]; then
  echo "pre-fetching weights for $MODEL (HF cache warm; excluded from timing)..."
  MODEL="$MODEL" .venv/bin/python -c \
    'import os; from huggingface_hub import snapshot_download; snapshot_download(os.environ["MODEL"])' \
    >/dev/null 2>&1 || echo "  pre-fetch failed (continuing; download may appear in weights)"
fi

stamp() { while IFS= read -r l; do printf '%s %s\n' "$(date +%s.%N)" "$l"; done; }

FIFO="$(mktemp -u "$LOGDIR/.p1fifo.XXXXXX")"; mkfifo "$FIFO"
( while IFS= read -r l; do printf '%s %s\n' "$(date +%s.%N)" "$l"; done < "$FIFO" >> "$VLOG" ) &
READER=$!

# shellcheck disable=SC2086
T0=$(date +%s.%N)
echo "launch_command: .venv/bin/vllm serve $MODEL --port $PORT --max-model-len $MAX_MODEL_LEN --gpu-memory-utilization $GPU_MEM_UTIL $EXTRA_ARGS" | tee -a "$VLOG"
setsid env PYTHONUNBUFFERED=1 .venv/bin/vllm serve "$MODEL" --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_MEM_UTIL" \
  $EXTRA_ARGS > "$FIFO" 2>&1 &
SRV=$!
cleanup() {
  kill -TERM -- "-$SRV" 2>/dev/null || kill "$SRV" 2>/dev/null
  sleep 3
  kill -KILL -- "-$SRV" 2>/dev/null || true
  kill "$READER" 2>/dev/null || true
  rm -f "$FIFO"
}
trap cleanup EXIT

echo "T0=$T0 waiting for /health..."
ready=0
for _ in $(seq 1 1800); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then ready=1; break; fi
  kill -0 "$SRV" 2>/dev/null || { echo "server exited early"; break; }
  sleep 0.5
done
T6=$(date +%s.%N)
echo "T6=$T6 ready=$ready"
echo "T0->T6 wall seconds: $(awk -v a="$T0" -v b="$T6" 'BEGIN{printf "%.3f", b-a}')"

if [ "$ready" != "1" ]; then
  echo "NOT READY — last log lines:"; tail -30 "$VLOG"
  exit 1
fi

# Time to first successful response.
R0=$(date +%s.%N)
curl -s "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"prompt\":\"$PROMPT\",\"max_tokens\":8,\"temperature\":0}" \
  | tee "$FLOG"
R1=$(date +%s.%N)
echo
echo "T0->first_response seconds: $(awk -v a="$T0" -v b="$R1" 'BEGIN{printf "%.3f", b-a}')"
echo "T6->first_response seconds: $(awk -v a="$T6" -v b="$R1" 'BEGIN{printf "%.3f", b-a}')"
{
  echo "=== harness_summary tag=$TAG ==="
  echo "T0=$T0"
  echo "T6=$T6 ready=$ready"
  echo "clear_cache=$CLEAR_CACHE preload=$PRELOAD"
  echo "T0_T6_wall_seconds=$(awk -v a="$T0" -v b="$T6" 'BEGIN{printf "%.3f", b-a}')"
  echo "T0_first_response_seconds=$(awk -v a="$T0" -v b="$R1" 'BEGIN{printf "%.3f", b-a}')"
  echo "T6_first_response_seconds=$(awk -v a="$T6" -v b="$R1" 'BEGIN{printf "%.3f", b-a}')"
} >> "$VLOG"

{
  echo "=== resources at T6 (tag=$TAG) ==="
  nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
  echo "=== process tree ==="
  ps -eo pid,ppid,rss,cmd --forest | grep -E "vllm|EngineCore|python" | grep -v grep
} | tee "$RLOG"

echo "=== vLLM startup duration lines ==="
grep -n -i -E "took|Loading model weights|Graph capturing|init engine|Available KV cache|GPU KV cache|compile" "$VLOG" | tee -a "$RLOG"
