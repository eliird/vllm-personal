#!/usr/bin/env bash
# Minimal CUDA checkpoint/kill/restore round-trip.
#
#   MODE=manual  cuda-checkpoint --toggle around criu dump/restore
#   MODE=plugin  let the CRIU CUDA plugin drive cuda-checkpoint
#   PROG=torch|nvcc  which minimal CUDA program to run (default torch)
#
# Usage: MODE=manual PROG=torch bash scripts/p0_cuda_roundtrip.sh
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1

MODE="${MODE:-manual}"
PROG="${PROG:-torch}"
D="${D:-/tmp/p0_cuda}"
WAIT="${WAIT:-60}"

if [ "$(id -u)" -eq 0 ]; then SUDO=""; else SUDO="sudo"; fi

# Manual mode must not also let the installed CRIU CUDA plugin act, or the CUDA
# state is toggled twice. Point CRIU at an empty plugin dir for manual mode.
LIBDIR_ARGS=()
if [ "$MODE" = "manual" ]; then
  LIBDIR_ARGS=(--libdir "$D/no_plugins")
fi

rm -rf "$D"; mkdir -p "$D/images" "$D/no_plugins"
: > "$D/prog.log"

if [ "$PROG" = "nvcc" ]; then
  [ -x scripts/p0_cuda_min ] || nvcc scripts/p0_cuda_min.cu -o scripts/p0_cuda_min
  CMD=(./scripts/p0_cuda_min "$D/state" 1000000000)
else
  CMD=(.venv/bin/python -u scripts/p0_cuda_min_torch.py "$D/state" 1000000000)
fi

"${CMD[@]}" > "$D/prog.log" 2>&1 &
PID=$!
echo "launched pid=$PID mode=$MODE prog=$PROG"

for _ in $(seq 1 "$WAIT"); do [ -s "$D/state" ] && break; sleep 0.2; done
[ -s "$D/state" ] || { echo "FAIL: program never wrote state"; cat "$D/prog.log"; exit 1; }
grep '^ready' "$D/prog.log" || true
sleep 3
before=$(cat "$D/state"); echo "before=$before"
before_count=$(grep -c '^ready' "$D/prog.log" || true)
echo "gpu_before: $(nvidia-smi --query --display=PIDS 2>/dev/null | grep -c "Process ID.*: $PID" || true) match(es)"

if [ "$MODE" = "manual" ]; then
  echo "== cuda-checkpoint --toggle (suspend) =="
  $SUDO cuda-checkpoint --toggle --pid "$PID"
  echo "== state after suspend =="
  $SUDO cuda-checkpoint --get-state --pid "$PID" || true
  echo "gpu_after_suspend: $(nvidia-smi --query --display=PIDS 2>/dev/null | grep -c "Process ID.*: $PID" || true) match(es)"
fi

echo "== criu dump =="
dump_start=$(date +%s%N)
$SUDO criu dump "${LIBDIR_ARGS[@]}" --shell-job --images-dir "$D/images" --tree "$PID"
dump_end=$(date +%s%N)

echo "== criu restore =="
restore_start=$(date +%s%N)
$SUDO criu restore "${LIBDIR_ARGS[@]}" --shell-job --restore-detached --images-dir "$D/images"
restore_end=$(date +%s%N)

if [ "$MODE" = "manual" ]; then
  echo "== cuda-checkpoint --toggle (resume) =="
  $SUDO cuda-checkpoint --toggle --pid "$PID"
  echo "== state after resume =="
  $SUDO cuda-checkpoint --get-state --pid "$PID" || true
fi

sleep 3
after=$(cat "$D/state"); echo "after=$after"
echo "dump_ms=$(( (dump_end - dump_start) / 1000000 )) restore_ms=$(( (restore_end - restore_start) / 1000000 ))"

b_ctr=${before%% *}; b_sum=${before##* }
a_ctr=${after%% *};  a_sum=${after##* }
if [ "$a_ctr" -gt "$b_ctr" ] 2>/dev/null && [ "$a_sum" = "$b_sum" ]; then
  echo "PASS"
else
  echo "FAIL"; exit 1
fi
