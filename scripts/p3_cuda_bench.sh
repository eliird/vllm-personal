#!/usr/bin/env bash
# Phase 3 CUDA device-memory sweep: measure checkpoint/restore time, host RSS
# peak (device->host copy), and state preservation as device memory grows.
#
# Env: SIZES="1024 4096 8192 12288"  MODE=manual|plugin  PROG=torch|nvcc
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

SIZES="${SIZES:-1024 4096 8192 12288}"
MODE="${MODE:-manual}"
PROG="${PROG:-torch}"
WAIT="${WAIT:-120}"

if [ "$(id -u)" -eq 0 ]; then SUDO=""; else SUDO="sudo"; fi

if [ "$PROG" = "nvcc" ]; then
  NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
  [ -x scripts/p3_cuda_sweep ] || "$NVCC" -O2 -arch="${NVCC_ARCH:-sm_89}" scripts/p3_cuda_sweep.cu -o scripts/p3_cuda_sweep
fi

LIBDIR_ARGS=()
for MB in $SIZES; do
  D="/tmp/p3_$MB"; rm -rf "$D"; mkdir -p "$D/images" "$D/no_plugins"
  LIBDIR_ARGS=(); [ "$MODE" = "manual" ] && LIBDIR_ARGS=(--libdir "$D/no_plugins")
  : > "$D/prog.log"
  if [ "$PROG" = "nvcc" ]; then
    ./scripts/p3_cuda_sweep "$MB" "$D/state" 1000000000 > "$D/prog.log" 2>&1 &
  else
    .venv/bin/python -u scripts/p3_cuda_sweep_torch.py "$MB" "$D/state" 1000000000 > "$D/prog.log" 2>&1 &
  fi
  PID=$!
  trap 'kill "$PID" 2>/dev/null || true' EXIT
  for _ in $(seq 1 "$WAIT"); do [ -s "$D/state" ] && break; sleep 0.5; done
  if [ ! -s "$D/state" ]; then echo "MB=$MB FAIL: probe not ready"; cat "$D/prog.log"; kill "$PID" 2>/dev/null; continue; fi
  sleep 3
  before=$(cat "$D/state")
  gpu_before=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)

  t0=$(date +%s%N)
  if [ "$MODE" = "manual" ]; then $SUDO cuda-checkpoint --toggle --pid "$PID"; fi
  t1=$(date +%s%N)
  rss_peak=$(awk '/VmHWM/{print $2}' "/proc/$PID/status" 2>/dev/null)

  $SUDO criu dump "${LIBDIR_ARGS[@]}" --shell-job --images-dir "$D/images" --tree "$PID" 2>&1 | tee "$D/dump.log"
  t2=$(date +%s%N)
  $SUDO criu restore "${LIBDIR_ARGS[@]}" --shell-job --restore-detached --images-dir "$D/images" 2>&1 | tee "$D/restore.log"
  t3=$(date +%s%N)
  if [ "$MODE" = "manual" ]; then $SUDO cuda-checkpoint --toggle --pid "$PID"; fi
  t4=$(date +%s%N)

  sleep 3
  after=$(cat "$D/state")
  gpu_after=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  img_kb=$(du -sk "$D/images" | cut -f1)

  checkpoint_ms=$(( (t1 - t0) / 1000000 ))
  dump_ms=$(( (t2 - t1) / 1000000 ))
  restore_ms=$(( (t3 - t2) / 1000000 ))
  resume_ms=$(( (t4 - t3) / 1000000 ))

  b_ctr=${before%% *}; b_sum=${before##* }
  a_ctr=${after%% *};  a_sum=${after##* }
  if [ "${a_ctr:-0}" -gt "${b_ctr:-0}" ] 2>/dev/null && [ "$a_sum" = "$b_sum" ]; then v=PASS; else v=FAIL; fi
  printf 'MB=%s mode=%s checkpoint_ms=%s dump_ms=%s restore_ms=%s resume_ms=%s rss_peak_kb=%s gpu_before=%s gpu_after=%s image_kb=%s before=[%s] after=[%s] %s\n' \
    "$MB" "$MODE" "$checkpoint_ms" "$dump_ms" "$restore_ms" "$resume_ms" "${rss_peak:-?}" "$gpu_before" "$gpu_after" "$img_kb" "$before" "$after" "$v"
  kill "$PID" 2>/dev/null || true
  sleep 2
done
