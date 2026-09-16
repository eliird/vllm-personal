#!/usr/bin/env bash
# Stop any running vLLM worker, hung CRIU restores, and free the GPU.
# Run this BEFORE each run to avoid port/GPU/PID conflicts (a leftover restored
# worker makes the next `criu restore` fail with "Can't fork for <pid>: File exists").
#
# Usage:
#   bash scripts/stop_vllm.sh [--shm] [PORT...]
#     --shm   also remove /dev/shm/link_remap.* and /dev/shm/sem.*
#             (ONLY before a snapshot, never between dump and restore)
set -uo pipefail

# Re-exec as root so all kills work (prompts once for sudo).
if [ "$(id -u)" -ne 0 ]; then
  exec sudo -E bash "$0" "$@"
fi

CLEAN_SHM=0
PORTS=()
for a in "$@"; do
  case "$a" in
    --shm) CLEAN_SHM=1 ;;
    *) PORTS+=("$a") ;;
  esac
done
[ ${#PORTS[@]} -eq 0 ] && PORTS=(8400 8401 8411 8420 8610)

# 1. hung CRIU restores
pkill -9 -x criu 2>/dev/null || true

# 2. anything listening on the known ports
if command -v fuser >/dev/null 2>&1; then
  for port in "${PORTS[@]}"; do fuser -k "${port}/tcp" 2>/dev/null || true; done
fi

# 3. vLLM API server / EngineCore / resource tracker.
#    Match both comm (EngineCore sets its thread name) and cmd. The patterns
#    live only in this script, so they cannot match an interactive shell.
pids=$(ps -eo pid=,comm=,cmd= | awk '
  $2 == "vllm" || $2 ~ /^VLLM::Engine/ ||
  /\.venv\/bin\/vllm serve/ || /multiprocessing.resource_tracker/ {print $1}')
[ -n "${pids:-}" ] && kill -9 $pids 2>/dev/null || true

# 4. anything still holding the GPU
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' '); do
  kill -9 "$p" 2>/dev/null || true
done

# 5. optionally clear /dev/shm semaphores (before a snapshot only)
if [ "$CLEAN_SHM" = "1" ]; then
  rm -f /dev/shm/link_remap.* /dev/shm/sem.* 2>/dev/null || true
fi

sleep 2
echo "stopped vLLM/criu; gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | tr -d ' ')"
