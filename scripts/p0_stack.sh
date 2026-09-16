#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
{
  echo "== date =="; date -u
  echo "== kernel =="; uname -a
  echo "== distro =="; . /etc/os-release && echo "$PRETTY_NAME"
  echo "== is WSL =="; grep -qi microsoft /proc/version && echo yes || echo no
  echo "== gpu =="; nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv 2>/dev/null || echo "no nvidia-smi / no driver"
  echo "== gpu count =="; nvidia-smi -L 2>/dev/null | wc -l || echo 0
  echo "== cuda toolkit =="; nvcc --version 2>/dev/null || echo "no nvcc"
  echo "== cuda-checkpoint =="; command -v cuda-checkpoint && cuda-checkpoint --help 2>&1 | head -5 || echo "cuda-checkpoint NOT found"
  echo "== criu =="; criu --version 2>/dev/null || echo "criu NOT found"
  echo "== criu plugins =="; ls -la /usr/lib/criu/ 2>/dev/null; ls -la /usr/local/lib/criu/ 2>/dev/null
  echo "== nvidia devices =="; ls /dev/nvidia* 2>/dev/null || echo none
  echo "== nvidia proc =="; ls /proc/driver/nvidia/gpus 2>/dev/null || echo absent
  echo "== vllm =="; .venv/bin/python -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "vllm not importable in .venv"
} | tee results/stack.txt
