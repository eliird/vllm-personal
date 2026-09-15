# Phase 0a — Stack pin, tooling install, and GPU-free CRIU sanity

**ID:** `phase-00a`
**Runs in:** WSL2 for the CPU/CRIU half; target host for stack capture of the GPU half.
**Depends on:** —
**Effort:** 0.5–1 day

## Objective

Pin and record the exact stack, install `criu` and build/install the CRIU CUDA
plugin, and prove that GPU-free CRIU round-trips a plain process. Stage the
Phase 0b CUDA scripts so the GPU gate is a short test once hardware is
available.

## Question

*Is the toolchain (CRIU + CUDA plugin + `cuda-checkpoint`) present, correctly
versioned, discoverable by CRIU, and is CRIU itself healthy on this host?*

## Background

See [`../README.md`](../README.md) shared background. Key points: the CUDA
plugin self-disables when no `/proc/driver/nvidia/gpus/` exists, so it can be
built, installed, and load-tested without a GPU. On WSL2 there is `/dev/dxg`
but no `/dev/nvidia*`; the plugin will report "CUDA plugin is disabled", which
is the expected and correct result.

## Prerequisites

- A shell on the WSL2/target host.
- `sudo` for package install and plugin placement.
- Network access to fetch packages/CRIU source.
- Create the workspace (once, if not already present):

```bash
mkdir -p results scripts snapshots logs results/verification
: > results/RESULTS.md
```

## Steps

### 1. Capture the stack

Write `scripts/p0_stack.sh` to collect and tee the following into
`results/stack.txt`. Include the GPU model/count placeholders even when absent.

```bash
#!/usr/bin/env bash
set -uo pipefail
{
  echo "== date =="; date -u
  echo "== kernel =="; uname -a
  echo "== distro =="; . /etc/os-release && echo "$PRETTY_NAME"
  echo "== is WSL =="; grep -qi microsoft /proc/version && echo yes || echo no
  echo "== gpu =="; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>/dev/null || echo "no nvidia-smi / no driver"
  echo "== gpu count =="; nvidia-smi -L 2>/dev/null | wc -l || echo 0
  echo "== cuda toolkit =="; nvcc --version 2>/dev/null || echo "no nvcc"
  echo "== cuda-checkpoint =="; command -v cuda-checkpoint && cuda-checkpoint --help 2>&1 | head -5 || echo "cuda-checkpoint NOT found"
  echo "== criu =="; criu --version 2>/dev/null || echo "criu NOT found"
  echo "== criu plugins =="; ls -la /usr/lib/criu/ 2>/dev/null; ls -la /usr/local/lib/criu/ 2>/dev/null
  echo "== nvidia devices =="; ls /dev/nvidia* 2>/dev/null || echo none
  echo "== nvidia proc =="; ls /proc/driver/nvidia/gpus 2>/dev/null || echo absent
  echo "== vllm =="; .venv/bin/python -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "vllm not importable in .venv"
} | tee results/stack.txt
```

Run it: `bash scripts/p0_stack.sh`.

### 2. Install CRIU and the CUDA plugin

```bash
# Ubuntu/Debian path:
sudo apt update
sudo apt install -y criu
criu --version

# If apt has no candidate, enable universe first:
#   sudo add-apt-repository universe && sudo apt update
# Otherwise build from source:
#   git clone https://github.com/checkpoint-restore/criu
#   cd criu && sudo apt install -y build-essential libprotobuf-dev libprotobuf-c-dev \
#     protobuf-c-compiler protobuf-compiler python3-protobuf pkg-config libnl-3-dev \
#     libcap-dev libnet1-dev libaio-dev libgnutls28-dev
#   make -j"$(nproc)"

# Build the CUDA plugin (from the CRIU source tree, criu-dev branch):
#   make -C plugins/cuda
#   sudo install -m 0755 plugins/cuda/cuda_plugin.so /usr/lib/criu/
#   ls -la /usr/lib/criu/
```

Place `cuda-checkpoint` somewhere on `$PATH` when operating on the GPU host
(NVIDIA ships prebuilt binaries under `bin/` in the repo). Record its exact
version in `results/stack.txt`.

### 3. Verify CRIU health and plugin discovery

```bash
sudo criu check --all 2>&1 | tee logs/p0_criu_check.log
# Expect a list of "looks good" plus any failing features; capture all of it.

# Confirm CRIU discovers/loads the plugin (with no GPU, expect self-disable):
sudo criu check --all -v4 2>&1 | grep -i cuda | tee logs/p0_plugin_load.log
# Look for: "initialized: cuda_plugin" or "No GPU device found; CUDA plugin is disabled"
```

If `criu check --all` reports failures, record each with its kernel-config
cause. Do not attempt to work around missing kernel features here; report for
human review.

### 4. GPU-free plain-process round-trip

Write `scripts/p0_plain_process.sh`: start a trivial counter process that
increments a value and writes it to a file every second, `criu dump`, kill,
`criu restore`, then verify the counter kept advancing.

```bash
#!/usr/bin/env bash
set -euo pipefail
PREFIX="${PREFIX:-/tmp/p0_plain}"
mkdir -p "$PREFIX"
cat > "$PREFIX/counter.sh" <<'EOF'
#!/usr/bin/env bash
n=0
while true; do n=$((n+1)); echo "$n" > "${PREFIX:-/tmp/p0_plain}/state"; sleep 1; done
EOF
chmod +x "$PREFIX/counter.sh"
PREFIX="$PREFIX" "$PREFIX/counter.sh" & PID=$!
sleep 3
before=$(cat "$PREFIX/state")
sudo criu dump --shell-job --images-dir "$PREFIX/images" --tree "$PID"
sleep 2
sudo criu restore --shell-job --restore-detached --images-dir "$PREFIX/images"
sleep 2
after=$(cat "$PREFIX/state")
echo "before=$before after=$after"
[ "$after" -gt "$before" ] && echo "PASS" || { echo "FAIL"; exit 1; }
```

Run it and append the PASS/FAIL plus timings to `results/RESULTS.md`.

### 5. Stage the Phase 0b CUDA artifacts (do not run without GPU)

Write but do not require execution of:

- `scripts/p0_cuda_min.cu` — allocates device memory, fills a known pattern,
  loops touching it, prints `ready` and a checksum.
- `scripts/p0_cuda_roundtrip.sh` — the exact lock/dump/restore/resume sequence,
  with a `MODE=manual|plugin` switch (pick one per run; see shared background).

```bash
# minimal build for the GPU host:
# nvcc scripts/p0_cuda_min.cu -o scripts/p0_cuda_min
```

## Deliverables

- `results/stack.txt` (complete; GPU fields may read "none" until hardware).
- `results/RESULTS.md` with a `## Phase 0a` section containing: `criu --version`,
  `criu check --all` summary, plugin load/discovery result, plain-process
  round-trip result + timings.
- `scripts/p0_stack.sh`, `scripts/p0_plain_process.sh`, `scripts/p0_cuda_min.cu`,
  `scripts/p0_cuda_roundtrip.sh`.
- `logs/p0_criu_check.log`, `logs/p0_plugin_load.log`.

## Evidence to capture

- Full `criu check --all` output (both stdout and stderr).
- Verbose log line proving the CUDA plugin was discovered.
- Plain-process dump/restore wall-clock and PASS line.

## Constraints / Do NOT

- Do not run CUDA scripts without a GPU/driver.
- Do not modify kernel config to make `criu check` pass; report instead.
- Do not store CRIU images on `/mnt/c` (WSL2); use ext4.
- Do not commit anything.

## Definition of done

`criu check --all` is captured; the CUDA plugin is discoverable by CRIU and
self-disables cleanly without a GPU; `results/stack.txt` is complete; the
plain-process round-trip passes. Phase 0b artifacts exist but are unrun.

## References

- `../checkpoint.md` Phase 0.
- `../README.md` shared background.
- NVIDIA `cuda-checkpoint`: https://github.com/NVIDIA/cuda-checkpoint
- CRIU CUDA plugin: https://github.com/checkpoint-restore/criu/tree/criu-dev/plugins/cuda
