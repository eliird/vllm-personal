# Phase 0b — Minimal CUDA checkpoint/kill/restore round-trip

**ID:** `phase-00b`
**Runs in:** bare-metal Linux with a real NVIDIA GPU and driver.
**Depends on:** `phase-00a`
**Effort:** 0.5–1 day

## Objective

Prove the single most fragile dependency works on this exact GPU + driver +
CRIU + plugin combination: a trivial CUDA process is checkpointed, killed, and
restored with its GPU state intact.

## Question

*Can `cuda-checkpoint` + CRIU round-trip a trivial CUDA process on this host at
all?*

## Background

See [`../README.md`](../README.md) shared background for the suspend/resume
mechanism and the two integration models (manual `--toggle` vs plugin-driven).
This is a **hard gate**: if it fails, later phases are moot and the issue is a
driver/plugin escalation, not something to engineer around.

**This cannot run on WSL2.** WSL exposes `/dev/dxg` and no `/dev/nvidia*` or
`/proc/driver/nvidia`, so `cuda-checkpoint` and the CUDA plugin will not work.

## Prerequisites

- Bare-metal GPU host; driver r555+ (prefer 570+ for CRIU 4.0 process-tree
  support; 580+ for migration; 610 for IPC).
- CRIU + `cuda_plugin.so` installed (Phase 0a), `cuda-checkpoint` on `$PATH`.
- CUDA toolkit (`nvcc`) or a Python+torch/CuPy fallback.
- No MIG, no MPS, no UVM allocations in the test process.

## Steps

### 1. Record the GPU stack and unsupported-feature guard

Append to `results/stack.txt`:

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total,mig.mode.current --format=csv | tee -a results/stack.txt
nvidia-smi -L | tee -a results/stack.txt
cuda-checkpoint --help 2>&1 | head -3 | tee -a results/stack.txt
```

- If `mig.mode.current` is anything other than `Disabled`, **stop**: MIG is
  unsupported for checkpointing. Record and escalate.
- Note whether an MPS daemon is running (`nvidia-cuda-mps-control -d` implies
  MPS); MPS is unsupported.

### 2. Build the minimal CUDA program

`scripts/p0_cuda_min.cu` must:

- allocate a device buffer (e.g. 256 MiB),
- fill it with a known pattern,
- hold a monotonically increasing device-side counter,
- every second write `<counter> <pattern_checksum>` to a status file,
- print `ready` once initialized.

```bash
nvcc scripts/p0_cuda_min.cu -o scripts/p0_cuda_min
```

### 3. Run the manual round-trip

`scripts/p0_cuda_roundtrip.sh` (use `MODE=manual`; pick one model per run):

```bash
#!/usr/bin/env bash
set -euo pipefail
MODE="${MODE:-manual}"
D="${D:-/tmp/p0_cuda}"; rm -rf "$D"; mkdir -p "$D"
scripts/p0_cuda_min "$D/state" & PID=$!
for _ in $(seq 50); do [ -s "$D/state" ] && break; sleep 0.2; done
sleep 3
before=$(cat "$D/state"); echo "before=$before"

if [ "$MODE" = manual ]; then
  cuda-checkpoint --toggle --pid "$PID"
fi
sudo criu dump --shell-job --images-dir "$D/images" --tree "$PID"
kill -0 "$PID" 2>/dev/null || true
sudo criu restore --shell-job --restore-detached --images-dir "$D/images"
if [ "$MODE" = manual ]; then
  cuda-checkpoint --toggle --pid "$PID"
fi
cuda-checkpoint --get-state --pid "$PID"

sleep 3
after=$(cat "$D/state"); echo "after=$after"

# Counter must continue from before, not restart; pattern checksum must be stable.
b_ctr=${before%% *}; b_sum=${before##* }
a_ctr=${after%% *};  a_sum=${after##* }
[ "$a_ctr" -gt "$b_ctr" ] && [ "$a_sum" = "$b_sum" ] && echo "PASS" || echo "FAIL"
```

Run: `MODE=manual bash scripts/p0_cuda_roundtrip.sh | tee logs/p0_cuda_roundtrip.log`.

If the plugin-driven path is installed, also try `MODE=plugin` (skip the manual
`--toggle` calls). Do not mix modes in one run.

### 4. Capture failure diagnostics

On failure, record: the `cuda-checkpoint` output, the `criu dump`/`restore`
error, plugin verbose log, and the exact driver/CRIU/cuda-checkpoint versions.

## Deliverables

- `results/RESULTS.md` `## Phase 0b` section: before/after `<counter> <checksum>`,
  PASS/FAIL, mode used, wall-clock.
- `logs/p0_cuda_roundtrip.log`, plus dump/restore verbose logs on failure.
- `scripts/p0_cuda_min.cu`, `scripts/p0_cuda_roundtrip.sh` finalized.
- Updated `results/stack.txt` with GPU model/driver/MIG status.

## Evidence to capture

- `cuda-checkpoint --get-state` output after resume (expect `running`).
- The status-file values before and after.
- `nvidia-smi --query --display=PIDS` showing the PID on the GPU before
  checkpoint and off the GPU after suspend (manual mode).

## Constraints / Do NOT

- Do not use UVM, MIG, MPS, or IPC memory in the test process.
- Do not use pre-dump (`criu pre-dump`); the CUDA plugin disables itself.
- Do not proceed to other phases if this gate fails.
- Do not commit anything.

## Definition of done

The counter continues (strictly greater than its pre-checkpoint value) and the
pattern checksum is unchanged after restore, in the mode recorded. Failure is
recorded verbatim with the stack version and escalated.

## References

- `../checkpoint.md` Phase 0.
- NVIDIA `cuda-checkpoint` README example: https://github.com/NVIDIA/cuda-checkpoint
- CRIU CUDA plugin: https://github.com/checkpoint-restore/criu/tree/criu-dev/plugins/cuda
