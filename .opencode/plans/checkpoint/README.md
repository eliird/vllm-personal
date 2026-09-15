# vLLM Snapshot/Restore — Phase Docs

Execution docs for building a non-Kubernetes snapshot/restore system for a warm
vLLM worker using CRIU + `cuda-checkpoint`, with progressive optimization of
restore latency.

The master narrative lives in [`../checkpoint.md`](../checkpoint.md). This
folder is the operational breakdown: one folder per phase, each with a
`task.md` (what a subagent implements) and a `verifier.md` (what a separate
subagent checks later).

## Headline metric

**Time from "start restore" → "first successful inference response."**
"Successful" means a correct response for a fixed prompt, not an HTTP 200.
Record every measured number in `results/RESULTS.md`.

## Hardware framing

Do **not** assume a specific GPU model. Each task records the actual hardware it
ran on. Use the placeholder `<TARGET_GPU>` in commands and fill it from
`results/stack.txt`. Multi-GPU phases use `<N_GPUS>` and scale progressively.

- Single-GPU work: whichever single GPU is available.
- Multi-GPU work: whichever multi-GPU box is available; record topology.
- Snapshot compatibility is keyed to driver version + GPU count/arch, not to a
  marketing name.

Some phases can run on WSL2 (CPU/CRIU/storage only); some require bare-metal
Linux with a real NVIDIA driver. WSL2 exposes `/dev/dxg` (GPU
paravirtualization) and **no** `/dev/nvidia*` or `/proc/driver/nvidia`, so
`cuda-checkpoint` and the CRIU CUDA plugin cannot function there. See the tag
column below.

## Conventions

Workspace (create once at the start of Phase 0a):

```bash
mkdir -p results scripts snapshots logs results/verification
: > results/RESULTS.md
```

| Path | Purpose |
| --- | --- |
| `results/RESULTS.md` | Running results log; append a section per phase. The surviving deliverable. |
| `results/stack.txt` | Exact recorded stack (driver, CUDA, CRIU, cuda-checkpoint, kernel, GPU model/count). |
| `results/restore_profile.md` | Phase 8 profiling output. |
| `results/verification/phase-NN.md` | Verifier verdicts. |
| `scripts/` | All executable scripts (one phase prefix per script, e.g. `p2_criu_bench.sh`). |
| `snapshots/` | CRIU/GPU snapshot images. |
| `logs/` | Raw command output and service logs. |

Store images on a local ext4 filesystem. On WSL2, never use `/mnt/c` (drvfs
does not support the required operations).

## Phase table

| Phase | Objective | Gate | Runs in | Depends on |
| --- | --- | --- | --- | --- |
| 0a | Install/pin CRIU + CUDA plugin, write scripts | `criu check --all` clean; plugin loads+self-disables without GPU; stack recorded | WSL + target prep | — |
| 0b | Minimal CUDA checkpoint/kill/restore round-trip | GPU pattern byte-identical after restore | bare-metal GPU | 0a |
| 1 | Cold-start baseline with init-vs-weight decomposition | Bolded rows populated; dominance conclusion written | bare-metal GPU | 0a |
| 2 | CRIU characterization (no CUDA) | State preserved across memory sweep; times/sizes recorded | WSL | 0a |
| 3 | `cuda-checkpoint` characterization | Restore verified at large device-memory size; scaling curve recorded | bare-metal GPU | 0b |
| 4 | Snapshot vLLM, unoptimized | Correct response on small + target model; blockers logged | bare-metal GPU (+ WSL sub-step) | 0b, 1 |
| 5 | Quiesce lifecycle + thin snapshot manager | One command → restorable snapshot | WSL CLI + GPU integration | 2, 4 |
| 6+7 | Sleep/KV-discard via vLLM sleep mode | Slept worker snapshots/restores correctly; effect quantified | bare-metal GPU | 4, 5 |
| 8 | Profile restore, attack storage bottleneck | Dominant cost named; one intervention shows measured reduction | WSL (host/storage) + GPU (device copy) | 5 |
| 9 | Separate process state from model weights | Concurrent restore+load reaches correct inference | WSL (design/host) + GPU (integration) | 8 |
| 10 | Multi-GPU / TP-EP | Correct inference at each targeted GPU count | bare-metal multi-GPU | 9 |
| 11 | Production-style snapshot manager | Mismatched (e.g. driver) restore refused | WSL-testable | 5, 9 |

### Recommended execution order

Front-load WSL-runnable work while the GPU environment is being prepared:

1. `0a` (stack prep), then `2` (CRIU characterization), `5` (manager CLI),
   `11` (production manager).
2. Host/storage halves of `8` and design half of `9` on WSL.
3. GPU phases once hardware is available: `0b` → `1` → `3` → `4` → `6+7` →
   device halves of `8`/`9` → `10`.

Do not start a phase whose dependency gate has not passed. Record the failure
and stop for human review instead.

## Shared background (read before writing/executing a task)

### cuda-checkpoint

NVIDIA's `cuda-checkpoint` utility (`github.com/NVIDIA/cuda-checkpoint`) toggles
a process's CUDA state. Actions: `--action lock|checkpoint|restore|unlock`,
plus `--toggle`, `--get-state`, `--get-restore-tid`. During suspend it locks
CUDA APIs, drains submitted work, copies device memory to host allocations, and
releases GPU resources; resume reverses this and remaps at the original
addresses. It does not suspend CPU threads.

Version-gated features (driver): 550 base, 555 CRIU plugin, 570 NVML + CRIU 4.0
process-tree integration + lock timeout, 580 GPU migration + container partial
passthrough, 595 ARM, 610 `cuIpcGetMemHandle` IPC + `--launch-job`. IPC memory
from `cuMemExportToShareableHandle()` and UVM are **not** supported.

### CRIU CUDA plugin

`checkpoint-restore/criu`, `plugins/cuda/cuda_plugin.c` (installs as
`cuda_plugin.so`, loaded from `/usr/lib/criu/`). It registers hooks:
`PAUSE_DEVICES → lock`, `CHECKPOINT_DEVICES → checkpoint`,
`RESUME_DEVICES_LATE → restore`/`unlock`. It is **per-PID** and walks the
process tree; it does not iterate GPUs itself — each `cuda-checkpoint` call
handles all contexts of that process.

- Requires `cuda-checkpoint` in `$PATH` and driver r555+.
- Self-disables if `/proc/driver/nvidia/gpus/` is absent, if `--action` is
  unsupported, or under pre-dump (`CR_PLUGIN_STAGE__PRE_DUMP`).
- Limitations: MIG and MPS unsupported; NVML unsupported (leftover
  `/dev/nvidia*` refs fail the dump; workaround via `DUMP_EXT_FILE` /
  `RESTORE_EXT_FILE` ignore hooks); fork-without-exec with no CUDA calls leaves
  refs; restore requires similar GPUs and same GPU count; GPU memory is copied
  into host RAM and then dumped by CRIU ("memory thrashing" on large usage).
- There are **two** integration models — **plugin-driven** (CRIU drives
  `cuda-checkpoint`) and **manual** (`cuda-checkpoint --toggle` around
  `criu dump`/`restore`, as in NVIDIA's README example). Pick one per
  experiment; do not do both. In the manual flow CRIU restores the original
  PID; do not assume a new PID.

### vLLM sleep mode and `SleepModeBackend`

- `vllm/device_allocator/sleep_mode_backend.py` defines `SleepModeBackend` and
  `SleepModeBackendFactory`, with capability flags (`preserves_communicators`,
  `preserves_compiled_artifacts`, `preserves_graphs_with_communicators`,
  `supports_durable_storage`). This is the intended extension point for
  CUDA-checkpoint / CRIU / durable-snapshot backends (RFC #34303). Third-party
  backends register via a `vllm.general_plugins` entry point.
- Default backend `cumem` wraps `CuMemAllocator`
  (`vllm/device_allocator/cumem.py`). Selected by
  `ModelConfig.sleep_mode_backend` (`vllm/config/model.py`).
- **Sleep semantics (verified):** `sleep(level=1)` calls
  `allocator.sleep(offload_tags=("weights",))` — weights are offloaded to
  pinned host RAM and **everything else is discarded** (KV cache). `level=2`
  discards weights and KV with no CPU backup. So a level-1 slept snapshot still
  carries the weight bytes, relocated from device dump to host RAM; only the KV
  portion shrinks.
- `vllm/v1/worker/gpu_worker.py` `sleep`/`wake_up`; `/sleep` endpoint under
  `vllm/entrypoints/serve/dev/sleep/api_router.py`; engine dispatch through
  `vllm/v1/engine/core.py` / `vllm/v1/executor/abstract.py`.
- `enable_nccl_comm_suspend` (`vllm/config/model.py`) drives
  `suspend_device_comms()` around sleep.
- Weight-cache IPC loader (`vllm/model_executor/model_loader/weight_cache/ipc_loader.py`)
  maps a weight-cache daemon's tensors via CUDA IPC; in `zero_copy` mode weights
  live in the daemon's CUDA IPC allocations and **sleep mode weight offloading
  must not be used** with it. Relevant to Phase 9.

### Known corrections embedded in the tasks

- Phase 1 ordering: CUDA context init precedes weight load; the timestamp
  sequence is process start → driver/context init → weight transfer →
  graph capture/compile → KV alloc → ready.
- Phase 6/7 must distinguish sleep level 1 vs level 2 as above.
- Phase 9 (weight separation) is the crux if Phase 1 shows weight-load
  dominates; treat it as essential, not optional.

## Agent rules

- Follow `AGENTS.md`: never use system `python3`/bare `pip`; use `uv` and
  `.venv/bin/python`; run lint/pre-commit as applicable.
- Do not add a large `--snapshot` feature to vLLM core. Prefer the
  `SleepModeBackend` plugin path.
- Do not modify CRIU until profiling proves CRIU is the bottleneck.
- Do not commit unless explicitly asked.
- Record raw evidence (logs, timings, checksums) under `results/` or `logs/`;
  a summary without raw evidence is not acceptable.
