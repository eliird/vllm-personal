# Phase 4 — Snapshot vLLM, unoptimized

**ID:** `phase-04`
**Runs in:** bare-metal GPU for the real runs; a CPU-only vLLM sub-step runs in WSL2.
**Depends on:** `phase-00b`, `phase-01`
**Effort:** 2–4 days

## Objective

Find and fix everything that stops a warm vLLM worker from
checkpointing/restoring cleanly. Do not optimize yet. Get one correct inference
after restore.

## Question

*What prevents vLLM from being checkpointed/restored, and can we get one
correct inference after restore?*

## Background

Read [`../README.md`](../README.md) shared background, especially the CRIU CUDA
plugin limitations that are expected blockers for vLLM:
NVML leftover `/dev/nvidia*` references, fork-without-exec references, and
MIG/MPS/UVM unsupported. Also review vLLM's process topology: V1 runs an
engine-core process plus worker process(es), communicating over ZMQ and shared
memory. Sleep/`/sleep` dispatch is in `vllm/v1/engine` and
`vllm/entrypoints/serve/dev/sleep/api_router.py`.

Start with a **small model** to iterate quickly on process-tree/socket issues,
then repeat with the target model.

## Prerequisites

- Phase 0b and Phase 1 gates passed.
- CRIU + CUDA plugin, `cuda-checkpoint` on `$PATH`.
- vLLM installed in `.venv` on the GPU host.
- Optional WSL sub-step: CPU-only vLLM (`VLLM_TARGET_DEVICE=cpu`) to exercise
  the process tree without a GPU.

## Steps

### 1. (WSL, optional but recommended) CPU-only topology walk

Run a small model with the CPU backend and capture:

```bash
.venv/bin/vllm serve "$SMALL_MODEL" --port 8000 & PID=$!
sleep 5; pstree -ap "$PID" | tee logs/p4_topology_cpu.log
ls -l /proc/$PID/fd | tee logs/p4_fds_cpu.log
```

Register a **stub `SleepModeBackend`** (a `vllm.general_plugins` entry point
that logs suspend/resume and does nothing else) and drive `/sleep` and
`/wake_up` to confirm the dispatch path and the quiesce state machine before
touching CUDA. Record the process tree and the socket/shm inventory.

### 2. GPU small-model snapshot/restore

Write `scripts/p4_snapshot_vllm.sh` and `scripts/p4_restore_vllm.sh` implementing:

```
start vLLM (small model) -> warm up (1 request) -> quiesce ->
cuda-checkpoint toggle -> criu dump (tree, shell-job) -> kill ->
criu restore -> cuda-checkpoint resume -> send request -> assert correct
```

Use one integration model per run (`MODE=manual|plugin`). Keep the exact flags
in the scripts.

### 3. Work the blocker list one at a time

Record each blocker and its fix in `results/RESULTS.md`:

- network sockets / listening port (API server) — `--tcp-established`,
  close/rebind strategy; ensure the old process/port is fully gone before
  restore.
- NCCL communicators / CUDA IPC handles (even single-GPU may init NCCL; see
  `enable_nccl_comm_suspend`).
- shared memory + multiprocessing (engine-core ↔ workers, ZMQ `ipc://`).
- CUDA contexts, FDs to `/dev/nvidia*`, threads.
- NVML leftover references (vLLM uses `pynvml`); apply `DUMP_EXT_FILE` /
  `RESTORE_EXT_FILE` ignore hooks for `/dev/nvidiactl`, `/dev/nvidia{0..N}` if
  the plugin reports leftover refs.
- fork-without-exec processes with no CUDA calls leaving refs.
- no K8s here — verify nothing depends on container/namespace assumptions.

### 4. Repeat with the target model

Once green on the small model, re-run with the target model and record any new
blockers.

## Deliverables

- `scripts/p4_snapshot_vllm.sh`, `scripts/p4_restore_vllm.sh`.
- `results/RESULTS.md` `## Phase 4` with a blocker→fix table and pass/fail for
  small and target models.
- `logs/p4_topology_*.log`, `logs/p4_fds_*.log`, snapshot logs.

## Evidence to capture

- One correct response after restore per model (raw response body).
- Process tree and FD inventory before/after.
- Verbatim CRIU/plugin errors for each blocker.

## Constraints / Do NOT

- Do not optimize; correctness only.
- Do not add a permanent `--snapshot` feature to vLLM core. Use `SleepModeBackend`
  plugin registration or external orchestration.
- Do not use pre-dump.
- Do not commit anything.

## Definition of done

`start → warm → checkpoint → terminate → restore → inference` yields a correct
response on both the small model and the target model, and every blocker and its
fix is logged.

## References

- `../checkpoint.md` Phase 4.
- `vllm/device_allocator/sleep_mode_backend.py`, `vllm/v1/worker/gpu_worker.py`.
- CRIU plugin limitations: https://github.com/checkpoint-restore/criu/tree/criu-dev/plugins/cuda
