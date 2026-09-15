# Phase 10 — Multi-GPU / TP-EP

**ID:** `phase-10`
**Runs in:** bare-metal multi-GPU.
**Depends on:** `phase-09`
**Effort:** 3–5 days

## Objective

Extend snapshot/restore to tensor/expert parallelism across GPUs. The model is
MoE, so expert parallel matters.

## Question

*Does the snapshot/restore survive NCCL communicators, CUDA IPC handles, and
multi-process topology?*

## Background

Known constraints from the tooling:

- Restore requires **similar GPUs and the same GPU count** (CRIU plugin docs).
- For GPU migration, **every CUDA-visible GPU must be enumerated**, even unused
  ones (`r580-migration-cli.c`, `--device-map`). The CRIU plugin's restore path
  does not pass `--device-map`, so the default CRIU path assumes same
  GPU set/UUIDs.
- vLLM V1 uses an engine-core process plus worker processes; DP/TP/EP add
  process topology, NCCL communicators, CUDA IPC, and shared memory. See
  `enable_nccl_comm_suspend` (`vllm/config/model.py`) and `suspend_device_comms`.

## Prerequisites

- Phase 9 gate passed.
- Multi-GPU host with a model config that fits. Record GPU count and topology.
- Driver/CRIU/plugin versions unchanged from the single-GPU run.

## Steps

### 1. Scale progressively

For each count in 1 → 2 → 4 → N (whatever the box has): init TP/EP → warm →
snapshot → restore → inference. Keep the exact launch config per count.

```bash
# e.g. .venv/bin/vllm serve $MODEL --tensor-parallel-size N --port 8000
```

### 2. Watch specifically

Record which of these break and the fix for each:

- NCCL communicators (may need `enable_nccl_comm_suspend`),
- CUDA contexts per device,
- CUDA IPC handles,
- shared memory,
- GPU affinity / process topology,
- network interfaces.

### 3. Device-map / enumeration

Confirm the snapshot/restore enumerates all CUDA-visible GPUs. If migration
between GPU sets is attempted, document the `--device-map` requirement and that
the CRIU plugin path does not currently pass it.

### 4. Record

Fill `results/RESULTS.md` per GPU count: pass/fail, snapshot size, restore time,
ready-to-serve, blockers + fixes.

## Deliverables

- Launch/snapshot/restore scripts parameterized by GPU count (`scripts/p10_*`).
- `results/RESULTS.md` `## Phase 10` per-count table with blockers and fixes.
- Communicator/IPC handling notes.

## Evidence to capture

- Process tree and GPU affinity per count.
- NCCL/IPC-related errors verbatim and their fixes.
- Correct inference at each targeted count.

## Constraints / Do NOT

- Do not skip counts; verify each targeted count.
- Do not assume cross-GPU migration works via the CRIU path without proving
  `--device-map` handling.
- Do not commit anything.

## Definition of done

Snapshot/restore yields correct inference at each GPU count targeted, with the
NCCL/IPC handling documented.

## References

- `../checkpoint.md` Phase 10.
- `r580-migration-cli.c`: https://github.com/NVIDIA/cuda-checkpoint
- `vllm/config/model.py` (`enable_nccl_comm_suspend`).
