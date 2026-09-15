# Phase 5 — Quiesce lifecycle + thin external snapshot manager

**ID:** `phase-05`
**Runs in:** WSL2 for the CLI/state-machine tests; bare-metal GPU for integration.
**Depends on:** `phase-02`, `phase-04`
**Effort:** 1 day

## Objective

Give the worker a clean snapshot-ready state, driven by a **thin external**
manager. Do not add a large snapshot feature to vLLM.

## Question

*Can an external manager reliably drive the worker into a checkpointable state
and snapshot it?*

## Background

The manager drives a state machine via signals and vLLM's **existing** APIs
(not a fork). The only in-process hook relied on is vLLM's sleep/discard API
(Phase 6/7). Prefer the `SleepModeBackend` plugin registration path over
external signal hacking where possible; the external CLI can still orchestrate
`cuda-checkpoint` + `criu dump`.

```
SERVING → (snapshot request) → QUIESCING → (finish active requests)
        → DISCARD_KV → SNAPSHOT_READY → [cuda-checkpoint + criu dump]
```

## Prerequisites

- Phase 2 (CRIU) and Phase 4 (vLLM snapshot correctness) gates passed.
- `criu`, `cuda-checkpoint`, and (for GPU runs) the plugin available.
- `.venv/bin/python` for the CLI on the GPU host.

## Steps

### 1. Build the thin CLI

`scripts/snapshot-manager` (Python, run via `.venv/bin/python`):

```
snapshot-manager snapshot --pid <vllm-pid> --output snapshots/<name> [--mode manual|plugin]
snapshot-manager status   --pid <vllm-pid>
```

Internally: quiesce (finish in-flight requests) → discard KV (Phase 6 API) →
`cuda-checkpoint` → `criu dump`. Keep it external; do not modify vLLM core.

Implementation notes:

- Discover the vLLM process tree (engine-core + workers) before dumping; dump
  the tree with `--shell-job`.
- Emit a machine-readable transition log (one line per state) to
  `logs/p5_manager.log`.
- Make the cuda-checkpoint/CRIU steps a `MODE` switch so the same CLI works for
  manual and plugin integration.
- On failure, leave the worker in a documented, recoverable state (unlock/resume
  if suspend had started).

### 2. WSL CLI validation (no GPU)

Validate the state machine and CLI plumbing against a **plain CRIU process**
from Phase 2: the CLI should quiesce (no-op), snapshot via `criu dump`, and
produce a restorable image. This isolates CLI bugs from CUDA issues.

### 3. GPU integration

Run the CLI against a warm vLLM worker, then restore via the Phase 4 restore
path and confirm one correct inference.

```bash
.venv/bin/python scripts/snapshot-manager snapshot --pid "$VLLM_PID" --output snapshots/p5_small
# then restore with scripts/p4_restore_vllm.sh and query the API
```

## Deliverables

- `scripts/snapshot-manager`.
- `results/RESULTS.md` `## Phase 5` with CLI validation result and GPU
  integration result.
- `logs/p5_manager.log`.

## Evidence to capture

- Transition log showing each state with timestamps.
- Snapshot image listing and size.
- Correct response after restore.

## Constraints / Do NOT

- Do not add a `--snapshot` feature to vLLM core; external orchestration plus
  the existing sleep/discard API only.
- Do not begin Phase 6 allocator changes here.
- Do not commit anything.

## Definition of done

One command takes a warm worker to a valid snapshot on disk, and the Phase 4
restore path restores it to a correct response. The same CLI works on a plain
CRIU process for plumbing tests.

## References

- `../checkpoint.md` Phase 5.
- `vllm/entrypoints/serve/dev/sleep/api_router.py`, `vllm/device_allocator/sleep_mode_backend.py`.
