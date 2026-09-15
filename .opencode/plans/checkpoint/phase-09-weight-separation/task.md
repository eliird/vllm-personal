# Phase 9 — Separate process state from model weights

**ID:** `phase-09`
**Runs in:** WSL2 for design/host-side prototyping; bare-metal GPU for integration.
**Depends on:** `phase-08`
**Effort:** 3–5 days

## Objective

Avoid re-reading the model weights through the snapshot on every restore by
keeping weights out of the CRIU/`cuda-checkpoint` image, so process restore and
weight load run concurrently.

## Question

*Can the process + CUDA context be restored while weights are supplied from a
separate resident/loaded artifact, so the two paths run concurrently?*

## Background

If Phase 1 showed weight-load dominates cold start, this is where most of the
restore win lives — treat it as essential, not optional. Note the verified
constraints:

- `cuda-checkpoint` migrates **all** device allocations of the process, so
  weights must be held in a form it does not capture, or re-mapped after
  restore.
- vLLM's weight-cache IPC loader
  (`vllm/model_executor/model_loader/weight_cache/ipc_loader.py`) maps a daemon's
  tensors via CUDA IPC; in `zero_copy` mode weights live in the daemon's CUDA
  IPC allocations and sleep-mode weight offloading must not be used.
- Level-1 sleep offloads weights to host RAM (still in the image); level-2
  discards them (reloaded from source). Neither alone achieves weight
  separation — this phase is the explicit split.

## Prerequisites

- Phase 8 gate passed; a working restore path and its measured baseline.
- A fast weight source: resident host pool, memory-mapped weight file, or
  GPU-direct load.

## Steps

### 1. Define the split

```text
process snapshot: CPU process state + CUDA context + small buffers   (small, fast)
model artifact:   GPU weight tensors                                  (large)
```

Document which exact allocations go to each side and how the restored process
finds the weights.

### 2. Host-side prototype (WSL)

Prototype the weight-artifact path without a GPU: e.g. a memory-mapped,
pre-processed weight file and a loader that maps/streams it. Measure raw load
bandwidth. This validates the artifact format and the rendezvous, independent of
CUDA.

### 3. Concurrent restore (GPU)

Restore the process/context via CRIU while loading weights via the fastest
available path, then rendezvous to READY. Ensure the concurrency is real
(overlapping wall-clock), not sequential start then wait.

```bash
# measure: t(process restore), t(weight load), t(ready) and show overlap
```

### 4. Measure

Record ready-to-serve vs the Phase 8 best, cold start, and slept-snapshot
restore.

## Deliverables

- Weight-artifact format + loader scripts/notes (`scripts/p9_*`).
- `results/RESULTS.md` `## Phase 9` with ready-to-serve for concurrent vs all
  prior configurations, plus the size accounting (weights excluded from image).
- Evidence that process restore and weight load overlap.

## Evidence to capture

- Image size with and without weight separation.
- Wall-clock timeline showing concurrency.
- Correct inference after the concurrent restore.

## Constraints / Do NOT

- Do not rely on `cuda-checkpoint` capturing and separately excluding weights
  without proving it; verify what is in the image.
- Do not use sleep-mode weight offloading with the IPC loader in `zero_copy`.
- Do not commit anything.

## Definition of done

Concurrent process-restore + weight-load reaches correct inference, with
ready-to-serve recorded against all prior configurations and weights
demonstrably absent from the process image.

## References

- `../checkpoint.md` Phase 9.
- `vllm/model_executor/model_loader/weight_cache/ipc_loader.py`.
