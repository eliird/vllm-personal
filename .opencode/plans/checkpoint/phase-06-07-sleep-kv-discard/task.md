# Phase 6+7 — KV discard via vLLM sleep, and the VMM/VA question

**ID:** `phase-06-07`
**Runs in:** bare-metal GPU.
**Depends on:** `phase-04`, `phase-05`
**Effort:** 1–2 days

## Objective

Use vLLM's **existing** sleep mode to discard memory before snapshot, and test
whether it composes with `cuda-checkpoint` + CRIU. Merged with the VMM/VA
question because sleep already preserves virtual addresses.

## Question

*Does vLLM's existing sleep mode compose cleanly with `cuda-checkpoint` + CRIU,
and how much does discarding memory shrink the snapshot — at each level?*

## Background (read carefully — corrects common assumptions)

vLLM's sleep mode is built on `CuMemAllocator` using CUDA VMM to free physical
pages while keeping virtual addresses, so CUDA graphs are not invalidated. The
open question is therefore not "can we preserve VA" (it exists) but "does a
*slept* worker checkpoint/restore correctly, and does waking reallocate cleanly."

**Verified sleep semantics** (`vllm/device_allocator/sleep_mode_backend.py`,
`vllm/device_allocator/cumem.py`):

- `sleep(level=1)` calls `allocator.sleep(offload_tags=("weights",))`:
  **weights are offloaded to pinned host RAM**, and **everything else (KV cache)
  is discarded**. Weights do **not** stay on GPU and are not freed from host.
- `sleep(level=2)` discards weights and KV with **no** CPU backup; weights are
  reloaded from the model source on resume.

Consequence: a level-1 slept snapshot still carries the **weight bytes** — now
written to the CRIU image from host RAM instead of from device dump. Only the
KV portion shrinks. So the honest comparison is **awake vs level 1 vs level 2**,
and the snapshot-size question must be answered for each. Level 1 pins roughly
the weight size in host RAM (`pin_memory=True`), which can exceed host limits.

## Prerequisites

- Phase 4 and 5 gates passed.
- vLLM sleep mode available; `ModelConfig.sleep_mode_backend` default `cumem`.
- Note: if the weight-cache IPC loader is in use (`zero_copy`), sleep-mode weight
  offloading must not be used (see `ipc_loader.py`).

## Steps

### 1. Sleep/wake alone (no CRIU)

Warm the worker, run `sleep(level=1)`, confirm KV/physical pages are freed and
VAs reserved, then `wake_up()` and serve. Record GPU memory before/after.

```bash
# via the /sleep and /wake_up endpoints or the engine API
curl -s -X POST http://127.0.0.1:8000/sleep?level=1
nvidia-smi --query-gpu=memory.used --format=csv
curl -s -X POST http://127.0.0.1:8000/wake_up
```

Repeat for `level=2` and record whether resume reloads weights.

### 2. Snapshot a slept worker

warm → sleep → `cuda-checkpoint` + `criu dump`; restore → wake → serve. Verify
graphs survive (no illegal-address / graph-replay errors) and outputs are
correct.

```bash
# MODE=manual|plugin, LEVEL=1|2
scripts/p5... snapshot with a pre-step that calls sleep, or add a small
scripts/p67_slept_snapshot.sh that wraps sleep + the manager.
```

### 3. Compare

Fill `results/RESULTS.md`:

```markdown
| Config | Snapshot size | Checkpoint s | Restore s | GPU mem captured | Weight bytes in image? |
| --- | ---: | ---: | ---: | ---: | --- |
| Awake (warm) | | | | | yes |
| sleep(level=1) | | | | | yes (host RAM) |
| sleep(level=2) | | | | | no (reloaded) |
```

### 4. Allocator changes only if needed

Only if sleep does **not** cleanly release/reallocate under checkpoint should
you consider modifying vLLM's allocator. Treat that as a separate, gated
experiment; do not begin it unless step 2 fails.

## Deliverables

- `scripts/p67_slept_snapshot.sh` (or documented manager integration).
- `results/RESULTS.md` `## Phase 6+7` with the comparison table and a conclusion
  on whether any vLLM allocator change is needed.
- Sleep/wake GPU-memory logs.

## Evidence to capture

- GPU memory before/after sleep per level.
- Whether weight bytes appear in the image (image size vs weight size).
- Correct responses after wake/restore; any graph errors verbatim.

## Constraints / Do NOT

- Do not modify vLLM's allocator unless step 2 fails.
- Do not use sleep with the IPC weight-cache loader in `zero_copy` mode.
- Do not commit anything.

## Definition of done

A slept worker snapshots and restores to correct inference; the size/time effect
of KV discard is quantified per level; a recorded conclusion states whether any
vLLM allocator change is needed.

## References

- `../checkpoint.md` Phases 6 and 7 (with the level-1/level-2 correction above).
- `vllm/device_allocator/sleep_mode_backend.py`, `vllm/device_allocator/cumem.py`,
  `vllm/model_executor/model_loader/weight_cache/ipc_loader.py`.
