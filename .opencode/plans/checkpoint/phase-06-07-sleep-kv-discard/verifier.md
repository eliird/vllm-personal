# Phase 6+7 — Verification

**Verifier role:** independent. Confirm the sleep levels actually behave as the
task claims; the common failure is mis-reporting what each level discards.

## Gate being verified

A slept worker snapshots and restores to correct inference, the size/time
reduction from discarding memory is quantified per level, and a conclusion is
recorded on whether any vLLM allocator change is needed.

## Setup

- Bare-metal GPU, same stack. Do not mutate `results/` except appending
  `results/verification/phase-06-07.md`.

## Artifact checks

1. `scripts/p67_slept_snapshot.sh` (or documented equivalent) exists.
2. `results/RESULTS.md` `## Phase 6+7` table has rows for awake, level 1, level 2
   with size/time/GPU memory.
3. Sleep/wake GPU-memory logs exist.

## Independent reproduction

### R1 — Sleep level semantics (critical)

For each level, capture GPU memory before/after sleep and the resulting snapshot
size:

- **level 1:** GPU memory drops by ~weights+KV; snapshot size should still be on
  the order of the **weight size** (weights offloaded to host RAM, still in the
  image). FAIL if the task claims level 1 eliminates weight bytes.
- **level 2:** snapshot size should be much smaller; resume should reload
  weights from the model source (evidence in logs). FAIL if no reload evidence.

### R2 — Wake correctness

After `wake_up()` with no checkpoint, serve a request (fixed prompt set).

- PASS: correct output, no illegal-address/graph-replay errors.
- FAIL otherwise.

### R3 — Slept-worker round-trip

Snapshot a level-1 (and level-2) slept worker, restore, wake, and serve.

- PASS: correct output; graphs survive.
- FAIL on graph-replay/illegal-address errors or wrong output.

### R4 — Comparison table recomputation

Recompute the table from raw image sizes and logs; verify `awake >= level1 >
level2` in snapshot size (level1 bounded below by weight size).

- FAIL if ordering is inconsistent or numbers cannot be reconstructed.

### R5 — Allocator-change decision

Confirm the recorded conclusion is justified: allocator changes were considered
**only** because step 2 failed, and if step 2 succeeded, no allocator change was
made. Any allocator change present in the repo without a failing result is a
FAIL.

## Numeric acceptance criteria

- Level-1 image size ≥ ~weight size (within tolerance for compression).
- Level-2 image size << level-1.
- Wake and slept-restore outputs correct; N≥5 restore loops for the slept case.
- GPU memory freed per level matches expectation (~weights+KV for L1).

## Anti-gaming checks

- Verify image sizes via `du -sh`, not the task's prose.
- Verify level-2 resume actually reloads weights (log evidence) rather than
  silently keeping them.
- Verify pinning: level-1 host RSS should reflect the pinned weight backup
  (check host RSS during sleep).

## Failure triage

- Graph errors after woke restore: FAIL; preserve logs, escalate.
- Level-1 image tiny (weights missing from image): FAIL — likely weights were
  discarded unexpectedly; verify with the weight-cache loader interaction.

## Verdict format

Write `results/verification/phase-06-07.md`:

```markdown
# Phase 6+7 verification — <date> — <GPU/model>
- R1 sleep level semantics: PASS/FAIL — L1 image=<x> L2 image=<y> weight=<w>
- R2 wake correctness: PASS/FAIL
- R3 slept round-trip: PASS/FAIL — <n>/<N> restores
- R4 table recomputation: PASS/FAIL
- R5 allocator-change decision: PASS/FAIL
- Overall: PASS/FAIL
- Notes / human action required: <text>
```
