# Phase 9 — Verification

**Verifier role:** independent. The two failure modes to catch are (1) weights
still present in the process image, and (2) "concurrent" restore that is
actually sequential.

## Gate being verified

Concurrent process-restore + weight-load reaches correct inference, with
ready-to-serve recorded against all prior configurations and weights absent from
the process image.

## Setup

- WSL2 for the host-side artifact check; bare-metal GPU for integration.
- Do not mutate `results/` except appending `results/verification/phase-09.md`.

## Artifact checks

1. `scripts/p9_*` exist and implement an artifact loader.
2. `results/RESULTS.md` `## Phase 9` records ready-to-serve for concurrent and
   all prior configurations, plus size accounting.
3. A timeline/log evidencing overlap exists.

## Independent reproduction

### R1 — Weights absent from the process image (critical)

Compare process-image size against the weight size:

- PASS: image size is on the order of process/context state, much smaller than
  the weights (report the ratio).
- FAIL if image size is ≈ weight size (weights still captured).

Also inspect image contents (e.g. `criu` page counts / `du`) and the weight
artifact separately.

### R2 — Concurrency is real

From the timeline logs, confirm the process-restore interval and weight-load
interval overlap in wall-clock time. Re-run once and reproduce the overlap.

- FAIL if there is no overlap (sequential restore then load).

### R3 — Correctness

Serve the fixed prompt set through the concurrently restored worker.

- PASS: correct output; no graph/illegal-address errors.
- FAIL otherwise.

### R4 — Ready-to-serve accounting

Recompute ready-to-serve for concurrent vs the recorded baselines (cold, Phase 8
best, slept restore). Confirm the concurrent number is not worse than Phase 8
best without explanation.

- FAIL if baselines are missing or the comparison is apples-to-oranges.

### R5 — Host artifact prototype

On WSL, verify the artifact loader maps/streams the weight file and reports a
plausible bandwidth.

## Numeric acceptance criteria

- Image/weight size ratio << 1 (weights excluded).
- Demonstrated wall-clock overlap > 0.
- Correct inference; N≥5 restore loops.
- Concurrent ready-to-serve ≤ Phase 8 best (or explained).

## Anti-gaming checks

- Verify the "concurrent" claim from timestamps, not prose.
- Verify the weight artifact is genuinely used (e.g. loader logs, mmap stats).
- Verify the process image was produced by CRIU with weights excluded, not by
  simply not snapshotting GPU state.

## Failure triage

- Weights in image: FAIL; separation not achieved.
- No overlap: FAIL; rename to sequential and require a real concurrent design.
- Correctness failure: FAIL; preserve logs.

## Verdict format

Write `results/verification/phase-09.md`:

```markdown
# Phase 9 verification — <date> — <host/GPU>
- R1 weights absent from image: PASS/FAIL — image=<x> weight=<y> ratio=<z>
- R2 concurrency real: PASS/FAIL — overlap=<x> ms
- R3 correctness: PASS/FAIL
- R4 ready-to-serve accounting: PASS/FAIL — concurrent=<x> baseline=<y>
- R5 host artifact prototype: PASS/FAIL
- Overall: PASS/FAIL
- Notes / human action required: <text>
```
