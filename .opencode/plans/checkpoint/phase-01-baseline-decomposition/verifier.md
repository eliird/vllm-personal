# Phase 1 — Verification

**Verifier role:** independent. Recompute the decomposition from the raw logs;
do not trust the table.

## Gate being verified

The weight-load portion and the init+graph+compile portion are populated with
real numbers, and `results/RESULTS.md` states which dominates.

## Setup

- GPU host with the same model/vLLM install. You may reuse the task's logs for
  recomputation, but you must run at least one independent cold start to
  confirm reproducibility.
- Do not mutate `results/` except appending `results/verification/phase-01.md`.

## Artifact checks

1. `results/RESULTS.md` `## Phase 1` table exists for both surrogate and target.
2. `logs/p1_vllm.log`, `logs/p1_first_response.json`, `logs/p1_resources.log`
   exist and are non-empty.
3. `scripts/p1_baseline.sh` exists and records T0 and readiness.

## Independent reproduction

### R1 — Timestamp integrity

From `logs/p1_vllm.log`, find lines supporting each boundary and confirm:

- `T0 <= T1 <= T2 <= T3 <= T4 <= T5 <= T6` (monotonic, no negatives).
- The weight-load window `T3 - T2` overlaps actual storage read activity
  (evidence: log lines about loading weights, or reduced I/O wait afterward).

- FAIL if boundaries are missing, non-monotonic, or if `T2 < T1` (ordering
  error — context init must precede weight load).

### R2 — Sum consistency

Sum the sub-intervals and confirm they equal `T6 - T0` within ~5% (instrument
gaps/overlap are acceptable if explained).

- FAIL if the sum diverges from the total by more than the stated tolerance
  without explanation.

### R3 — Weight-load plausibility

Cross-check `weight-load bytes` against achievable storage bandwidth:

```
bytes = on-disk model size (from stack.txt)
implied_BW = bytes / (T3 - T2)
```

- PASS if `implied_BW` is within a plausible range for the storage device
  (record the number). A value above device link speed is a FAIL.
- Warm page cache can inflate bandwidth; note whether the run was cold-cache
  (recommended: drop caches or read from a cold file).

### R4 — Independent cold start

Run `scripts/p1_baseline.sh` once yourself. Confirm total startup and the
dominant portion direction match the task within ~20%.

- FAIL if the dominant portion flips (weight-load vs init) between runs and the
  task did not report both.

### R5 — First response is real

Open `logs/p1_first_response.json`; confirm it is valid JSON with non-empty
generated text and that the elapsed time matches the reported time-to-first.

- FAIL if the response is an error payload or empty.

## Numeric acceptance criteria

- All boundaries present and monotonic.
- Sum vs total within ~5% (or explained).
- Implied weight-load bandwidth <= storage link speed.
- Independent run agrees on the dominant portion.

## Anti-gaming checks

- Verify the model on-disk size in `results/stack.txt` matches `du -sh` of the
  model directory.
- Verify `T2` is anchored to a real "loading weights" log event, not a guess.
- Verify the dominance conclusion is a sentence in `RESULTS.md`, not just an
  unfilled table.

## Failure triage

- Non-monotonic or missing boundaries: FAIL; the decomposition is unusable.
- Implausible bandwidth: FAIL; likely warm cache or mis-attributed boundary.
- If the model does not fit the GPU, FAIL and require a correctly sized
  surrogate; do not accept offloaded/CPU weight-load numbers as the target
  baseline.

## Verdict format

Write `results/verification/phase-01.md`:

```markdown
# Phase 1 verification — <date> — <GPU/model>
- R1 timestamp integrity: PASS/FAIL
- R2 sum consistency: PASS/FAIL — sum=<x> total=<y>
- R3 weight-load plausibility: PASS/FAIL — implied_BW=<x>
- R4 independent cold start: PASS/FAIL
- R5 first response real: PASS/FAIL
- Dominant portion (task / verifier): <x> / <y>
- Overall: PASS/FAIL
- Notes / human action required: <text>
```
