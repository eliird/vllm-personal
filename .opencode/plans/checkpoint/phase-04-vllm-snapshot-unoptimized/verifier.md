# Phase 4 — Verification

**Verifier role:** independent. Reproduce the restore and run the correctness
harness. Do not accept a single correct response as proof.

## Gate being verified

`start → warm → checkpoint → terminate → restore → inference` yields a correct
response on both the small model and the target model, and every blocker and
its fix is logged.

## Setup

- Bare-metal GPU, same stack as the task.
- Do not mutate `results/` except appending `results/verification/phase-04.md`.

## Artifact checks

1. `scripts/p4_snapshot_vllm.sh`, `scripts/p4_restore_vllm.sh` exist.
2. `results/RESULTS.md` `## Phase 4` has a blocker→fix table and pass/fail rows
   for both models.
3. Logs for topology/FDs and snapshot exist.

## Independent reproduction

### R1 — Blocker log completeness

For each blocker the task claims fixed, confirm the fix is actually present in
the scripts (e.g. `--tcp-established`, ignore hooks, comm suspend). A claimed
fix absent from the scripts is a FAIL.

### R2 — One clean round-trip (target model)

Run the snapshot then restore scripts yourself with a fresh snapshot dir.
Confirm:

- restore succeeds and the API responds,
- the process tree after restore matches before (engine-core + workers),
- no evidence of a fresh model load in the restore logs (the whole point is to
  skip it).

- FAIL if the restored server reloads weights from disk (check logs for model
  loading) or the topology differs.

### R3 — Correctness harness (applies from Phase 4 onward)

Run all of the following against the restored worker and a cold-start worker:

1. **Fixed prompt/seed set** — at least 8 prompts; compare restored vs cold
   output with a tolerance (MoE kernels are not bit-exact; use logit/token
   tolerance and report the method).
2. **Repeated restores** — restore N≥5 times in a loop; confirm every restore
   yields a correct response (catches nondeterministic corruption).
3. **Soak** — serve ≥100 requests through a restored worker with varying
   prompt lengths to exercise fresh KV allocation after wake.

- FAIL if any restore produces wrong/empty output, a crash, or an illegal-address
  / graph-replay error.

### R4 — GPU state actually preserved

Confirm the restored worker's GPU memory footprint matches the pre-snapshot
footprint (via `nvidia-smi`), and that no weight re-load occurred.

## Numeric acceptance criteria

- R2: restore succeeds; topology match.
- R3: N≥5 restore loops all correct; soak ≥100 requests all HTTP-successful and
  non-empty; restored-vs-cold agreement within the stated tolerance.
- R4: GPU memory within ~10% of pre-snapshot.

## Anti-gaming checks

- Verify the "restore" script is not secretly doing a cold start: grep for model
  loading / `vllm serve` in the restore path, and check the restore logs.
- Verify the PID restoration: with `--shell-job` CRIU restores the original PID;
  if the script uses a different PID for `cuda-checkpoint` resume, FAIL.
- Verify prompts are fixed and recorded, not cherry-picked after the fact.

## Failure triage

- Any intermittent corruption across N restores: FAIL; this is the exact class
  of bug this gate exists to catch. Record the failing prompt and seed.
- Illegal-address/graph-replay errors: FAIL; preserve full logs and stop for
  human review.
- Blocker with no fix: FAIL; do not partial-pass.

## Verdict format

Write `results/verification/phase-04.md`:

```markdown
# Phase 4 verification — <date> — <GPU/model>
- R1 blocker fixes present: PASS/FAIL — <missing items>
- R2 clean round-trip: PASS/FAIL — weights reloaded? <yes/no>
- R3a fixed prompt set: PASS/FAIL — agreement=<x>
- R3b repeated restores (N): PASS/FAIL — <a>/<N>
- R3c soak (M requests): PASS/FAIL — <a>/<M>
- R4 GPU state preserved: PASS/FAIL
- Overall: PASS/FAIL
- Notes / human action required: <text>
```
