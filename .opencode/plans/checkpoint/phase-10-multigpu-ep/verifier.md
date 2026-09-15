# Phase 10 — Verification

**Verifier role:** independent. Verify each targeted GPU count; do not accept a
single-count result.

## Gate being verified

Snapshot/restore yields correct inference at each GPU count targeted, with the
NCCL/IPC handling documented.

## Setup

- Bare-metal multi-GPU host, same GPU count(s) targeted by the task.
- Do not mutate `results/` except appending `results/verification/phase-10.md`.

## Artifact checks

1. `scripts/p10_*` exists and is parameterized by GPU count.
2. `results/RESULTS.md` `## Phase 10` has a row per targeted count with
   pass/fail, sizes/times, and blockers+fixes.
3. Communicator/IPC notes exist.

## Independent reproduction

### R1 — GPU count and topology

For each targeted count, confirm the launch actually used that many GPUs
(`nvidia-smi` / process GPU affinity), and that the process tree matches the
expected engine-core + workers topology.

- FAIL if the count is mislabeled or GPUs are not actually used.

### R2 — Round-trip per count

Run snapshot → restore → inference at each targeted count (at least the largest,
plus one other).

- PASS: correct inference, no NCCL/IPC errors.
- FAIL on any count.

### R3 — Device enumeration

Confirm all CUDA-visible GPUs are enumerated for the dump/restore. For any
migration attempt, confirm `--device-map` is used and correct; for same-GPU-set
restore, confirm GPU UUIDs/count match.

- FAIL if a subset of GPUs is enumerated while migration is claimed.

### R4 — NCCL/IPC claims

For each blocker the task claims fixed, confirm the fix appears in the scripts
(e.g. `enable_nccl_comm_suspend`, comm re-init) and that inference works with it.

### R5 — Same-count restore constraint

Attempt (or confirm documented) restore on a different GPU count; it should
fail or be refused, matching the tooling's constraint.

- PASS if the constraint is observed and documented.

## Numeric acceptance criteria

- Correct inference at every targeted count (N≥5 loops at largest).
- Recorded snapshot size/time per count.
- No unhandled NCCL/IPC errors.

## Anti-gaming checks

- Verify actual GPU usage per count via `nvidia-smi`, not config args alone.
- Verify blocker fixes are real (present in scripts, inference passes with them).
- Verify restore is CRIU-based, not a cold start with TP.

## Failure triage

- Any count fails: FAIL for that count; record the NCCL/IPC error.
- Migration claimed without `--device-map`: FAIL.

## Verdict format

Write `results/verification/phase-10.md`:

```markdown
# Phase 10 verification — <date> — <GPUs/topology>
- R1 counts/topology: PASS/FAIL — <per count>
- R2 round-trip per count: PASS/FAIL — <count=status>
- R3 device enumeration: PASS/FAIL
- R4 NCCL/IPC fixes: PASS/FAIL
- R5 same-count constraint: PASS/FAIL
- Overall: PASS/FAIL
- Notes / human action required: <text>
```
