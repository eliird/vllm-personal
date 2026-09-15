# Phase 5 — Verification

**Verifier role:** independent. Exercise the CLI on a plain process yourself,
then inspect the GPU integration artifacts.

## Gate being verified

One command takes a warm worker to a valid snapshot on disk, and the Phase 4
restore path restores it to a correct response.

## Setup

- WSL2 for the CLI plumbing test; bare-metal GPU for integration verification.
- Do not mutate `results/` except appending `results/verification/phase-05.md`.

## Artifact checks

1. `scripts/snapshot-manager` exists, is executable, and supports
   `snapshot` and `status` subcommands.
2. `logs/p5_manager.log` contains one line per state transition with timestamps.
3. `results/RESULTS.md` `## Phase 5` has both the WSL CLI result and the GPU
   integration result.

## Independent reproduction

### R1 — CLI on a plain process (WSL)

Start a Phase 2 probe, run the manager against it, then restore with CRIU and
verify the counter continued.

- PASS: manager quiesces/no-ops, dumps, and the restored process continues.
- FAIL if the CLI errors on a non-CUDA process (plumbing is not isolated).

### R2 — State-machine integrity

From `logs/p5_manager.log`, confirm the transition order is exactly
`SERVING → QUIESCING → DISCARD_KV → SNAPSHOT_READY` before the dump, with no
skipped or out-of-order states.

- FAIL on missing/ordered transitions.

### R3 — Snapshot validity

Confirm the output directory contains a complete CRIU image (inventory/pages
files) and is non-empty.

- FAIL if the dump is a stub or empty.

### R4 — GPU integration (target model)

Run the manager, restore via the Phase 4 path, and query the API.

- PASS: restore succeeds and the response is correct.
- FAIL if restore requires a cold start or the response is wrong.

### R5 — Failure recovery

Run a snapshot where the dump is forced to fail (e.g. invalid images dir) and
confirm the manager leaves the worker recoverable (resumes if it had suspended),
not stuck locked.

## Numeric acceptance criteria

- R1 counter continues after restore.
- R3 image size > 0 and contains the expected CRIU inventory.
- R4 correct response.
- R5 worker still serving after a failed snapshot.

## Anti-gaming checks

- Verify the manager does not restart vLLM: grep the code for `vllm serve`;
  the restore path must be CRIU-based.
- Verify the transition log is emitted by the manager at runtime, not
  hand-written.

## Failure triage

- CLI works only on GPU and fails on plain process: FAIL (plumbing).
- Failure recovery leaves the worker locked: FAIL; record the state and fix.

## Verdict format

Write `results/verification/phase-05.md`:

```markdown
# Phase 5 verification — <date> — <host/GPU>
- R1 CLI on plain process: PASS/FAIL
- R2 state-machine integrity: PASS/FAIL — <transitions>
- R3 snapshot validity: PASS/FAIL — size=<x>
- R4 GPU integration: PASS/FAIL
- R5 failure recovery: PASS/FAIL
- Overall: PASS/FAIL
- Notes / human action required: <text>
```
