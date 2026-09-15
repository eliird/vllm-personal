# Phase 11 — Verification

**Verifier role:** independent. Exercise the guard adversarially; a guard that
only checks the driver field or always passes is a failure.

## Gate being verified

The manager creates/restores/lists/deletes snapshots, and a deliberately
mismatched (e.g. driver) restore is refused rather than attempted.

## Setup

- WSL2 is sufficient for the metadata/guard logic; a GPU host is needed only to
  confirm a matching restore succeeds.
- Do not mutate `results/` except appending `results/verification/phase-11.md`.

## Artifact checks

1. `scripts/snapshot-manager` supports `create`, `list`, `restore`, `delete`,
   `status`.
2. Each snapshot has a machine-readable metadata file with all schema fields.
3. `results/RESULTS.md` `## Phase 11` records guard tests per field.

## Independent reproduction

### R1 — Lifecycle

Create, list, status, restore (matching), and delete a snapshot. Confirm each
subcommand works and returns sensible exit codes.

- FAIL if any subcommand is a stub.

### R2 — Guard matrix (adversarial)

For each field in {driver_version, gpu, gpu_count, tensor_parallel,
expert_parallel, vllm_version, cuda_version}, mutate the metadata value and
attempt a restore.

- PASS: every mutated field causes a refusal with a nonzero exit and a message
  naming the field.
- FAIL if any field is unchecked or the restore proceeds.

### R3 — No side effects on refusal

On a refused restore, confirm the manager does **not** begin a dump/restore or
leave the worker altered.

### R4 — Matching restore on GPU

On a GPU host with matching metadata, confirm create → restore produces correct
inference.

### R5 — Documentation

Confirm the tool's help/docs state snapshots are ephemeral across driver
upgrades.

## Numeric acceptance criteria

- All seven fields guarded.
- Refusal exit code nonzero; matching restore exit code zero.
- No partial restore on refusal.

## Anti-gaming checks

- Cross-check metadata fields are populated from `results/stack.txt`, not
  hardcoded constants.
- Verify the guard compares against the **live** host at restore time, not
  against stored metadata only.
- Verify delete actually removes the snapshot (and list no longer shows it).

## Failure triage

- Any unguarded field: FAIL.
- Guard passes but restore still proceeds: FAIL (side-effect check).
- Matching restore fails on GPU: FAIL.

## Verdict format

Write `results/verification/phase-11.md`:

```markdown
# Phase 11 verification — <date> — <host/GPU>
- R1 lifecycle: PASS/FAIL
- R2 guard matrix: PASS/FAIL — unguarded fields=<list>
- R3 no side effects on refusal: PASS/FAIL
- R4 matching restore (GPU): PASS/FAIL
- R5 documentation: PASS/FAIL
- Overall: PASS/FAIL
- Notes / human action required: <text>
```
