# Phase 11 — Production-style snapshot manager

**ID:** `phase-11`
**Runs in:** WSL2-testable (metadata/versioning logic); bare-metal GPU for real create/restore.
**Depends on:** `phase-05`, `phase-09`
**Effort:** 1–2 days

## Objective

Package the proven mechanism into a usable tool with a compatibility guard, so
users can create/list/restore/delete snapshots safely and incompatible restores
are refused.

## Question

*Can a user create/list/restore/delete snapshots safely, with incompatible
restores refused?*

## Background

Snapshots are **ephemeral across driver upgrades** — a driver update invalidates
GPU checkpoints. The manager must hard-fail on mismatch rather than attempt a
restore. Build on the Phase 5 CLI.

## Prerequisites

- Phase 5 CLI and Phase 9 restore path available.
- Phase 5 gate passed.

## Steps

### 1. Extend the CLI

```
snapshot-manager create <name>
snapshot-manager list
snapshot-manager restore <name>
snapshot-manager delete <name>
snapshot-manager status
```

### 2. Metadata + hard-fail guard

Write a metadata file per snapshot and **hard-fail** on mismatch at restore:

```yaml
model: <name>
vllm_version: ...
cuda_version: ...
driver_version: ...      # MOST fragile — hard-fail on mismatch
gpu: <model>
gpu_count: ...
tensor_parallel: ...
expert_parallel: ...
snapshot_version: ...
created_at: ...
```

Populate fields from `results/stack.txt` at create time and compare against the
live host at restore time. On any mismatch, refuse with a clear message and a
nonzero exit code. State plainly in the help/docs that snapshots are ephemeral
across driver upgrades.

### 3. Test the guard (WSL)

The metadata/compare logic is testable without a GPU: create a snapshot
directory with fake metadata, mutate a field, and confirm restore is refused.
Test each field (driver, gpu_count, tensor_parallel, vllm_version).

## Deliverables

- Extended `scripts/snapshot-manager` with the five subcommands.
- Metadata schema + guard, plus a short usage doc.
- `results/RESULTS.md` `## Phase 11` with guard test results per field.

## Evidence to capture

- `list`/`status` output.
- Refusal messages for each mismatched field (with exit codes).
- A successful create/restore on matching metadata (GPU host).

## Constraints / Do NOT

- Do not attempt a restore on driver mismatch; refuse.
- Do not store metadata only in prose; use a machine-readable file.
- Do not commit anything.

## Definition of done

The manager creates/restores/lists/deletes snapshots, and a deliberately
mismatched (e.g. driver) restore is refused rather than attempted.

## References

- `../checkpoint.md` Phase 11.
- Phase 5 manager and Phase 9 restore path.
