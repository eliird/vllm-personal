# Phase 2 — Verification

**Verifier role:** independent. Re-run at least one sweep size yourself with a
fresh probe; do not trust the summary table.

## Gate being verified

A plain Linux process checkpoints and restores with state preserved across a
memory sweep, and dump/restore time and image size are recorded for each size.

## Setup

- CPU-only host (WSL2 fine). `sudo` for CRIU. Local ext4 for images.
- Do not mutate `results/` except appending `results/verification/phase-02.md`.

## Artifact checks

1. `scripts/p2_ram.c` and `scripts/p2_criu_bench.sh` exist and are non-trivial.
2. `results/RESULTS.md` `## Phase 2` table has a row per swept size with dump
   time, restore time, image size, and before/after state.
3. Notes on FD/socket/shm behavior are present.

## Independent reproduction

### R1 — Probe correctness

Compile `scripts/p2_ram.c` and run it briefly. Confirm:

- the checksum is stable across ticks,
- the counter strictly increases,
- the reported memory was actually allocated (cross-check RSS with `ps` or
  `/proc/<pid>/status` VmRSS ≈ requested size).

- FAIL if the checksum is unstable or RSS is far from the requested size.

### R2 — Re-run one size (largest swept)

```bash
SIZES=<largest_swept_MB> bash scripts/p2_criu_bench.sh
```

- PASS: `after_counter > before_counter` and `after_checksum == before_checksum`.
- FAIL on any reset/mismatch or CRIU error.

### R3 — Image size plausibility

- Confirm image size for each size is at least a meaningful fraction of the
  resident memory (CRIU may compress; note if image << memory and verify the
  pages were not sparse). A tiny image for a large touched region is a FAIL
  unless a clear reason (e.g. zero pages) is documented.

### R4 — No swap during dump

Repeat R2 while watching `free`/`vmstat`. Confirm the largest size did not push
the host into swap; if it did, mark the affected row invalid.

## Numeric acceptance criteria

- State preserved at every swept size.
- Each row's `restore_time > 0` and image size > 0.
- Largest size does not swap.

## Anti-gaming checks

- Probe memory must be genuinely resident (R1 RSS check), not a sparse mmap.
- The counter must be embedded in the large buffer (grep the source), so
  continuation proves the region survived, not just an external file.

## Failure triage

- Restore failure or state loss: FAIL; capture the CRIU error and host free
  memory.
- Swapping observed: mark the row invalid and require a smaller sweep.

## Verdict format

Write `results/verification/phase-02.md`:

```markdown
# Phase 2 verification — <date> — <host>
- R1 probe correctness: PASS/FAIL — RSS=<x> MiB for request=<y> MiB
- R2 re-run largest size: PASS/FAIL — before=<ctr sum> after=<ctr sum>
- R3 image size plausibility: PASS/FAIL
- R4 no swap: PASS/FAIL
- Overall: PASS/FAIL
- Notes / human action required: <text>
```
