# Phase 3 — Verification

**Verifier role:** independent. Re-run the largest size yourself; recompute the
scaling from raw logs.

## Gate being verified

A CUDA process is checkpointed and restored without reinitializing GPU state at
a device-memory size comparable to a real model, with a linear-ish time-vs-size
curve recorded.

## Setup

- Bare-metal GPU, same driver/CRIU/plugin as the task. `sudo` for CRIU.
- Do not mutate `results/` except appending `results/verification/phase-03.md`.

## Artifact checks

1. `scripts/p3_cuda_sweep.cu` and `scripts/p3_cuda_bench.sh` exist and are
   non-trivial (size CLI arg, `MODE` switch).
2. `results/RESULTS.md` `## Phase 3` table has ≥3 sizes with all columns filled.
3. Raw logs `logs/p3_*.log` exist.

## Independent reproduction

### R1 — Environment guard

Confirm MIG `Disabled`, no MPS, driver r555+, and that sizes do not exceed ~3/4
of HBM.

### R2 — Re-run the largest size

Run the bench at the largest swept size (fresh temp dir) in the same mode.

- PASS: `after_counter > before_counter`, `after_checksum == before_checksum`,
  restore succeeds.
- FAIL on mismatch/reset/error.

### R3 — Host-RAM copy behavior

Confirm host RSS peak during dump is on the order of the device allocation
(the documented device→host copy). A host peak far below the device size
suggests the memory was not actually resident — check with `nvidia-smi`.

- FAIL if device memory was not actually allocated/resident.

### R4 — Scaling recomputation

From raw logs, recompute checkpoint/restore time vs size. Fit and report the
implied bandwidth. Confirm the task's "linear-ish" claim.

- PASS if R² of a linear fit is reasonable (≥ ~0.9) **or** the task explained a
  non-linear regime.
- FAIL if the reported curve cannot be reproduced from logs.

### R5 — Interpolation gap

Run one intermediate, non-reported size and check its time falls between the
neighboring reported points (monotonic trend).

- FAIL if timings are non-monotonic.

## Numeric acceptance criteria

- Counter continues and checksum stable at the largest size.
- Device memory allocated equals the requested size (within MiB, via
  `nvidia-smi`).
- Host RSS peak within ~2× of the device size.
- Linear-fit R² ≥ ~0.9 (or explained).

## Anti-gaming checks

- No negative control is needed here (Phase 0b covered continuation), but verify
  the largest run actually used the GPU by checking device memory via
  `nvidia-smi` during the run.
- Cross-check `results/stack.txt` HBM total to ensure sweep sizes are within
  bounds.

## Failure triage

- Restore/state failure at large size: FAIL; escalate (driver/plugin). Record
  size and error.
- Non-monotonic or non-reproducible timings: FAIL; likely measurement error
  (background load, thermal). Require a quiet host.

## Verdict format

Write `results/verification/phase-03.md`:

```markdown
# Phase 3 verification — <date> — <GPU/driver>
- R1 environment guard: PASS/FAIL
- R2 largest-size round-trip: PASS/FAIL — before=<ctr sum> after=<ctr sum>
- R3 host-RAM copy: PASS/FAIL — RSS peak=<x> MiB vs device=<y> MiB
- R4 scaling: PASS/FAIL — fit R2=<x>, implied BW=<x>
- R5 interpolation monotonic: PASS/FAIL
- Overall: PASS/FAIL
- Notes / human action required: <text>
```
