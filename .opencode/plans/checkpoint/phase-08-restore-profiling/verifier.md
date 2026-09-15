# Phase 8 — Verification

**Verifier role:** independent. Recompute the dominant-cost claim from raw logs
and re-measure at least one intervention.

## Gate being verified

`results/restore_profile.md` names the dominant cost with numbers, and at least
one intervention (lazy-pages and/or parallel I/O) shows a measured
restore-time reduction. Any CRIU modification is justified by the profile.

## Setup

- WSL2 for host/storage verification; GPU host for device-copy numbers.
- Do not mutate `results/` except appending `results/verification/phase-08.md`.

## Artifact checks

1. `results/restore_profile.md` exists and contains: dominant cost, the
   a/b split (host vs device), and A/B results for at least one intervention.
2. `logs/p8_*.log` exist.
3. `scripts/p8_profile_restore.sh` exists and performs the measurements.

## Independent reproduction

### R1 — Bytes-read plausibility

From the profile and image listing, confirm total bytes read on restore ≈ image
size (within compression/overhead). Compute achieved bandwidth and compare to
the storage device link speed.

- FAIL if bytes read is inconsistent with image size.

### R2 — Dominant cost recomputation

Recompute host-restore vs device-copy from the raw timing deltas. Confirm the
named dominant cost matches within ~20%.

- FAIL if the conclusion is not supported by the numbers.

### R3 — Independent intervention measurement

Re-run the chosen intervention (lazy-pages on WSL, and/or parallel I/O) with a
fresh restore, at least 3 times each, and report medians.

- PASS: median time-to-first-response improves by a reproducible margin.
- FAIL if improvement is within noise or non-reproducible.

### R4 — No unjustified CRIU change

Confirm no CRIU patch/config change exists unless the profile identifies CRIU
(not storage/CPU) as the dominant cost, with a benchmark backing it.

## Numeric acceptance criteria

- Achieved bandwidth ≤ storage link speed.
- Intervention median improvement > noise (report the distribution).
- If CRIU changed: an independent benchmark shows the CRIU-bound portion was
  dominant before the change.

## Anti-gaming checks

- Verify multiple runs; a single-run win is not acceptable (I/O noise).
- Verify lazy-pages actually engaged (userfaultfd/page-fault counts drop) — not
  just the flag being passed.
- Verify the device-copy number was measured on GPU hardware, not inferred.

## Failure triage

- Intervention shows no improvement: FAIL; the gate requires a measured
  reduction, not a null result.
- CRIU modification without profile justification: FAIL.

## Verdict format

Write `results/verification/phase-08.md`:

```markdown
# Phase 8 verification — <date> — <host/GPU>
- R1 bytes-read plausibility: PASS/FAIL — bytes=<x> image=<y> BW=<z>
- R2 dominant cost: PASS/FAIL — task=<x> verifier=<y>
- R3 intervention: PASS/FAIL — baseline=<x> with=<y> (n=3 medians)
- R4 CRIU change justified: PASS/FAIL
- Overall: PASS/FAIL
- Notes / human action required: <text>
```
