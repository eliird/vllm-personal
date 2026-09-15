# Phase 8 — Profile restore, then attack the storage bottleneck

**ID:** `phase-08`
**Runs in:** WSL2 for the host-memory/storage half; bare-metal GPU for the device-copy half.
**Depends on:** `phase-05`
**Effort:** 2 days

## Objective

Find where restore time actually goes, then reduce it. Do not modify CRIU until
profiling proves it is the bottleneck.

## Question

*Is restore bound by CRIU, storage, CPU, page faults, or filesystem — and can
the host-memory restore be overlapped/streamed?*

## Background

The two costs have different fixes: (a) CPU/host-memory restore via CRIU, and
(b) GPU device-memory copy-back via `cuda-checkpoint`. Per the CRIU plugin docs,
the device copy-back is eager, so its cost is bounded by the weight bytes
(connect to Phase 1's conclusion). `--lazy-pages`/userfaultfd helps the **host**
half only.

## Prerequisites

- Phase 5 gate passed; a real snapshot image available.
- Profiling tools: `/usr/bin/time -v`, `pidstat`/`sysstat`, `iostat`, `fio`,
  and CRIU's lazy-pages support.
- On WSL2: host/storage profiling only; run the device half on the GPU host.

## Steps

### 1. Profile a restore

`scripts/p8_profile_restore.sh` capturing:

- NVMe read bandwidth achieved and total bytes read (`iostat -x 1`,
  `/proc/diskstats`),
- number and size of image files (`find snapshots/<name> -type f -printf ...`),
- CPU utilization (`pidstat -u 1`),
- page-fault counts (`/usr/bin/time -v criu restore ...`),
- time split: CRIU memory restore vs `cuda-checkpoint` device copy (wall-clock
  deltas around each step).

Write `results/restore_profile.md` with numbers and a clearly named dominant
cost.

### 2. Separate the two costs

Explicitly report (a) host-memory restore time and (b) device copy-back time.
Do not conflate them.

### 3. Host memory: lazy-pages / userfaultfd

```bash
# CRIU lazy-pages (verify flags for your version):
sudo criu lazy-pages -D snapshots/<name> --daemon
time sudo criu restore --lazy-pages --shell-job --images-dir snapshots/<name> ...
```

Measure time-to-first-response with and without.

> Caveat to verify and record: lazy-pages helps host memory only; the GPU
> device-memory copy-back remains eager and bounded by weight bytes. On WSL2,
> `vm.unprivileged_userfaultfd=0`; run CRIU as root.

### 4. Storage: parallel I/O

Compare serial (`read→wait→read`) with multiple outstanding reads (e.g. `fio`
with `--iodepth` sweep, or parallel readers over the image files). Record
achieved bandwidth and restore-time effect.

### 5. CRIU modification only if proven

Only if profiling shows CRIU itself (not storage/CPU) is the bottleneck,
prototype a small CRIU patch/config change and benchmark it independently. Do
not assume.

## Deliverables

- `scripts/p8_profile_restore.sh`.
- `results/restore_profile.md`: dominant cost with numbers, the a/b split, and
  A/B results for lazy-pages and/or parallel I/O.
- Raw profiling logs in `logs/p8_*.log`.

## Evidence to capture

- Achieved read bandwidth vs device link speed.
- Page-fault counts with/without lazy-pages.
- Time-to-first-response for each intervention.

## Constraints / Do NOT

- Do not modify CRIU unless the profile proves CRIU is the bottleneck.
- Do not run the device-copy half on WSL.
- Do not commit anything.

## Definition of done

`results/restore_profile.md` names the dominant cost with numbers, and at least
one intervention (lazy-pages and/or parallel I/O) shows a measured restore-time
reduction. Any CRIU modification is justified by the profile.

## References

- `../checkpoint.md` Phase 8.
- CRIU lazy-pages: https://criu.org (lazy-pages / userfaultfd docs for your version).
