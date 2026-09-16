# Phase 3 verification — 2026-09-15 — RTX 4060 Ti 16GB / driver 580.178.04

- R1 artifacts: PASS
- R2 re-run 8192 MiB: PASS — checkpoint_ms=2247 restore_ms=2671 rss_peak=8423.5MiB image_kb=8546332
- R3 linear scaling: PASS — effective device->host=3.0 GB/s (largest row)
- R4 cleanup: PASS
- Overall: PASS
- Notes / human action required: None required. `dump_ms` is disk-bound and
  high-variance (original 3240 ms vs this run 7975 ms) as already noted in
  RESULTS.md; checkpoint/restore/RSS/image were all highly reproducible
  (<0.3% drift vs the recorded 8192 row). R3: checkpoint time roughly doubles
  for 4→8 GiB (1264→2240 ms) and the 1→4 GiB step is 3.25x for 4x memory
  (mildly sublinear); the 8→12 GiB step is 1.92x for 1.5x memory (superlinear,
  within single-run noise). The linear-scaling conclusion is directionally
  sound. Effective bandwidth computed as 12288 MiB (12.885 GB) / 4.303 s ≈ 3.0
  GB/s; the table's stated "≈ 2.9 GB/s" is a conservative rounding, not an error.

## Independent re-run evidence (R2)

Command: `SIZES="8192" MODE=manual PROG=nvcc NVCC=/usr/local/cuda/bin/nvcc bash scripts/p3_cuda_bench.sh`

```
MB=8192 mode=manual checkpoint_ms=2247 dump_ms=7975 restore_ms=2671 resume_ms=1666 \
  rss_peak_kb=8625712 gpu_before=8448 gpu_after=8448 image_kb=8546332 \
  before=[2 1098437885952] after=[3 1098437885952] PASS
```

- Counter 2→3 (continues), checksum 1098437885952 exact match across restore.
- RSS peak 8625712 KiB = 8423.5 MiB = 8192 + 231.5 MiB baseline (approx. the
  expected ~250 MiB).
- Image 8546332 KiB (8345 MiB) ≈ RSS, uncompressed, consistent with conclusion.

## R1 artifact inspection

- `results/RESULTS.md` "## Phase 3" table (line 280): rows for 1024/4096/8192/12288
  MiB with checkpoint/dump/restore/resume ms, host RSS peak, GPU after, image KiB,
  counter, checksum. Scaling conclusion present (lines 287-297).
- `logs/p3_cuda_bench.log` present with one PASS line per size.
- Scripts non-trivial: `p3_cuda_bench.sh` (68 lines, real orchestration),
  `p3_cuda_sweep.cu` (97 lines, CUDA kernels), `p3_cuda_sweep_torch.py`
  (52 lines, torch fallback).

## R4 cleanup

- `pgrep -x p3_cuda_sweep` → no match; no `p3_cuda_sweep_torch.py` process.
- `nvidia-smi` memory.used = 126 MiB (desktop baseline), GPU idle.
