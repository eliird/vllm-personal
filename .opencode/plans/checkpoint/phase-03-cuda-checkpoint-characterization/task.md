# Phase 3 — `cuda-checkpoint` characterization (no vLLM)

**ID:** `phase-03`
**Runs in:** bare-metal GPU.
**Depends on:** `phase-00b`
**Effort:** 1 day

## Objective

Measure the CUDA checkpoint path properly: profile checkpoint/restore time and
footprint as device-memory allocation grows, and confirm state survives at a
size comparable to a real model.

## Question

*How do device-memory size, checkpoint time, restore time, and host footprint
scale, and does state survive at realistic size?*

## Background

Phase 0b proved feasibility; this profiles it. Per the CRIU plugin docs, GPU
memory is copied into **host RAM** and then dumped by CRIU, so host RSS grows by
roughly the device-memory size during dump — this is a capacity risk for large
models and the motivation for Phases 8/9. Capacities differ per GPU; use
`<FRACTION>` of HBM rather than absolute sizes.

## Prerequisites

- Bare-metal GPU, driver r555+ (prefer 570+), CRIU + plugin, `cuda-checkpoint`.
- Phase 0b passed. No MIG/MPS/UVM in the test process.

## Steps

### 1. Extend the minimal program

Extend `scripts/p0_cuda_min.cu` into `scripts/p3_cuda_sweep.cu` so the device
allocation size is a CLI argument, and it reports a pattern checksum plus a
device counter (same continuation trick as Phase 0b).

```bash
# for MB in 1024 8192 32768 $((HBM_MB/2)) $((HBM_MB*3/4)); do ...
nvcc -O2 scripts/p3_cuda_sweep.cu -o scripts/p3_cuda_sweep
```

### 2. Sweep and measure

`scripts/p3_cuda_bench.sh`: for each size, start the probe, record baseline,
suspend + dump, measure host RSS peak, kill, restore + resume, verify, and
record metrics.

```bash
# per size, capture:
#   checkpoint time                 = t(dump) - t(suspend start)
#   restore time                    = t(resume done) - t(restore start)
#   host RSS peak during dump       (ps / /proc/<pid>/status VmHWM)
#   device memory used after resume (nvidia-smi)
#   before/after counter + checksum
```

Use one integration model per run (`MODE=manual|plugin`, default manual).

### 3. Record the scaling

Fill in `results/RESULTS.md`:

```markdown
| Device mem (MiB) | Checkpoint s | Restore s | Host RSS peak (MiB) | GPU mem after (MiB) | Counter continues | Checksum stable |
| --- | ---: | ---: | ---: | ---: | --- | --- |
```

State whether time scales roughly linearly with device-memory size and note the
implied effective bandwidth (device→host and host→device).

## Deliverables

- `scripts/p3_cuda_sweep.cu`, `scripts/p3_cuda_bench.sh`.
- `results/RESULTS.md` `## Phase 3` table + scaling conclusion.
- Raw logs in `logs/p3_*.log`.

## Evidence to capture

- `nvidia-smi` before/after for each size.
- Host RSS peak per size (proves the host-RAM copy behavior).
- Before/after counter and checksum.

## Constraints / Do NOT

- Do not exceed ~3/4 of HBM; leave headroom for the driver and runtime.
- Do not use UVM/MIG/MPS/IPC memory.
- Do not proceed if Phase 0b failed.
- Do not commit anything.

## Definition of done

At a device-memory size comparable to a real model allocation, checkpoint and
restore succeed with the counter continuing and checksum stable, and a
linear-ish time-vs-size curve is recorded.

## References

- `../checkpoint.md` Phase 3.
- CRIU plugin known limitations (host-RAM copy): https://github.com/checkpoint-restore/criu/tree/criu-dev/plugins/cuda
