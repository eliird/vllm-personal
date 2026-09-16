# Phase 2 verification — 2026-09-15 — yuya-sakamoto

- R1 probe correctness: PASS — RSS=513.6 MiB for request=512 MiB (100.3%, fresh
  `gcc -O2` build). Counter strictly increased 1→5, checksum constant at
  `288230381453312000` across all ticks. Source confirmed at `scripts/p2_ram.c:26`:
  the checksum loop starts at `i=1`, so `buf[0]` (the counter) is excluded.
- R2 re-run largest size: PASS — before=`1 9223372034707292160`
  after=`2 9223372034707292160` (counter advanced, checksum byte-identical),
  RSS=16,778,904 kB, dump=9,229 ms, restore=5,193 ms. Evidence:
  `logs/verify_p2_largest.log`.
- R3 image size plausibility: PASS — image=16,777,476 KB (16.0 GiB; `pages-1.img`
  = 17,179,992,064 B), ~99.99% of RSS. Probe fills every word with a nonzero
  pattern, so no zero-page dominance.
- R4 no swap: PASS — 77 `vmstat` samples during R2 all show `swpd=256` (baseline),
  `si=0`, `so=0`; min `free`=2,533 MiB. Evidence: `logs/verify_p2_memwatch.log`.
- Overall: PASS
- Notes / human action required:
  - R2 functional gate met, but CRIU still emits the documented non-fatal noise
    `cuda_plugin: Failed to launch cuda-checkpoint to retrieve restore tid ...`
    (twice per process) on every CPU-only dump/restore. It did not block
    dump/restore or affect image integrity. The literal "no CRIU error" wording is
    not satisfied; if the phase gate requires zero CRIU log errors, the CUDA plugin
    must be disabled for CPU probes. Human decision requested on whether this
    counts.
  - My first R2 launch hung because the bench script's bare `sudo criu ...` had no
    cached credential on the background tty (stuck in `n_tty_read`); re-ran after
    priming `sudo -v` in the same session. No repo change needed, but note the
    script assumes a warm sudo timestamp.
  - Fresh 16 GB dump took 9,229 ms vs the implementer's 144,013 ms (warm page
    cache/host I/O variance, as already documented). Image size and before/after
    state exactly match the RESULTS.md table.
  - Artifacts verified: `results/RESULTS.md:173` "## Phase 2" with 3 size rows
    (lines 189–191) plus FD/socket/shm notes (`:217`); `logs/p2_criu_bench.log`,
    and the first-run FAIL log (checksum off by exactly the counter delta)
    confirming the probe fix.
  - All started processes killed and all `/tmp/p2_*` images removed.
