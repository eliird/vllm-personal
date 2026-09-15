# Phase 2 — CRIU characterization (no CUDA)

**ID:** `phase-02`
**Runs in:** WSL2 or any Linux (CPU only).
**Depends on:** `phase-00a`
**Effort:** 0.5 day

## Objective

Characterize CRIU on this host without GPU noise: measure dump/restore time and
image size for memory-heavy processes, and note FD/socket/shared-memory
behavior for later phases.

## Question

*What are CRIU dump/restore time and image size for a plain memory-heavy
process as memory grows?*

## Background

Phase 0a proved CRIU works; this measures it and establishes the host-memory
half of the restore cost model (the device-memory half is Phase 3/8). Images
must live on local ext4; on WSL2 never use `/mnt/c`.

## Prerequisites

- `criu` installed (Phase 0a), `sudo` available.
- Enough free RAM for the sweep; cap the largest size to roughly half of
  available RAM. On a 31 GB WSL2 box, use 1/8/16 GB.

## Steps

### 1. Build the memory-heavy probe

`scripts/p2_ram.c`: allocate N MiB of anonymous memory, fill it with a known
pattern, keep a counter word that increments every second, and write
`<counter> <pattern_checksum>` to a status file each second. The counter proves
continuation; the checksum proves content preservation.

```c
// gcc -O2 scripts/p2_ram.c -o scripts/p2_ram
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
int main(int argc, char **argv) {
    size_t mb = (argc > 1) ? strtoul(argv[1], 0, 10) : 1024;
    const char *status = (argc > 2) ? argv[2] : "/tmp/p2_state";
    size_t bytes = mb << 20;
    uint32_t *buf = malloc(bytes);
    if (!buf) { perror("malloc"); return 1; }
    for (size_t i = 0; i < bytes / 4; i++) buf[i] = (uint32_t)(i * 2654435761u);
    uint64_t counter = 0;
    for (;;) {
        buf[0] = (uint32_t)counter;           // counter embedded in the region
        uint64_t sum = 0;
        for (size_t i = 0; i < bytes / 4; i++) sum += buf[i];
        FILE *f = fopen(status, "w");
        if (f) { fprintf(f, "%llu %llu\n", (unsigned long long)counter, (unsigned long long)sum); fclose(f); }
        counter++; sleep(1);
    }
}
```

### 2. Sweep and measure

`scripts/p2_criu_bench.sh`: for each size, start the probe, read the baseline,
`criu dump`, kill, `criu restore`, re-read, and record
dump time / restore time / image size.

```bash
#!/usr/bin/env bash
set -euo pipefail
SIZES="${SIZES:-1024 8192 16384}"
for MB in $SIZES; do
  D="/tmp/p2_$MB"; rm -rf "$D"; mkdir -p "$D"
  scripts/p2_ram "$MB" "$D/state" & PID=$!
  for _ in $(seq 50); do [ -s "$D/state" ] && break; sleep 0.2; done
  sleep 3
  before=$(cat "$D/state")
  t0=$(date +%s.%N)
  sudo criu dump --shell-job --images-dir "$D/images" --tree "$PID"
  t1=$(date +%s.%N)
  sudo criu restore --shell-job --restore-detached --images-dir "$D/images"
  t2=$(date +%s.%N)
  sleep 3
  after=$(cat "$D/state")
  img=$(du -sh "$D/images" | cut -f1)
  echo "MB=$MB dump=$(echo "$t1-$t0"|bc)s restore=$(echo "$t2-$t1"|bc)s image=$img before=[$before] after=[$after]"
done
```

### 3. Note FD/socket/shm behavior

Run a small process holding an open TCP listener and a POSIX shared-memory
segment; attempt a dump and record which flags are needed (e.g.
`--tcp-established`) and what CRIU reports for each resource. Record in
`results/RESULTS.md` for use in Phase 4.

## Deliverables

- `scripts/p2_ram.c`, `scripts/p2_criu_bench.sh`.
- `results/RESULTS.md` `## Phase 2` table: per size, dump time, restore time,
  image size, before/after state.
- Notes on FD/socket/shm handling.

## Evidence to capture

- Raw `criu dump`/`restore` output and the measured times.
- `before`/`after` status values proving the counter continued and the checksum
  was preserved.

## Constraints / Do NOT

- Do not exceed ~half of available RAM; the probe must not swap (check
  `vmstat`/`free`).
- Do not store images on `/mnt/c`.
- Do not commit anything.

## Definition of done

Every size in the sweep checkpoints and restores with `after_counter >
before_counter` and `after_checksum == before_checksum`; times and image sizes
are recorded across the sweep.

## References

- `../checkpoint.md` Phase 2.
- `../README.md` conventions.
