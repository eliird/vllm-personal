# Phase 0a — Verification

**Verifier role:** independent of the implementing subagent. Do not trust
`results/RESULTS.md` summaries; reproduce from raw artifacts and re-run the
CPU-side checks yourself.

## Gate being verified

CRIU is installed and healthy; the CRIU CUDA plugin is discoverable and
self-disables cleanly with no GPU; `results/stack.txt` is complete; a plain
process checkpoints and restores with state preserved.

## Setup

- Access to the same host and workspace root.
- `sudo` available (CRIU dump/restore need it).
- Do not mutate artifacts under `results/` except appending your verdict to
  `results/verification/phase-00a.md`.

## Artifact checks

1. `results/stack.txt` exists, non-empty, and contains:
   - kernel line, distro line, is-WSL line,
   - `criu` version (not "NOT found"),
   - a `cuda-checkpoint` line (found or explicitly not found),
   - a `criu plugins` listing showing `cuda_plugin.so` **or** an explicit note
     that the plugin is absent.
2. `results/RESULTS.md` has a `## Phase 0a` section.
3. `scripts/p0_stack.sh`, `scripts/p0_plain_process.sh`, `scripts/p0_cuda_min.cu`,
   `scripts/p0_cuda_roundtrip.sh` exist, are non-empty, and are executable where
   appropriate.
4. `logs/p0_criu_check.log` and `logs/p0_plugin_load.log` exist and are non-empty.

## Independent reproduction

### R1 — CRIU installed and version

```bash
criu --version
sudo criu check --all 2>&1 | tee /tmp/verify_criu_check.log
```

- PASS: command runs and returns a feature report. Enumerate every line
  containing "not" / "fail" / "error" and judge whether it blocks the CPU-side
  gates (Phase 2). Anything blocking is a FAIL.

### R2 — plugin actually loaded by CRIU

```bash
ls -la /usr/lib/criu/cuda_plugin.so
sudo criu check --all -v4 2>&1 | grep -i "cuda_plugin\|CUDA plugin"
```

- PASS: the verbose log references `cuda_plugin` (initialized or disabled).
- Confirm the self-disable message is due to absent GPU, not a missing binary:
  it should read "No GPU device found; CUDA plugin is disabled" or similar. If
  it says the plugin was never attempted, FAIL.

### R3 — re-run an independent plain-process round-trip

Do **not** just rerun the task's script. Write a fresh counter with a distinct
marker and confirm the value after restore is greater than before and that the
process is actually alive after restore.

```bash
D=/tmp/verify_p0_plain; rm -rf "$D"; mkdir -p "$D"
printf '#!/usr/bin/env bash\nn=0\nwhile true; do n=$((n+1)); echo "$n" > %s/state; sleep 1; done\n' "$D" > "$D/c.sh"
chmod +x "$D/c.sh"; "$D/c.sh" & PID=$!
sleep 3
b=$(cat "$D/state")
sudo criu dump --shell-job --images-dir "$D/images" --tree "$PID"
sleep 1; sudo criu restore --shell-job --restore-detached --images-dir "$D/images"
sleep 2; a=$(cat "$D/state")
echo "before=$b after=$a"; [ "$a" -gt "$b" ] && echo PASS || echo FAIL
```

- PASS: prints PASS. FAIL if the counter reset, the restore errors, or the
  process is not running after restore.

### R4 — Phase 0b artifacts staged, not fabricated

- `scripts/p0_cuda_min.cu` contains a device allocation and a checksum/pattern
  (grep for `cudaMalloc`, `__global__`).
- `scripts/p0_cuda_roundtrip.sh` contains a `MODE` switch and both
  `cuda-checkpoint` and `criu dump`/`criu restore`.
- FAIL if either is a stub or if `results/RESULTS.md` claims GPU results that
  cannot have been produced without a GPU.

## Numeric acceptance criteria

- Plain-process round-trip: `after > before` (strictly).
- `criu check --all`: no failing feature that blocks socket/FD/namespace
  checkpointing (needed by Phase 2/4). Document each failure either way.

## Anti-gaming checks

- Independently list `/usr/lib/criu/`; the plugin file must exist with nonzero
  size.
- The verbose `criu check` evidence must show the plugin was loaded by CRIU, not
  merely present on disk.
- Stack fields must be real: cross-check `criu --version` and `uname -a` against
  `results/stack.txt`.

## Failure triage

- If `criu check --all` shows blocking failures: record the exact feature and
  kernel-config cause; do **not** attempt kernel reconfiguration as verifier;
  mark FAIL and stop for human review.
- If the plugin is present but CRIU never loads it: FAIL (Phase 0b will not
  work on the GPU host either).
- If the plain-process round-trip fails on WSL: capture the CRIU error verbatim.

## Verdict format

Write `results/verification/phase-00a.md`:

```markdown
# Phase 0a verification — <date> — <host>
- R1 CRIU health: PASS/FAIL — <one line + evidence path>
- R2 plugin load: PASS/FAIL — <one line>
- R3 plain-process round-trip: PASS/FAIL — before=<x> after=<y>
- R4 staged artifacts: PASS/FAIL
- Blocking criu check failures: <list or none>
- Overall: PASS/FAIL
- Notes / human action required: <text>
```
