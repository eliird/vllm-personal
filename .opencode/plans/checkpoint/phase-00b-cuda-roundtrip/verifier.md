# Phase 0b — Verification

**Verifier role:** independent. Never trust the task's PASS line; reproduce the
round-trip yourself and run the negative control.

## Gate being verified

A trivial CUDA process is checkpointed, killed, restored, and its GPU state is
preserved: the device-side counter continues from its pre-checkpoint value and
the pattern checksum is unchanged.

## Setup

- Bare-metal GPU host, same driver/CRIU/plugin as the task run.
- `sudo` for CRIU.
- Do not mutate `results/` except appending your verdict to
  `results/verification/phase-00b.md`.

## Artifact checks

1. `results/RESULTS.md` `## Phase 0b` section has before/after values, mode, and
   PASS/FAIL.
2. `logs/p0_cuda_roundtrip.log` exists and contains the before/after lines.
3. `results/stack.txt` records GPU model, driver version, GPU count, and MIG
   mode.
4. `scripts/p0_cuda_min.cu` and `scripts/p0_cuda_roundtrip.sh` are non-trivial
   (contain allocation, a device kernel, `cuda-checkpoint`, `criu dump`,
   `criu restore`, and a `MODE` switch).

## Independent reproduction

### R1 — Environment guard

```bash
nvidia-smi --query-gpu=name,driver_version,mig.mode.current --format=csv
cuda-checkpoint --help 2>&1 | head -3
criu --version
```

- PASS: MIG `Disabled`, no MPS daemon, driver r555+.
- FAIL (escalate) if MIG/MPS active or driver unsupported.

### R2 — Prove the test program is live

Run `scripts/p0_cuda_min` standalone for ~5 s and confirm the status file's
counter strictly increases and the checksum is stable.

- FAIL if the counter is static (the checkpoint test would be meaningless).

### R3 — Negative control (critical)

Run the same launch as the task but **replace the checkpoint/dump/restore with a
plain kill and restart**. The counter must reset to its initial value and the
new process must not continue the old sequence.

- PASS: counter resets (proves continuation after R4 comes from preserved GPU
  state, not from the file).
- FAIL if the counter appears to "continue" without checkpointing.

### R4 — Reproduce the real round-trip

Run `MODE=manual bash scripts/p0_cuda_roundtrip.sh` (fresh temp dir) and capture
before/after. Additionally capture `cuda-checkpoint --get-state --pid <pid>`
after resume.

- PASS: `after_counter > before_counter`, `after_checksum == before_checksum`,
  and `--get-state` reports `running`.
- FAIL on any checksum change (GPU corruption), counter reset, or restore error.

### R5 — Suspend actually removed GPU usage (manual mode)

Immediately after the `--toggle` suspend and before dump, capture
`nvidia-smi --query --display=PIDS` and confirm the PID is absent; confirm it is
present again after resume.

## Numeric acceptance criteria

- `after_counter > before_counter` (strict).
- `after_checksum == before_checksum` (exact string equality).
- `cuda-checkpoint --get-state` == `running` after resume.
- Negative control: counter resets.

## Anti-gaming checks

- R2 and R3 together: the file alone cannot produce continuation, so a PASS in
  R4 is only possible via preserved device state.
- Cross-check `results/stack.txt` against live `nvidia-smi` / `criu --version`.
- If the task used `MODE=plugin`, confirm the plugin was actually loaded
  (`criu check -v4 | grep cuda_plugin`); otherwise mark the mode claim FAIL.

## Failure triage

- Any checksum mismatch or counter reset: FAIL; record driver/CRIU/plugin
  versions and the `cuda-checkpoint` error; stop for human review (driver/plugin
  escalation).
- If `criu restore` errors, capture full stderr; do not retry blindly more than
  twice (the plugin documents a rare `cuInit` race where a retry is warranted;
  note if that was the cause).

## Verdict format

Write `results/verification/phase-00b.md`:

```markdown
# Phase 0b verification — <date> — <GPU/driver>
- R1 environment guard: PASS/FAIL
- R2 program is live: PASS/FAIL
- R3 negative control (counter resets): PASS/FAIL
- R4 round-trip: PASS/FAIL — before=<ctr sum> after=<ctr sum>
- R5 suspend removed GPU usage: PASS/FAIL
- Overall: PASS/FAIL
- Notes / human action required: <text>
```
