# Phase 0b verification — 2026-09-15 — NVIDIA RTX 4060 Ti 16GB / driver 580.178.04 (sm_89)

- R1 environment guard: PASS
- R2 program is live: PASS
- R3 negative control (counter resets): PASS
- R4 round-trip: PASS — before=4 after=8 (manual), before=4 after=7 (plugin)
- R5 suspend removed GPU usage: PASS
- Overall: PASS
- Notes / human action required:
  - Independently reproduced with the verifier's own runs; implementer PASS lines not trusted.
  - R1: one GPU, MIG `[N/A]` (unsupported on consumer Ada), no MPS daemon
    (`pgrep -a nvidia-cuda-mps` empty), `cuda-checkpoint` 580.178.04, CRIU 4.2.1,
    driver 580.178.04 (r555+).
  - R2: launched `p0_cuda_min_torch.py` 60 iters; counter strictly increased
    1→5 and checksum stayed `34326184.000`; verifier killed it.
  - R3: fresh run reached counter `4`, plain `kill` (no checkpoint/dump/restore),
    a fresh copy restarted at counter `2` — reset proves R4 continuation is from
    preserved GPU state, not the status file.
  - R4 manual: `MODE=manual` with `--libdir <empty>` (plugin disabled). before=4
    after=8, checksums equal, `--get-state` after suspend=`checkpointed` and after
    resume=`running`. R4 plugin: before=4 after=7, checksums equal, after restore
    `--get-state`=`running`, PID present in `nvidia-smi --query --display=PIDS`,
    process alive.
  - R5: manual log records `gpu_after_suspend: 0 match(es)`; verifier observed the
    same `0 match(es)` in its own manual run.
  - Artifacts: `results/RESULTS.md` has a `## Phase 0b` section with modes,
    before/after, PASS; `logs/p0_cuda_roundtrip_manual_clean.log` and
    `logs/p0_cuda_roundtrip_plugin.log` exist and contain before/after;
    `results/stack.txt` includes MIG mode (`[N/A]`).
  - Caution: during verification, `/tmp/p0_cuda` and the two round-trip logs were
    rewritten concurrently (mtimes ~10:39–10:40), and an abandoned root-owned
    `checkpointed` process (PID 40115, parent = `systemd --user`, state file
    counter 11) was found. It was not launched by the verifier; it was killed and
    `/tmp/p0_cuda` removed to leave a clean GPU. If another agent/run was active
    in parallel, re-run this verification on a quiescent host to be certain.
  - `nvcc` absent; the torch program was used, as the implementer documented.
