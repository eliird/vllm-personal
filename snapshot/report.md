# vLLM Snapshot/Restore — Phase Report

Narrative per-phase results. Raw evidence: `snapshot/README.md`, `snapshot/logs/`,
`snapshot/results/verification/`. Each phase is executed and then independently verified
by a separate subagent whose verdict lands in `snapshot/results/verification/phase-NN.md`.

Host: `yuya-sakamoto` — Ubuntu 22.04.5, kernel 6.8.0-138, **1× RTX 4060 Ti 16 GB
(sm_89)**, driver 580.178.04, CRIU 4.2.1 + CUDA plugin, `cuda-checkpoint`
580.178.04. vLLM `0.29.1rc1.dev135+g39651f804.precompiled` (editable, cu130
wheel; `.venv` Python 3.12).

> This host is **not** the GB300/500 GB target in the plan. Single 16 GB GPU →
> small substitute model; Phase 10 (multi-GPU) N/A.

---

## Environment / tooling (Stages A–B) — DONE

- Python venv recreated at 3.12.14 with `uv` (was 3.10).
- vLLM installed editable with `VLLM_USE_PRECOMPILED=1`, no local build
  (`VLLM_PRECOMPILED_WHEEL_COMMIT=8263ea12bd…`, variant `cu130`).
- CRIU 4.2.1 built from source + `cuda_plugin.so` installed to `/usr/lib/criu`
  (CRIU's hardcoded default plugin dir; the `make install` default `/usr/local`
  path would not have been discovered).
- `cuda-checkpoint` 580.178.04 (prebuilt NVIDIA binary under `.bin/`) installed.
- CUDA verified: torch 2.13.0+cu132, matmul on the 4060 Ti OK.

## Phase 0a — Stack, tooling, GPU-free CRIU sanity — **PASS** (verified)

- `criu check --all`: "Looks good but some kernel features are missing" — only
  non-blocking warnings (no libnftables locking, `STATMOUNT_BY_FD` unavailable).
- CUDA plugin loads during real dump/restore (proven via `cuda_plugin.c:221`
  lines); `criu check` itself does not load plugins.
- Plain-process round-trip: `before=3 → after=7` (verifier: `before=3 → after=6`),
  dump 370 ms / restore 12 ms.
- Deviations: `nvcc` absent → torch-equivalent CUDA program added; plugin logs
  "Failed to launch cuda-checkpoint to retrieve restore tid" even for non-CUDA
  processes (harmless here).
- Verdict: `snapshot/results/verification/phase-00a.md` (Overall PASS).

## Phase 0b — Minimal CUDA checkpoint/kill/restore round-trip — **PASS** (verified)

- Torch-equivalent CUDA program (256 MiB pattern + device counter), checksum
  stable at `34326184.000`.
- **Manual mode** (plugin disabled with `--libdir <empty>`): suspend →
  `checkpointed` and the PID leaves the GPU; dump/restore; resume → `running`;
  counter `4→8`; dump 816 ms / restore 667 ms.
- **Plugin-driven mode**: counter `4→7`, `--get-state` = `running`, PID back on
  GPU; dump 1359 ms / restore 1122 ms.
- Hazard found and fixed: with the plugin installed, running the manual toggles
  *as well* double-toggles CUDA, leaving the process `checkpointed`. Manual mode
  now uses an empty `--libdir`. **Never mix modes.**
- Verifier reproduced everything incl. negative control (plain kill resets the
  counter) and independently observed the suspend removing GPU usage.
- Verdict: `snapshot/results/verification/phase-00b.md` (Overall PASS).

## Phase 1 — Cold vs warm startup decomposition — **PASS** (verified)

Measured three ways per model: cold (vLLM + torch/inductor compile cache
cleared), warm (compile cache reused), and (later, Phase 4/5) snapshot restore.
Weights are pre-fetched into the HF cache before timing and any HF **download**
is reported separately; it is network transfer and must not be counted as
weight load. Timeline: `snapshot/plots/startup_timeline.png`.

- **gpt-oss-20b** (MoE; needs `--cpu-offload-gb 6` + `--enforce-eager` to fit
  16 GB; Marlin MXFP4 on Ada): cold ready **104.3 s**, warm ready **43.0 s**.
  `init engine` **57.1 s cold → 3.3 s warm**; the safetensors read is
  3.5 s → 2.1 s (page cache), but the **MXFP4 Marlin repack/alloc/copy
  (~8.5 s) is not cacheable** and is re-paid every start.
- **Qwen2.5-1.5B** (fits, default compile + CUDA graphs): warm ready **30.9 s**;
  `torch.compile` **9.8 s cold → 0.1 s warm**; the `p1_qwen_cold` run also
  contains a one-time **42.1 s HF download**, now excluded from weight load
  (net **2.2 s**). **CUDA graph capture ~7 s is re-paid every start** (not
  cached) — a prime snapshot target.
- **Conclusion:** init + kernel warmup/compile dominates cold start, so a
  CRIU + `cuda-checkpoint` snapshot has a large guaranteed win. Phase 9 (weight
  separation) is **optional** here, though a naive snapshot still moves the
  weight bytes (read + repack) through disk.
- Verdict: `snapshot/results/verification/phase-01.md`.

## Phase 2 — CRIU characterization (CPU) — **PASS** (verified)

- Sweep 1/8/16 GiB; all checkpoint/restore with counter continued and checksum
  preserved. Image ≈ RSS (uncompressed).
- dump: 0.89 s / 4.5 s / 144 s (high I/O variance: verifier saw 9.2 s for the
  same 16 GiB); restore: 0.38 s / 2.66 s / 5.25 s. No swap.
- Socket flags: listening socket alone dumps with no flags; in-flight backlog
  needs `--skip-in-flight`; an accepted ESTABLISHED socket needs
  `--tcp-established`.
- POSIX shm: CRIU stores the `/dev/shm` **path, not contents** — the file must
  exist at restore.
- Probe bug found and fixed: original `p2_ram.c` included the embedded counter
  in the checksum; checksum now skips `buf[0]` (verifier confirmed
  `snapshot/scripts/dev/p2_ram.c:26`).
- Verdict: `snapshot/results/verification/phase-02.md` (Overall PASS).




## Phase 3 — `cuda-checkpoint` characterization — **PASS**

- Swept 1/4/8/12 GiB device memory (manual mode, native probe). All restore with
  counter continued and deterministic integer checksum stable.
- **Checkpoint (device→host) scales linearly**: 0.39 / 1.26 / 2.24 / 4.30 s →
  **≈2.9 GB/s**; restore ≈3 GB/s.
- **Host RSS peak ≈ device size + ~250 MiB** (device memory is copied to host
  before CRIU dumps it) — the weight-bytes cost model for Phases 8/9.
- CRIU dump is disk-bound with high variance (12 GiB: 48.7–66 s).
- Fixes: device-buffer counter (not `__device__` global) fixes a
  `cuda-checkpoint` illegal-access error; integer checksum removes float
  non-determinism; chunked torch pattern avoids allocator bloat.

## Revised model set and Phase 4 results (device-resident)

`gpt-oss-20b` is excluded (13.8 GB MXFP4 does not fit 16 GB without offload).
Workloads are now `Qwen/Qwen3-4B` (dense bf16, 7.56 GiB) and
`Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4` (4-bit MoE, 7.91 GiB).

- **Phase 1:** cold/warm ready — Qwen3-4B 130.5 → 40.5 s; MoE 108.8 → 36.4 s.
  Compile/warmup dominates cold and is cached; weight load and ~9–13 s of CUDA
  graph capture survive into warm.
- **Phase 4:** snapshot→restore→correct inference — Qwen3-4B **8.44 s**
  (4.8× vs warm, 15.5× vs cold); MoE **8.58 s** (4.2× / 12.7×). Images ≈ 17 GB
  each (device footprint + host, includes idle KV).
- Plots: `snapshot/plots/startup_breakdown_{qwen3_4b,moe}.png`.
- Blocker: `/dev/shm` link-remap is one-shot (clean before snapshot, re-snapshot
  for another restore).
