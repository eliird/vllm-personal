# vLLM Snapshot/Restore — Results

Running results log. One section per phase. Raw evidence lives in `logs/` and
`results/verification/`.

Headline metric: **start restore → first successful (correct) inference response.**

---

## Environment (Stages A–B, pre-phase)

- Host: `yuya-sakamoto`, Ubuntu 22.04.5, bare metal (not WSL/container).
- Kernel: `6.8.0-138-generic` (was `5.15.0-170` before reboot).
- GPU: **1× NVIDIA GeForce RTX 4060 Ti, 16380 MiB, compute capability 8.9 (Ada/sm_89)**.
- Driver: `580.178.04`, CUDA `13.0` reported by `nvidia-smi`; torch `2.13.0+cu132`.
- Tooling: `cuda-checkpoint` `580.178.04` (`/usr/local/bin`), `criu` `4.2.1`
  (built from source, `33482a1`) with `cuda_plugin.so` in `/usr/lib/criu/`.
- Python: `.venv` recreated at **3.12.14** via `uv`; vLLM installed editable
  with `VLLM_USE_PRECOMPILED=1` (pinned commit `8263ea12bd…`, variant `cu130`),
  version `0.29.1rc1.dev135+g39651f804.precompiled`. No local vLLM build.
- Full capture: `results/stack.txt`.

> **Deviation from the plan's assumed hardware:** the plan targets a GB300 with
> ~500 GB weights. This host is a single 16 GB consumer GPU with 62 GiB RAM, so
> the target model is substituted; Phase 10 (multi-GPU) is not applicable.

---

## Phase 0a — Stack, tooling, GPU-free CRIU sanity

**Gate:** CRIU installed/healthy; CUDA plugin discoverable; stack recorded; plain
process checkpoints/restores with state preserved.

**Result: PASS**

| Check | Result | Evidence |
| --- | --- | --- |
| `criu --version` | `4.2.1` (`GitID 33482a1`) | `results/stack.txt` |
| `criu check --all` | "Looks good but some kernel features are missing" | `logs/p0_criu_check.log` |
| CUDA plugin installed | `/usr/lib/criu/cuda_plugin.so`, 50112 bytes | `results/stack.txt` |
| CUDA plugin loaded by CRIU | loaded + invoked at dump/restore (calls `cuda-checkpoint`) | `logs/p0_plugin_load.log` |
| Plain-process round-trip | **PASS**, `before=3 after=7` | `logs/p0_plain_roundtrip.log` |
| dump / restore wall-clock | `dump_ms=370`, `restore_ms=12` (trivial process) | `logs/p0_plain_roundtrip.log` |

Non-blocking `criu check` warnings:
- no `libnftables` support (nftables-based locking unavailable);
- `STATMOUNT_BY_FD` unavailable (files on unmounted bind mounts unsupported).

Note: `criu check` does **not** load plugins; plugin initialization happens during
`dump`/`restore`. Evidence of loading is from a real dump (`cuda_plugin.c` lines
in `logs/p0_plain_roundtrip.log`). On this host `/proc/driver/nvidia/gpus` exists,
so the plugin is **enabled** (it would self-disable only on GPU-less hosts).

Artifacts produced: `scripts/p0_stack.sh`, `scripts/p0_plain_process.sh`,
`scripts/p0_cuda_min.cu`, `scripts/p0_cuda_min_torch.py`,
`scripts/p0_cuda_roundtrip.sh`; logs `p0_criu_check.log`, `p0_plugin_load.log`,
`p0_plain_roundtrip.log`.

Deviation: the plan assumes `nvcc` for Phase 0b. `nvcc` is absent, so a
torch-equivalent program (`p0_cuda_min_torch.py`) was added and will be used;
the `.cu` source is staged for hosts with a toolkit.

---

## Phase 0b — Minimal CUDA checkpoint/kill/restore round-trip

**Gate:** a trivial CUDA process is checkpointed, killed, restored, and its GPU
state is preserved (device counter continues, pattern checksum byte-identical).

**Result: PASS** (both integration modes)

Environment guard: one GPU, MIG `[N/A]` (unsupported on consumer Ada), no MPS
daemon, driver 580.178.04 (r555+).

Program: 256 MiB device buffer with a fixed pattern, device-side counter
incremented per iteration, writes `<counter> <checksum>` to a status file each
second. Checksum stable at `34326184.000` throughout.

| Mode | before | after | checksum | get-state @ suspend → resume | dump / restore |
| --- | ---: | ---: | --- | --- | --- |
| manual (`cuda-checkpoint --toggle`, plugin disabled via `--libdir`) | `4` | `8` | `34326184.000` = | `checkpointed` → `running` | 816 / 667 ms |
| plugin-driven (CRIU CUDA plugin drives cuda-checkpoint) | `4` | `7` | `34326184.000` = | `running` after restore | 1359 / 1122 ms |

Evidence:
- `logs/p0_cuda_roundtrip_manual_clean.log` — suspend removes the PID from the
  GPU (`gpu_after_suspend: 0 match(es)`), resume reports `running`.
- `logs/p0_cuda_roundtrip_plugin.log` — plugin path; `cuda-checkpoint
  --get-state` = `running`, PID back on the GPU, process alive.
- `logs/p0_cuda_roundtrip.log` — first attempt. Kept as evidence of the
  **mode-mixing hazard**: with the plugin installed, the plugin acted during
  dump/restore *and* the script toggled manually, so `--get-state` after "resume"
  read `checkpointed`. Fixed by passing `--libdir <empty>` in manual mode.
  **Do not mix modes.**

Notes:
- `nvcc` was initially absent → torch path used. A CUDA toolkit was later
  installed (**nvcc 13.2**, `/usr/local/cuda`) so the `.cu` can be built and
  gpt-oss JIT warmup can compile.
- `criu dump` requires the `--images-dir` to pre-exist; both scripts create it.

---

## Phase 1 — Cold-start baseline with init-vs-weight decomposition

**Gate:** the weight-load portion and the init+compile portion are populated with
real numbers, plus a written conclusion on which dominates.

**Result: PASS** — init + kernel warmup/compile dominates cold start; weight load
is the smaller term.

Two models were measured across a **cold** start (vLLM + torch/inductor
compile/autotune caches cleared) and a **warm** start (those caches reused),
because the point is to separate what caching/snapshotting eliminates from what
it does not.

Weights are pre-fetched into the HuggingFace cache before timing, and any HF
**download** is reported as its own term — it is network transfer, not weight
load, and must not be attributed to storage or cache. Decomposition and a
timeline plot: `results/startup_breakdown.md`, `results/startup_timeline.png`.

### gpt-oss-20b (MXFP4 MoE, sm_89 → Marlin backend)

Requires CPU weight offload to fit 16 GB: `--cpu-offload-gb 6` → ≈7.7 GiB on GPU
+ ≈6 GiB pinned host, `--enforce-eager`, `--max-model-len 2048`. First response
correct (" Paris.").

| Metric | Cold | Warm (cache) | Δ |
| --- | ---: | ---: | ---: |
| T0→ready (total startup) | **104.25 s** | **43.02 s** | −61.2 s |
| Model loading (`Model loading took`) | 18.54 s | 17.21 s | ≈0 |
| └ safetensors read | 3.52 s | 2.13 s | page cache |
| └ MXFP4 Marlin repack/alloc/copy | 8.54 s | 8.54 s | **not cacheable** |
| `init engine` (profile+KV+warmup) | **57.10 s** | **3.32 s** | **−53.8 s** |
| T6→first response | 1.27 s | 1.27 s | 0 |
| Time to first response (T0→) | 105.5 s | 44.3 s | −61.2 s |
| GPU memory at ready | 14.53 GiB | 14.53 GiB | — |

### Qwen2.5-1.5B-Instruct (dense, fully resident, default compile + CUDA graphs)

| Metric | Cold | Warm (cache) | Δ |
| --- | ---: | ---: | ---: |
| T0→ready (total startup) | 147.2 s | **30.87 s** | — |
| HF download (network, one-time) | 42.14 s | — | excluded from weight load |
| weight load (net of download) | 2.19 s | 2.13 s | storage-bound |
| └ safetensors read | 0.42 s | 0.42 s | already cached |
| └ repack/alloc/copy | 0.41 s | 0.41 s | not cacheable |
| `torch.compile` | **9.84 s** | **0.11 s** | cacheable |
| CUDA graph capture | 3 s + 4 s = **~7 s** | 3 s + 4 s = **~7 s** | **NOT cached** |
| `init engine` | **75.10 s** | **9.26 s** | −65.8 s |
| T6→first response | 0.32 s | 0.12 s | — |

The graph-capture time is re-paid on every start (not cached) — a prime target
for a CUDA-context snapshot.

### Conclusion (explicit, per gate)

For a **cold** start, **init + kernel warmup/compile dominates** (gpt-oss: 57 s
init vs 3.5 s weight read; Qwen: 75 s init vs 0.42 s weight read). Caching
removes most of that, but a fresh process still re-pays the safetensors read
(3.5 s for gpt-oss; 2.1 s warm due to page cache), a **non-cacheable MXFP4
Marlin repack/alloc/copy (~8.5 s for gpt-oss)**, and CUDA graph capture (~7 s).

Therefore:
- A CRIU + `cuda-checkpoint` snapshot has a **large guaranteed win**: it should
  eliminate the surviving init and graph-capture cost.
- **Phase 9 (weight separation) is optional** for these models — weight load is
  not the dominant term. However, a *naive* snapshot still copies the weight
  bytes to disk and back, so gpt-oss restore will still pay the weight bytes
  (~3.5 s read + ~8.5 s repack) unless Phase 9 is done.

Raw evidence: `logs/p1_gptoss_cold_vllm.log`, `logs/p1_gptoss_warm_vllm.log`,
`logs/p1_qwen_cold_vllm.log`, `logs/p1_qwen_warm_vllm.log`,
`logs/p1_*_resources.log`, `logs/p1_*_first_response.json`.

Notes / deviations:
- `nvcc` absent initially; installed 13.2 mid-phase (gpt-oss warmup needs it).
- gpt-oss does **not** fit 16 GB without CPU offload (OOM during weight load) and
  uses the **Marlin** MXFP4 backend on Ada. This is recorded, not hidden.
- `--enforce-eager` for gpt-oss disables torch.compile/CUDA graphs, so its
  T3→T4 row is N/A; graph capture is measured on Qwen instead.

---

## Phase 2 — CRIU characterization (CPU only, no CUDA)

**Gate:** every sweep size checkpoints/restores with `after_counter >
before_counter` and `after_checksum == before_checksum`; times and image sizes
recorded.

**Result: PASS (all three sizes)** — after fixing a bug in the supplied probe
(see deviation) and re-running. No shape changed the restored bytes.

Probe: `scripts/p2_ram` allocates N MiB anonymous memory, writes a fixed
pattern, embeds a per-second counter in word 0, and writes
`<counter> <checksum>` to a status file each second. Sweep: `SIZES="1024 8192
16384" bash scripts/p2_criu_bench.sh`. Images on local ext4 `/tmp` only.

| MB | RSS (MiB) | dump_ms | restore_ms | image_kb | before (counter checksum) | after (counter checksum) | verdict |
| ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 1024 | 1026 | 887 | 382 | 1,048,752 | `2 576460758611656704` | `5 576460758611656704` | **PASS** |
| 8192 | 8194 | 4,502 | 2,660 | 8,388,820 | `1 4611686008763711488` | `2 4611686008763711488` | **PASS** |
| 16384 | 16386 | 144,013 | 5,250 | 16,777,472 | `0 9223372034707292160` | `2 9223372034707292160` | **PASS** |

Interpretation: counter advanced in every case (memory region survived and the
process kept running); checksum byte-identical (region contents preserved).
`image_kb` ≈ 1.0× RSS plus a small metadata overhead; dump and image size scale
roughly linearly with the memory region above ~1 GiB.

**Anomaly — 16 GB dump is disk-bound and high-variance.** The same 16 GB dump
took **144.0 s** on the valid run vs **47.4 s** on the pre-fix run (≥3× spread);
8 GB was 3.4–4.5 s and 1 GB 0.64–0.89 s. `logs/p2_mem_watch.log` (`vmstat 2`)
shows **zero swap in/out** (swpd stayed at the 256 K baseline) but free RAM fell
to ~390 MiB and I/O wait reached 27% during the 16 GB phase, with sustained
block-out near 580 MB/s. So the 16 GB cost is dominated by writing 16 GB of
images to disk under page-cache pressure on a shared host, not by CRIU itself.
16 GB = 26% of the 62 GiB host, well under the ~half-RAM cap; no size was
skipped and no size swapped.

**Deviation from the supplied probe (important).** `scripts/p2_ram.c` as given
writes `buf[0] = counter` and then includes `buf[0]` in the checksum, so the
"preserved" checksum necessarily changes by exactly the counter delta each
second. The first run therefore reported `FAIL` with `after_sum - before_sum ==
after_counter - before_counter` (1024: 4=4; 8192: 2=2; 16384: 1=1) — i.e. the
restores were in fact correct. The checksum now skips `buf[0]`, making the
gate satisfiable and meaningful. First-run raw output is preserved in
`logs/p2_criu_bench_firstrun.log`.

### FD / socket / shared-memory behavior

Evidence: `logs/p2_fd_socket.log` (all attempts) and
`logs/p2_fd_socket_raw/`. Probe: `scripts/p2_fd_socket.py` (TCP listener +
one POSIX shm segment; `multiprocessing.shared_memory` also spawns a
resource-tracker child, so each instance is a 2-process tree).

**Which flags are needed for sockets** (this is the explicit answer):

| Socket state | `criu dump` without flags | Required flag |
| --- | --- | --- |
| Listening socket, no connections | succeeds | none |
| Listening socket with an **in-flight** (unaccepted, in backlog) connection | fails: `inet: In-flight connection (l) … can be ignored with the --skip-in-flight option.` | **`--skip-in-flight`** (`--tcp-established` does **not** help) |
| A normally **accepted/connected** ESTABLISHED socket held by an fd | fails: `inet: Connected TCP socket, consider using --tcp-established option.` | **`--tcp-established`** |

So both flags are distinct and non-interchangeable: `--tcp-established` is for
established connections owned by the dumped process, `--skip-in-flight` is for
half-open/backlogged connections. `--skip-in-flight` dumps successfully but
drops the in-flight connection (the peer sees a reset); for later phases prefer
accepting/owning the connection and using `--tcp-established`.

**POSIX shared memory handling:**
- The segment lives at `/dev/shm/psm_<name>` (mode 0600) and is dumped as a
  **memory map / open file reference**, *not* as a copy of its contents:
  `files.img` records the path and the pages are in `pages-*.img` only while
  mapped.
- Dump+restore with the file left in place **succeeds**: counter continued
  (`4` → `7`) and `shm0=42` was preserved; CRIU re-opens `/dev/shm/<name>` by
  path on restore.
- If `/dev/shm/psm_<name>` is **deleted between dump and restore**, restore
  fails hard: `Can't open file dev/shm/psm_<name> on restore: No such file or
  directory` → `Restoring FAILED`. CRIU does not back up the shm contents.
  Any future phase that snapshots a process using `shared_memory` must persist
  or recreate the `/dev/shm` files alongside the CRIU images.
- After a successful dump kills the owner, the `/dev/shm/psm_*` file is
  **left behind** (CRIU does not unlink it); stale segments accumulated across
  attempts. Cleanup must remove them explicitly.

**Recurring non-fatal noise:** with the CUDA plugin installed, every CPU-only
dump/restore emits `cuda_plugin: Failed to launch cuda-checkpoint to retrieve
restore tid: Could not find restore thread for process ID <pid>` (twice per
process). It did not block any dump/restore here because the processes had no
CUDA state; the plugin is enabled host-wide by `/proc/driver/nvidia/gpus`.
Worth remembering for Phase 4: plugin presence adds this noise to CPU probes.

Raw evidence: `logs/p2_criu_bench.log`, `logs/p2_criu_bench_firstrun.log`,
`logs/p2_criu_per_size.log`, `logs/p2_mem_watch.log`, `logs/p2_fd_socket.log`.
No commit made; all started processes killed.



---

## Phase 3 — `cuda-checkpoint` characterization (device-memory sweep)

**Gate:** checkpoint/restore succeed at a device-memory size comparable to a real
model allocation, counter continues, checksum stable, linear-ish scaling.

**Result: PASS** (manual mode, native probe).

Probe: integer pattern + device-buffer counter; deterministic integer checksum.
Sweep on a 16 GiB card (desktop uses ~160 MiB).

| Device mem (MiB) | checkpoint ms | dump ms | restore ms | resume ms | host RSS peak (MiB) | GPU after (MiB) | image KiB | counter | checksum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1024 | 389 | 3347 | 426 | 336 | 1256 | 1280 | 1,206,268 | 1→3 | stable |
| 4096 | 1264 | 1691 | 1364 | 974 | 4328 | 4352 | 4,352,008 | 2→5 | stable |
| 8192 | 2240 | 3240 | 2676 | 1489 | 8424 | 8448 | 8,546,340 | 2→3 | stable |
| 12288 | 4303 | 66012 | 3999 | 2199 | 12520 | 12544 | 12,740,660 | 1→3 | stable |

Conclusions:
- **Checkpoint (device→host copy) scales linearly** with device memory:
  0.39 / 1.26 / 2.24 / 4.30 s → **≈ 2.9 GB/s** effective; restore/resume
  host→device is similar (≈ 3 GB/s). This is the device-memory half of the
  restore cost model and the motivation for Phase 9.
- **Host RSS peak ≈ device size + ~250 MiB baseline** — confirms the CRIU plugin
  copies all device memory into host RAM before dumping ("memory thrashing" risk
  on large models).
- CRIU dump time is disk-bound and **high variance** (1 GiB: 0.56 s vs 3.3 s
  across runs; 12 GiB: 48.7 s vs 66 s).
- Image size ≈ RSS (uncompressed).

Probe fixes found during this phase:
- Native `.cu` initially used a `__device__` global counter → `cuda-checkpoint`
  failed with an illegal memory access; moved the counter to a device buffer.
- Float `atomicAdd` checksum was non-deterministic across restores → switched to
  an integer sum (recover `k = round(p*1000)`), making equality exact.
- Torch probe initially built the pattern with a full-size `int64` range and
  cached temporaries, inflating device memory that gets checkpointed; now built
  in chunks + `empty_cache()`.

Raw evidence: `logs/p3_cuda_bench.log`, `logs/p3_cuda_bench_torch.log`,
`logs/p3_cuda_bench.log` (native). Artifacts: `scripts/p3_cuda_sweep.cu`,
`scripts/p3_cuda_sweep_torch.py`, `scripts/p3_cuda_bench.sh`.
