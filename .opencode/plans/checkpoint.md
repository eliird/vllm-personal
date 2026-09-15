# vLLM Snapshot/Restore — Execution Plan (Claude Code Runnable)

**Target system:** single GB300 host, bare metal (non-Kubernetes), NVMe local storage.
**Model under test:** DeepSeek-V4-Flash (MoE, ~500 GB class weights). Substitute a smaller model for early phases where noted.
**Goal:** a non-Kubernetes snapshot/restore system for a warm vLLM worker using CRIU + `cuda-checkpoint`, with progressive optimization of restore latency. The success metric is **time from "start restore" → "first successful inference response."**

## How to use this document

This is written to be executed by an agent (Claude Code) **on the target GB300 host**, not in a generic cloud container. Work top to bottom. Each phase has:

- an **Objective** and the single **Question** it answers,
- **Steps** (write these scripts/commands and run them on the target),
- **Artifacts** to produce under `./results/` and `./scripts/`,
- a **Gate** — a hard go/no-go condition. Do not proceed past a failed gate; record the failure and stop for human review.

Create the workspace once at the start:

```bash
mkdir -p results scripts snapshots logs
: > results/RESULTS.md   # running results log; append a section per phase
```

Record every measured number into `results/RESULTS.md` as you go, and keep each script under `scripts/`. Treat `results/RESULTS.md` as the deliverable that survives even if later phases stall.

### Guiding principle

Do **not** make "build the snapshot manager" the goal. Each phase answers one falsifiable question. Even if you hit a wall at CUDA VMM, NCCL, or storage, you will still have produced a working prototype and a defensible benchmark table.

---

## Phase 0 — Pin the stack and de-risk `cuda-checkpoint` FIRST

**Objective:** Prove the single most fragile, newest dependency works on *this exact* GB300 + driver + CRIU + plugin combination before investing in anything else.
**Question:** *Can `cuda-checkpoint` + CRIU round-trip a trivial CUDA process on this host at all?*

> Rationale: GB300 is new silicon. `cuda-checkpoint` toggling a live CUDA context (lock → checkpoint → restore) is tightly coupled to the exact NVIDIA driver and the CRIU `cuda_plugin`. If this does not work, most of the plan is moot — and you want to know on day one.

### Steps

1. Record the exact stack into `results/stack.txt`:
   ```bash
   {
     echo "== date =="; date -u
     echo "== nvidia driver =="; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
     echo "== cuda toolkit =="; nvcc --version 2>/dev/null || echo "no nvcc"
     echo "== cuda-checkpoint =="; which cuda-checkpoint && cuda-checkpoint --version 2>/dev/null || echo "cuda-checkpoint NOT found"
     echo "== criu =="; criu --version 2>/dev/null || echo "criu NOT found"
     echo "== criu plugins =="; ls -la /usr/lib/criu/ 2>/dev/null; ls -la /usr/local/lib/criu/ 2>/dev/null
     echo "== kernel =="; uname -a
     echo "== vllm =="; python -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "vllm not importable"
   } | tee results/stack.txt
   ```
2. If `cuda-checkpoint`, `criu`, or the CUDA CRIU plugin is missing, install/build them and record the versions. The plugin must be the CUDA/GPU plugin that CRIU loads to drive `cuda-checkpoint`. Confirm CRIU actually loads it (`criu check --all` and inspect plugin discovery).
3. Prove GPU-free CRIU works (fast sanity, no CUDA): write `scripts/p0_plain_process.sh` that starts a trivial counter process, dumps it with `criu dump`, kills it, `criu restore`s it, and verifies the counter kept advancing. This isolates "is CRIU itself healthy" from "does GPU checkpoint work."
4. Prove the minimal CUDA round-trip: write `scripts/p0_cuda_min.cu` (or a tiny Python+CuPy/torch equivalent) that allocates GPU memory, fills it with a known pattern, and loops touching it. Then:
   ```bash
   # pseudocode of the sequence to script:
   #   launch cuda_min ; wait until it prints "ready"
   #   cuda-checkpoint --toggle --pid <pid>        # lock/checkpoint the context
   #   criu dump   --tree <pid> --images-dir snapshots/p0 --shell-job
   #   kill process
   #   criu restore --images-dir snapshots/p0 --shell-job &
   #   cuda-checkpoint --toggle --pid <newpid>     # resume/restore the context
   #   verify GPU memory still holds the known pattern (checksum matches)
   ```
   Capture exact flags in the script; the toggle direction and CRIU tree/shell-job flags are what most often go wrong.

### Artifacts
`results/stack.txt`, `scripts/p0_plain_process.sh`, `scripts/p0_cuda_min.*`, checksum before/after in `results/RESULTS.md`.

### Gate (HARD)
> A trivial CUDA process is checkpointed, killed, restored, and its GPU memory pattern verifies byte-identical.

If this fails: stop. This is a driver/plugin escalation, not something to work around in later phases. Record the exact error and the stack versions.

---

## Phase 1 — Cold-start baseline WITH init-vs-weight decomposition

**Objective:** Establish the baseline and — critically — split cold start into "CUDA init + graph capture + compile" vs. "weight load from storage."
**Question:** *Of cold-start time, how much is CUDA init + graph/compile (which snapshotting definitely eliminates) vs. weight-load bytes (which a naive CRIU snapshot does NOT eliminate)?*

> This decomposition is the hinge of the whole project. `cuda-checkpoint` dumps **device** memory → host → disk, so the ~500 GB of weights go to disk and back on restore. Restore is therefore I/O-bound on those bytes, same order as cold-start weight loading. The guaranteed win is skipping init/graph/compile. If weight-load dominates, the real lever is a weights-resident restore (Phase 9), and you need this number to know that now.

### Steps

1. Run the real model/config normally. Instrument the phase boundaries and log wall-clock at each:
   ```text
   T0 process start
   T1 model load begins
   T2 model loaded (weights resident on GPU)
   T3 CUDA init complete
   T4 CUDA graph capture / torch.compile complete
   T5 KV cache allocated
   T6 server ready (first /health OK)
   ```
   Prefer vLLM's own startup logs / timing hooks; supplement with a wrapper that timestamps log lines. Write `scripts/p1_baseline.sh` to launch, poll `/health`, then hit `/v1/completions` with a fixed prompt and record time-to-first-successful-response.
2. Capture the resource snapshot at T6: GPU memory used, host RSS, GPU util, number of processes (vLLM V1 spawns an engine-core process + TP worker processes — list the tree), number of GPUs, model size on disk, KV cache size.
3. Fill this table in `results/RESULTS.md`:

   | Metric | Cold start |
   | --- | ---: |
   | T0→T2 process → weights loaded | |
   | T2→T3 CUDA init | |
   | T3→T4 graph capture / compile | |
   | T4→T5 KV allocation | |
   | **weight-load portion (storage-bound)** | |
   | **init+graph+compile portion (snapshot eliminates)** | |
   | Total startup (T0→T6) | |
   | GPU memory at ready | |
   | Host RSS at ready | |

### Gate
> The two bolded rows are populated with real numbers, and `results/RESULTS.md` states which dominates. This determines the project's ceiling and whether Phase 9 is optional or essential — write that conclusion down explicitly.

---

## Phase 2 — CRIU independently (non-CUDA)

**Objective:** Characterize CRIU on this host without GPU noise. (Phase 0 step 3 proved it works; here you measure it.)
**Question:** *What are CRIU dump/restore time and image size for a plain memory-heavy process?*

### Steps
1. Write `scripts/p2_criu_bench.sh`: a process that allocates a configurable amount of RAM (sweep e.g. 1/8/32 GB), touches it, then idles. For each size: `criu dump`, kill, `criu restore`; verify state; measure dump time, restore time, image size.
2. Note file-descriptor, socket, and shared-memory handling behavior for reference in Phase 4.

### Gate
> A plain Linux process checkpoints and restores with state preserved; dump/restore time and size recorded across the memory sweep.

---

## Phase 3 — `cuda-checkpoint` characterization (still no vLLM)

**Objective:** Measure the CUDA checkpoint path properly (Phase 0 proved feasibility; here you profile it and stress it).
**Question:** *What do GPU-memory size, checkpoint time, and restore time look like as GPU allocation grows, and does state survive under realistic size?*

### Steps
1. Extend the Phase 0 CUDA program to allocate a configurable amount of device memory (sweep up toward a large fraction of GB300 HBM), run kernels, keep a verifiable pattern.
2. For each size, run the lock → `criu dump` → kill → `criu restore` → resume sequence. Measure: CPU checkpoint size, GPU checkpoint size, checkpoint time, restore time, GPU memory after restore, pattern checksum.
3. Record how checkpoint/restore time scales with device-memory size — this is the model for the weight-bytes cost in later phases.

### Gate
> A CUDA process is checkpointed and restored without reinitializing GPU state, verified at a device-memory size comparable to a real model, with a linear-ish time-vs-size curve recorded.

---

## Phase 4 — Snapshot vLLM, unoptimized

**Objective:** Find and fix everything that stops a warm vLLM worker from checkpointing/restoring cleanly. Do not optimize.
**Question:** *What prevents vLLM from being checkpointed/restored, and can we get one correct inference after restore?*

> Start with a **small model** to iterate fast on the process-tree/socket problems, then repeat with the target model.

### Steps
1. Write `scripts/p4_snapshot_vllm.sh` and `scripts/p4_restore_vllm.sh`:
   ```text
   start vLLM (small model) → warm up (1 request) → quiesce →
   cuda-checkpoint toggle → criu dump (tree, shell-job) → kill →
   criu restore → cuda-checkpoint resume → send request → assert correct
   ```
2. Work the known problem list one at a time, recording the fix for each in `results/RESULTS.md`:
   - network sockets / listening ports (API server) — `--tcp-established`, close/rebind strategy
   - NCCL communicators / IPC handles (even single-GPU may init NCCL)
   - shared memory + multiprocessing (V1 engine-core ↔ workers, ZMQ)
   - CUDA contexts, file descriptors to `/dev/nvidia*`, threads
   - container/namespace assumptions (there is no K8s here — verify nothing depends on it)
3. Once green on the small model, repeat with the target model.

### Gate
> `start → warm → checkpoint → terminate → restore → inference` yields a correct response on both the small model and the target model. Each blocker and its fix is logged.

---

## Phase 5 — Quiesce lifecycle + external snapshot manager (thin)

**Objective:** Give the worker a clean snapshot-ready state, driven by an **external** manager. Do not add a large `--snapshot` feature to vLLM.
**Question:** *Can an external manager reliably drive the worker into a checkpointable state and snapshot it?*

### Steps
1. Define the state machine the manager drives (via signals + vLLM's existing APIs, not a fork):
   ```text
   SERVING → (snapshot request) → QUIESCING → (finish active requests)
           → DISCARD_KV → SNAPSHOT_READY → [cuda-checkpoint + criu dump]
   ```
2. Build a thin CLI `scripts/snapshot-manager` with just:
   ```bash
   snapshot-manager snapshot --pid <vllm-pid> --output snapshots/<name>
   ```
   Internally: quiesce → discard KV (Phase 6 API) → `cuda-checkpoint` → `criu dump`. Keep it external; the only in-process hook you rely on is vLLM's sleep/discard API from Phase 6.

### Gate
> One command takes a warm worker to a valid snapshot on disk, and the Phase 4 restore path restores it to a correct response.

---

## Phase 6 + 7 (MERGED) — KV discard via vLLM sleep, and the VMM/VA question

**Objective:** Use vLLM's **existing** sleep mode to discard the KV cache before snapshot, and test whether it composes with checkpoint/restore. Merged because vLLM's sleep already solves most of the "VA preservation" problem.
**Question:** *Does vLLM's existing `sleep(level=1)` (VMM-based physical-page release with VA preservation) compose cleanly with `cuda-checkpoint` + CRIU — and how much does it shrink the snapshot?*

> Key insight: vLLM's sleep mode is built on a CuMemAllocator that uses the CUDA VMM API to free **physical** pages while keeping **virtual** addresses, precisely so CUDA graphs are not invalidated. This is the "reserve VA, drop physical, don't break graphs" behavior. So the open question is **not** "can we build VA preservation" — it exists — but "does a *slept* worker checkpoint and restore correctly, and does waking reallocate cleanly?"

### Steps
1. Test vLLM sleep/wake alone first (no CRIU): warm → `sleep(level=1)` → confirm KV/physical freed and VAs reserved → `wake_up()` → serve correctly. Record GPU memory before/after sleep.
2. Then snapshot a **slept** worker: warm → sleep → `cuda-checkpoint` + `criu dump`. Restore → wake → serve. Verify graphs survive (no illegal-address / graph-replay errors) and outputs are correct.
3. Compare:

   | | Normal (warm) | KV discarded (slept) |
   | --- | ---: | ---: |
   | Snapshot size | | |
   | Checkpoint time | | |
   | Restore time | | |
   | GPU memory captured | | |

4. Only if sleep does **not** cleanly release/reallocate under checkpoint should you consider modifying vLLM's allocator. Treat that as a separate, gated experiment — do not begin it unless step 2 fails.

### Gate
> A slept worker snapshots and restores to correct inference, and the size/time reduction from KV discard is quantified. Conclusion recorded on whether any vLLM allocator change is even needed.

---

## Phase 8 — Profile restore, then attack the storage bottleneck (incl. lazy-pages)

**Objective:** Find where restore time actually goes, then reduce it. Do not modify CRIU until profiling proves it is the bottleneck.
**Question:** *Is restore bound by CRIU, storage, CPU, page faults, or filesystem — and can the CPU-memory restore be overlapped/streamed?*

### Steps
1. Profile a restore: NVMe read bandwidth achieved, total bytes read, number/size of image files, CPU utilization, page-fault counts, time in CRIU memory restore vs. `cuda-checkpoint` device copy. Write `results/restore_profile.md`.
2. Separate the two costs explicitly: (a) CPU/host memory restore, (b) GPU device-memory copy-back. They have different fixes.
3. For host memory: test CRIU **`--lazy-pages` / userfaultfd (post-copy)** so the process can start executing and fault pages in on demand instead of blocking on a full reload. This directly attacks time-to-first-response.
   > Caveat to verify and record: lazy-pages helps the **host** memory restore only. The GPU device-memory copy-back from `cuda-checkpoint` is eager, so its cost is still bounded by the weight bytes (tie back to Phase 1's conclusion).
4. For storage: test multiple outstanding NVMe reads (parallel I/O) vs. serial `read→wait→read`. Only if CRIU itself is the proven bottleneck, prototype a small CRIU patch/config change and benchmark it independently.

### Gate
> `results/restore_profile.md` names the dominant cost with numbers, and at least one intervention (lazy-pages and/or parallel I/O) shows a measured restore-time reduction. Any CRIU modification is justified by the profile, not assumed.

---

## Phase 9 — Separate process state from model weights (the real lever, if Phase 1 said so)

**Objective:** Avoid re-reading ~500 GB of weights through the snapshot on every restore by keeping weights out of the CRIU/`cuda-checkpoint` image.
**Question:** *Can the process + CUDA context be restored while weights are supplied from a separate resident/loaded artifact, so the two paths run concurrently?*

> Do **not** start here. But if Phase 1 showed weight-load dominates cold start, this is where most of the restore-time win actually lives — treat it as essential, not optional. Everything before this is the scaffolding that lets you measure whether you need it.

### Steps
1. Split the target state:
   ```text
   process snapshot: CPU process state + CUDA context + small buffers   (small, fast)
   model artifact:   GPU weight tensors                                  (large)
   ```
2. Prototype concurrent restore: restore the process/context via CRIU while loading weights via the fastest available path (resident host pool, GPU-direct, or memory-mapped weight file), then rendezvous to READY.
3. Measure ready-to-serve vs. the Phase 8 best. Compare against cold start and slept-snapshot restore.

### Gate
> Concurrent process-restore + weight-load reaches correct inference, with ready-to-serve time recorded against all prior configurations.

---

## Phase 10 — Multi-GPU (only after single-GPU is solid)

**Objective:** Extend to TP/EP across GPUs. DeepSeek is MoE, so expert parallel matters.
**Question:** *Does the snapshot/restore survive NCCL communicators, IPC handles, and multi-process topology?*

### Steps
1. Scale progressively: 1 → 2 → 4 → 8 GPU. For each: init TP/EP → warm → snapshot → restore → inference.
2. Watch specifically: NCCL communicators, CUDA contexts per device, CUDA IPC handles, shared memory, GPU affinity, process topology, network interfaces. Record which break and the fix.

### Gate
> Snapshot/restore yields correct inference at each GPU count you target, with the NCCL/IPC handling documented.

---

## Phase 11 — Production-style snapshot manager

**Objective:** Package the proven mechanism into a usable tool with a compatibility guard.
**Question:** *Can a user create/list/restore/delete snapshots safely, with incompatible restores refused?*

### Steps
1. Extend the CLI:
   ```bash
   snapshot-manager create <name>
   snapshot-manager list
   snapshot-manager restore <name>
   snapshot-manager delete <name>
   snapshot-manager status
   ```
2. Write metadata per snapshot and **hard-fail** on mismatch at restore:
   ```yaml
   model: DeepSeek-V4-Flash
   vllm_version: ...
   cuda_version: ...
   driver_version: ...      # MOST fragile — a driver upgrade invalidates GPU checkpoints. Hard-fail on mismatch.
   gpu: GB300
   gpu_count: ...
   tensor_parallel: ...
   expert_parallel: ...
   snapshot_version: ...
   created_at: ...
   ```
   State plainly in docs that snapshots are **ephemeral across driver upgrades**.

### Gate
> The manager creates/restores/lists/deletes snapshots, and a deliberately mismatched (e.g. driver) restore is refused rather than attempted.

---

## Correctness hardening (applies from Phase 4 onward — do not skip)

One correct response proves very little; GPU-state corruption tends to be subtle and intermittent. For every phase that restores a real worker:

1. **Fixed prompt/seed set**, compared cold-vs-restored output (near/bit-exact as config allows).
2. **Repeated restores in a loop** (restore N times) to catch nondeterministic corruption.
3. **Soak test**: serve many requests through a restored worker so fresh KV allocation after wake is exercised, not just the first request.

Log pass/fail counts in `results/RESULTS.md` per phase.

---

## Central benchmark (the deliverable)

Fill this as phases complete. The headline number is **restore → first successful inference**.

| Configuration | Snapshot size | Snapshot time | Restore time | Ready-to-serve |
| --- | ---: | ---: | ---: | ---: |
| Cold start | N/A | N/A | — | |
| CRIU only | | | | |
| CUDA + CRIU | | | | |
| + KV discard (sleep) | | | | |
| + async / lazy-pages restore | | | | |
| + separate weights | | | | |

---

## Prior art — read before building the mechanism (do not reinvent)

- **NVIDIA Dynamo snapshot** — its snapshot path is essentially CRIU + `cuda-checkpoint`; study its implementation as reference. Your differentiation is the **non-K8s packaging** and the **async / weight-separation** optimizations, not the checkpoint toggle sequence.
- **`cuda-checkpoint`** (NVIDIA) and the **CRIU CUDA/GPU plugin** — the canonical integration point; use it rather than re-deriving context toggling.
- **vLLM sleep mode / CuMemAllocator** — the existing VMM-based KV/weight release used in Phase 6/7.
- **CRIU `--lazy-pages` / userfaultfd** — the canonical post-copy mechanism for Phase 8.

> ⚠️ Versions of `cuda-checkpoint`, the CRIU plugin, and vLLM sleep internals on Blackwell/GB300 move quickly. Confirm current versions and docs against your actual stack (recorded in `results/stack.txt`) before relying on any specific flag or behavior above.

## Assumptions made (adjust if wrong)

- Runs on the target GB300 host; the agent has permission to install `criu`/`cuda-checkpoint`, build the CUDA test programs, and start/kill vLLM.
- Local NVMe is the snapshot store; paths above (`snapshots/`) are placeholders.
- Early phases may use a smaller model to iterate; target-model runs are called out where they matter.