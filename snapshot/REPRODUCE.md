# Reproduce the 3 runs: cold / warm / snapshot-restore (Qwen3-4B)

Follow-along runbook for the headline experiment. You will produce:

1. **Cold start** — no kernel/compile cache (HF download prefetched and excluded).
2. **Warm start** — compile/autotune cache reused.
3. **Snapshot restore** — `criu dump` a warm worker, `criu restore` it, and get a
   correct inference.

Then render the plot of how long each startup takes.

All work lives under `snapshot/`:
- scripts: `snapshot/scripts/`
- logs (gitignored): `snapshot/logs/`
- plots (gitignored): `snapshot/plots/`
- deliverables: `snapshot/results/`

Metric: **start restore → first correct inference response.**

---

## 0. Prerequisites (already set up on this host)

| Component | Expected |
| --- | --- |
| GPU | 1× RTX 4060 Ti 16 GB (`sm_89`) |
| Driver | 580.178.04 (`nvidia-smi` works) |
| `criu` | 4.2.1 at `/usr/local/sbin/criu` + `/usr/lib/criu/cuda_plugin.so` |
| `cuda-checkpoint` | 580.178.04 at `/usr/local/bin/cuda-checkpoint` |
| `nvcc` | 13.2 at `/usr/local/cuda/bin/nvcc` |
| venv | `.venv` (Python 3.12) at the repo root, vLLM editable precompiled |
| model | `Qwen/Qwen3-4B` cached in `~/.cache/huggingface` |

Sanity + free the GPU and `/dev/shm`:

```bash
cd ~/work/vllm-personal
nvidia-smi --query-gpu=memory.used --format=csv,noheader          # expect ~130-500 MiB (desktop only)
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader  # expect empty
# kill any leftovers (use PIDs from the line above; avoid pkill patterns that match your own shell)
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' '); do
  echo irdali | sudo -S -p '' kill -9 "$p" 2>/dev/null
done
```

> `sudo` password on this host is `irdali`. All CRIU/cuda-checkpoint steps need root.

---

## 1. Cold start (caches cleared)

```bash
cd ~/work/vllm-personal
MODEL=Qwen/Qwen3-4B PORT=8400 TAG=qwen3_4b_cold \
  CLEAR_CACHE=1 MAX_MODEL_LEN=4096 GPU_MEM_UTIL=0.90 \
  bash snapshot/scripts/p1_baseline.sh 2>&1 | tee snapshot/logs/run_cold.log
```

`CLEAR_CACHE=1` wipes `~/.cache/vllm`, `~/.cache/torch/inductor`,
`~/.cache/triton`, `~/.cache/flashinfer`. `PRELOAD=1` (default) prefetches the
HF weights before T0, so "cold" means compile-cache-cold, not first fetch.

Expected (measured): **T0→ready ≈ 130 s**, of which `init engine` ≈ 98 s
(`torch.compile` ≈ 24 s, CUDA graph capture ≈ 14 s).

Key lines:

```bash
grep -E "T0->|init engine|torch.compile|Graph capturing|Available KV cache" snapshot/logs/p1_qwen3_4b_cold_vllm.log
```

---

## 2. Warm start (cache reused)

Start it **without** clearing caches, so the AOT compile cache is reused:

```bash
cd ~/work/vllm-personal
MODEL=Qwen/Qwen3-4B PORT=8401 TAG=qwen3_4b_warm \
  CLEAR_CACHE=0 MAX_MODEL_LEN=4096 GPU_MEM_UTIL=0.90 \
  bash snapshot/scripts/p1_baseline.sh 2>&1 | tee snapshot/logs/run_warm.log
```

Expected (measured): **T0→ready ≈ 40 s**, `torch.compile` ≈ 0.1 s (cached),
CUDA graph capture still ≈ 13 s (re-paid every start), weight load ≈ 3 s
(storage-bound).

---

## 3. Snapshot + restore

### 3a. Snapshot a warm worker

IMPORTANT: clean `/dev/shm` **before** the snapshot (never between dump and
restore). A leftover `link_remap.*` / `sem.*` from a previous run breaks the
dump or restore.

```bash
cd ~/work/vllm-personal
echo irdali | sudo -S -p '' rm -f /dev/shm/link_remap.* /dev/shm/sem.*

echo irdali | sudo -S -p '' bash -c '
  timeout 900 env MODEL=Qwen/Qwen3-4B PORT=8411 TAG=qwen3_4b MODE=plugin \
    IMG=/tmp/p4_snap_qwen3_4b MAX_MODEL_LEN=4096 GPU_MEM_UTIL=0.90 \
    bash snapshot/scripts/p4_snapshot_vllm.sh' 2>&1 | tee snapshot/logs/run_snapshot.log
```

The script starts vLLM, warms one request, then runs (plugin mode, so the CRIU
CUDA plugin drives `cuda-checkpoint`):

```
criu dump --shell-job --tree <pid> --tcp-established --link-remap --images-dir /tmp/p4_snap_qwen3_4b
```

with `UV_USE_IO_URING=0` set for the server (CRIU cannot dump `io_uring`
mappings).

Expected (measured): **dump ≈ 15 s**, image **≈ 17–18 GB** (includes the KV
cache), and the dumped process is terminated by the dump.

### 3b. Restore and verify one inference

Do **not** touch `/dev/shm` first.

```bash
cd ~/work/vllm-personal
echo irdali | sudo -S -p '' bash -c '
  timeout 600 env MODEL=Qwen/Qwen3-4B PORT=8411 TAG=qwen3_4b MODE=plugin \
    IMG=/tmp/p4_snap_qwen3_4b EXPECT=Paris \
    bash snapshot/scripts/p4_restore_vllm.sh' 2>&1 | tee snapshot/logs/run_restore.log
```

Expected (measured): **restore → first response ≈ 8.4 s**, and a response
containing `Paris` (e.g. `" Paris. The capital of Germany is Berlin"`), i.e.
`PASS: response contains 'Paris'`. The restore script also writes
`snapshot/logs/p4_qwen3_4b_restore_times.json`.

---

## 4. Render the startup plot

```bash
cd ~/work/vllm-personal
.venv/bin/python snapshot/scripts/p1_visualize_breakdown.py \
  --cold-log snapshot/logs/p1_qwen3_4b_cold_vllm.log \
  --warm-log snapshot/logs/p1_qwen3_4b_warm_vllm.log \
  --restore-json snapshot/logs/p4_qwen3_4b_restore_times.json \
  --label Qwen3-4B \
  --out snapshot/plots/startup_breakdown_qwen3_4b.png
```

Output: `snapshot/plots/startup_breakdown_qwen3_4b.png` — three stacked timelines
(cold / warm / restore) showing where the startup time goes.

Optional: a per-run timeline over all logs (also prints a table to stdout):

```bash
.venv/bin/python snapshot/scripts/p1_parse_breakdown.py \
  --plot snapshot/plots/startup_timeline.png \
  'snapshot/logs/p1_*_vllm.log'
```

---

## 5. Expected results

| Phase | Total | process/API | weight load | compile | graph capture | warmup/other |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cold | 130.5 s | 28.9 | 3.1 | 24.2 | 14.0 | 60.4 |
| Warm | 40.5 s | 20.7 | 2.9 | 0.1 | 13.0 | 3.7 |
| Restore | **8.4 s** | — | — | — | — | — (criu restore 8.2 + ready→first 0.3) |

Speedups: restore = **4.8×** vs warm, **15.5×** vs cold.

Interpretation: cold start is dominated by init + kernel warmup/compile; caching
removes most of it but warm still re-pays weight load and ~13 s of CUDA graph
capture; **snapshot restore eliminates the whole init/compile block** and is
I/O-bound on the ~17 GB image.

---

## 6. Cleanup

```bash
cd ~/work/vllm-personal
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' '); do
  echo irdali | sudo -S -p '' kill -9 "$p" 2>/dev/null
done
echo irdali | sudo -S -p '' rm -rf /tmp/p4_snap_qwen3_4b
echo irdali | sudo -S -p '' rm -f /dev/shm/link_remap.* /dev/shm/sem.*
nvidia-smi --query-gpu=memory.used --format=csv,noheader
```

---

## Caveats / gotchas

- **`/dev/shm` link-remap is one-shot.** Python multiprocessing creates POSIX
  semaphores (`/dev/shm/sem.*`); CRIU needs `--link-remap`, but the
  `link_remap.*` temp lives only in `/dev/shm` and is **not** stored in the
  images. So a snapshot can be restored **once**; restoring again requires a
  fresh snapshot. Clean `/dev/shm` before snapshot, never between dump/restore.
- **`UV_USE_IO_URING=0` is required.** uvloop's libuv otherwise creates an
  `io_uring` mapping that CRIU 4.2 cannot dump.
- **Do not mix integration modes.** The CRIU CUDA plugin drives
  `cuda-checkpoint`; the manual `--toggle` path must use an empty
  `--libdir` so the plugin does not also act.
- **`gpt-oss-20b` is excluded**: MXFP4 weights ~13.8 GB do not fit 16 GB without
  `--cpu-offload-gb`; it is not device-resident.
- Use **fresh `TAG`s/PORTs** per run; the port must be free before restore.
- The `cuda_plugin ... Failed to launch cuda-checkpoint ... restore tid` lines
  for non-CUDA helper processes are harmless noise.
