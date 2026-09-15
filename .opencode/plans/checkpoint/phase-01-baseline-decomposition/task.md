# Phase 1 — Cold-start baseline with init-vs-weight decomposition

**ID:** `phase-01`
**Runs in:** bare-metal GPU.
**Depends on:** `phase-00a`
**Effort:** 0.5 day

## Objective

Establish the cold-start baseline and split it into "driver/CUDA init + graph
capture/compile" (which snapshotting can eliminate) versus "weight load from
storage" (which a naive snapshot does **not** eliminate).

## Question

*Of cold-start time, how much is driver/CUDA init + graph/compile versus
weight-load bytes?*

## Background

This decomposition is the hinge of the project. A naive CRIU+`cuda-checkpoint`
snapshot still carries the weight bytes to disk and back, so restore is
I/O-bound on those bytes. The guaranteed win is skipping init/graph/compile. If
weight load dominates, the real lever is a weights-resident restore (Phase 9),
and this number tells you that now.

**Ordering correction (do not copy the master doc's list verbatim):** CUDA
context/driver init precedes weight load because weights are allocated into the
context. Use this sequence:

```
T0 process start
T1 driver + CUDA context init complete
T2 weight load begins (first storage read)
T3 weights resident on GPU
T4 CUDA graph capture / torch.compile complete
T5 KV cache allocated
T6 server ready (first /health OK)
```

## Prerequisites

- GPU host, vLLM installed in `.venv` (see `AGENTS.md`), a model that fits.
- Use a small surrogate model first, then the target model.
- Record model identity and on-disk size in `results/stack.txt`.

## Steps

### 1. Launch and instrument

Write `scripts/p1_baseline.sh`. Prefer vLLM's own startup logs; supplement with
a wrapper that timestamps log lines. Suggested launch:

```bash
MODEL="${MODEL:?set MODEL}"; PORT="${PORT:-8000}"
: > logs/p1_vllm.log
T0=$(date +%s.%N)
.venv/bin/vllm serve "$MODEL" --port "$PORT" \
  > >(ts '[%H:%M:%S]' >> logs/p1_vllm.log) 2>&1 &
echo "T0=$T0"
until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; do sleep 0.2; done
T6=$(date +%s.%N); echo "T6=$T6 ready=$ (echo "$T6 - $T0" | bc)"
```

Derive T1..T5 from the log (or add temporary timing around the relevant calls
without modifying vLLM core). Do not add permanent instrumentation to vLLM.

### 2. Time-to-first-successful-response

```bash
curl -s "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
  -d '{"model":"'"$MODEL"'","prompt":"The capital of France is","max_tokens":8,"temperature":0}' \
  | tee logs/p1_first_response.json
```

Record the elapsed time from T0 and from T6.

### 3. Resource snapshot at T6

```bash
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv | tee -a logs/p1_resources.log
ps -eo pid,ppid,rss,cmd --forest | grep -E "vllm|python" | tee -a logs/p1_resources.log
du -sh "$MODEL_DIR" 2>/dev/null | tee -a logs/p1_resources.log
```

List the full process tree (engine-core + workers). Record GPU count, KV cache
size (from vLLM logs), model size on disk.

### 4. Fill the table in `results/RESULTS.md`

```markdown
| Metric | Cold start |
| --- | ---: |
| T0→T2 process → weights loading | |
| T1→T0 driver + CUDA context init | |
| T2→T3 weight load (storage-bound) | |
| T3→T4 graph capture / compile | |
| T4→T5 KV allocation | |
| T5→T6 ready | |
| **weight-load portion (storage-bound)** | |
| **init+graph+compile portion (snapshot eliminates)** | |
| Total startup (T0→T6) | |
| Time to first response | |
| GPU memory at ready | |
| Host RSS at ready | |
```

## Deliverables

- `scripts/p1_baseline.sh`.
- `results/RESULTS.md` `## Phase 1` with the completed table for the surrogate
  and the target model.
- `logs/p1_vllm.log`, `logs/p1_first_response.json`, `logs/p1_resources.log`.
- `results/stack.txt` updated with model identity, on-disk size, GPU count.

## Evidence to capture

- Raw log lines backing each timestamp boundary.
- The exact launch command and model revision.

## Constraints / Do NOT

- Do not add permanent instrumentation to vLLM core; wrap externally.
- Do not proceed if the model does not fit; size the run to the available GPU.
- Do not commit anything.

## Definition of done

Both bolded rows are populated with real numbers, the table is complete, and
`results/RESULTS.md` explicitly states which portion dominates and therefore
whether Phase 9 is essential or optional.

## References

- `../checkpoint.md` Phase 1 (with the ordering correction above).
- `../README.md` shared background.
