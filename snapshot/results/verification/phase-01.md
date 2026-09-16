# Phase 1 verification — 2026-09-15 — yuya-sakamoto

Independent verifier re-inspected all raw logs and re-ran a fresh Qwen2.5-1.5B
cold/warm pair on the RTX 4060 Ti 16 GB. Implementer summaries were not trusted;
every number below is from raw log lines or the verifier's own run.

- Artifacts: **FAIL** — every sub-check passes except `results/stack.txt`, which
  does not mention either model (`grep -iE "gpt-oss|Qwen|model" results/stack.txt`
  → no match). It only records vLLM/CUDA/CRIU versions. See notes.
- Log cross-check (gpt-oss + Qwen deltas): **PASS** — numbers match RESULTS.md
  (details below). One methodological gap: `T0→T6` is printed to the harness's
  stdout and is **not persisted in the raw vLLM logs**, so only the component
  metrics could be cross-checked directly from logs.
- Independent cold/warm reproduction: **PASS** — cold_ready=102.23s
  warm_ready=30.88s, cold_init=74.45s warm_init=9.39s (Qwen2.5-1.5B, len 4096,
  gpu_util 0.90).
- Conclusion consistency (init/compile dominates; Phase 9 optional): **PASS**.
- Overall: **PASS** (one non-substantive documentation gap: `stack.txt`).
- Notes / human action required: add model identity + on-disk size for
  `openai/gpt-oss-20b` and `Qwen/Qwen2.5-1.5B-Instruct` to `results/stack.txt`
  (required by the phase's own deliverable list). Optionally have the harness
  tee its `T0→T6` line into the log so wall time is archived; today it exists
  only in transient stdout.

## A. Artifacts

| Check | Verdict | Evidence |
| --- | --- | --- |
| `scripts/p1_baseline.sh` non-trivial | PASS | 93 lines; `setsid` + process-group kill, FIFO line timestamping, `/health` poll loop, fixed prompt, resource capture |
| `RESULTS.md` `## Phase 1` with both portions populated + explicit conclusion | PASS | gpt-oss + Qwen tables carry weight-load (`Model loading`/`Loading weights`) and init+compile (`init engine`, `torch.compile`) rows; explicit `### Conclusion` states init/warmup dominates and Phase 9 is optional |
| `results/stack.txt` mentions the model(s) | **FAIL** | no `gpt-oss`/`Qwen`/`model` string anywhere in the file |
| Four required logs exist | PASS | `logs/p1_gptoss_cold_vllm.log`, `p1_gptoss_warm_vllm.log`, `p1_qwen_cold_vllm.log`, `p1_qwen_warm_vllm.log` all present |

Minor deviation (not a failure): RESULTS.md does not use the literal bolded rows
named "weight-load portion" / "init+compile portion" from the task template; it
reports the underlying component rows, which is a superset of what the gate
needs.

## B. Log cross-check

### gpt-oss-20b (raw log lines)

| Metric | Cold | Warm | Source |
| --- | ---: | ---: | --- |
| `init engine ... took` | **57.10 s** | **3.32 s** | cold L49 / warm L49 |
| `Model loading took` | 18.538 s | 17.214 s | cold L39 / warm L39 |
| `Loading weights took` | 3.52 s | 2.13 s | cold L36 / warm L36 |
| T0→T6 wall (RESULTS only) | 104.25 s | 43.02 s | not in raw logs |

Warm `init engine` is ~17× smaller (Δ −53.78 s); weight loading is ~unchanged
(Δ −1.32 s, storage-bound). Matches RESULTS.md. Consistency check: the readiness
marker `Application startup complete` is at `1789470169.70` (cold) and
`1789470218.63` (warm), and the first-response JSON `created` values are
`1789470170` / `1789470218`, consistent with the reported ~1.27 s
`T6→first_response`.

### Qwen2.5-1.5B-Instruct (raw log lines)

| Metric | Cold | Warm | Source |
| --- | ---: | ---: | --- |
| `torch.compile took` | **9.84 s** | **0.11 s** | cold L37 / warm L32 |
| CUDA graph capture | 3 s + 4 s (≈7 s) | 3 s + 4 s (≈7 s) | cold L41/L49, warm L36/L44 |
| `init engine ... took` (compilation) | 75.10 s (9.84 s) | 9.26 s (0.11 s) | cold L54 / warm L49 |
| `Model loading took` | 44.329 s (incl. first download) | 2.129 s | cold L28 / warm L27 |
| `Loading weights took` | 0.42 s | 0.42 s | cold L27 / warm L26 |

Cold `torch.compile` ≫ warm (9.84 → 0.11 s), graph capture ~equal (~7 s, not
cached), init engine 75.10 → 9.26 s. All match RESULTS.md exactly; no
contradiction found.

## C. Independent reproduction

Fresh pair run by the verifier (different ports/tags, `MAX_MODEL_LEN=4096`,
`GPU_MEM_UTIL=0.90`), logs at `logs/p1_verify_qwen_cold_*` and
`logs/p1_verify_qwen_warm_*`:

| Metric | Cold (verify) | Warm (verify) |
| --- | ---: | ---: |
| T0→ready | **102.231 s** | **30.876 s** |
| `init engine` | **74.45 s** | **9.39 s** |
| `torch.compile` | **9.37 s** | **0.11 s** |
| CUDA graph capture | 3 + 4 = 7 s | 3 + 4 = 7 s |
| `Model loading took` | 2.165 s | 2.113 s |
| `T6→first_response` | 0.318 s | 0.315 s |
| First response correct | " Paris. The capital of Italy is Rome" | same |

Warm startup (30.88 s) is well under cold (102.23 s); warm init/compile collapse
(74.45→9.39 s, 9.37→0.11 s) while graph capture and weight load stay ~constant.
The verifier's warm ready time (30.876 s) reproduces RESULTS.md's 30.87 s almost
exactly. The verifier's cold run lacks the ~40 s first-download the original cold
run included (model already in HF cache), so its cold ready is 102.2 s vs the
reported 147.2 s — expected, not a contradiction.

## D. Cleanup

After both runs: `ps` shows no `vllm serve`/`EngineCore`; `nvidia-smi` reports
126 MiB / 16380 MiB and 0% util (desktop baseline only). No stray processes.
