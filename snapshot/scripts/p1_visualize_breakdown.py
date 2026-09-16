#!/usr/bin/env python3
"""Render three startup timelines (cold / warm / snapshot restore) as a single
matplotlib image, with the component breakdown stacked along the time axis.

Cold and warm are parsed from vLLM logs by p1_parse_breakdown; the restore row
comes from the JSON written by snapshot/scripts/p4_restore_vllm.sh.

Usage:
    python snapshot/scripts/p1_visualize_breakdown.py \
        --cold-log  snapshot/logs/p1_qwen3_4b_cold_vllm.log \
        --warm-log  snapshot/logs/p1_qwen3_4b_warm_vllm.log \
        --restore-json snapshot/logs/p4_qwen3_4b_restore_times.json \
        --label Qwen3-4B \
        --out snapshot/plots/startup_breakdown_qwen3_4b.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p1_parse_breakdown import parse  # noqa: E402

COLORS = {
    "process/API/import": "#9e9e9e",
    "weight load (storage)": "#ff9800",
    "torch.compile": "#e53935",
    "CUDA graph capture": "#8e24aa",
    "kernel warmup / other": "#1e88e5",
    "criu restore (image read + cuda-checkpoint)": "#00897b",
    "ready -> first response": "#43a047",
}


def startup_segments(b) -> list[tuple[str, float]]:
    return [
        ("process/API/import", b.inferred_proc or 0.0),
        ("weight load (storage)", b.model_load or 0.0),
        ("torch.compile", b.compilation or 0.0),
        ("CUDA graph capture", b.graph_capture or 0.0),
        ("kernel warmup / other", b.init_other or 0.0),
    ]


def draw(ax, title, subtitle, segments, total):
    left = 0.0
    for name, val in segments:
        if val <= 0:
            continue
        ax.barh(0, val, left=left, height=0.5, color=COLORS[name],
                edgecolor="white", linewidth=0.8, label=name)
        if val / total > 0.04:
            ax.text(left + val / 2, 0, f"{val:.1f}", ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")
        left += val
    ax.text(left, 0.42, f"total {total:.1f} s", ha="right", va="bottom",
            fontsize=10, fontweight="bold")
    ax.set_xlim(0, total * 1.02)
    ax.set_ylim(-0.6, 0.9)
    ax.set_yticks([])
    ax.set_title(f"{title}\n{subtitle}", fontsize=11, loc="left")
    ax.set_xlabel("seconds since start")
    ax.grid(axis="x", alpha=0.25)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cold-log", required=True)
    ap.add_argument("--warm-log", required=True)
    ap.add_argument("--restore-json", required=True)
    ap.add_argument("--label", default="model")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cold = parse(args.cold_log, {})
    warm = parse(args.warm_log, {})
    for b in (cold, warm):
        if b.total is None:
            sys.exit(f"missing wall-clock total in {b.path}; run via p1_baseline.sh")

    with open(args.restore_json) as f:
        r = json.load(f)
    r_call = r["restore_call_seconds"]
    r_ready = r.get("restore_to_ready_seconds", r_call)
    r_first = r["restore_to_first_response_seconds"]

    restore_segments = [
        ("criu restore (image read + cuda-checkpoint)", r_call),
        ("ready -> first response", max(r_first - r_ready, 0.0)),
    ]
    restore_total = r_first

    fig, axes = plt.subplots(3, 1, figsize=(12, 7.2))
    draw(axes[0], "1. Cold start",
         "(no kernel/compile cache; model already downloaded, so download excluded)",
         startup_segments(cold), cold.total)
    draw(axes[1], "2. Warm start",
         "(compile/autotune cache reused; weights still loaded from storage)",
         startup_segments(warm), warm.total)
    draw(axes[2], "3. Snapshot restore",
         f"(CRIU + cuda-checkpoint, mode={r.get('mode', 'plugin')}; "
         f"correct={r.get('correct')})",
         restore_segments, restore_total)

    handles, labels = axes[0].get_legend_handles_labels()
    handles2, labels2 = axes[2].get_legend_handles_labels()
    fig.legend(handles + handles2, labels + labels2, loc="lower center",
               ncol=4, fontsize=9, frameon=False)
    fig.suptitle(f"{args.label}: startup time breakdown — cold vs warm vs snapshot restore",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")
    print(f"cold={cold.total:.2f}s warm={warm.total:.2f}s restore={restore_total:.2f}s "
          f"speedup vs warm={warm.total / restore_total:.1f}x vs cold={cold.total / restore_total:.1f}x")


if __name__ == "__main__":
    main()
