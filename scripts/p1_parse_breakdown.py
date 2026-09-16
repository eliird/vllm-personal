#!/usr/bin/env python3
"""Parse vLLM startup logs into a cold/warm/snapshot time breakdown and validate
that the component numbers are consistent with the measured wall clock.

The cold/warm axis is the vLLM + torch/inductor compile cache (cleared vs
reused). HuggingFace weight *downloads* are a separate network cost, not part of
"weight load", and are parsed as their own field so they never inflate the
weight-load row.

Works on logs produced by scripts/p1_baseline.sh (timestamp-prefixed) and on the
older logs already in logs/, using either:
  - a `=== harness_summary ===` block inside the log (future runs), or
  - the fallback file logs/p1_wall_clock_summary.txt (old runs).

Usage:
    .venv/bin/python scripts/p1_parse_breakdown.py logs/p1_*_vllm.log
    .venv/bin/python scripts/p1_parse_breakdown.py --summary logs/p1_wall_clock_summary.txt logs/p1_*_vllm.log
    .venv/bin/python scripts/p1_parse_breakdown.py --markdown results/startup_breakdown.md \
        --plot results/startup_timeline.png logs/p1_*_vllm.log
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from dataclasses import dataclass, field

TS = re.compile(r"^\s*(\d+\.\d+)\s")
RE_MODEL = re.compile(r"model='([^']+)'")
RE_MODEL2 = re.compile(r"vllm serve (\S+)")
RE_LOAD_WEIGHTS = re.compile(r"Loading weights took ([\d.]+) seconds")
RE_DOWNLOAD = re.compile(r"Time spent downloading weights for [^:]+: ([\d.]+) seconds")
RE_MODEL_LOAD = re.compile(r"Model loading took ([\d.]+) GiB memory and ([\d.]+) seconds")
RE_INIT = re.compile(
    r"init engine \(profile, create kv cache, warmup model\) took ([\d.]+) s"
    r"(?: \(compilation: ([\d.]+) s\))?"
)
RE_COMPILE = re.compile(r"torch.compile took ([\d.]+) s in total")
RE_GRAPH = re.compile(r"Graph capturing finished in ([\d.]+) secs")
RE_KV = re.compile(r"Available KV cache memory: ([\d.]+) GiB")
RE_READY = re.compile(r"Application startup complete")
RE_SUMMARY_T6 = re.compile(r"T0_T6_wall_seconds=([\d.]+)")
RE_SUMMARY_FIRST = re.compile(r"T0_first_response_seconds=([\d.]+)")
RE_SUMMARY_T0 = re.compile(r"^T0=([\d.]+)", re.MULTILINE)
RE_SUMMARY_CACHE = re.compile(r"clear_cache=(\d)")
RE_FALLBACK = re.compile(
    r"\(([A-Za-z0-9_]+)\):\s*T0->T6=([\d.]+)s(?:\s*T0->first=([\d.]+)s)?"
)

# Timeline phases, in display order, with a coherent palette (blue = I/O,
# orange = network, purple/pink = compilation, green = kernel warmup).
PHASES = {
    "startup": ("#78909C", "Process / API / import"),
    "download": ("#F57C00", "HF weight download (network)"),
    "read": ("#1E88E5", "Weight loading (safetensors read)"),
    "post": ("#00897B", "Weight finalize (repack / copy to GPU)"),
    "compile": ("#8E24AA", "torch.compile (inductor)"),
    "graph": ("#D81B60", "CUDA graph capture"),
    "init": ("#43A047", "Model warmup / kernel JIT tuning"),
    "tail": ("#B0BEC5", "Post-init to ready"),
}
COLORS = {k: v[0] for k, v in PHASES.items()}


def _num(patterns: list[re.Pattern[str]], text: str) -> float | None:
    for p in patterns:
        m = p.search(text)
        if m:
            return float(m.group(1))
    return None


def _nums(p: re.Pattern[str], text: str) -> list[tuple[float, ...]]:
    return [tuple(float(g) for g in m.groups()) for m in p.finditer(text)]


def _event_ts(pattern: re.Pattern[str], text: str) -> float | None:
    """Timestamp of the first line matching `pattern` (log lines are prefixed)."""
    for line in text.splitlines():
        if pattern.search(line):
            m = TS.match(line)
            if m:
                return float(m.group(1))
    return None


@dataclass
class Breakdown:
    path: str
    tag: str
    model: str = "?"
    total: float | None = None          # T0 -> ready (harness / fallback)
    first_response: float | None = None  # T0 -> first response
    log_span: float | None = None       # first -> ready log timestamp
    load_weights: float | None = None   # safetensors read only
    download: float | None = None       # HF hub download (network, not weight load)
    model_load: float | None = None     # Model loading took: download+read+repack+copy
    model_gib: float | None = None
    init_engine: float | None = None
    compilation: float | None = None
    graph_capture: float | None = None
    kv_gib: float | None = None
    cache_cleared: bool | None = None
    t0: float | None = None
    ready_ts: float | None = None
    download_ts: float | None = None
    weights_ts: float | None = None
    model_load_ts: float | None = None
    init_ts: float | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def net_model_load(self) -> float | None:
        """Model-load time excluding any HuggingFace download."""
        if self.model_load is None:
            return None
        return max(0.0, self.model_load - (self.download or 0.0))

    @property
    def inferred_proc(self) -> float | None:
        # model_load is the full elapsed model-loading span (download included),
        # so it (not net_model_load) is what remains outside of proc/import.
        if self.total is None or self.model_load is None or self.init_engine is None:
            return None
        return self.total - self.model_load - self.init_engine

    @property
    def init_other(self) -> float | None:
        if self.init_engine is None:
            return None
        return self.init_engine - (self.compilation or 0.0) - (self.graph_capture or 0.0)

    @property
    def cache_label(self) -> str:
        if self.cache_cleared is not None:
            return "cleared" if self.cache_cleared else "reused"
        return {"cold": "cleared", "warm": "reused"}.get(self.tag.split("_")[-1], "?")


def parse(path: str, fallback: dict[str, tuple[float, float | None]]) -> Breakdown:
    tag = os.path.basename(path).replace("_vllm.log", "")
    b = Breakdown(path=path, tag=tag)
    with open(path, errors="replace") as f:
        text = f.read()

    m = RE_MODEL.search(text) or RE_MODEL2.search(text)
    if m:
        b.model = m.group(1).rsplit("/", 1)[-1]

    stamps: list[float] = []
    for line in text.splitlines():
        ts = TS.match(line)
        if ts:
            stamps.append(float(ts.group(1)))
    b.ready_ts = _event_ts(RE_READY, text)
    if stamps and b.ready_ts is not None:
        b.log_span = b.ready_ts - stamps[0]

    b.load_weights = _num([RE_LOAD_WEIGHTS], text)
    b.weights_ts = _event_ts(RE_LOAD_WEIGHTS, text)
    b.download = _num([RE_DOWNLOAD], text)
    b.download_ts = _event_ts(RE_DOWNLOAD, text)
    ml = RE_MODEL_LOAD.search(text)
    if ml:
        b.model_gib = float(ml.group(1))
        b.model_load = float(ml.group(2))
    b.model_load_ts = _event_ts(RE_MODEL_LOAD, text)
    init = RE_INIT.search(text)
    if init:
        b.init_engine = float(init.group(1))
        if init.group(2):
            b.compilation = float(init.group(2))
    b.init_ts = _event_ts(RE_INIT, text)
    if b.compilation is None:
        b.compilation = _num([RE_COMPILE], text)
    graphs = [g[0] for g in _nums(RE_GRAPH, text)]
    if graphs:
        b.graph_capture = sum(graphs)
    b.kv_gib = _num([RE_KV], text)

    s6 = RE_SUMMARY_T6.search(text)
    if s6:
        b.total = float(s6.group(1))
    sfirst = RE_SUMMARY_FIRST.search(text)
    if sfirst:
        b.first_response = float(sfirst.group(1))
    if b.total is None and tag in fallback:
        b.total = fallback[tag][0]
        b.first_response = fallback[tag][1]
    mt0 = RE_SUMMARY_T0.search(text)
    if mt0:
        b.t0 = float(mt0.group(1))
    elif b.ready_ts is not None and b.total is not None:
        b.t0 = b.ready_ts - b.total
    elif stamps:
        b.t0 = stamps[0]
    mc = RE_SUMMARY_CACHE.search(text)
    if mc:
        b.cache_cleared = mc.group(1) == "1"

    # Validation.
    if b.total is not None and b.log_span is not None and b.log_span > b.total + 0.5:
        b.warnings.append(
            f"log span {b.log_span:.2f}s > harness total {b.total:.2f}s"
        )
    if b.inferred_proc is not None and b.inferred_proc < 0:
        b.warnings.append(
            f"component sum exceeds total (proc={b.inferred_proc:.2f}s) — "
            "likely overlapping timers"
        )
    if b.init_engine is not None and (b.compilation or 0) + (b.graph_capture or 0) > b.init_engine + 0.5:
        b.warnings.append("compile+graph capture exceeds init engine")
    if b.total is None:
        b.warnings.append("no wall-clock total found (no harness_summary and no fallback entry)")
    if b.model_load is not None and b.load_weights is not None:
        if b.model_load_ts is not None and b.weights_ts is not None:
            gap = b.model_load_ts - b.weights_ts
        else:
            gap = b.model_load - (b.download or 0.0) - b.load_weights
        if gap > 5.0:
            b.warnings.append(
                f"weight post-processing (repack/alloc/copy) is {gap:.2f}s after "
                f"the {b.load_weights:.2f}s safetensors read — not a download, and "
                "not cacheable."
            )
    if b.download:
        b.warnings.append(
            f"excluded {b.download:.2f}s HuggingFace download from weight load "
            "(network, not storage/cache); pre-fetch weights before cold runs to "
            "make cold/warm a pure compile-cache comparison."
        )
    return b


def load_fallback(path: str) -> dict[str, tuple[float, float | None]]:
    out: dict[str, tuple[float, float | None]] = {}
    if not os.path.exists(path):
        return out
    with open(path, errors="replace") as f:
        for line in f:
            m = RE_FALLBACK.search(line)
            if m:
                out[m.group(1)] = (float(m.group(2)),
                                   float(m.group(3)) if m.group(3) else None)
    return out


def bar(seconds: float | None, scale: float, char: str = "#") -> str:
    if not seconds or seconds <= 0:
        return ""
    return char * max(1, int(round(seconds / scale)))


def fmt(v: float | None, unit: str = "s") -> str:
    return "n/a" if v is None else f"{v:.3f}{unit}"


def dominant(b: Breakdown) -> str:
    if b.net_model_load is None or b.init_engine is None:
        return "unknown"
    if b.init_engine >= b.net_model_load:
        return "init/compile (snapshot eliminates)"
    return "weight load (storage-bound; Phase 9)"


def timeline_segments(
    b: Breakdown, include_download: bool = False
) -> list[tuple[float, float, str, str]] | None:
    """(start, end, label, color) segments on an absolute-time axis.

    Downloads are network transfer, not startup work, so they are excluded and
    compressed out of the axis by default.
    """
    if b.t0 is None or b.weights_ts is None or b.model_load_ts is None:
        return None
    segs: list[tuple[float, float, str, str]] = []
    weight_start = b.weights_ts - (b.load_weights or 0.0)
    has_dl = b.download_ts is not None and bool(b.download)
    dl_start = (b.download_ts - b.download) if has_dl else None
    first = dl_start if dl_start is not None else weight_start
    if first > b.t0 + 0.05:
        segs.append((b.t0, first, PHASES["startup"][1], COLORS["startup"]))
    if has_dl:
        segs.append((dl_start, b.download_ts,
                     PHASES["download"][1], COLORS["download"]))
    if b.load_weights:
        segs.append((weight_start, b.weights_ts,
                     PHASES["read"][1], COLORS["read"]))
    post = b.model_load_ts - b.weights_ts
    if post > 0.05:
        segs.append((b.weights_ts, b.model_load_ts,
                     PHASES["post"][1], COLORS["post"]))
    if b.init_ts is not None and b.init_engine:
        cur = b.init_ts - b.init_engine
        for val, key in (
            (b.compilation, "compile"),
            (b.graph_capture, "graph"),
            (b.init_other, "init"),
        ):
            if val and val > 0:
                segs.append((cur, cur + val, PHASES[key][1], COLORS[key]))
                cur += val
    tail_ok = (b.ready_ts is not None and b.init_ts is not None
               and b.ready_ts > b.init_ts + 0.05)
    if tail_ok:
        segs.append((b.init_ts, b.ready_ts, PHASES["tail"][1], COLORS["tail"]))

    if has_dl and not include_download:
        def adj(ts: float) -> float:
            return ts - b.download if ts >= b.download_ts else ts

        segs = [(adj(s), adj(e), lab, c) for s, e, lab, c in segs
                if c != COLORS["download"]]
    return segs


def plot_timeline(rows: list[Breakdown], path: str) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    usable = [b for b in rows if timeline_segments(b)]
    if not usable:
        print("no runs with enough timestamps to plot")
        return 0

    fig, ax = plt.subplots(figsize=(13, max(3.0, 1.1 * len(usable) + 1.8)))
    ylabels: list[str] = []
    present: set[str] = set()
    for i, b in enumerate(usable):
        segs = timeline_segments(b) or []
        t0 = b.t0 or 0.0
        for start, end, _label, color in segs:
            ax.broken_barh([(start - t0, end - start)], (i - 0.34, 0.68),
                           facecolors=color, edgecolor="white", linewidth=0.6)
            present.add(color)
        ylabels.append(f"{b.tag}\n({b.model}, cache {b.cache_label})")

    ax.set_yticks(range(len(usable)))
    ax.set_yticklabels(ylabels, fontsize=9)
    ax.set_xlabel("seconds from process launch (HF download excluded)")
    ax.set_title("vLLM startup timeline — cold vs warm (compile cache on/off)",
                 fontsize=11)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_ylim(-0.6, len(usable) - 0.4)

    handles = [Patch(facecolor=col, label=lab)
               for _k, (col, lab) in PHASES.items() if col in present]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              fontsize=8.5, ncol=4, frameon=False, handlelength=1.4,
              columnspacing=1.6)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")
    return len(usable)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", help="vLLM log files (globs allowed)")
    ap.add_argument("--summary", default="logs/p1_wall_clock_summary.txt",
                    help="fallback wall-clock file for old runs")
    ap.add_argument("--scale", type=float, default=2.0, help="seconds per bar cell")
    ap.add_argument("--markdown", default=None, help="also write a markdown report")
    ap.add_argument("--plot", default=None, help="also write a PNG timeline plot")
    args = ap.parse_args()

    paths: list[str] = []
    for pat in args.logs:
        paths.extend(sorted(glob.glob(pat)))
    paths = [p for p in dict.fromkeys(paths) if os.path.exists(p)]
    fallback = load_fallback(args.summary)

    rows = [parse(p, fallback) for p in paths]

    lines: list[str] = []
    def out(s: str = "") -> None:
        print(s)
        lines.append(s)

    for b in rows:
        out(f"### {b.tag}  ({b.model})")
        out(f"  file                 : {b.path}")
        out(f"  compile cache        : {b.cache_label}   (vLLM + torch/inductor)")
        out(f"  T0 -> ready (wall)   : {fmt(b.total)}")
        out(f"  T0 -> first response : {fmt(b.first_response)}")
        out(f"  log first->ready span: {fmt(b.log_span)}")
        out(f"  process/API/import   : {fmt(b.inferred_proc)}"
            f"   (inferred = total - model load - init)")
        out(f"  HF download          : {fmt(b.download)}"
            "   (network; inside model load, excluded from weight load)")
        out(f"  weight load (net)    : {fmt(b.net_model_load)}   "
            f"(model load {fmt(b.model_load)} - download {fmt(b.download)}; "
            f"safetensors read {fmt(b.load_weights)})")
        out(f"  init engine          : {fmt(b.init_engine)}   "
            f"(compile {fmt(b.compilation)}, graph capture {fmt(b.graph_capture)}, "
            f"warmup/JIT tuning {fmt(b.init_other)})")
        out(f"  KV cache             : {fmt(b.kv_gib, ' GiB')}")
        out(f"  dominates            : {dominant(b)}")
        out(f"  bar                  : |{bar(b.inferred_proc, args.scale)}"
            f"{bar(b.download, args.scale, 'D')}{bar(b.net_model_load, args.scale)}"
            f"{bar(b.init_engine, args.scale)}|"
            f"   (proc^ downloadD weight# init/compile#)")
        for w in b.warnings:
            out(f"  WARN: {w}")
        out()

    # Compact comparison table.
    out("| tag | cache | model | T0->ready | proc/API | download | "
        "weight load (net) | read | init engine | compile | graph | T0->first |")
    out("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: |")
    for b in rows:
        out(f"| {b.tag} | {b.cache_label} | {b.model} | {fmt(b.total)} | "
            f"{fmt(b.inferred_proc)} | {fmt(b.download)} | "
            f"{fmt(b.net_model_load)} | {fmt(b.load_weights)} | "
            f"{fmt(b.init_engine)} | {fmt(b.compilation)} | "
            f"{fmt(b.graph_capture)} | {fmt(b.first_response)} |")

    if args.markdown:
        with open(args.markdown, "w") as f:
            f.write("# Startup time breakdown\n\nAuto-generated by "
                    "`scripts/p1_parse_breakdown.py`.\n\n")
            f.write("`download` is HuggingFace network transfer and is **not** "
                    "part of\n`weight load (net)`. The cold/warm axis is the "
                    "vLLM + torch/inductor\ncompile cache (cleared vs reused).\n\n")
            if args.plot:
                f.write(f"Timeline: `{os.path.relpath(args.plot)}`\n\n")
            f.write("\n".join(lines) + "\n")
        print(f"\nwrote {args.markdown}")

    if args.plot:
        plot_timeline(rows, args.plot)


if __name__ == "__main__":
    main()
