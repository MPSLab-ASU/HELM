#!/usr/bin/env python3
"""
Replicate the paper's tables and figures as PNGs + markdown companions.

For artifact evaluators: render each table/figure of the paper from the
canonical data committed in results/paper/, so the reproduction can be put
side-by-side with the paper's version. With --run-root, the measured outputs
of the corresponding reproduce.sh stages are discovered under that directory
and overlaid next to the paper values, labeled "paper" vs "measured".

    python results/make_figures.py all|table2|table3|fig4|fig5|fig6 \
        [--run-root <dir>] [--out-dir results/figures]

Artifacts (per stage, written to --out-dir, default results/figures/):

  table2  Table II feasibility + decode-throughput/TTFT grid
          source: results/paper/table2_decode_throughput.csv
          -> table2_decode_throughput.png + .md  (heatmap matrix; OOM and
             FAIL_INIT cells drawn as hatched, labeled failure cells)
  table3  llama.cpp comparison (Llama-2-13B, RTX 3090)
          source: results/paper/table3_llamacpp_comparison.csv
          -> table3_llamacpp.png + .md
  fig4    TTFT vs prompt length (Qwen3-32B, RTX 3090, cpu(43u)+cuda(23u))
          source: results/paper/fig4_ttft_prompt_scaling.json.  The canonical
          JSON ships the two anchor points the paper text commits to (128 and
          4096 tokens, the 20.8x claim); the paper-reference rendition plots
          those anchors plus the ideal linear-scaling guide, and a --run-root
          overlay adds the full measured curve.
          -> fig4_ttft_prompt_scaling.png + .md
  fig5    Max output tokens per backend, log2 bars, one panel per GPU
          source: results/paper/fig5_max_output_tokens.json (cells the paper
          reports only graphically are NOT_REPORTED there and rendered "NR")
          -> fig5_max_output_tokens.png + .md
  fig6    Long-context paging decode throughput vs context length
          source: results/paper/fig6_paging_throughput.json
          -> fig6_paging_throughput.png + .md

--run-root discovery (matches the exact output layout each reproduce.sh
stage produces; see reproduce.sh and REPRODUCING.md):

  table2  experiments/run_paper_experiments.sh runs:
          <root>/**/<Model-slug>/<backend>/A_latency/paper_results.json
          (any *paper_results.json with a latency_sweep is accepted;
          E_feasibility probes contribute measured OOM/FAIL_INIT verdicts)
  table3  experiments/llama_cpp_bench.py:  <root>/**/results.json keyed
          "<gguf>::<tag>::ngl_<n>"; the measured HELM row is taken from any
          Llama-2-13B *paper_results.json latency sweep under the root
  fig4    experiments/ttft_prompt_scaling.sh:
          <root>/<backend>/in<len>/paper_results.json
          (parsed via experiments/ttft_prompt_scaling_summarize.collect)
  fig5    SECTIONS=F experiments/run_paper_experiments.sh:
          <root>/**/<backend>/F_max_decode/paper_results.json
          (max_decode_length section of any *paper_results.json)
  fig6    experiments/longctx_paging.sh:
          <root>/ctx_<len>/paper_results.json
          (parsed via experiments/longctx_paging_summarize.collect)

Every PNG gets a markdown companion (same basename, .md) which is also
printed to the console, so evaluators without a display still get the
paper-vs-measured comparison. Requires matplotlib (dev extras:
`pip install -e ".[dev]"` or `uv sync`); no seaborn, no GPU, no torch.
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = RESULTS_DIR.parent
sys.path.insert(0, str(RESULTS_DIR))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from validate_results import PAPER_DIR, _num, load_table2, measured_from_run  # noqa: E402
from plot_comparison import (  # noqa: E402
    COLOR_MEASURED,
    COLOR_PAPER,
    FAILURE_LABELS,
    GRID,
    INK,
    INK_MUTED,
)
from ttft_prompt_scaling_summarize import collect as collect_ttft_rows  # noqa: E402
from longctx_paging_summarize import collect as collect_paging_rows  # noqa: E402

# Backend display order and colors: validated 4-slot categorical palette
# (dataviz reference palette slots 1-4, all-pairs safe; the two low-contrast
# slots always carry direct labels). Identity never rides on color alone:
# every bar/cell is direct-labeled.
BACKEND_ORDER = ["HELM", "Accelerate", "DeepSpeed", "vLLM"]
BACKEND_COLORS = {
    "HELM": "#2a78d6",
    "Accelerate": "#008300",
    "DeepSpeed": "#e87ba4",
    "vLLM": "#eda100",
}
BACKEND_COLS = {"HELM": "helm", "Accelerate": "accelerate", "DeepSpeed": "deepspeed", "vLLM": "vllm"}

# Sequential blue ramp (magnitude encoding for the Table II heatmap).
SEQ_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
            "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
COLOR_FAIL_TEXT = "#d03b3b"

# Fig. 5 failure-cause codes (results/paper/fig5_max_output_tokens.json).
FIG5_CODES = {"WEIGHT_OOM": "W", "FAIL_INIT": "I", "KV_OOM": "KV",
              "FOOTPRINT": "F", "NOT_REPORTED": "NR", "OOM": "OOM"}
FIG5_FLOOR = 65  # log2 axis floor used by the paper for infeasible bars

# Model-name aliases: paper_bench config.model -> Table II short name.
MODEL_ALIASES = {
    "mistral-nemo-instruct-2407": "Mistral-Nemo-12B",
    "llama-2-13b-hf": "Llama-2-13B",
    "llama-2-13b": "Llama-2-13B",
    "meta-llama-3.1-8b": "Llama-3.1-8B",
    "llama-3.1-8b-instruct": "Llama-3.1-8B",
    "olmo-2-1124-13b": "OLMo-2-13B",
    "gemma-2-9b": "Gemma-2-9B",
    "gemma-2-27b": "Gemma-2-27B",
}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def short_model(model_id):
    short = (model_id or "").split("/")[-1]
    return MODEL_ALIASES.get(short.lower(), short)


def fmt_tokens(v):
    return f"{v // 1024}K" if v >= 1024 else str(v)


def write_companion(out_dir, name, lines):
    path = out_dir / f"{name}.md"
    text = "\n".join(lines) + "\n"
    path.write_text(text)
    print(text)
    print(f"wrote {path}")
    return path


def save_fig(fig, out_dir, name):
    path = out_dir / f"{name}.png"
    fig.savefig(path)
    print(f"wrote {path}")
    return path


# ──────────────────────────────────────────────────────────────────────────
# --run-root discovery
# ──────────────────────────────────────────────────────────────────────────

def iter_paper_results(run_root):
    """Yield (path, parsed_json) for every *paper_results.json under run_root."""
    for path in sorted(run_root.rglob("*paper_results.json")):
        try:
            yield path, json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  [skip] {path}: {exc}", file=sys.stderr)


# Paper GPU classes; measured runs are attributed to a class by substring
# match on the run's reported gpu_name so an overlay only lands on the
# matching Table II rows / Fig. 5 panel.
GPU_CLASS_TOKENS = {"4060": "RTX 4060", "3090": "RTX 3090", "l40s": "L40S"}


def gpu_class(name):
    n = (name or "").lower().replace(" ", "")
    for token, cls in GPU_CLASS_TOKENS.items():
        if token in n:
            return cls
    return None


def discover_gpu_name(run_root):
    """GPU of the measured run, from the hardware.json run_paper_experiments.sh
    writes (or the hardware section of any paper_results.json)."""
    for path in sorted(run_root.rglob("hardware.json")):
        try:
            name = json.loads(path.read_text()).get("gpu_name")
            if name:
                return name
        except (OSError, json.JSONDecodeError):
            pass
    for _, data in iter_paper_results(run_root):
        name = (data.get("hardware") or {}).get("gpu_name")
        if name:
            return name
    return None


def discover_table2_measured(run_root, output_len=128):
    """(gpu_class, model_short_lower, backend_col) -> {tok_s, ttft_s, src}
    plus measured OOM/FAIL_INIT verdicts from feasibility probes (tok_s None,
    status set). Runs whose GPU matches no paper class go under class None."""
    measured = {}
    for path, data in iter_paper_results(run_root):
        model = short_model((data.get("config") or {}).get("model"))
        if not model:
            continue
        cls = gpu_class((data.get("hardware") or {}).get("gpu_name"))
        rank = 0 if "A_latency" in str(path) else 1  # prefer the latency-sweep section
        for backend in (data.get("latency_sweep") or {}):
            tok_s, ttft_s, _ = measured_from_run(path, backend, output_len)
            if tok_s is None:
                continue
            key = (cls, model.lower(), "helm" if backend == "helm-router" else backend)
            if key not in measured or rank < measured[key]["rank"]:
                measured[key] = {"tok_s": tok_s, "ttft_s": ttft_s, "src": path, "rank": rank}
        for backend, fd in (data.get("feasibility") or {}).items():
            if isinstance(fd, dict) and not fd.get("fits_in_memory"):
                key = (cls, model.lower(), backend)
                status = fd.get("status") or "FAIL"
                measured.setdefault(key, {"tok_s": None, "ttft_s": None,
                                          "status": status, "src": path, "rank": 2})
    return measured


def discover_fig5_measured(run_root):
    """gpu_class -> model_short -> backend -> max_decode_length record."""
    grids = {}
    for _, data in iter_paper_results(run_root):
        model = short_model((data.get("config") or {}).get("model"))
        cls = gpu_class((data.get("hardware") or {}).get("gpu_name"))
        for backend, info in (data.get("max_decode_length") or {}).items():
            if isinstance(info, dict) and info:
                grids.setdefault(cls, {}).setdefault(model, {})[backend] = info
    return grids


def discover_table3_measured(run_root):
    """llama.cpp rows {(ngl, tag): tok_s_p50} + measured HELM Llama-2-13B tok/s."""
    lcpp = {}
    for path in sorted(run_root.rglob("results.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        for rec in data.values():
            if not (isinstance(rec, dict) and "n_gpu_layers" in rec):
                continue
            summary = rec.get("summary") or {}
            tok_s = _num(summary.get("decode_tok_s_p50"))
            if tok_s is None:
                samples = sorted(v for v in (rec.get("decode_tok_per_s") or []) if v)
                tok_s = samples[len(samples) // 2] if samples else None
            if tok_s is not None:
                lcpp[(int(rec["n_gpu_layers"]), str(rec.get("variant", "")).lower())] = tok_s
    helm_tok_s = None
    for path, data in iter_paper_results(run_root):
        if "llama-2-13b" not in short_model((data.get("config") or {}).get("model")).lower():
            continue
        tok_s, _, _ = measured_from_run(path, "helm", 128)
        if tok_s is not None:
            helm_tok_s = tok_s
    return lcpp, helm_tok_s


# ──────────────────────────────────────────────────────────────────────────
# Table II
# ──────────────────────────────────────────────────────────────────────────

def render_table2(run_root, out_dir):
    plt = _plt()
    rows = load_table2()
    measured = discover_table2_measured(run_root) if run_root else {}
    gpu_name = discover_gpu_name(run_root) if run_root else None

    # Markdown / console companion.
    lines = ["## Table II — decode throughput (tok/s) / TTFT p50 (s), output length 128, batch=1",
             "", "Source: results/paper/table2_decode_throughput.csv"
             + (f"; measured overlay from {run_root}" + (f" ({gpu_name})" if gpu_name else "")
                if run_root else " (paper reference, no measured run)"), ""]
    header = "| GPU | Model | " + " | ".join(BACKEND_ORDER) + " | HELM vs best |"
    lines += [header, "|---|---|" + "---|" * (len(BACKEND_ORDER) + 1)]
    for row in rows:
        cells = []
        for backend in BACKEND_ORDER:
            col = BACKEND_COLS[backend]
            tok = row[f"{col}_tok_s"]
            cell = f"{tok} / {row[f'{col}_ttft_s']}" if _num(tok) is not None else tok
            m = measured.get((row["gpu"], row["model"].lower(), col))
            if m:
                if m["tok_s"] is not None:
                    cell += f" **(meas {m['tok_s']:.1f} / {m['ttft_s']:.3f})**"
                elif m.get("status"):
                    cell += f" **(meas {m['status']})**"
            cells.append(cell)
        lines.append(f'| {row["gpu"]} | {row["model"]} | ' + " | ".join(cells)
                     + f' | {row["helm_vs_best_baseline"]}x |')
    unmatched = sorted((mdl, bk) for (cls, mdl, bk) in measured if cls is None)
    if unmatched:
        lines += ["", "_Measured results on a GPU outside the paper's classes "
                  "(not overlaid): " + ", ".join(f"{m}/{b}" for m, b in unmatched) + "._"]
    if run_root and not measured:
        lines += ["", f"_No usable latency/feasibility results found under {run_root}._"]
    companion = write_companion(out_dir, "table2_decode_throughput", lines)

    # PNG: matrix/heatmap. Rows = GPU x model, columns = backends; cell fill
    # encodes decode tok/s (log scale, sequential blue); OOM / FAIL_INIT cells
    # are hatched gray with a labeled failure token.
    n_rows, n_cols = len(rows), len(BACKEND_ORDER)
    tok_values = [_num(r[f"{c}_tok_s"]) for r in rows for c in BACKEND_COLS.values()
                  if _num(r[f"{c}_tok_s"]) is not None]
    import math
    lo, hi = math.log(min(tok_values)), math.log(max(tok_values))

    def ramp_color(v):
        frac = (math.log(v) - lo) / (hi - lo) if hi > lo else 0.5
        return SEQ_RAMP[round(frac * (len(SEQ_RAMP) - 1))]

    fig, ax = plt.subplots(figsize=(11.5, 0.46 * n_rows + 1.9), dpi=150)
    ax.set_xlim(0, n_cols + 1.15)
    ax.set_ylim(0, n_rows)
    ax.invert_yaxis()
    ax.axis("off")
    fig.suptitle("Table II — decode throughput (tok/s) / TTFT p50 (s), output 128, batch=1",
                 color=INK, fontsize=12)
    for j, backend in enumerate(BACKEND_ORDER):
        ax.text(j + 0.5, -0.35, backend, ha="center", va="center", color=INK,
                fontsize=10, fontweight="bold")
    ax.text(n_cols + 0.575, -0.35, "HELM vs best", ha="center", va="center",
            color=INK, fontsize=9, fontweight="bold")
    prev_gpu = None
    for i, row in enumerate(rows):
        if row["gpu"] != prev_gpu:
            ax.axhline(i, color=INK_MUTED, linewidth=1.2, xmin=0.0, xmax=1.0)
            prev_gpu = row["gpu"]
        ax.text(-0.12, i + 0.5, f'{row["gpu"]}  ·  {row["model"]}', ha="right",
                va="center", color=INK, fontsize=8.5)
        for j, backend in enumerate(BACKEND_ORDER):
            col = BACKEND_COLS[backend]
            tok, ttft = _num(row[f"{col}_tok_s"]), row[f"{col}_ttft_s"]
            m = measured.get((row["gpu"], row["model"].lower(), col))
            if tok is not None:
                idx = SEQ_RAMP.index(ramp_color(tok))
                text_color = "white" if idx >= 7 else INK
                ax.add_patch(plt.Rectangle((j + 0.03, i + 0.05), 0.94, 0.9,
                                           color=ramp_color(tok), linewidth=0))
                label = f"{tok:g} / {ttft}"
                if m and m["tok_s"] is not None:
                    ax.text(j + 0.5, i + 0.32, label, ha="center", va="center",
                            fontsize=7.5, color=text_color)
                    ax.text(j + 0.5, i + 0.7, f"meas {m['tok_s']:.1f} / {m['ttft_s']:.3f}",
                            ha="center", va="center", fontsize=7,
                            color=text_color, fontstyle="italic")
                else:
                    ax.text(j + 0.5, i + 0.5, label, ha="center", va="center",
                            fontsize=8, color=text_color)
            else:
                cell = row[f"{col}_tok_s"]  # OOM | FAIL_INIT
                hatch = "///" if cell == "OOM" else "xxx"
                token_color = COLOR_FAIL_TEXT if cell == "OOM" else INK_MUTED
                ax.add_patch(plt.Rectangle((j + 0.03, i + 0.05), 0.94, 0.9,
                                           facecolor="#f0efec", edgecolor="#d8d7d0",
                                           hatch=hatch, linewidth=0.6))
                token = FAILURE_LABELS.get(cell, cell)
                if m and m.get("status"):
                    token += f"  (meas {FAILURE_LABELS.get(m['status'], m['status'])})"
                ax.text(j + 0.5, i + 0.5, token, ha="center", va="center",
                        fontsize=8, color=token_color, fontweight="bold")
            if m:  # measured-data marker: ring the cell in the measured color
                ax.add_patch(plt.Rectangle((j + 0.03, i + 0.05), 0.94, 0.9,
                                           fill=False, edgecolor=COLOR_MEASURED,
                                           linewidth=1.6))
        ax.text(n_cols + 0.575, i + 0.5, f'{row["helm_vs_best_baseline"]}x',
                ha="center", va="center", fontsize=8.5, color=INK)
    caption = ("cell fill = decode tok/s (log scale) · hatched = infeasible "
               "(/// OOM, xxx init failure)")
    if measured:
        caption += f" · green ring = measured under {run_root.name}" \
                   + (f" ({gpu_name})" if gpu_name else "")
    ax.text((n_cols + 1.15) / 2, n_rows + 0.75, caption, ha="center", va="center",
            fontsize=8.5, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    png = save_fig(fig, out_dir, "table2_decode_throughput")
    plt.close(fig)
    return [png, companion]


# ──────────────────────────────────────────────────────────────────────────
# Table III
# ──────────────────────────────────────────────────────────────────────────

def render_table3(run_root, out_dir):
    plt = _plt()
    with open(PAPER_DIR / "table3_llamacpp_comparison.csv") as fh:
        rows = list(csv.DictReader(fh))
    lcpp_meas, helm_meas = discover_table3_measured(run_root) if run_root else ({}, None)

    def measured_for(row):
        if row["config"].startswith("HELM"):
            return helm_meas
        m = re.search(r"ngl=(-?\d+)", row["config"])
        return lcpp_meas.get((int(m.group(1)), row["precision"].lower())) if m else None

    lines = ["## Table III — HELM vs llama.cpp (Llama-2-13B, RTX 3090, in=out=128)",
             "", "Source: results/paper/table3_llamacpp_comparison.csv"
             + (f"; measured overlay from {run_root}" if run_root
                else " (paper reference, no measured run)"), "",
             "| Config | Precision | paper tok/s | measured tok/s |", "|---|---|---|---|"]
    matched = {}
    for row in rows:
        m = measured_for(row)
        matched[row["config"]] = m
        lines.append(f'| {row["config"]} | {row["precision"]} | {row["decode_tok_s"]} '
                     f'| {f"{m:.2f}" if m is not None else "—"} |')
    if run_root:
        used = set()
        for r in rows:
            g = re.search(r"ngl=(-?\d+)", r["config"])
            if g:
                used.add((int(g.group(1)), r["precision"].lower()))
        for (ngl, tag), tok in sorted(lcpp_meas.items()):
            if (ngl, tag) not in used:
                lines.append(f"| llama.cpp ngl={ngl} (measured only) | {tag} | — | {tok:.2f} |")
    companion = write_companion(out_dir, "table3_llamacpp", lines)

    fig, ax = plt.subplots(figsize=(8.6, 0.62 * len(rows) + 1.8), dpi=150)
    labels = [f'{r["config"]}  [{r["precision"]}]' for r in rows]
    y = range(len(rows))
    for yi, row in enumerate(rows):
        val = _num(row["decode_tok_s"])
        is_helm = row["config"].startswith("HELM")
        color = BACKEND_COLORS["HELM"] if is_helm else (
            "#eda100" if row["precision"].lower() == "q4_k_m" else "#9ec5f4")
        ax.barh(yi, val, 0.62, color=color, edgecolor="white", linewidth=1)
        ax.annotate(f"{val:g}", (val, yi), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK,
                    fontweight="bold" if is_helm else "normal")
        m = matched.get(row["config"])
        if m is not None:
            ax.plot([m], [yi], marker="D", markersize=7, color=INK,
                    markerfacecolor=COLOR_MEASURED, markeredgecolor="white", zorder=5)
            ax.annotate(f"meas {m:.1f}", (m, yi), xytext=(0, -13),
                        textcoords="offset points", ha="center", fontsize=7.5,
                        color=INK_MUTED)
    ax.set_yticks(list(y), labels, fontsize=9, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("decode throughput (tok/s)", fontsize=9, color=INK)
    ax.set_title("Table III — HELM (auto partition) vs hand-tuned llama.cpp ngl sweep\n"
                 "Llama-2-13B, RTX 3090, in=out=128", fontsize=11, color=INK)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    handles = [plt.Rectangle((0, 0), 1, 1, color=BACKEND_COLORS["HELM"]),
               plt.Rectangle((0, 0), 1, 1, color="#9ec5f4"),
               plt.Rectangle((0, 0), 1, 1, color="#eda100")]
    names = ["HELM (auto, fp16)", "llama.cpp fp16", "llama.cpp Q4_K_M"]
    if any(v is not None for v in matched.values()):
        handles.append(plt.Line2D([], [], marker="D", linestyle="", color=INK,
                                  markerfacecolor=COLOR_MEASURED, markeredgecolor="white"))
        names.append("measured (this machine)")
    ax.legend(handles, names, loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout()
    png = save_fig(fig, out_dir, "table3_llamacpp")
    plt.close(fig)
    return [png, companion]


# ──────────────────────────────────────────────────────────────────────────
# Fig. 4
# ──────────────────────────────────────────────────────────────────────────

def render_fig4(run_root, out_dir):
    plt = _plt()
    data = json.loads((PAPER_DIR / "fig4_ttft_prompt_scaling.json").read_text())
    anchors = sorted((int(k), v) for k, v in data["ttft_p50_s"].items())
    prompt_lengths = data["prompt_lengths"]

    measured = {}  # backend -> sorted [(input_len, ttft_s)]
    if run_root:
        for r in collect_ttft_rows(run_root):
            if r["ttft_p50"] is not None and r["input_len"] is not None:
                measured.setdefault(r["backend"], {})[int(r["input_len"])] = r["ttft_p50"] / 1000.0
        measured = {b: sorted(pts.items()) for b, pts in measured.items()}

    lines = ["## Fig. 4 — TTFT vs prompt length (Qwen3-32B, RTX 3090, "
             f'{data["partition"]}, n={data["n_per_point"]})',
             "", "Source: results/paper/fig4_ttft_prompt_scaling.json (canonical anchors: "
             "the 128- and 4096-token points behind the paper's 20.8x claim)"
             + (f"; measured curve from {run_root}" if run_root else ""), "",
             "| prompt tokens | paper TTFT p50 (s) | " +
             " | ".join(f"measured {b} (s)" for b in measured) + (" |" if measured else "|")]
    lines.append("|---|---|" + "---|" * len(measured))
    anchor_map = dict(anchors)
    all_x = sorted({x for x in prompt_lengths} | set(anchor_map)
                   | {x for pts in measured.values() for x, _ in pts})
    for x in all_x:
        row = [str(x), f"{anchor_map[x]:g}" if x in anchor_map else "—"]
        for b in measured:
            pts = dict(measured[b])
            row.append(f"{pts[x]:.1f}" if x in pts else "—")
        lines.append("| " + " | ".join(row) + " |")
    x0, y0 = anchors[0]
    x1, y1 = anchors[-1]
    lines += ["", f"Paper claim: TTFT scales {y1 / y0:.1f}x over the {x1 // x0}x input range "
              "(prefill on a heavy CPU partition grows with prompt length, not flat)."]
    companion = write_companion(out_dir, "fig4_ttft_prompt_scaling", lines)

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    # Ideal linear-scaling guide through the first anchor.
    guide_x = [prompt_lengths[0], prompt_lengths[-1]]
    ax.plot(guide_x, [y0 * gx / x0 for gx in guide_x], color=GRID, linewidth=1.4,
            linestyle=":", zorder=1)
    ax.annotate("linear scaling", (guide_x[-1], y0 * guide_x[-1] / x0), xytext=(-6, 6),
                textcoords="offset points", ha="right", fontsize=8, color=INK_MUTED)
    ax.plot([a[0] for a in anchors], [a[1] for a in anchors], color=COLOR_PAPER,
            linestyle="--", linewidth=2, marker="o", markersize=8,
            markeredgecolor="white", zorder=3, label="paper (canonical anchors)")
    for x, yv in anchors:
        ax.annotate(f"{yv:g} s", (x, yv), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=9, color=INK)
    for backend, pts in measured.items():
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        label = "measured" if backend == "helm" else f"measured ({backend})"
        color = COLOR_MEASURED if backend == "helm" else INK_MUTED
        ax.plot(xs, ys, color=color, linewidth=2, marker="s", markersize=6,
                markeredgecolor="white", zorder=4, label=label)
        ax.annotate(f"{ys[-1]:.1f} s", (xs[-1], ys[-1]), xytext=(0, -14),
                    textcoords="offset points", ha="center", fontsize=8, color=color)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(prompt_lengths, [str(x) for x in prompt_lengths])
    ax.set_xlabel("prompt length (tokens, padded)", fontsize=9, color=INK)
    ax.set_ylabel("TTFT p50 (s)", fontsize=9, color=INK)
    ax.set_title(f'Fig. 4 — TTFT vs prompt length\nQwen3-32B, RTX 3090, {data["partition"]}, '
                 f'{data["cpu_threads"]} CPU threads, n={data["n_per_point"]}',
                 fontsize=11, color=INK)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.annotate(f"{y1 / y0:.1f}x over a {x1 // x0}x input range", (x1, y1),
                xytext=(-10, -22), textcoords="offset points", ha="right",
                fontsize=9, color=COLOR_PAPER)
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    fig.tight_layout()
    png = save_fig(fig, out_dir, "fig4_ttft_prompt_scaling")
    plt.close(fig)
    return [png, companion]


# ──────────────────────────────────────────────────────────────────────────
# Fig. 5
# ──────────────────────────────────────────────────────────────────────────

FIG5_PANELS = [("rtx4060_8gb", "(a) RTX 4060 (8 GB)", "RTX 4060"),
               ("rtx3090_24gb", "(b) RTX 3090 (24 GB)", "RTX 3090"),
               ("l40s_48gb", "(c) L40S (48 GB)", "L40S")]


def render_fig5(run_root, out_dir):
    plt = _plt()
    data = json.loads((PAPER_DIR / "fig5_max_output_tokens.json").read_text())
    models = list(data[FIG5_PANELS[0][0]].keys())
    measured = discover_fig5_measured(run_root) if run_root else {}
    gpu_name = discover_gpu_name(run_root) if run_root else None
    overlaid = any(cls in measured for _, _, cls in FIG5_PANELS)

    def cell_str(cell):
        return fmt_tokens(cell) if isinstance(cell, (int, float)) else FIG5_CODES.get(cell, cell)

    lines = ["## Fig. 5 — maximum supported output length per backend (log2 scale)",
             "", "Source: results/paper/fig5_max_output_tokens.json "
             "(W=weight-load OOM, I=init failure, KV=KV-allocation OOM, "
             "F=footprint > CPU+GPU memory, NR=reported only graphically in the paper)"
             + (f"; measured overlay from {run_root}" + (f" ({gpu_name})" if gpu_name else "")
                if run_root else ""), ""]
    for panel_key, title, cls in FIG5_PANELS:
        panel_meas = measured.get(cls) or {}
        lines += [f"### {title}", "", "| Model | " + " | ".join(BACKEND_ORDER) + " |",
                  "|---|" + "---|" * len(BACKEND_ORDER)]
        for model in models:
            cells = []
            for backend in BACKEND_ORDER:
                cell = cell_str(data[panel_key][model][backend])
                m = (panel_meas.get(model) or {}).get(BACKEND_COLS[backend])
                if m:
                    mx = m.get("max_output_len")
                    cells.append(cell + f" **(meas {fmt_tokens(int(mx)) if mx else 'OOM'})**")
                else:
                    cells.append(cell)
            lines.append(f"| {model} | " + " | ".join(cells) + " |")
        lines.append("")
    if measured and not overlaid:
        lines += [f"_Measured max-decode results found under {run_root} but the run's GPU "
                  f"({gpu_name or 'unknown'}) matches no paper panel; see the tables in "
                  "the run directory._", ""]
    ratios = data["headline_ratios"]
    lines.append("Headline: geometric-mean context extension "
                 "5.9x over feasible Qwen3 pairs, max "
                 f"{max(v for v in ratios.values() if isinstance(v, (int, float))):g}x "
                 "(Qwen3-14B on RTX 3090: HELM 32K vs Accelerate 128).")
    companion = write_companion(out_dir, "fig5_max_output_tokens", lines)

    fig, axes = plt.subplots(3, 1, figsize=(8.2, 9.6), dpi=150, sharex=True)
    width = 0.19
    for ax, (panel_key, title, cls) in zip(axes, FIG5_PANELS):
        panel_meas = measured.get(cls) or {}
        for bi, backend in enumerate(BACKEND_ORDER):
            offset = (bi - (len(BACKEND_ORDER) - 1) / 2) * width
            for mi, model in enumerate(models):
                cell = data[panel_key][model][backend]
                xpos = mi + offset
                if isinstance(cell, (int, float)):
                    ax.bar(xpos, cell, width * 0.92, color=BACKEND_COLORS[backend],
                           edgecolor="white", linewidth=1)
                    ax.annotate(fmt_tokens(int(cell)), (xpos, cell), xytext=(0, 2),
                                textcoords="offset points", ha="center", fontsize=7,
                                color=INK, rotation=45)
                else:
                    ax.bar(xpos, FIG5_FLOOR, width * 0.92, color=BACKEND_COLORS[backend],
                           alpha=0.35, edgecolor=BACKEND_COLORS[backend],
                           linewidth=0.8, linestyle="--")
                    ax.annotate(FIG5_CODES.get(cell, cell), (xpos, FIG5_FLOOR),
                                xytext=(0, 2), textcoords="offset points", ha="center",
                                fontsize=7, color=INK_MUTED, rotation=45)
                m = (panel_meas.get(model) or {}).get(BACKEND_COLS[backend])
                if m:
                    mx = m.get("max_output_len") or FIG5_FLOOR
                    ax.plot([xpos], [max(mx, FIG5_FLOOR)], marker="D", markersize=6,
                            color=INK, markerfacecolor=COLOR_MEASURED,
                            markeredgecolor="white", zorder=5)
        ax.set_yscale("log", base=2)
        ax.set_ylim(50, 262144)
        ax.set_yticks([128, 512, 2048, 8192, 32768], ["128", "512", "2K", "8K", "32K"])
        ax.set_ylabel("max output tokens", fontsize=9, color=INK)
        ax.set_title(title, fontsize=10, color=INK, fontweight="bold")
        ax.set_xticks(range(len(models)), models, fontsize=9, color=INK)
        ax.tick_params(colors=INK_MUTED, labelsize=8)
        ax.grid(axis="y", color=GRID, linewidth=0.8, linestyle="--")
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(GRID)
    handles = [plt.Rectangle((0, 0), 1, 1, color=BACKEND_COLORS[b]) for b in BACKEND_ORDER]
    names = list(BACKEND_ORDER)
    if overlaid:
        handles.append(plt.Line2D([], [], marker="D", linestyle="", color=INK,
                                  markerfacecolor=COLOR_MEASURED, markeredgecolor="white"))
        names.append(f"measured ({gpu_name})" if gpu_name else "measured")
    fig.legend(handles, names, loc="lower center", ncol=min(len(names), 5),
               frameon=False, fontsize=9)
    fig.suptitle("Fig. 5 — maximum supported output length per backend (log2 scale)",
                 fontsize=11, color=INK)
    fig.text(0.5, 0.945, "faded dashed bars at the floor = infeasible\n"
             "(W weight OOM, I init failure, KV KV OOM, F footprint, NR not reported)",
             ha="center", va="top", fontsize=8.5, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.045, 1, 0.9))
    png = save_fig(fig, out_dir, "fig5_max_output_tokens")
    plt.close(fig)
    return [png, companion]


# ──────────────────────────────────────────────────────────────────────────
# Fig. 6
# ──────────────────────────────────────────────────────────────────────────

def render_fig6(run_root, out_dir):
    plt = _plt()
    data = json.loads((PAPER_DIR / "fig6_paging_throughput.json").read_text())
    paper_pts = sorted((int(k), v) for k, v in data["decode_tok_s_by_context"].items())
    measured_pts = []
    if run_root:
        measured_pts = [(int(r["ctx"]), r["tok_s"]) for r in collect_paging_rows(run_root)
                        if r["ctx"] is not None and r["tok_s"] is not None]

    lines = ["## Fig. 6 — decode throughput vs context length, KV paging active "
             f'({short_model(data["model"])}, {data["gpu"]}, {data["partition"]})',
             "", "Source: results/paper/fig6_paging_throughput.json"
             + (f"; measured curve from {run_root}" if run_root else ""), "",
             "| context (tokens) | paper tok/s |" + (" measured tok/s |" if run_root else "")]
    lines.append("|---|---|" + ("---|" if run_root else ""))
    m_map = dict(measured_pts)
    for ctx in sorted({c for c, _ in paper_pts} | set(m_map)):
        p = dict(paper_pts).get(ctx)
        row = [str(ctx), f"{p:g}" if p is not None else "—"]
        if run_root:
            row.append(f"{m_map[ctx]:.2f}" if ctx in m_map else "—")
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", f'Paging onset: {data["paging_onset"]}; at 16K the run evicts '
              f'{data["kv_evicted_to_host_gb_at_16k"]} GB and cumulatively prefetches '
              f'{data["kv_prefetched_gb_cumulative_at_16k"]} GB of KV pages while peak GPU '
              f'memory stays bounded ({data["peak_gpu_memory_gb"]["at_1k"]} -> '
              f'{data["peak_gpu_memory_gb"]["at_16k"]} GB); vLLM fails KV allocation at 16K.']
    companion = write_companion(out_dir, "fig6_paging_throughput", lines)

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    xs = [p[0] for p in paper_pts]
    ys = [p[1] for p in paper_pts]
    # Paging regime shading (past 4K per the canonical description).
    ax.axvspan(4096, max(xs + [x for x, _ in measured_pts] or [16384]) * 1.3,
               color="#f0efec", zorder=0)
    ax.plot(xs, ys, color=COLOR_PAPER, linewidth=2, marker="o", markersize=7,
            markeredgecolor="white", zorder=3, label="paper")
    for x, yv in paper_pts:
        ax.annotate(f"{yv:g}", (x, yv), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8.5, color=INK)
    if measured_pts:
        measured_pts.sort()
        mx = [p[0] for p in measured_pts]
        my = [p[1] for p in measured_pts]
        ax.plot(mx, my, color=COLOR_MEASURED, linewidth=2, marker="s", markersize=6,
                markeredgecolor="white", zorder=4, label="measured")
        ax.annotate(f"{my[-1]:.2f}", (mx[-1], my[-1]), xytext=(0, -14),
                    textcoords="offset points", ha="center", fontsize=8,
                    color=COLOR_MEASURED)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(xs, [fmt_tokens(x) for x in xs])
    ax.set_xlabel("context length (tokens)", fontsize=9, color=INK)
    ax.set_ylabel("decode throughput (tok/s)", fontsize=9, color=INK)
    ax.set_title("Fig. 6 — decode throughput vs context length with KV paging\n"
                 f'{short_model(data["model"])}, {data["gpu"]}, {data["partition"]}',
                 fontsize=11, color=INK)
    ax.annotate("KV paging active\n(peak GPU memory bounded, 22.3 -> 23.4 GB)",
                (8192, ys[0]), ha="center", fontsize=8, color=INK_MUTED)
    ax.annotate("vLLM: KV-allocation OOM at 16K", (16384, ys[-1]), xytext=(0, 24),
                textcoords="offset points", ha="right", fontsize=8, color=COLOR_FAIL_TEXT)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    if measured_pts:
        ax.legend(loc="lower left", frameon=False, fontsize=9)
    fig.tight_layout()
    png = save_fig(fig, out_dir, "fig6_paging_throughput")
    plt.close(fig)
    return [png, companion]


# ──────────────────────────────────────────────────────────────────────────

STAGES = {
    "table2": render_table2,
    "table3": render_table3,
    "fig4": render_fig4,
    "fig5": render_fig5,
    "fig6": render_fig6,
}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["all", *STAGES],
                   help="which table/figure to render (all = every one)")
    p.add_argument("--run-root", metavar="DIR", default=None,
                   help="directory containing measured reproduce.sh stage outputs "
                        "to overlay (see module docstring for per-stage layouts)")
    p.add_argument("--out-dir", default=str(REPO_ROOT / "results" / "figures"),
                   help="output directory for PNGs and markdown companions "
                        "(default: results/figures)")
    args = p.parse_args()

    run_root = None
    if args.run_root:
        run_root = Path(args.run_root).expanduser().resolve()
        if not run_root.is_dir():
            raise SystemExit(f"--run-root is not a directory: {run_root}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for name in (list(STAGES) if args.stage == "all" else [args.stage]):
        print(f"\n=== {name} ===")
        written += STAGES[name](run_root, out_dir)
    print(f"\n{len(written)} files written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
