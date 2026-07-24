#!/usr/bin/env python3
"""
Plot measured benchmark runs side-by-side against the paper's Table II cell.

For artifact evaluators: after running one or more backends with
experiments/verify_cell.sh (or the full sweep), render a direct visual
comparison of decode throughput and TTFT against the canonical paper numbers,
plus a console table with the measured/paper ratios.

    python results/plot_comparison.py --gpu "RTX 3090" --model Qwen3-14B \
        --run helm=<out>/paper_results.json \
        --run accelerate=<out>/paper_results.json \
        [--out results/comparison_rtx3090_qwen3-14b.png] [--output-len 128]

Backends without a --run still appear with the paper's value (or failure
verdict), labeled "not run", so the chart always shows the full row context.
Requires matplotlib (installed by the dev extras: `pip install -e ".[dev]"`).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_results import _num, load_table2, measured_from_run, table2_row  # noqa: E402

BACKENDS = [("helm", "HELM"), ("accelerate", "Accelerate"), ("deepspeed", "DeepSpeed"), ("vllm", "vLLM")]
FAILURE_LABELS = {"OOM": "OOM", "FAIL_INIT": "init ✗"}

# Validated categorical palette (dataviz reference palette, light mode):
# slot 1 = paper reference, slot 2 = this machine. Values are also printed
# as direct labels and a console table, so identity never rides on color alone.
COLOR_PAPER = "#2a78d6"
COLOR_MEASURED = "#1baf7a"
INK = "#1a1a19"
INK_MUTED = "#6f6e66"
GRID = "#e4e3dc"


def parse_runs(pairs):
    runs = {}
    for pair in pairs or []:
        backend, _, path = pair.partition("=")
        if not path:
            raise SystemExit(f"--run expects backend=path, got {pair!r}")
        if backend not in dict(BACKENDS):
            raise SystemExit(f"unknown backend {backend!r}; choose from {[b for b, _ in BACKENDS]}")
        runs[backend] = path
    return runs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpu", required=True, help='Table II GPU name ("RTX 4060" | "RTX 3090" | "L40S")')
    p.add_argument("--model", required=True, help="Table II model name (e.g. Qwen3-14B)")
    p.add_argument("--run", action="append", metavar="BACKEND=PAPER_RESULTS_JSON",
                   help="measured run to overlay; repeatable (helm=..., accelerate=..., ...)")
    p.add_argument("--output-len", type=int, default=128)
    p.add_argument("--out", default=None, help="output PNG path (default: results/comparison_<gpu>_<model>.png)")
    args = p.parse_args()

    row = table2_row(load_table2(), args.gpu, args.model)
    if row is None:
        raise SystemExit(f"no Table II row for gpu={args.gpu!r} model={args.model!r}")
    runs = parse_runs(args.run)
    if not runs:
        raise SystemExit("provide at least one --run backend=paper_results.json")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels, paper_tok, paper_ttft, meas_tok, meas_ttft, notes = [], [], [], [], [], []
    print(f"\n{args.gpu} / {args.model} (output length {args.output_len}, batch=1)")
    print(f"{'backend':<12} {'paper tok/s':>12} {'measured':>10} {'ratio':>7}   note")
    for backend, label in BACKENDS:
        col = "helm" if backend in ("helm", "helm-router") else backend
        p_tok, p_ttft = _num(row[f"{col}_tok_s"]), _num(row[f"{col}_ttft_s"])
        m_tok = m_ttft = None
        note = ""
        if backend in runs:
            m_tok, m_ttft, _ = measured_from_run(runs[backend], backend, args.output_len)
            note = "no usable samples in run" if m_tok is None else ""
        elif p_tok is None:
            note = f"paper: {row[f'{col}_tok_s']} (nothing to run)"
        else:
            note = "not run"
        if p_tok is not None and m_tok is not None:
            note = f"measured/paper = {m_tok / p_tok:.2f}x"
        labels.append(label)
        paper_tok.append(p_tok)
        paper_ttft.append(p_ttft)
        meas_tok.append(m_tok)
        meas_ttft.append(m_ttft)
        notes.append(note)
        print(f"{label:<12} {p_tok if p_tok is not None else row[f'{col}_tok_s']:>12} "
              f"{f'{m_tok:.2f}' if m_tok else '-':>10} "
              f"{f'{m_tok / p_tok:.2f}' if (m_tok and p_tok) else '-':>7}   {note}")

    # HELM-vs-baseline ratio check: host-independent form of the speedup claim.
    if meas_tok[0]:
        for i, (backend, label) in enumerate(BACKENDS[1:], start=1):
            if meas_tok[i] and paper_tok[i] and paper_tok[0]:
                print(f"\nHELM vs {label}: measured {meas_tok[0] / meas_tok[i]:.1f}x on this host "
                      f"(paper: {paper_tok[0] / paper_tok[i]:.1f}x)")

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), dpi=160)
    fig.suptitle(f"{args.gpu} / {args.model} — paper vs. this machine "
                 f"(output {args.output_len}, batch=1)", color=INK, fontsize=11)
    panels = [
        (axes[0], "Decode throughput (tok/s)", paper_tok, meas_tok, "{:.1f}"),
        (axes[1], "TTFT p50 (s)", paper_ttft, meas_ttft, "{:.2f}"),
    ]
    x = range(len(labels))
    width = 0.36
    for ax, title, paper_vals, meas_vals, fmt in panels:
        for xi, (pv, mv) in enumerate(zip(paper_vals, meas_vals)):
            if pv is not None:
                ax.bar(xi - width / 2, pv, width, color=COLOR_PAPER, edgecolor="white", linewidth=1)
                ax.annotate(fmt.format(pv), (xi - width / 2, pv), ha="center", va="bottom",
                            fontsize=8, color=INK)
            else:
                cell = row[f"{BACKENDS[xi][0]}_tok_s"]
                token = FAILURE_LABELS.get(cell, cell)
                ax.annotate(token, (xi - width / 2, 0), ha="center", va="bottom",
                            fontsize=8, color=INK_MUTED, rotation=90, xytext=(0, 4),
                            textcoords="offset points")
            if mv is not None:
                ax.bar(xi + width / 2, mv, width, color=COLOR_MEASURED, edgecolor="white", linewidth=1)
                ax.annotate(fmt.format(mv), (xi + width / 2, mv), ha="center", va="bottom",
                            fontsize=8, color=INK)
            else:
                ax.annotate("not run", (xi + width / 2, 0), ha="center", va="bottom",
                            fontsize=8, color=INK_MUTED, rotation=90, xytext=(0, 4),
                            textcoords="offset points")
        ax.set_title(title, color=INK, fontsize=10)
        ax.set_xlim(-0.7, len(labels) - 1 + 0.7)
        ax.set_xticks(list(x), labels, color=INK, fontsize=9)
        ax.tick_params(colors=INK_MUTED, labelsize=8)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(GRID)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR_PAPER),
               plt.Rectangle((0, 0), 1, 1, color=COLOR_MEASURED)]
    fig.legend(handles, ["Paper (Table II)", "This machine"], loc="lower center",
               ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))

    out = args.out or f"results/comparison_{args.gpu.lower().replace(' ', '')}_{args.model.lower()}.png"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
