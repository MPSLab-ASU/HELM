#!/usr/bin/env python3
"""
Validate the canonical paper results shipped in results/paper/ and compare
fresh benchmark runs against them.

Modes
-----
1. Self-check (default, no arguments, stdlib only, no GPU needed):

       python results/validate_results.py

   Recomputes every derived number in the paper from the committed raw
   tables and asserts the headline claims: per-row speedups, the 6.2x
   geometric-mean decode-throughput improvement over the nine overflow
   configurations, the 5.9x geometric-mean / 256x max context extension,
   monotonicity of the paging and page-size sweeps, and the llama.cpp
   comparison. Exits non-zero on any inconsistency.

2. Render tables:

       python results/validate_results.py --print

   Pretty-prints Tables II-IV and the figure data as Markdown.

3. Compare a fresh run (artifact evaluators: run this after
   experiments/paper_bench.py or experiments/run_paper_experiments.sh):

       python results/validate_results.py --check-run OUT/paper_results.json \
           --gpu "RTX 4060" --model Qwen3-4B [--backend helm] [--tolerance 0.25]

   Extracts p50 decode throughput and TTFT from the run's latency sweep and
   checks them against the corresponding Table II cell. The default +/-25%
   tolerance absorbs host-hardware variation (decode throughput on overflow
   models tracks the CPU of the machine, see paper Section VI-E); it can be
   tightened on hardware matching the paper's testbeds.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parent / "paper"

# The nine overflow configurations of the paper (weights exceed VRAM and at
# least one baseline is feasible); Section VI reports a 6.2x geometric-mean
# HELM speedup over the strongest feasible baseline across exactly these.
OVERFLOW_CONFIGS = [
    ("RTX 4060", "Qwen3-4B"),
    ("RTX 4060", "Qwen3-8B"),
    ("RTX 3090", "Qwen3-14B"),
    ("RTX 3090", "Qwen3-32B"),
    ("RTX 3090", "Llama-2-13B"),
    ("RTX 3090", "OLMo-2-13B"),
    ("RTX 3090", "Gemma-2-27B"),
    ("L40S", "Qwen3-32B"),
    ("L40S", "Gemma-2-27B"),
]

EXPECTED_THROUGHPUT_GEOMEAN = 6.2
EXPECTED_CONTEXT_GEOMEAN = 5.9
EXPECTED_CONTEXT_MAX = 256.0

FAILURE_TOKENS = {"OOM", "FAIL_INIT", "FOOTPRINT", "KV_OOM", "WEIGHT_OOM", "NOT_REPORTED"}

_errors = []


def check(ok, message):
    tag = "ok  " if ok else "FAIL"
    print(f"[{tag}] {message}")
    if not ok:
        _errors.append(message)


def _num(cell):
    try:
        return float(cell)
    except (TypeError, ValueError):
        return None


def geomean(values):
    return math.exp(sum(math.log(v) for v in values) / len(values))


def load_table2():
    with open(PAPER_DIR / "table2_decode_throughput.csv") as fh:
        return list(csv.DictReader(fh))


def table2_row(rows, gpu, model):
    for row in rows:
        if row["gpu"].lower() == gpu.lower() and row["model"].lower() == model.lower():
            return row
    return None


# ──────────────────────────────────────────────────────────────────────────
# Self-check
# ──────────────────────────────────────────────────────────────────────────

def self_check():
    rows = load_table2()
    check(len(rows) == 22, f"Table II has 22 model x GPU rows (found {len(rows)})")

    # Per-row: HELM has the highest feasible throughput, and the printed
    # speedup is consistent with the one-decimal rounding of each cell
    # (the paper computes it from unrounded measurements, so we bound the
    # claimed value by the interval the rounded cells allow).
    for row in rows:
        helm = _num(row["helm_tok_s"])
        claimed = _num(row["helm_vs_best_baseline"])
        baselines = {
            "Accelerate": _num(row["accelerate_tok_s"]),
            "DeepSpeed": _num(row["deepspeed_tok_s"]),
            "vLLM": _num(row["vllm_tok_s"]),
        }
        feasible = {k: v for k, v in baselines.items() if v is not None}
        label = f'{row["gpu"]} / {row["model"]}'
        check(helm is not None, f"{label}: HELM ran (no OOM/init failure)")
        check(bool(feasible), f"{label}: at least one baseline is feasible")
        if helm is None or not feasible:
            continue
        best = max(feasible.values())
        check(helm >= best * 0.99, f"{label}: HELM ({helm}) is fastest feasible backend (best baseline {best})")
        # Both the throughput cells and the printed speedup are rounded to one
        # decimal (the paper computes the ratio from unrounded measurements),
        # so bound the claim by the interval the rounded cells allow, widened
        # by the claim's own +/-0.05 quantization.
        lo = (helm - 0.05) / (best + 0.05) - 0.05
        hi = (helm + 0.05) / max(best - 0.05, 1e-9) + 0.05
        check(
            lo - 1e-9 <= claimed <= hi + 1e-9,
            f"{label}: claimed speedup {claimed}x within rounding bounds [{lo:.2f}, {hi:.2f}]",
        )

    # Headline: 6.2x geometric mean over the nine overflow configurations.
    speedups = []
    for gpu, model in OVERFLOW_CONFIGS:
        row = table2_row(rows, gpu, model)
        check(row is not None, f"overflow config present in Table II: {gpu} / {model}")
        if row:
            speedups.append(_num(row["helm_vs_best_baseline"]))
    gm = geomean(speedups)
    check(
        abs(gm - EXPECTED_THROUGHPUT_GEOMEAN) < 0.05,
        f"geomean speedup over {len(speedups)} overflow configs = {gm:.2f}x (paper: {EXPECTED_THROUGHPUT_GEOMEAN}x)",
    )
    check(
        max(speedups) == 10.4,
        f"max speedup = {max(speedups)}x (paper: 10.4x, Qwen3-32B on L40S)",
    )

    # In-VRAM rows dispatch to the vLLM fast path and match it (1.0x).
    for row in rows:
        if _num(row["helm_vs_best_baseline"]) == 1.0 and _num(row["vllm_tok_s"]) is not None:
            helm, vllm = _num(row["helm_tok_s"]), _num(row["vllm_tok_s"])
            check(
                abs(helm - vllm) <= 0.2,
                f'{row["gpu"]} / {row["model"]}: fast-path row matches vLLM ({helm} vs {vllm} tok/s)',
            )

    # Fig. 5: context-extension ratios. Re-derive each ratio from the raw
    # per-GPU grid (HELM max tokens / best numerically-reported baseline)
    # and require it to match the committed headline ratio, so a corrected
    # grid cell cannot silently coexist with a stale headline number.
    fig5 = json.loads((PAPER_DIR / "fig5_max_output_tokens.json").read_text())
    ratios = {k: v for k, v in fig5["headline_ratios"].items() if isinstance(v, (int, float))}
    for key, claimed_ratio in ratios.items():
        panel, model = key.split("_", 1)
        panel_key = {"rtx4060": "rtx4060_8gb", "rtx3090": "rtx3090_24gb", "l40s": "l40s_48gb"}[panel]
        cells = fig5[panel_key][model]
        helm_tokens = cells["HELM"]
        baselines = [v for k, v in cells.items() if k != "HELM" and isinstance(v, (int, float))]
        derived = helm_tokens / max(baselines) if baselines else None
        check(
            derived is not None and abs(derived - claimed_ratio) < 1e-9,
            f"Fig. 5 {key}: headline ratio {claimed_ratio}x matches grid ({helm_tokens} / best baseline -> {derived})",
        )
    gm_ctx = geomean(list(ratios.values()))
    check(
        abs(gm_ctx - EXPECTED_CONTEXT_GEOMEAN) < 0.05,
        f"geomean context extension over {len(ratios)} feasible Qwen3 pairs = {gm_ctx:.2f}x (paper: {EXPECTED_CONTEXT_GEOMEAN}x)",
    )
    check(
        max(ratios.values()) == EXPECTED_CONTEXT_MAX,
        f"max context extension = {max(ratios.values())}x (paper: 256x, Qwen3-14B on RTX 3090)",
    )
    grid = fig5["rtx3090_24gb"]["Qwen3-14B"]
    check(
        grid["HELM"] == 32768 and grid["Accelerate"] == 128,
        "Fig. 5 RTX 3090 / Qwen3-14B: HELM 32K vs Accelerate 128 (the 256x cell)",
    )

    # Fig. 6: decode throughput falls monotonically as paging traffic grows.
    fig6 = json.loads((PAPER_DIR / "fig6_paging_throughput.json").read_text())
    pts = sorted(((int(k), v) for k, v in fig6["decode_tok_s_by_context"].items()))
    values = [v for _, v in pts]
    check(
        all(a > b for a, b in zip(values, values[1:])),
        f"Fig. 6 throughput decreases monotonically with context: {values}",
    )

    # Fig. 4: TTFT grows with prompt length on a CPU-heavy partition.
    fig4 = json.loads((PAPER_DIR / "fig4_ttft_prompt_scaling.json").read_text())
    t128, t4096 = fig4["ttft_p50_s"]["128"], fig4["ttft_p50_s"]["4096"]
    check(
        abs(t4096 / t128 - 20.8) < 0.1,
        f"Fig. 4 TTFT scaling 128->4096 tokens = {t4096 / t128:.1f}x (paper: 20.8x)",
    )

    # Table III: automatic HELM beats every hand-tuned fp16 llama.cpp split.
    with open(PAPER_DIR / "table3_llamacpp_comparison.csv") as fh:
        t3 = list(csv.DictReader(fh))
    helm_fp16 = next(_num(r["decode_tok_s"]) for r in t3 if r["config"].startswith("HELM"))
    best_lcpp_fp16 = max(
        _num(r["decode_tok_s"]) for r in t3 if r["precision"] == "fp16" and not r["config"].startswith("HELM")
    )
    check(
        helm_fp16 > best_lcpp_fp16,
        f"Table III: HELM auto fp16 ({helm_fp16} tok/s) beats best manual llama.cpp fp16 ({best_lcpp_fp16} tok/s)",
    )

    # Table IV: within the paging regime, larger pages are never slower.
    with open(PAPER_DIR / "table4_kv_page_size.csv") as fh:
        t4 = list(csv.DictReader(fh))
    for row in t4:
        series = [_num(row[c]) for c in ("p64", "p128", "p256", "p512", "p1024")]
        series = [v for v in series if v is not None]
        check(
            all(a <= b for a, b in zip(series, series[1:])),
            f'Table IV {row["gpu"]} / {row["model"]}: throughput rises monotonically with page size {series}',
        )

    return summarize()


def summarize():
    print()
    if _errors:
        print(f"SELF-CHECK FAILED: {len(_errors)} inconsistencies")
        for e in _errors:
            print(f"  - {e}")
        return 1
    print("SELF-CHECK PASSED: committed paper results are internally consistent")
    return 0


# ──────────────────────────────────────────────────────────────────────────
# Fresh-run comparison
# ──────────────────────────────────────────────────────────────────────────

def measured_from_run(path, backend, output_len):
    """Extract (tok_s, ttft_s, entry) from a paper_bench paper_results.json.

    Table II reports p50-derived decode throughput (1000 / decode_lat_p50 ms)
    and TTFT p50 in seconds (the JSON stores milliseconds). Returns
    (None, None, entry) when the run has no usable successful samples.
    """
    data = json.loads(Path(path).read_text())
    sweep = (data.get("latency_sweep") or {}).get(backend) or {}
    entry = sweep.get(str(output_len)) or {}
    if not entry or not entry.get("n_success"):
        return None, None, entry
    decode_lat_p50 = entry.get("decode_lat_p50") or 0.0
    if decode_lat_p50 <= 0 or entry.get("ttft_p50") is None:
        return None, None, entry
    return 1000.0 / decode_lat_p50, entry["ttft_p50"] / 1000.0, entry


def check_run(args):
    rows = load_table2()
    row = table2_row(rows, args.gpu, args.model)
    if row is None:
        print(f"error: no Table II row for gpu={args.gpu!r} model={args.model!r}")
        print("       valid GPUs:  " + ", ".join(sorted({r['gpu'] for r in rows})))
        print("       valid models:" + ", ".join(sorted({r['model'] for r in rows})))
        return 2

    measured_tok_s, measured_ttft_s, entry = measured_from_run(args.check_run, args.backend, args.output_len)
    if measured_tok_s is None:
        if not entry or not entry.get("n_success"):
            print(f"error: no successful {args.backend} entries for output length {args.output_len} in {args.check_run}")
        else:
            print(f"error: entry has no usable decode_lat_p50/ttft_p50 (decode_lat_p50={entry.get('decode_lat_p50')!r}); "
                  "the run produced successes but no per-token latency samples")
        return 2

    col = "helm" if args.backend in ("helm", "helm-router") else args.backend
    expected_tok_s = _num(row[f"{col}_tok_s"])
    expected_ttft_s = _num(row[f"{col}_ttft_s"])

    # In-VRAM rows (speedup column 1.0) report HELM's vLLM fast path: the
    # helm-router backend dispatches all-GPU plans to vLLM (paper Section IV).
    # A plain `helm` backend run reads far lower there by design.
    if args.backend == "helm" and _num(row["helm_vs_best_baseline"]) == 1.0:
        print("note: this is an in-VRAM fast-path row - the paper's HELM column equals the")
        print("      vLLM fast path (paper_bench backend `helm-router`). If this check fails,")
        print("      re-run the cell with BACKENDS_OVERRIDE=helm-router (or compare --backend vllm);")
        print("      plain `helm` bypasses the router dispatch and reads lower by design.")
    if expected_tok_s is None:
        print(f"error: the paper reports no successful {args.backend} run for this configuration "
              f"(cell = {row.get(args.backend + '_tok_s', 'n/a')})")
        return 2

    tol = args.tolerance
    ok_tok = abs(measured_tok_s - expected_tok_s) <= tol * expected_tok_s
    check(ok_tok,
          f"decode throughput {measured_tok_s:.2f} tok/s within +/-{tol * 100:.0f}% of paper {expected_tok_s} tok/s "
          f"({args.gpu} / {args.model} / {args.backend})")
    if expected_ttft_s is not None:
        ok_ttft = abs(measured_ttft_s - expected_ttft_s) <= tol * expected_ttft_s
        check(ok_ttft,
              f"TTFT p50 {measured_ttft_s:.3f} s within +/-{tol * 100:.0f}% of paper {expected_ttft_s} s")
    else:
        print(f"note: paper reports no TTFT for this cell; TTFT comparison skipped")

    print()
    if _errors:
        print("RUN CHECK FAILED - measured numbers deviate from the paper beyond tolerance.")
        print("Note: absolute throughput on overflow models tracks the host CPU/DRAM of the")
        print("machine (paper Section VI-E); compare ratios across backends on YOUR hardware")
        print("before concluding a mismatch.")
        return 1
    print("RUN CHECK PASSED: measurements consistent with the paper within tolerance")
    return 0


# ──────────────────────────────────────────────────────────────────────────
# Markdown rendering
# ──────────────────────────────────────────────────────────────────────────

def print_tables():
    rows = load_table2()
    print("## Table II - decode throughput (tok/s) / TTFT p50 (s), output length 128, batch=1\n")
    print("| GPU | Model | HELM | Accelerate | DeepSpeed | vLLM | HELM vs best |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        def cell(prefix):
            tok, ttft = r[f"{prefix}_tok_s"], r[f"{prefix}_ttft_s"]
            return f"{tok} / {ttft}" if _num(tok) is not None else tok
        print(f'| {r["gpu"]} | {r["model"]} | {cell("helm")} | {cell("accelerate")} '
              f'| {cell("deepspeed")} | {cell("vllm")} | {r["helm_vs_best_baseline"]}x |')

    for name, title in [
        ("table3_llamacpp_comparison.csv", "Table III - HELM vs llama.cpp (Llama-2-13B, RTX 3090, in=out=128)"),
        ("table4_kv_page_size.csv", "Table IV - decode throughput (tok/s) vs KV page size, paging active"),
    ]:
        with open(PAPER_DIR / name) as fh:
            reader = csv.reader(fh)
            header = next(reader)
            print(f"\n## {title}\n")
            print("| " + " | ".join(header) + " |")
            print("|" + "---|" * len(header))
            for row in reader:
                print("| " + " | ".join(row) + " |")

    for name in ("fig4_ttft_prompt_scaling.json", "fig5_max_output_tokens.json", "fig6_paging_throughput.json"):
        data = json.loads((PAPER_DIR / name).read_text())
        print(f"\n## {name}\n")
        print("```json\n" + json.dumps(data, indent=2) + "\n```")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--print", dest="print_tables", action="store_true",
                   help="render the canonical tables/figures as Markdown and exit")
    p.add_argument("--check-run", metavar="PAPER_RESULTS_JSON",
                   help="path to a fresh experiments/paper_bench.py paper_results.json to compare")
    p.add_argument("--gpu", help="Table II GPU name for --check-run (RTX 4060 | RTX 3090 | L40S)")
    p.add_argument("--model", help="Table II model name for --check-run (e.g. Qwen3-8B)")
    p.add_argument("--backend", default="helm", help="backend column to compare (default: helm)")
    p.add_argument("--output-len", type=int, default=128, help="latency-sweep output length (default: 128)")
    p.add_argument("--tolerance", type=float, default=0.25,
                   help="relative tolerance for --check-run (default: 0.25)")
    args = p.parse_args()

    if args.print_tables:
        return print_tables()
    if args.check_run:
        if not (args.gpu and args.model):
            p.error("--check-run requires --gpu and --model")
        return check_run(args)
    return self_check()


if __name__ == "__main__":
    sys.exit(main())
