#!/usr/bin/env python3
"""Summarize a TTFT-vs-input-length sweep (paper Fig. 4).

Walks an output root produced by experiments/ttft_prompt_scaling.sh, reads every
paper_results.json, and prints a per-backend TTFT-vs-input-length table plus a
flatness verdict: does TTFT stay flat as the prompt grows on the large CPU
partition? The verdict states the measured spread; it never reports "flat"
unless the numbers support it.

Usage:
    python experiments/ttft_prompt_scaling_summarize.py <OUT_ROOT>

Stdlib only. Safe to run anywhere the JSONs are present.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def collect(out_root: Path):
    """Return rows: list of dicts with backend, input_len, ttft_p50/p95, plan, ..."""
    rows = []
    for results in sorted(out_root.rglob("paper_results.json")):
        try:
            d = json.loads(results.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  [skip] {results}: {exc}", file=sys.stderr)
            continue
        cfg = d.get("config", {})
        input_len = cfg.get("input_len")
        sweep = d.get("latency_sweep", {})
        for backend, by_outlen in sweep.items():
            if not isinstance(by_outlen, dict):
                continue
            for outlen, s in by_outlen.items():
                if not isinstance(s, dict):
                    continue
                rows.append(
                    {
                        "backend": backend,
                        "input_len": input_len if input_len is not None else cfg.get("input_len"),
                        "output_len": _num(outlen),
                        "ttft_p50": _num(s.get("ttft_p50")),
                        "ttft_p95": _num(s.get("ttft_p95")),
                        "decode_lat_p50": _num(s.get("decode_lat_p50")),
                        "decode_tok_s": _num(s.get("decode_tok_per_s_mean")),
                        "peak_gpu_mb": _num(s.get("peak_gpu_mb_mean")),
                        "peak_cpu_mb": _num(s.get("peak_cpu_mb_mean")),
                        "n_success": s.get("n_success"),
                        "stage_plan": s.get("stage_plan", "?"),
                        "path": str(results),
                    }
                )
    return rows


def fmt(v, suffix="", nd=1):
    if v is None:
        return "?"
    return f"{v:.{nd}f}{suffix}"


def report(rows):
    if not rows:
        print("No paper_results.json files found. Did the sweep run?")
        return
    backends = sorted({r["backend"] for r in rows})
    for backend in backends:
        brows = sorted(
            (r for r in rows if r["backend"] == backend),
            key=lambda r: (r["input_len"] is None, r["input_len"] or 0),
        )
        plans = {r["stage_plan"] for r in brows}
        print(f"\n=== {backend} ===")
        print(f"stage plan(s): {', '.join(sorted(plans))}")
        if len(plans) > 1:
            print("  WARNING: stage plan changed across input lengths; "
                  "the study needs ONE fixed large partition. Investigate.")

        header = f"{'input_len':>9} | {'TTFT p50':>10} | {'TTFT p95':>10} | {'decode p50':>11} | {'tok/s':>6} | {'n_ok':>4}"
        print(header)
        print("-" * len(header))
        ttfts = []
        for r in brows:
            if r["ttft_p50"] is not None:
                ttfts.append(r["ttft_p50"])
            print(
                f"{str(r['input_len']):>9} | "
                f"{fmt(r['ttft_p50'],' ms'):>10} | "
                f"{fmt(r['ttft_p95'],' ms'):>10} | "
                f"{fmt(r['decode_lat_p50'],' ms'):>11} | "
                f"{fmt(r['decode_tok_s'],'',2):>6} | "
                f"{str(r['n_success']):>4}"
            )

        # Flatness verdict.
        if len(ttfts) >= 2:
            lo, hi = min(ttfts), max(ttfts)
            spread = (hi - lo) / lo * 100.0 if lo else float("inf")
            verdict = "FLAT" if spread < 10 else ("MILD SLOPE" if spread < 50 else "SCALES WITH INPUT")
            print(f"\n  TTFT spread across input lengths: "
                  f"min {lo:.0f} ms -> max {hi:.0f} ms = {spread:.1f}%  => {verdict}")
            if spread >= 10:
                print("  NOTE: TTFT is NOT flat on this partition; it scales with prompt length.")

    # Paste-ready LaTeX-ish table for HELM (the partition under question).
    helm = sorted(
        (r for r in rows if r["backend"] == "helm" and r["ttft_p50"] is not None),
        key=lambda r: r["input_len"] or 0,
    )
    if helm:
        print("\n--- paste-ready (HELM TTFT vs input length) ---")
        print("Input & TTFT p50 (ms) \\\\")
        for r in helm:
            print(f"{r['input_len']} & {r['ttft_p50']:.0f} \\\\")


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    out_root = Path(sys.argv[1]).expanduser()
    if not out_root.exists():
        print(f"OUT_ROOT does not exist: {out_root}", file=sys.stderr)
        sys.exit(1)
    report(collect(out_root))


if __name__ == "__main__":
    main()
