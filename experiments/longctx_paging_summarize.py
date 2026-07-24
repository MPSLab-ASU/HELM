#!/usr/bin/env python3
"""Summarize a paging-active long-context sweep (paper Fig. 6).

Reads ctx_*/paper_results.json from a longctx_paging run and prints a
throughput-and-paging-vs-context table, plus where paging first activates and
whether throughput degrades once KV streams over PCIe.

Stdlib only.

Usage:
    python experiments/longctx_paging_summarize.py <OUT_ROOT>
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
    rows = []
    for results in sorted(out_root.rglob("paper_results.json")):
        try:
            d = json.loads(results.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        cfg = d.get("config", {})
        ctx = cfg.get("input_len")
        helm = d.get("latency_sweep", {}).get("helm", {})
        for outlen, s in helm.items():
            if not isinstance(s, dict):
                continue
            rows.append({
                "ctx": ctx,
                "actual_input": _num(s.get("input_tokens_p50") or s.get("input_tokens_mean")),
                "tok_s": _num(s.get("decode_tok_per_s_mean")),
                "ttft_ms": _num(s.get("ttft_p50")),
                "peak_gpu_mb": _num(s.get("peak_gpu_mb_mean")),
                "peak_cpu_mb": _num(s.get("peak_cpu_mb_mean")),
                "plan": s.get("stage_plan", "?"),
                "evict_calls": s.get("kv_evict_calls", 0),
                "pages_evicted": s.get("kv_pages_evicted", 0),
                "bytes_evicted": s.get("kv_bytes_evicted", 0),
                "prefetch_calls": s.get("kv_prefetch_calls", 0),
                "pages_prefetched": s.get("kv_pages_prefetched", 0),
                "bytes_prefetched": s.get("kv_bytes_prefetched", 0),
            })
    return sorted(rows, key=lambda r: (r["ctx"] is None, r["ctx"] or 0))


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    out_root = Path(sys.argv[1]).expanduser()
    rows = collect(out_root)
    if not rows:
        print("No paper_results.json found. Did the sweep run?")
        return

    print(f"\n=== Long-context PAGING sweep ===  plan: {rows[0]['plan']}")
    hdr = (f"{'context':>8} | {'actual':>8} | {'tok/s':>7} | {'TTFT ms':>8} | "
           f"{'GPU MB':>7} | {'CPU MB':>7} | {'evict_calls':>11} | "
           f"{'evicted':>8} | {'MB evicted':>10} | {'prefetch_calls':>14} | "
           f"{'prefetched':>10} | {'MB streamed':>11}")
    print(hdr)
    print("-" * len(hdr))
    first_paging = None
    for r in rows:
        mb_prefetched = (r["bytes_prefetched"] or 0) / 1e6
        mb_evicted = (r["bytes_evicted"] or 0) / 1e6
        if first_paging is None and (r["pages_prefetched"] or 0) > 0:
            first_paging = r["ctx"]
        tok_s_str = "?" if r["tok_s"] is None else "{:.2f}".format(r["tok_s"])
        ttft_str = "?" if r["ttft_ms"] is None else "{:.0f}".format(r["ttft_ms"])
        actual_str = "?" if r["actual_input"] is None else "{:.0f}".format(r["actual_input"])
        gpu_str = "?" if r["peak_gpu_mb"] is None else "{:.0f}".format(r["peak_gpu_mb"])
        cpu_str = "?" if r["peak_cpu_mb"] is None else "{:.0f}".format(r["peak_cpu_mb"])
        print(
            "{:>8} | {:>8} | {:>7} | {:>8} | {:>7} | {:>7} | {:>11} | "
            "{:>8} | {:>10.1f} | {:>14} | {:>10} | {:>11.1f}".format(
                str(r["ctx"]), actual_str, tok_s_str, ttft_str,
                gpu_str, cpu_str, str(r["evict_calls"]),
                str(r["pages_evicted"]), mb_evicted, str(r["prefetch_calls"]),
                str(r["pages_prefetched"]), mb_prefetched,
            )
        )

    # ── Guard: did the context axis actually take effect? ────────────────────
    # If the measured prefill length does not track the requested context, the
    # sweep ran the same (short) prompt at every point: flat tok/s, flat peak
    # GPU, prefetched=evicted=0. That is a HARNESS no-op, NOT a "KV fits on GPU"
    # result, and must never be reported as a long-context measurement.
    measured = [r for r in rows if r["actual_input"] is not None
                and r["ctx"] is not None]
    axis_ok = True
    if len(measured) >= 2:
        max_ctx = max(r["ctx"] for r in measured)
        max_actual = max(r["actual_input"] for r in measured)
        min_actual = min(r["actual_input"] for r in measured)
        # Real sweep: longest prefill should track the largest requested ctx and
        # actual lengths should span a range. Allow tokenizer slack (>=70%).
        if max_ctx > 0 and (max_actual < 0.7 * max_ctx
                            or (max_actual - min_actual) < 0.25 * max_ctx):
            axis_ok = False
            print()
            print("!" * 72)
            print("WARNING: context axis appears to be a NO-OP.")
            print(f"  requested max context = {max_ctx}, but measured prefill "
                  f"ranged {min_actual:.0f}..{max_actual:.0f} tokens.")
            print("  The prompt was likely only truncated, never padded, so every")
            print("  context ran the same short prompt. Flat tok/s and "
                  "prefetched=evicted=0")
            print("  here mean NOTHING about paging. DO NOT report these as a "
                  "long-context")
            print("  result. Re-run paper_bench with --pad-to-input-len "
                  "(and, to force the")
            print("  paging regime, HELM_GPU_KV_WATERMARK_MB=<MB below context "
                  "footprint>).")
            print("!" * 72)

    paged = [r for r in rows if (r["pages_prefetched"] or 0) > 0 and r["tok_s"] is not None]
    nonpaged = [r for r in rows if (r["pages_prefetched"] or 0) == 0 and r["tok_s"] is not None]
    print()
    if not axis_ok:
        print("Paging status: INCONCLUSIVE — context axis was a no-op (see warning above).")
    elif first_paging is not None:
        print(f"Paging first activates at context = {first_paging} "
              f"(pages_prefetched > 0).")
    else:
        print("Paging never activated: KV still fit on GPU at all tested contexts. "
              "Increase context or use a model with less GPU headroom.")
    if paged and nonpaged:
        base = max(r["tok_s"] for r in nonpaged)
        worst = min(r["tok_s"] for r in paged)
        drop = (base - worst) / base * 100 if base else 0
        print(f"Throughput in paging regime: {worst:.2f} tok/s vs {base:.2f} GPU-resident "
              f"=> {drop:.1f}% drop (the PCIe KV-streaming cost).")
    print("\nInterpretation: past the GPU KV watermark W, each decode step "
          "streams (context - W) tokens of KV per layer over PCIe; prefetch volume grows "
          "linearly with context, and throughput degrades once that PCIe traffic exceeds "
          "the per-step weight-read bandwidth.")


if __name__ == "__main__":
    main()
