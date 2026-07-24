#!/usr/bin/env python3
"""Dump the cost-model partition curve.

Explains why the planner picks a given partition (e.g. the large Qwen3-32B auto
plan stage0@cpu(43u)): the roofline cost model scores every feasible CPU/GPU
split and the selected k minimizes decode latency. This script captures the FULL
curve the planner already computes, showing:

  * decode latency vs CPU-unit count, with the minimum at the selected k, and
  * prefill latency (TTFT) rising as more layers move to the CPU stage,

i.e. the decode-vs-TTFT tradeoff behind the chosen split.

Implementation: we wrap StrategySelector._log_selection (which already receives
every feasible candidate during select()) and serialize each candidate. No
planner logic is duplicated; this is the same search the compiler runs.

Only FEASIBLE candidates appear (the planner excludes splits that OOM, recording
the reason). Among feasible plans, the selected k is the decode-latency minimum.

Run on the GPU host so device_profiler measures the real GPU/CPU/PCIe.

Usage:
    python experiments/plan_curve.py --model Qwen/Qwen3-32B \
        --out experiments/results/plan_curve_32b.json
    # then inspect the printed table / JSON.

Deps: torch, transformers, helm (same env as paper_bench.py).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helm.compiler.compiler import HelmCompileOptions, compile_graph  # noqa: E402
from helm.compiler.importers.decode_tracer import DecodeTracer  # noqa: E402
from helm.compiler.optimization.strategy_selector import StrategySelector  # noqa: E402

_CAPTURED: list[dict] = []


def _install_capture() -> None:
    """Wrap _log_selection so every feasible candidate is recorded."""
    orig = StrategySelector._log_selection

    def patched(self, candidates, best_plan, best_cost):
        for plan, cost in candidates:
            cpu_units = sum(len(s.units) for s in plan.stages if "cpu" in s.device_id)
            gpu_units = sum(len(s.units) for s in plan.stages if "cpu" not in s.device_id)
            descs = ", ".join(
                f"stage{s.stage_id}@{s.device_id}({len(s.units)}u)" for s in plan.stages
            )
            _CAPTURED.append(
                {
                    "cpu_units": cpu_units,
                    "gpu_units": gpu_units,
                    "decode_ms": cost.decode_token_latency_s * 1000.0,
                    "prefill_ms": cost.prefill_latency_s * 1000.0,
                    "total_ms": cost.total_latency_s * 1000.0,
                    "throughput_tok_s": getattr(cost, "throughput_tokens_per_s", None),
                    "max_stage_mem_mb": cost.max_stage_memory_bytes / 1e6,
                    "is_best": plan is best_plan,
                    "stage_plan": descs,
                }
            )
        return orig(self, candidates, best_plan, best_cost)

    StrategySelector._log_selection = patched


def _load_and_trace(model_name: str, dtype: torch.dtype):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, low_cpu_mem_usage=True
    ).eval()
    dummy = DecodeTracer.build_dummy_inputs(device="cpu", batch_size=1, dtype=torch.float16)
    gm = DecodeTracer(model).trace()
    return model, gm, dummy


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-32B")
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--objective", default="decode_latency",
                   choices=["decode_latency", "prefill_latency", "total_latency", "throughput"])
    p.add_argument("--kv-offload", action="store_true")
    p.add_argument("--out", default="experiments/results/plan_curve.json")
    args = p.parse_args()

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    _install_capture()
    print(f"[plan-curve] loading + tracing {args.model} ...", flush=True)
    model, gm, dummy = _load_and_trace(args.model, dtype)

    opts = HelmCompileOptions(
        mode="decode",
        objective=args.objective,
        plan_mode="auto",
        allow_cpu=True,
        allow_gpu=True,
        lower_stages=False,            # plan only; no FX lowering needed for the curve
        graph_kind="decode",
        model_name=args.model,
        kv_offload=args.kv_offload,
        allow_static_analysis_fallback=True,
    )
    print("[plan-curve] running auto partition search ...", flush=True)
    compile_graph(gm=gm, example_inputs=dummy, model=model, options=opts)

    if not _CAPTURED:
        print("[plan-curve] NO candidates captured. Did the planner run? "
              "Check that plan_mode=auto and at least one feasible split exists.", file=sys.stderr)
        sys.exit(1)

    rows = sorted(_CAPTURED, key=lambda r: r["cpu_units"])
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"model": args.model, "objective": args.objective, "kv_offload": args.kv_offload,
         "candidates": rows}, indent=2))

    best = next((r for r in rows if r["is_best"]), None)
    print(f"\n=== Partition curve: {args.model} (objective={args.objective}) ===")
    hdr = f"{'cpu_u':>6} {'gpu_u':>6} | {'decode_ms':>9} | {'TTFT_ms':>9} | {'mem_MB':>8} | best"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        mark = "  <==" if r["is_best"] else ""
        print(f"{r['cpu_units']:>6} {r['gpu_units']:>6} | {r['decode_ms']:>9.1f} | "
              f"{r['prefill_ms']:>9.0f} | {r['max_stage_mem_mb']:>8.0f} |{mark}")

    if best:
        decodes = [r["decode_ms"] for r in rows]
        print(f"\nSelected k = cpu({best['cpu_units']}u): decode minimum "
              f"({best['decode_ms']:.1f} ms) across {len(rows)} feasible splits "
              f"[decode range {min(decodes):.1f}-{max(decodes):.1f} ms].")
        print(f"TTFT at selected k = {best['prefill_ms']:.0f} ms. Note TTFT rises with cpu_units: "
              f"this is the decode-vs-TTFT tradeoff the planner accepts to minimize per-token latency.")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
