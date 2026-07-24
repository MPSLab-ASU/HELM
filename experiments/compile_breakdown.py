#!/usr/bin/env python3
"""Break the HELM one-time compile cost into stages (paper §VI-D).

Measures hardware profiling, analysis, partition search, graph construction,
and kernel JIT separately. The first-run C++ kernel JIT is a one-time cost
(cached afterwards) and could be removed by shipping a precompiled binary.

Method: wrap the module-level phase functions in helm.compiler.compiler with
perf_counter timers (no core edits; we reassign module globals that compile_graph
resolves by name at call time), and time model load, FX trace, and the AVX2+F16C
kernel JIT (torch.utils.cpp_extension.load) at the driver level.

Stages reported:
  kernel_jit         AVX2+F16C extension load/compile (helm.kernels._load_ext)
  model_load         transformers from_pretrained
  fx_trace           DecodeTracer.trace
  ir_build           _build_helm_graph
  analysis           _run_analysis (HybridAnalyzer)
  partition_units    _build_partition_units
  device_profiling   profile_devices (subset of plan_total)
  partition_search   plan_total - device_profiling (StrategySelector.select)
  graph_construction lower_to_runtime_stages (StageFXBuilder)
  compile_total      whole compile_graph call

COLD vs WARM: the kernel JIT compiles once and caches under
~/.cache/torch_extensions. For a true COLD number, clear that dir first:
    rm -rf ~/.cache/torch_extensions   # then run with --cold-note
The script reports whatever cache state it finds and labels it.

Run on the GPU host (device_profiling needs the real GPU). Run for a large model
(Qwen3-32B) AND a mid model (NousResearch/Llama-2-13b-hf) so the breakdown covers
both regimes.

Usage:
    python experiments/compile_breakdown.py --model Qwen/Qwen3-32B \
        --out experiments/results/compile_breakdown_32b.json

Deps: torch, transformers, helm (same env as paper_bench.py).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import helm.compiler.compiler as C  # noqa: E402
from helm.compiler.compiler import HelmCompileOptions, compile_graph  # noqa: E402
from helm.compiler.importers.decode_tracer import DecodeTracer  # noqa: E402

_TIMINGS: dict[str, float] = {}


def _wrap_phase(attr: str, label: str) -> None:
    """Wrap a module-level function C.<attr> to accumulate wall time under label."""
    orig = getattr(C, attr)

    def timed(*a, **k):
        t0 = time.perf_counter()
        try:
            return orig(*a, **k)
        finally:
            _TIMINGS[label] = _TIMINGS.get(label, 0.0) + (time.perf_counter() - t0)

    setattr(C, attr, timed)


def _time(label: str, fn, *a, **k):
    t0 = time.perf_counter()
    r = fn(*a, **k)
    _TIMINGS[label] = _TIMINGS.get(label, 0.0) + (time.perf_counter() - t0)
    return r


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-32B")
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--kv-offload", action="store_true")
    p.add_argument("--out", default="experiments/results/compile_breakdown.json")
    args = p.parse_args()

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    # Cache state note for the kernel JIT.
    ext_cache = Path.home() / ".cache" / "torch_extensions"
    jit_state = "WARM (cache present)" if ext_cache.exists() else "COLD (no cache)"

    # 1) Kernel JIT: triggers torch.utils.cpp_extension.load (compile on cold, load on warm).
    try:
        import helm.kernels as K
        avail = _time("kernel_jit", K.is_available)
        kernel_note = f"available={avail}; err={K.load_error()}"
    except Exception as exc:  # never abort the breakdown on kernel issues
        kernel_note = f"kernel load raised: {exc}"

    # Wrap compile phases BEFORE compile_graph runs.
    _wrap_phase("_build_helm_graph", "ir_build")
    _wrap_phase("_run_analysis", "analysis")
    _wrap_phase("_build_partition_units", "partition_units")
    _wrap_phase("profile_devices", "device_profiling")
    _wrap_phase("build_partition_plan", "plan_total")  # includes device_profiling + search
    _wrap_phase("lower_to_runtime_stages", "graph_construction")

    # 2) Model load.
    from transformers import AutoModelForCausalLM
    print(f"[breakdown] loading {args.model} ...", flush=True)
    model = _time("model_load", lambda: AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, low_cpu_mem_usage=True).eval())

    # 3) FX trace (decode step).
    dummy = DecodeTracer.build_dummy_inputs(device="cpu", batch_size=1, dtype=torch.float16)
    print("[breakdown] tracing decode graph ...", flush=True)
    gm = _time("fx_trace", lambda: DecodeTracer(model).trace())

    # 4) compile_graph (includes wrapped sub-phases + graph construction).
    opts = HelmCompileOptions(
        mode="decode",
        objective="decode_latency",
        plan_mode="auto",
        allow_cpu=True,
        allow_gpu=True,
        lower_stages=True,             # include graph construction (StageFXBuilder)
        graph_kind="decode",
        model_name=args.model,
        kv_offload=args.kv_offload,
        allow_static_analysis_fallback=True,
    )
    print("[breakdown] compiling ...", flush=True)
    _time("compile_total", lambda: compile_graph(
        gm=gm, example_inputs=dummy, model=model, options=opts))

    # Derive partition search = plan_total - device_profiling.
    _TIMINGS["partition_search"] = max(
        0.0, _TIMINGS.get("plan_total", 0.0) - _TIMINGS.get("device_profiling", 0.0))

    # Report.
    order = [
        "kernel_jit", "model_load", "fx_trace", "ir_build", "analysis",
        "partition_units", "device_profiling", "partition_search",
        "graph_construction", "compile_total",
    ]
    warm_compile = _TIMINGS.get("compile_total", 0.0)
    out = {
        "model": args.model,
        "dtype": args.dtype,
        "kv_offload": args.kv_offload,
        "kernel_jit_state": jit_state,
        "kernel_note": kernel_note,
        "timings_s": {k: round(_TIMINGS.get(k, 0.0), 4) for k in order},
        "notes": {
            "compile_total": "compile_graph only; excludes model_load, fx_trace, kernel_jit",
            "partition_search": "plan_total - device_profiling",
            "warm_vs_cold": "kernel_jit is one-time; clear ~/.cache/torch_extensions for a COLD number",
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    print(f"\n=== Compile-time breakdown: {args.model} ===")
    print(f"kernel JIT cache state at start: {jit_state}  ({kernel_note})")
    print(f"{'stage':>20} | {'seconds':>9}")
    print("-" * 33)
    for k in order:
        print(f"{k:>20} | {_TIMINGS.get(k, 0.0):>9.3f}")
    print(f"\ncompile_graph total (warm, excl. model_load/trace/JIT): {warm_compile:.3f} s")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
