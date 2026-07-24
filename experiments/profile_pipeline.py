"""
HELM Pipeline Profiler
======================
Measures per-stage, per-operation, and per-token timing to identify
exactly where time is spent in the heterogeneous CPU+GPU pipeline.

Output sections:
  1. Per-stage breakdown (setup/move + forward)
  2. Prefill vs decode latency split
  3. Per-token timing over N decode steps (shows warmup effect)
  4. CPU stage sub-operation breakdown via torch.profiler
  5. PCIe transfer bandwidth (activation cross-stage)
  6. Memory snapshot (VRAM + RAM usage per stage)

Usage:
  uv run python experiments/profile_pipeline.py \
      --model Qwen/Qwen3-4B-Instruct-2507 \
      --decode-steps 20
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List

import torch

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _ram_gb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024**3
    except Exception:
        return 0.0


def _vram_gb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0.0


def _header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────────────────────
# Build runtime (reuse benchmark setup logic)
# ─────────────────────────────────────────────────────────────────────────────

def build_runtime(model_name: str, dtype_str: str = "float16", cpu_threads: int = 8):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import psutil

    from helm.compiler.compiler import HelmCompileOptions, compile_graph
    from helm.compiler.importers.decode_tracer import DecodeTracer
    from helm.compiler.importers.patch_fx import apply_fx_patch
    from helm.runtime.executor import StageRuntimeExecutor
    from helm.runtime.pipeline_runtime import PipelineRuntime
    from helm.runtime.kv_offload import KVOffloadManager, KVOffloadConfig
    apply_fx_patch()

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(dtype_str, torch.float16)

    torch.set_num_threads(cpu_threads)

    print(f"[profile] Loading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    gpu_total_mb = torch.cuda.get_device_properties(0).total_memory / 1024**2 if torch.cuda.is_available() else 0
    gpu_budget = f"{int(gpu_total_mb * 0.80)}MiB" if gpu_total_mb else None
    ram_total_mb = psutil.virtual_memory().total / 1024**2
    cpu_budget = f"{int(ram_total_mb - 2048)}MiB"
    max_memory = {"cpu": cpu_budget}
    if gpu_budget:
        max_memory[0] = gpu_budget

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map="auto",
        max_memory=max_memory,
        low_cpu_mem_usage=True,
        use_cache=False,
    )
    model.eval()

    try:
        from accelerate.hooks import remove_hook_from_submodules
        remove_hook_from_submodules(model)
    except ImportError:
        pass

    # Build dummy inputs
    prompt = "Explain how a compiler works."
    tok_out = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
    input_ids = tok_out["input_ids"]
    seq_len = input_ids.shape[1]

    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    cache_position = torch.arange(seq_len, dtype=torch.long)
    min_val = torch.finfo(torch.float16).min
    causal = torch.triu(torch.full((seq_len, seq_len), min_val, dtype=torch.float16), diagonal=1)
    inv_mask = (1.0 - tok_out["attention_mask"].float()) * min_val
    attn_mask = causal[None, None, :, :] + inv_mask[:, None, None, :]
    dummy_inputs = (input_ids, attn_mask, position_ids, cache_position)

    workload = {"batch_size": 1, "prefill_seq_len": seq_len, "decode_context_len": seq_len,
                "decode_tokens": 64, "dtype_size": 2}

    # Capture and compile graphs
    from experiments.dev_pipeline import capture_fx_graph, capture_decode_fx_graph

    print("[profile] Tracing prefill ...")
    prefill_gm = capture_fx_graph(
        model,
        dummy_inputs,
        run_dir=None,
        allow_fallback=True,
        trace_policy="explicit_leaf",
    )
    opts_common = dict(mode="both", objective="decode_latency", plan_mode="auto",
                       lower_stages=True, model_name=model_name, workload=workload)
    prefill_art = compile_graph(
        gm=prefill_gm, example_inputs=dummy_inputs, model=model, tokenizer=tokenizer,
        options=HelmCompileOptions(**opts_common, graph_kind="prefill"), artifacts_dir=None)

    print("[profile] Tracing decode ...")
    dec_dummy = DecodeTracer.build_dummy_inputs(device="cpu", batch_size=1, dtype=torch.float16)
    decode_gm = capture_decode_fx_graph(model, dec_dummy, run_dir=None)
    decode_wrapper = getattr(model, "_helm_decode_wrapper", None)
    decode_art = compile_graph(
        gm=decode_gm, example_inputs=dec_dummy, model=model, tokenizer=tokenizer,
        options=HelmCompileOptions(**opts_common, graph_kind="decode"),
        partition_plan_override=prefill_art.partition_plan, artifacts_dir=None)

    prefill_exec = StageRuntimeExecutor(prefill_art.stage_graphs)
    decode_exec = StageRuntimeExecutor(decode_art.stage_graphs)
    kv_cfg = KVOffloadConfig.from_model(model)
    kv_mgr = KVOffloadManager(model, kv_cfg, batch_size=1)
    kv_mgr.patch_module_roots(
        *(stage.module for stage in prefill_exec.stages),
        *(stage.module for stage in decode_exec.stages),
    )
    runtime = PipelineRuntime(
        prefill_exec, decode_exec,
        tokenizer=tokenizer, dtype=dtype,
        kv_offload_mgr=kv_mgr, decode_wrapper=decode_wrapper,
    )

    return runtime, tokenizer, input_ids, decode_exec


# ─────────────────────────────────────────────────────────────────────────────
# Section 1: Per-stage per-token timing
# ─────────────────────────────────────────────────────────────────────────────

def profile_per_stage(runtime, input_ids, decode_steps: int = 16):
    """
    Manually time each stage separately by intercepting StageRuntimeExecutor.run().
    Reports: stage_id, device, input→output transfer, forward pass, total.
    """
    import helm.runtime.executor as _exec_mod
    _exec_mod._PROFILE = True
    _exec_mod._call_count = 0

    _header("Per-Stage Per-Token Timing (first decode token)")

    runtime._reset_decode_cache()
    if runtime.kv_offload_mgr:
        runtime.kv_offload_mgr.reset()

    # Warm up: prefill
    _sync()
    t0 = time.perf_counter()
    logits = runtime.prefill(input_ids.clone())
    _sync()
    prefill_ms = (time.perf_counter() - t0) * 1000
    print(f"\n  Prefill: {prefill_ms:.1f} ms  (seq_len={input_ids.shape[1]})")

    print(f"\n  Decode steps (executor prints per-stage timing):")
    seq_len = input_ids.shape[1]
    for step in range(decode_steps):
        next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        _sync()
        logits = runtime.decode_step(next_tok, seq_len + step)
        _sync()
        if step >= 3:
            break  # first few tokens for warmup visibility

    _exec_mod._PROFILE = False


# ─────────────────────────────────────────────────────────────────────────────
# Section 2: Per-token timing over N steps
# ─────────────────────────────────────────────────────────────────────────────

def profile_per_token(runtime, input_ids, decode_steps: int = 20):
    _header(f"Per-Token Latency Over {decode_steps} Decode Steps")

    runtime._reset_decode_cache()
    if runtime.kv_offload_mgr:
        runtime.kv_offload_mgr.reset()

    _sync()
    t0 = time.perf_counter()
    logits = runtime.prefill(input_ids.clone())
    _sync()
    prefill_ms = (time.perf_counter() - t0) * 1000

    seq_len = input_ids.shape[1]
    token_times_ms = []

    for step in range(decode_steps):
        next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        _sync()
        t0 = time.perf_counter()
        logits = runtime.decode_step(next_tok, seq_len + step)
        _sync()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        token_times_ms.append(elapsed_ms)

    print(f"\n  Prefill latency   : {prefill_ms:.1f} ms")
    print(f"\n  {'Step':>4}  {'ms':>8}  {'tok/s':>8}")
    print(f"  {'-'*24}")
    for i, t in enumerate(token_times_ms):
        marker = " ← warmup" if i == 0 else ""
        print(f"  {i+1:>4}  {t:>8.1f}  {1000/t:>8.2f}{marker}")

    steady = token_times_ms[2:]  # skip first 2 for warmup
    avg_ms = sum(steady) / len(steady)
    print(f"\n  Steady-state avg  : {avg_ms:.1f} ms/token  ({1000/avg_ms:.2f} tok/s)")
    print(f"  Min               : {min(steady):.1f} ms")
    print(f"  Max               : {max(steady):.1f} ms")

    return token_times_ms


# ─────────────────────────────────────────────────────────────────────────────
# Section 3: Stage-by-stage timing with manual instrumentation
# ─────────────────────────────────────────────────────────────────────────────

def profile_stage_breakdown(decode_exec, input_ids, decode_steps: int = 5):
    """
    Patch StageRuntimeExecutor.run() to time each stage independently:
    separately measure input-move time and forward time per stage.
    """
    _header("Stage-Level Breakdown (move vs forward)")

    from transformers.cache_utils import DynamicCache
    from helm.runtime.pipeline_runtime import PipelineRuntime

    runtime = decode_exec

    # Build decode inputs manually (seq=1 input)
    device = "cpu"
    input_ids_dec = torch.randint(0, 1000, (1, 1), dtype=torch.long)
    position_ids = torch.tensor([[20]], dtype=torch.long)
    cache_position = torch.tensor([20], dtype=torch.long)
    decode_mask = torch.zeros((1, 1, 1, 21), dtype=torch.float16)

    inputs = {
        "input_ids": input_ids_dec,
        "attention_mask": decode_mask,
        "position_ids": position_ids,
        "cache_position": cache_position,
    }

    # Time each stage independently
    stage_stats = {i: {"move_ms": [], "fwd_ms": []} for i in range(len(runtime.stages))}

    # First run to warm up devices
    _ = runtime.run(inputs.copy())

    for rep in range(decode_steps):
        env = dict(inputs)
        past_kv = DynamicCache()

        for stage_idx, stage in enumerate(runtime.stages):
            import helm.runtime.executor as _exec_mod
            from helm.runtime.tensor_transfer import move_tensor

            target_device = torch.device(stage.device)
            kwargs = {}
            for node in stage.module.graph.nodes:
                if node.op == 'placeholder':
                    if node.target in env:
                        val = env[node.target]
                    else:
                        try:
                            obj = stage.module
                            for part in node.target.split('.'):
                                obj = getattr(obj, part)
                            val = obj
                        except AttributeError:
                            continue
                    if isinstance(val, torch.Tensor):
                        kwargs[node.target] = val
                    else:
                        kwargs[node.target] = val

            # Time: input tensor moves
            _sync()
            t_move0 = time.perf_counter()
            for k, v in list(kwargs.items()):
                if isinstance(v, torch.Tensor) and v.device != target_device:
                    kwargs[k] = v.to(stage.device)
            _sync()
            move_ms = (time.perf_counter() - t_move0) * 1000

            # Inject cache
            wrapper = _exec_mod.StageRuntimeExecutor._find_wrapper(
                _exec_mod.StageRuntimeExecutor, stage.module)
            if wrapper is not None and past_kv is not None:
                wrapper.past_key_values = past_kv

            # Time: forward pass
            _sync()
            t_fwd0 = time.perf_counter()
            with torch.no_grad():
                import helm.runtime.executor as _exec_mod2
                _exec_mod2.DynamicCache  # ensure in scope
                stage.module.forward.__globals__["DynamicCache"] = _exec_mod.DynamicCache
                outputs = stage.module(**kwargs)
            _sync()
            fwd_ms = (time.perf_counter() - t_fwd0) * 1000

            stage_stats[stage_idx]["move_ms"].append(move_ms)
            stage_stats[stage_idx]["fwd_ms"].append(fwd_ms)

            if stage_idx < len(runtime.stages) - 1:
                if isinstance(outputs, dict):
                    env.update(outputs)
            if wrapper is not None and hasattr(wrapper, 'past_key_values'):
                past_kv = wrapper.past_key_values

    print(f"\n  {'Stage':>6}  {'Device':>8}  {'InputMove':>12}  {'Forward':>10}  {'Total':>10}")
    print(f"  {'-'*52}")
    for i, stage in enumerate(runtime.stages):
        moves = stage_stats[i]["move_ms"][1:]  # skip first
        fwds = stage_stats[i]["fwd_ms"][1:]
        avg_move = sum(moves)/len(moves) if moves else 0
        avg_fwd = sum(fwds)/len(fwds) if fwds else 0
        print(f"  {i:>6}  {stage.device:>8}  {avg_move:>10.1f}ms  {avg_fwd:>8.1f}ms  "
              f"{avg_move+avg_fwd:>8.1f}ms")


# ─────────────────────────────────────────────────────────────────────────────
# Section 4: Memory snapshot
# ─────────────────────────────────────────────────────────────────────────────

def profile_memory(runtime, input_ids):
    _header("Memory Usage")

    runtime._reset_decode_cache()
    if runtime.kv_offload_mgr:
        runtime.kv_offload_mgr.reset()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    logits = runtime.prefill(input_ids.clone())
    _sync()
    next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    _ = runtime.decode_step(next_tok, input_ids.shape[1])
    _sync()

    print(f"\n  VRAM allocated    : {_vram_gb():.2f} GB")
    if torch.cuda.is_available():
        print(f"  VRAM peak         : {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
        print(f"  VRAM reserved     : {torch.cuda.memory_reserved()/1024**3:.2f} GB")
    print(f"  Process RSS (RAM) : {_ram_gb():.2f} GB")

    if runtime.kv_offload_mgr:
        report = runtime.kv_offload_mgr.report()
        print(f"\n  KV cache pool:")
        print(f"    GPU pages used  : {report.get('active_gpu_pages', 0)} "
              f"({report.get('gpu_used_bytes',0)/1024**2:.1f} MB)")
        print(f"    CPU pages used  : {report.get('active_cpu_pages', 0)} "
              f"({report.get('cpu_used_bytes',0)/1024**2:.1f} MB)")


# ─────────────────────────────────────────────────────────────────────────────
# Section 5: torch.profiler trace on CPU stage
# ─────────────────────────────────────────────────────────────────────────────

def profile_cpu_ops(decode_exec, input_ids):
    _header("CPU Stage Op-Level Profile (torch.profiler, 1 decode step)")

    from transformers.cache_utils import DynamicCache

    # Prepare a single decode step input
    input_ids_dec = torch.randint(0, 1000, (1, 1), dtype=torch.long)
    position_ids = torch.tensor([[20]], dtype=torch.long)
    cache_position = torch.tensor([20], dtype=torch.long)
    decode_mask = torch.zeros((1, 1, 1, 21), dtype=torch.float16)
    inputs = {
        "input_ids": input_ids_dec,
        "attention_mask": decode_mask,
        "position_ids": position_ids,
        "cache_position": cache_position,
    }

    # Warm up
    decode_exec.run(inputs.copy())

    # Profile
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
        with_stack=False,
    ) as prof:
        for _ in range(3):
            decode_exec.run(inputs.copy())

    # Print top ops by self CPU time
    print("\n  Top 20 ops by self CPU time:")
    print(prof.key_averages().table(
        sort_by="self_cpu_time_total", row_limit=20))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--cpu-threads", type=int, default=8)
    p.add_argument("--decode-steps", type=int, default=20)
    p.add_argument("--sections", nargs="+",
                   default=["per_token", "stage_breakdown", "memory", "cpu_ops"],
                   choices=["per_stage", "per_token", "stage_breakdown", "memory", "cpu_ops"],
                   help="Which profiling sections to run")
    args = p.parse_args()

    runtime, tokenizer, input_ids, decode_exec = build_runtime(
        args.model, args.dtype, args.cpu_threads)

    if "per_stage" in args.sections:
        profile_per_stage(runtime, input_ids, decode_steps=args.decode_steps)

    if "per_token" in args.sections:
        profile_per_token(runtime, input_ids, decode_steps=args.decode_steps)

    if "stage_breakdown" in args.sections:
        profile_stage_breakdown(decode_exec, input_ids, decode_steps=6)

    if "memory" in args.sections:
        profile_memory(runtime, input_ids)

    if "cpu_ops" in args.sections:
        profile_cpu_ops(decode_exec, input_ids)


if __name__ == "__main__":
    main()
