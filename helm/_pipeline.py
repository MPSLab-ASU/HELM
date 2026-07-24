from __future__ import annotations

import os
import json
import time
import argparse

import torch
import torch.fx

from helm.compiler.partition.partition_units import PartitionUnitBuilder
from helm.compiler.importers.patch_fx import apply_fx_patch
from helm.compiler.compiler import HelmCompileOptions, compile_graph
from helm.runtime.pipeline_runtime import PipelineRuntime


def configure_cpu_threads(num_threads: int):
    torch.set_num_threads(num_threads)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(num_threads)
        except RuntimeError:
            pass  # can only be set before parallel work starts; ignore if too late
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)

def get_run_dir(base_dir="artifacts"):
    os.makedirs(base_dir, exist_ok=True)
    existing_runs = [d for d in os.listdir(base_dir) if d.startswith("run_") and os.path.isdir(os.path.join(base_dir, d))]
    run_idx = len(existing_runs) + 1
    run_dir = os.path.join(base_dir, f"run_{run_idx:03d}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def save_hardware_summary(run_dir, cpu_threads, load_device, trace_device, execution_device):
    data = {
        "cpu_threads": cpu_threads,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "load_device": load_device,
        "trace_device": trace_device,
        "execution_device": execution_device,
    }
    with open(os.path.join(run_dir, "hardware.json"), "w") as f:
        json.dump(data, f, indent=2)


def _build_compile_options(args, *, lower_stages: bool, graph_kind: str) -> HelmCompileOptions:
    return HelmCompileOptions(
        mode="both",
        objective="decode_latency",
        plan_mode=args.plan,
        cpu_layers=args.cpu_layers,
        gpu_layers=args.gpu_layers,
        lower_stages=lower_stages,
        graph_kind=graph_kind,
        model_name=args.model,
        kv_offload=getattr(args, "kv_offload", False),
        workload={
            "batch_size": 1,
            "prefill_seq_len": args.max_input_tokens,
            "decode_context_len": args.max_input_tokens,
            "decode_tokens": args.max_new_tokens,
            "dtype_size": 2,
        },
    )


def load_model_and_tokenizer(model_name: str, dtype: torch.dtype, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"\n[1] Loading {model_name} on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        use_cache=False,
        device_map=device,
    )
    model.eval()

    return model, tokenizer

def run_baseline(model, tokenizer, prompt, max_input_tokens, device):
    print("\n[2] --- Running Baseline ---")
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    ).to(device)

    start_time = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    end_time = time.time()

    elapsed = end_time - start_time
    print(f"  Baseline forward pass completed in {elapsed:.4f} seconds")
    print(f"  Logits shape: {logits.shape}")

    return logits, elapsed, inputs

def _capture_hf_symbolic_trace(model):
    from transformers.utils.fx import symbolic_trace as hf_symbolic_trace

    apply_fx_patch()
    return hf_symbolic_trace(model, input_names=["input_ids", "attention_mask"])


def _capture_leaf_module_trace(model, example_inputs):
    apply_fx_patch()

    import inspect
    import torch.fx as fx
    import torch.nn as nn

    class HelmTracer(fx.Tracer):
        def is_leaf_module(self, module, module_qualified_name):
            if module.__class__.__name__.endswith("DecoderLayer"):
                return True
            if "embed_tokens" in module_qualified_name:
                return True
            if "rotary_emb" in module_qualified_name:
                return True
            if "lm_head" in module_qualified_name:
                return True
            return super().is_leaf_module(module, module_qualified_name)

    class TracerWrapper(nn.Module):
        def __init__(self, inner, input_ids, attention_mask, position_ids, cache_position):
            super().__init__()
            self.inner = inner
            self.attention_mask = attention_mask
            self.position_ids = position_ids
            self.cache_position = cache_position

            first_layer = inner.model.layers[0]
            layer_sig = inspect.signature(first_layer.forward)
            self._use_pos_embed = "position_embeddings" in layer_sig.parameters

            first_param = next(first_layer.parameters(), None)
            first_dtype = first_param.dtype if first_param is not None else torch.float32
            first_device = (
                first_param.device
                if first_param is not None and not getattr(first_param, "is_meta", False)
                else torch.device("cpu")
            )
            hidden_size = int(
                getattr(inner.config, "hidden_size", 0)
                or getattr(inner.model.embed_tokens, "embedding_dim", 0)
                or 1
            )
            _hidden = torch.zeros(1, 1, hidden_size, dtype=first_dtype, device=first_device)
            _pos = torch.zeros(1, 1, dtype=torch.long, device=first_device)
            _cache_pos = torch.zeros(1, dtype=torch.long, device=first_device)
            _mask = torch.zeros(1, 1, 1, 1, dtype=_hidden.dtype, device=first_device)
            _kwargs = {
                "attention_mask": _mask,
                "position_ids": _pos,
                "cache_position": _cache_pos,
                "use_cache": False,
            }
            if self._use_pos_embed and hasattr(inner.model, "rotary_emb"):
                _kwargs["position_embeddings"] = inner.model.rotary_emb(
                    _hidden, position_ids=_pos
                )
            with torch.no_grad():
                _out = first_layer(_hidden, **_kwargs)
            self._layer_returns_tuple = isinstance(_out, (tuple, list))
            model_type = getattr(inner.config, "model_type", "").lower()
            self._embed_scale = (
                float(hidden_size ** 0.5)
                if model_type in {"gemma", "gemma2", "gemma3"}
                else None
            )

        def forward(self, input_ids):
            hidden_states = self.inner.model.embed_tokens(input_ids)
            if self._embed_scale is not None:
                hidden_states = hidden_states * self._embed_scale
            attention_mask = self.attention_mask.to(hidden_states.dtype)

            position_embeddings = None
            if self._use_pos_embed and hasattr(self.inner.model, "rotary_emb"):
                position_embeddings = self.inner.model.rotary_emb(
                    hidden_states, position_ids=self.position_ids
                )

            for layer in self.inner.model.layers:
                layer_kwargs = {
                    "attention_mask": attention_mask,
                    "position_ids": self.position_ids,
                    "cache_position": self.cache_position,
                    "use_cache": False,
                }
                if self._use_pos_embed and position_embeddings is not None:
                    layer_kwargs["position_embeddings"] = position_embeddings

                layer_out = layer(hidden_states, **layer_kwargs)
                if self._layer_returns_tuple:
                    hidden_states = layer_out[0]
                else:
                    hidden_states = layer_out

            if hasattr(self.inner.model, "norm"):
                hidden_states = self.inner.model.norm(hidden_states)

            logits = self.inner.lm_head(hidden_states)
            final_softcap = getattr(self.inner.config, "final_logit_softcapping", None)
            if final_softcap is not None:
                logits = logits / final_softcap
                logits = torch.tanh(logits)
                logits = logits * final_softcap

            return logits

    inps = example_inputs if isinstance(example_inputs, (list, tuple)) else [example_inputs]
    if isinstance(inps[0], tuple):
        inps = inps[0]

    wrapped = TracerWrapper(
        model,
        input_ids=inps[0],
        attention_mask=inps[1],
        position_ids=inps[2] if len(inps) > 2 else None,
        cache_position=inps[3] if len(inps) > 3 else None,
    )
    tracer = HelmTracer()
    graph = tracer.trace(wrapped)
    return torch.fx.GraphModule(wrapped, graph)


def capture_fx_graph(
    model,
    example_inputs,
    run_dir=None,
    allow_fallback=False,
    trace_policy="strict_hf",
):
    print("\n[3] --- Capturing FX Graph ---")

    if trace_policy not in {"strict_hf", "explicit_leaf"}:
        raise ValueError(f"unsupported trace_policy: {trace_policy}")

    if trace_policy == "explicit_leaf":
        print("  Using explicit leaf-module HelmTracer")
        gm = _capture_leaf_module_trace(model, example_inputs)
    else:
        try:
            gm = _capture_hf_symbolic_trace(model)
            print("  Used transformers.utils.fx.symbolic_trace")
        except Exception as e:
            if run_dir is not None:
                with open(os.path.join(run_dir, "trace_error.txt"), "w") as f:
                    f.write(str(e))
            if not allow_fallback:
                raise
            print(f"  HF tracing failed: {e}")
            print("  Falling back to custom HelmTracer with leaf modules")
            gm = _capture_leaf_module_trace(model, example_inputs)

    print(f"  Captured FX Graph with {len(gm.graph.nodes)} nodes")

    if run_dir is not None:
        fx_path = os.path.join(run_dir, "prefill_fx.txt")
        with open(fx_path, "w") as f:
            f.write(str(gm.graph))

    return gm

def capture_decode_fx_graph(model, example_inputs, run_dir=None):
    from helm.compiler.importers.decode_tracer import DecodeTracer
    tracer = DecodeTracer(model)
    gm = tracer.trace()

    if run_dir is not None:
        fx_path = os.path.join(run_dir, "decode_fx.txt")
        with open(fx_path, "w") as f:
            f.write(str(gm.graph))

    return gm

def execute_generation(prefill_stages, decode_stages, inputs, max_new_tokens, tokenizer=None, dtype=torch.bfloat16, run_dir=None, kv_offload_mgr=None):
    from helm.runtime.executor import StageRuntimeExecutor
    print("\n[8] --- Generation Loop Execution ---")
    if kv_offload_mgr is not None:
        print("  KV offload: enabled (paged cache + streaming attention)")
    prefill_executor = StageRuntimeExecutor(prefill_stages)
    decode_executor = StageRuntimeExecutor(decode_stages)
    if kv_offload_mgr is not None:
        kv_offload_mgr.patch_module_roots(
            *(stage.module for stage in prefill_executor.stages),
            *(stage.module for stage in decode_executor.stages),
        )

    runtime = PipelineRuntime(prefill_executor, decode_executor, tokenizer=tokenizer, dtype=dtype, kv_offload_mgr=kv_offload_mgr)

    input_ids = None
    if isinstance(inputs, tuple) and len(inputs) > 0:
        input_ids = inputs[0]
    elif isinstance(inputs, torch.Tensor):
        input_ids = inputs

    start = time.perf_counter()
    generated_tokens = runtime.generate(input_ids, max_new_tokens)
    elapsed = time.perf_counter() - start
    num_new = generated_tokens.shape[-1]  # generate() returns only the new tokens
    if num_new > 0 and elapsed > 0:
        print(f"  Generated {num_new} tokens in {elapsed:.2f} s "
              f"({num_new / elapsed:.2f} tok/s end-to-end, including prefill)")

    decoded = None
    if tokenizer is not None:
        decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        prompts = tokenizer.batch_decode(input_ids, skip_special_tokens=True) if input_ids is not None else []
        for i, text in enumerate(decoded):
            if i < len(prompts):
                print(f"\n  Prompt: {prompts[i]}")
            print(f"  Output: {text}")

    # Save outputs artifact
    if run_dir is not None and decoded is not None:
        with open(os.path.join(run_dir, "outputs.json"), "w") as f:
            json.dump({
                "tokens": generated_tokens.tolist(),
                "text": decoded,
            }, f, indent=2)
        print(f"  Saved generation outputs to {run_dir}/outputs.json")

    return generated_tokens

def compare_outputs(reference, test_output, run_dir):
    print("\n[9] --- Comparing Outputs ---")

    # Ensure devices match
    if isinstance(test_output, torch.Tensor) and isinstance(reference, torch.Tensor):
        test_output = test_output.to(reference.device)

    # Compare with torch.allclose
    max_diff = (reference - test_output).abs().max().item()
    matched = torch.allclose(reference, test_output, atol=1e-3, rtol=1e-3)
    status = "success" if matched else "mismatch"

    report = {
        "status": status,
        "max_diff": max_diff,
        "matched": matched
    }
    report_path = os.path.join(run_dir, "outputs.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"  Output comparison saved to {report_path}")
    print(f"  Match: {matched} | Max Diff: {max_diff:.8f}")

def main(argv=None):
    parser = argparse.ArgumentParser(description="Compiler integration benchmark")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--prompt", type=str, default="Explain what a compiler does.")
    parser.add_argument("--max-input-tokens", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--mode", type=str, choices=["baseline", "import", "units", "plan", "lower", "execute_stagewise", "dry_run"], required=True)
    parser.add_argument("--plan", type=str, choices=["auto", "manual"], default="auto")
    parser.add_argument("--cpu-layers", type=str, help="e.g. 0:11 (manual plan only)")
    parser.add_argument("--gpu-layers", type=str, help="e.g. 12:27 (manual plan only)")
    parser.add_argument("--dtype", type=str, default="float16")

    # Devices arguments
    parser.add_argument("--load-device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--trace-device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--execution-device", type=str, default="cpu", choices=["cpu", "cuda"])

    # Threading
    parser.add_argument("--cpu-threads", type=int, default=6)

    parser.add_argument("--print-graph-summary", action="store_true")
    parser.add_argument("--print-partition-units", action="store_true")
    parser.add_argument("--print-plan", action="store_true")
    parser.add_argument("--save-artifacts-dir", type=str, default="artifacts")
    parser.add_argument("--kv-offload", action="store_true",
                        help="Use paged KV cache with CPU offloading and streaming attention")

    # Baseline control for explicit execution modes
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--baseline-on-cpu", action="store_true")

    args = parser.parse_args(argv)

    configure_cpu_threads(args.cpu_threads)

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    req_dtype = dtype_map.get(args.dtype, torch.float16)

    # Output directories
    run_dir = get_run_dir(args.save_artifacts_dir)
    print(f"\n[INIT] Base artifacts saved to: {run_dir}")

    # Hardware summary
    save_hardware_summary(run_dir, args.cpu_threads, args.load_device, args.trace_device, args.execution_device)

    metrics = {
        "mode": args.mode,
        "model": args.model,
        "dtype": args.dtype,
        "cpu_threads": args.cpu_threads,
        "load_device": args.load_device,
        "trace_device": args.trace_device,
        "execution_device": args.execution_device,
        "plan_mode": args.plan,
        "cpu_layers": args.cpu_layers,
        "gpu_layers": args.gpu_layers,
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
    }

    if args.mode == "dry_run":
        with open(os.path.join(run_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        print("\n[DONE] Dry run completed.")
        return

    # 1. Loading
    model, tokenizer = load_model_and_tokenizer(args.model, req_dtype, args.load_device)

    # Helpers for the graph
    def do_baseline():
        baseline_device = "cpu" if args.baseline_on_cpu else args.execution_device
        if baseline_device != args.load_device:
            model.to(baseline_device)
        return run_baseline(model, tokenizer, args.prompt, args.max_input_tokens, baseline_device)

    def do_import(lower_stages: bool = False, graph_kind: str = "prefill"):
        if args.trace_device != args.load_device:
            model.to(args.trace_device)
        dummy_inputs_dict = tokenizer(args.prompt, return_tensors="pt", truncation=True, max_length=args.max_input_tokens).to(args.trace_device)
        seq_len = dummy_inputs_dict["input_ids"].shape[1]
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=args.trace_device).unsqueeze(0)
        cache_position = torch.arange(0, seq_len, dtype=torch.long, device=args.trace_device)

        b, s = dummy_inputs_dict["input_ids"].shape
        mask_dtype = torch.float16
        min_val = torch.finfo(mask_dtype).min
        c_mask = torch.full((s, s), min_val, device=args.trace_device, dtype=mask_dtype)
        c_mask = torch.triu(c_mask, diagonal=1)
        expanded_mask = dummy_inputs_dict["attention_mask"][:, None, None, :]
        inv_mask = (1.0 - expanded_mask.to(mask_dtype)) * min_val
        causal_mask = c_mask[None, None, :, :] + inv_mask

        dummy_inputs = (dummy_inputs_dict["input_ids"], causal_mask, position_ids, cache_position)

        gm = capture_fx_graph(
            model,
            dummy_inputs,
            run_dir,
            allow_fallback=True,
            trace_policy="explicit_leaf",
        )

        compile_options = _build_compile_options(
            args,
            lower_stages=lower_stages,
            graph_kind=graph_kind,
        )

        artifact = compile_graph(
            gm=gm,
            example_inputs=dummy_inputs,
            model=model,
            tokenizer=tokenizer,
            options=compile_options,
            artifacts_dir=run_dir,
        )

        helm_graph = artifact.helm_graph
        if args.print_graph_summary:
            helm_graph.summary()
        return gm, helm_graph, dummy_inputs, artifact

    def do_units(artifact):
        units = artifact.partition_units
        if args.print_partition_units:
            builder = PartitionUnitBuilder(artifact.helm_graph)
            builder.units = units
            builder.print_units()
        return units

    def do_plan(artifact):
        plan_data = artifact.plan_data
        plan_obj = artifact.partition_plan
        if plan_obj is not None and getattr(plan_obj, "stages", None):
            summary = " + ".join(
                f"stage{st.stage_id}@{st.device_id}({len(st.units)}u)" for st in plan_obj.stages
            )
            print(f"  Partition plan: {summary}")
        if args.print_plan:
            print(json.dumps(plan_data, indent=2))

        return plan_data, plan_obj

    # Execution branching
    ref_logits = None

    if args.mode == "baseline":
        ref_logits, baseline_time, _ = do_baseline()
        metrics["baseline_time_s"] = baseline_time

    elif args.mode == "import":
        do_import()

    elif args.mode == "units":
        _, _, _, artifact = do_import()
        do_units(artifact)

    elif args.mode == "plan":
        _, _, _, artifact = do_import()
        do_units(artifact)
        do_plan(artifact)

    elif args.mode == "lower":
        _, _, _, artifact = do_import(lower_stages=True)
        do_units(artifact)
        _, plan_obj = do_plan(artifact)
        prefill_stages = artifact.stage_graphs
        print(f"  Lower mode complete. {len(prefill_stages)} prefill stages lowered.")

    elif args.mode == "execute_stagewise":
        if not args.skip_baseline:
            ref_logits, baseline_time, _ = do_baseline()
            metrics["baseline_time_s"] = baseline_time

        _, _, dummy_inputs, prefill_artifact = do_import(lower_stages=True, graph_kind="prefill")
        do_units(prefill_artifact)
        _, plan_obj = do_plan(prefill_artifact)

        from helm.compiler.importers.decode_tracer import DecodeTracer
        # Force strict evaluation over the raw parameters (Qwen2 CPU checkpoint enforces Half precision internally)
        mask_dtype = torch.float16
        decode_dummy_inputs = DecodeTracer.build_dummy_inputs(device=args.trace_device, batch_size=1, dtype=mask_dtype)
        decode_gm = capture_decode_fx_graph(model, decode_dummy_inputs, run_dir)

        decode_compile_options = _build_compile_options(
            args,
            lower_stages=True,
            graph_kind="decode",
        )
        decode_artifact = compile_graph(
            gm=decode_gm,
            example_inputs=decode_dummy_inputs,
            model=model,
            tokenizer=tokenizer,
            options=decode_compile_options,
            partition_plan_override=plan_obj,
            artifacts_dir=os.path.join(run_dir, "decode"),
        )

        prefill_stages = prefill_artifact.stage_graphs
        decode_stages = decode_artifact.stage_graphs

        # Build KV offload manager if requested
        kv_offload_mgr = None
        if args.kv_offload:
            from helm.runtime.kv_offload import KVOffloadManager, KVOffloadConfig
            kv_cfg = KVOffloadConfig.from_model(model)
            kv_offload_mgr = KVOffloadManager(model, kv_cfg)
            print(f"  KV offload config: layers={kv_cfg.num_layers} "
                  f"kv_heads={kv_cfg.num_kv_heads} head_dim={kv_cfg.head_dim} "
                  f"page_size={kv_cfg.page_size} "
                  f"gpu_watermark={kv_cfg.gpu_watermark_bytes//1024}KB")

        # Run generation with real KV cache
        mask_dtype = torch.float16
        execute_generation(
            prefill_stages, decode_stages, dummy_inputs,
            args.max_new_tokens, tokenizer=tokenizer, dtype=mask_dtype,
            run_dir=run_dir, kv_offload_mgr=kv_offload_mgr,
        )

    # Save metrics configuration
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[DONE] Finished running mode: {args.mode}")

if __name__ == "__main__":
    main()
