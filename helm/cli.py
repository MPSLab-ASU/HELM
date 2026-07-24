from __future__ import annotations

import argparse
from typing import List

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="helm",
        description="HELM: compile an LLM into CPU/GPU stages and run it heterogeneously.",
        epilog="Example: helm --model Qwen/Qwen3-4B --mode execute_stagewise --kv-offload",
    )

    parser.add_argument("--model", type=str, required=True,
                        help="Hugging Face model ID or local model directory")
    parser.add_argument("--mode", type=str, default="plan", choices=["generate", "baseline", "import", "units", "plan", "lower", "execute_stagewise", "dry_run"],
                        help="generate: run inference and print completions; "
                             "execute_stagewise: compile + run with diagnostics and saved artifacts; "
                             "plan: print/save the partition plan without generating; "
                             "import/units/lower: stop after that compiler phase; "
                             "baseline: plain Hugging Face forward pass; dry_run: config only "
                             "(default: %(default)s)")
    parser.add_argument("--prompt", type=str, default="Explain what a compiler does.",
                        help="Prompt text (default: %(default)r)")
    parser.add_argument("--max-input-tokens", type=int, default=64,
                        help="Maximum prompt length in tokens (default: %(default)s)")
    parser.add_argument("--max-new-tokens", type=int, default=8,
                        help="Number of tokens to generate (default: %(default)s)")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"],
                        help="Model weight dtype (default: %(default)s)")

    compiler = parser.add_argument_group("Compiler Flags")
    compiler.add_argument("--compiler-plan", type=str, default="auto", choices=["auto", "manual"],
                          help="auto: profile the hardware and pick the best split; "
                               "manual: use --compiler-cpu-layers/--compiler-gpu-layers "
                               "(default: %(default)s)")
    compiler.add_argument("--compiler-cpu-layers", type=str, help="Manual plan CPU layer range, e.g. 0:13")
    compiler.add_argument("--compiler-gpu-layers", type=str, help="Manual plan GPU layer range, e.g. 14:27")
    compiler.add_argument("--print-graph-summary", action="store_true",
                          help="Print the HELM IR graph summary")
    compiler.add_argument("--print-partition-units", action="store_true",
                          help="Print the partition units")
    compiler.add_argument("--print-plan", action="store_true",
                          help="Print the full partition plan as JSON")

    runtime = parser.add_argument_group("Runtime Flags")
    runtime.add_argument("--runtime-skip-baseline", action="store_true",
                         help="execute_stagewise: skip the reference Hugging Face forward pass")
    runtime.add_argument("--runtime-baseline-on-cpu", action="store_true",
                         help="Run the reference forward pass on CPU")
    runtime.add_argument("--kv-offload", action="store_true",
                         help="Enable paged KV cache with CPU offloading for long-context generation")

    infra = parser.add_argument_group("Backend Flags")
    infra.add_argument("--backend", choices=["auto", "helm", "router"], default="auto",
                       help="generate mode: auto (default) runs models that fit in VRAM on vLLM "
                            "when it is installed and everything else on HELM's CPU+GPU pipeline; "
                            "router requires vLLM for models that fit; helm always uses HELM's "
                            "native executor")
    infra.add_argument("--load-device", type=str, default="cpu", choices=["cpu", "cuda"],
                       help="Device the model is loaded on before partitioning (default: %(default)s)")
    infra.add_argument("--trace-device", type=str, default="cpu", choices=["cpu", "cuda"],
                       help="Device used for FX tracing (default: %(default)s)")
    infra.add_argument("--execution-device", type=str, default="cpu", choices=["cpu", "cuda"],
                       help="Device for the reference baseline pass (default: %(default)s)")
    infra.add_argument("--cpu-threads", type=int, default=6,
                       help="PyTorch CPU threads for CPU stages (default: %(default)s)")
    infra.add_argument("--save-artifacts-dir", type=str, default="artifacts",
                       help="Directory for per-run compiler artifacts (default: %(default)s)")

    return parser


def _to_benchmark_argv(args: argparse.Namespace) -> List[str]:
    argv: List[str] = [
        "--mode",
        args.mode,
        "--model",
        args.model,
        "--prompt",
        args.prompt,
        "--max-input-tokens",
        str(args.max_input_tokens),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--dtype",
        args.dtype,
        "--plan",
        args.compiler_plan,
        "--load-device",
        args.load_device,
        "--trace-device",
        args.trace_device,
        "--execution-device",
        args.execution_device,
        "--cpu-threads",
        str(args.cpu_threads),
        "--save-artifacts-dir",
        args.save_artifacts_dir,
    ]

    if args.compiler_cpu_layers:
        argv.extend(["--cpu-layers", args.compiler_cpu_layers])
    if args.compiler_gpu_layers:
        argv.extend(["--gpu-layers", args.compiler_gpu_layers])
    if args.print_graph_summary:
        argv.append("--print-graph-summary")
    if args.print_partition_units:
        argv.append("--print-partition-units")
    if args.print_plan:
        argv.append("--print-plan")
    if args.runtime_skip_baseline:
        argv.append("--skip-baseline")
    if args.runtime_baseline_on_cpu:
        argv.append("--baseline-on-cpu")
    if args.kv_offload:
        argv.append("--kv-offload")

    return argv


def _to_inference_config(args: argparse.Namespace):
    import torch

    from helm.runtime.inference import HelmInferenceConfig

    return HelmInferenceConfig(
        model_id=args.model,
        dtype=getattr(torch, args.dtype),
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        kv_offload=args.kv_offload,
        plan_mode=args.compiler_plan,
        cpu_layers=args.compiler_cpu_layers,
        gpu_layers=args.compiler_gpu_layers,
        cpu_threads=args.cpu_threads,
        route_to_vllm_when_all_gpu={"auto": None, "router": True, "helm": False}[args.backend],
    )


def main(argv=None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.mode == "generate":
        from helm.runtime.inference import HelmInference

        try:
            config = _to_inference_config(args)
        except ValueError as exc:
            parser.error(str(exc))
        with HelmInference(config) as inference:
            plan = inference.partition_plan
            stages = " + ".join(
                f"stage{st.stage_id}@{st.device_id}({len(st.units)}u)" for st in plan.stages
            ) if plan is not None and plan.stages else "?"
            engine = "vLLM" if inference.routed_to_vllm else "HELM pipeline"
            print(f"[helm] partition plan: {stages}; running on: {engine}")
            for completion in inference.generate([args.prompt]):
                print(completion)
        return

    from helm._pipeline import main as pipeline_main
    pipeline_main(_to_benchmark_argv(args))


if __name__ == "__main__":
    main()
