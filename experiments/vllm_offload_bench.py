#!/usr/bin/env python3
"""
vLLM CPU-offload benchmark harness (paper Section III / vLLM `cpu_offload_gb`).

Runs vLLM with partial weight offloading (`cpu_offload_gb`) either over the
paper's 12B/13B overflow model list or over one explicit ``--model-id`` smoke
target, and records load time plus decode throughput per repetition.

vLLM imports live inside :func:`bench_one` so unit tests can exercise the
target-selection logic (:func:`build_targets`) without initializing vLLM.

Requires the research environment (research/README.md). Example smoke run:

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn python -u experiments/vllm_offload_bench.py \
        --model-id Qwen/Qwen2.5-0.5B-Instruct --name Qwen2.5-0.5B-Instruct-smoke \
        --offload-gb 1 --input-len 8 --output-len 2 --reps 1 --warmup-runs 0 \
        --max-model-len 64 --gpu-util 0.35 --out /tmp/helm_vllm_smoke_results.json
"""

import argparse
import json
import time
from pathlib import Path

# (label, hf_model_id, cpu_offload_gb) for the paper's overflow sweep.
PAPER_TARGETS = [
    ("Mistral-Nemo-12B", "mistralai/Mistral-Nemo-Instruct-2407", 8),
    ("LLaMA-2-13B", "NousResearch/Llama-2-13b-hf", 8),
    ("OLMo-2-13B", "allenai/OLMo-2-1124-13B-Instruct", 8),
]


def build_targets(args):
    """Return the list of (label, model_id, cpu_offload_gb) targets to run.

    An explicit ``--model-id`` takes precedence and yields a single smoke
    target; otherwise the paper model list applies, optionally narrowed by
    the comma-separated ``--only`` label filter.
    """
    if args.model_id:
        label = args.name or args.model_id
        return [(label, args.model_id, args.offload_gb)]

    targets = list(PAPER_TARGETS)
    if args.only:
        wanted = {token.strip() for token in args.only.split(",") if token.strip()}
        targets = [t for t in targets if t[0] in wanted]
    return targets


def bench_one(label, model_id, offload_gb, args):
    """Load one model under vLLM with CPU offload and time decode runs."""
    from vllm import LLM, SamplingParams  # deferred so unit tests avoid vLLM

    prompt = "word " * max(1, args.input_len - 1)
    sampling = SamplingParams(temperature=0.0, max_tokens=args.output_len)

    t0 = time.perf_counter()
    llm = LLM(
        model=model_id,
        dtype="float16",
        cpu_offload_gb=offload_gb,
        gpu_memory_utilization=args.gpu_util,
        max_model_len=args.max_model_len,
        enforce_eager=True,
    )
    load_s = time.perf_counter() - t0

    for _ in range(args.warmup_runs):
        llm.generate([prompt], sampling)

    runs = []
    for _ in range(args.reps):
        t0 = time.perf_counter()
        outputs = llm.generate([prompt], sampling)
        elapsed = time.perf_counter() - t0
        n_tokens = len(outputs[0].outputs[0].token_ids)
        runs.append(
            {
                "elapsed_s": elapsed,
                "output_tokens": n_tokens,
                "tok_per_s": n_tokens / elapsed if elapsed > 0 else 0.0,
            }
        )

    return {
        "label": label,
        "model_id": model_id,
        "cpu_offload_gb": offload_gb,
        "load_s": load_s,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "runs": runs,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--model-id", default=None,
                   help="Explicit HF model id for a one-off smoke run (skips the paper list).")
    p.add_argument("--name", default=None,
                   help="Label for the --model-id smoke target (defaults to the model id).")
    p.add_argument("--offload-gb", type=float, default=0.0,
                   help="cpu_offload_gb for the --model-id smoke target.")
    p.add_argument("--only", default=None,
                   help="Comma-separated labels to keep from the paper model list.")
    p.add_argument("--input-len", type=int, default=128)
    p.add_argument("--output-len", type=int, default=128)
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--warmup-runs", type=int, default=1)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gpu-util", type=float, default=0.90)
    p.add_argument("--out", default="experiments/results/vllm_offload_results.json")
    args = p.parse_args()

    results = []
    for label, model_id, offload_gb in build_targets(args):
        print(f"[vllm_offload_bench] {label}: {model_id} (cpu_offload_gb={offload_gb})", flush=True)
        results.append(bench_one(label, model_id, offload_gb, args))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[vllm_offload_bench] wrote {out_path}")


if __name__ == "__main__":
    main()
