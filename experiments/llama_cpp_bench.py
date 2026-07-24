"""llama.cpp comparison harness (paper Table III).

Benchmarks llama.cpp (FP16 + Q4_K_M) on the same model + hardware that HELM ran on,
sweeping manual ``--n-gpu-layers`` splits for comparison with HELM's automatic
partition.

Design: keep the per-request loop identical in shape to ``paper_bench.run_experiment``
so numbers are directly comparable.

Outputs JSON to ``--output-dir/results.json`` (incremental save after every run).
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import median
from typing import Optional

logger = logging.getLogger(__name__)

# Map HF model ids to GGUF repos. Each (filename, tag) pair is one variant we benchmark.
GGUF_VARIANTS: dict[str, tuple[str, list[tuple[str, str]]]] = {
    # Repos + Q4_K_M filenames verified on HF 2026-05 (bartowski's "<org>_<model>" naming;
    # the previous "bartowski/Qwen3-8B-GGUF" / "Qwen3-8B-f16.gguf" entries were wrong and would
    # 404). fp16 dropped: Qwen3 ships BF16 (not F16); the Qwen llama.cpp point of interest is the
    # Q4_K_M quantization, and the fp16 head-to-head is already covered by LLaMA-2-13B.
    "Qwen/Qwen3-8B": (
        "bartowski/Qwen_Qwen3-8B-GGUF",
        [
            ("Qwen_Qwen3-8B-Q4_K_M.gguf", "q4_k_m"),
        ],
    ),
    "Qwen/Qwen3-4B": (
        "bartowski/Qwen_Qwen3-4B-GGUF",
        [
            ("Qwen_Qwen3-4B-Q4_K_M.gguf", "q4_k_m"),
        ],
    ),
    # Optional breadth point: a Qwen-family llama.cpp comparison so the
    # llama.cpp head-to-head is not LLaMA-only. Q4_K_M only: the 14B fp16 GGUF is sharded,
    # which this single-file loader does not handle. Repo + filename verified on HF 2026-05
    # (bartowski's newer "<org>_<model>" naming convention: bartowski/Qwen_Qwen3-14B-GGUF).
    "Qwen/Qwen3-14B": (
        "bartowski/Qwen_Qwen3-14B-GGUF",
        [
            ("Qwen_Qwen3-14B-Q4_K_M.gguf", "q4_k_m"),
        ],
    ),
    "meta-llama/Llama-3.1-8B-Instruct": (
        "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        [
            ("Meta-Llama-3.1-8B-Instruct-f16.gguf", "fp16"),
            ("Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", "q4_k_m"),
        ],
    ),
    "meta-llama/Llama-3.2-3B-Instruct": (
        "bartowski/Llama-3.2-3B-Instruct-GGUF",
        [
            ("Llama-3.2-3B-Instruct-f16.gguf", "fp16"),
            ("Llama-3.2-3B-Instruct-Q4_K_M.gguf", "q4_k_m"),
        ],
    ),
}


@dataclass
class BenchResult:
    model: str
    variant: str
    n_gpu_layers: int
    n_threads: int
    input_len: int
    output_len: int
    num_requests: int
    ttft_ms: list[float] = field(default_factory=list)
    decode_tok_per_s: list[float] = field(default_factory=list)
    error: Optional[str] = None

    def summary(self) -> dict[str, float]:
        if not self.decode_tok_per_s:
            return {"error": self.error or "no measurements"}
        return {
            "ttft_ms_p50": median(self.ttft_ms),
            "ttft_ms_mean": sum(self.ttft_ms) / len(self.ttft_ms),
            "decode_tok_s_p50": median(self.decode_tok_per_s),
            "decode_tok_s_mean": sum(self.decode_tok_per_s) / len(self.decode_tok_per_s),
            "num_requests": self.num_requests,
        }


def build_prompt(input_len: int) -> str:
    """Make a deterministic prompt with at least input_len tokens. Token count is
    approximate, but we use the same prompt across every backend so comparisons hold.
    """
    base = (
        "You are a helpful assistant. Please write a short, factual response. "
        "Topic: explain how memory hierarchies and bandwidth interact in modern "
        "processors. Be concrete. "
    )
    repeats = max(1, input_len // 20)
    return (base * repeats).strip()


def run_one(
    model_path: Path,
    n_gpu_layers: int,
    n_threads: int,
    input_len: int,
    output_len: int,
    num_requests: int,
    num_warmup: int = 1,
    n_ctx: Optional[int] = None,
    n_batch: int = 512,
) -> BenchResult:
    from llama_cpp import Llama  # lazy import — keeps this script importable for tests

    result = BenchResult(
        model=str(model_path),
        variant="",
        n_gpu_layers=n_gpu_layers,
        n_threads=n_threads,
        input_len=input_len,
        output_len=output_len,
        num_requests=num_requests,
    )

    try:
        llm = Llama(
            model_path=str(model_path),
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx or max(2048, input_len + output_len + 256),
            n_threads=n_threads,
            n_batch=n_batch,
            verbose=False,
            logits_all=False,
        )
    except Exception as e:  # noqa: BLE001
        result.error = f"load failed: {e}"
        logger.exception("Load failed for %s", model_path)
        return result

    prompt = build_prompt(input_len)

    try:
        for _ in range(num_warmup):
            llm.create_completion(prompt, max_tokens=min(32, output_len), stream=False)

        for req_i in range(num_requests):
            t_start = time.perf_counter()
            first_tok_t: Optional[float] = None
            tokens_emitted = 0
            for chunk in llm.create_completion(
                prompt, max_tokens=output_len, stream=True
            ):
                if first_tok_t is None:
                    first_tok_t = time.perf_counter()
                tokens_emitted += 1
            t_end = time.perf_counter()

            if first_tok_t is None or tokens_emitted == 0:
                logger.warning("req %d produced 0 tokens", req_i)
                continue
            ttft = (first_tok_t - t_start) * 1000.0
            decode_s = t_end - first_tok_t
            decode_tok_s = (tokens_emitted - 1) / decode_s if decode_s > 0 else 0.0

            result.ttft_ms.append(ttft)
            result.decode_tok_per_s.append(decode_tok_s)
            logger.info(
                "  req %d: ttft=%.1f ms, decode=%.2f tok/s (%d toks)",
                req_i,
                ttft,
                decode_tok_s,
                tokens_emitted,
            )
    except Exception as e:  # noqa: BLE001
        result.error = f"runtime failure: {e}"
        logger.exception("Runtime error during bench")
    finally:
        del llm
        gc.collect()

    return result


def maybe_download(repo: str, filename: str) -> Optional[Path]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        logger.error("huggingface_hub not installed")
        return None
    try:
        path = hf_hub_download(
            repo_id=repo,
            filename=filename,
            local_dir=None,
            resume_download=True,
        )
        return Path(path)
    except Exception as e:  # noqa: BLE001
        logger.error("Failed to download %s/%s: %s", repo, filename, e)
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=[])
    parser.add_argument("--local-gguf", default=None, help="Benchmark an existing GGUF file without HF download")
    parser.add_argument("--local-tag", default="local", help="Variant tag for --local-gguf results")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-requests", type=int, default=10)
    parser.add_argument("--num-warmup", type=int, default=1)
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--n-ctx", type=int, default=None, help="Override llama.cpp context length")
    parser.add_argument("--n-batch", type=int, default=512, help="llama.cpp prompt batch size")
    parser.add_argument(
        "--n-gpu-layers",
        nargs="+",
        type=int,
        default=[-1],
        help="-1 = all layers on GPU; 0 = all CPU; integer = first N layers on GPU.",
    )
    parser.add_argument("--n-threads", type=int, default=8)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["fp16", "q4_k_m"],
        help="Subset of GGUF variant tags to benchmark.",
    )
    args = parser.parse_args()
    if not args.local_gguf and not args.models:
        parser.error("provide --models and/or --local-gguf")

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.json"
    all_results: dict[str, dict] = {}
    if results_path.exists():
        all_results = json.loads(results_path.read_text())
        logger.info("Loaded %d existing results from %s", len(all_results), results_path)

    if args.local_gguf:
        path = Path(args.local_gguf).expanduser()
        if not path.is_file():
            parser.error(f"--local-gguf does not exist: {path}")
        for n_gpu in args.n_gpu_layers:
            key = f"{path.stem}::{args.local_tag}::ngl_{n_gpu}"
            if key in all_results and "error" not in all_results[key]:
                logger.info("[skip] %s already benched", key)
                continue
            logger.info("=== %s ===", key)
            r = run_one(
                path,
                n_gpu_layers=n_gpu,
                n_threads=args.n_threads,
                input_len=args.input_len,
                output_len=args.output_len,
                num_requests=args.num_requests,
                num_warmup=args.num_warmup,
                n_ctx=args.n_ctx,
                n_batch=args.n_batch,
            )
            r.variant = args.local_tag
            rec = {**asdict(r), "summary": r.summary()}
            all_results[key] = rec
            results_path.write_text(json.dumps(all_results, indent=2))
            logger.info("[saved] %s → %s", key, r.summary())

    for model in args.models:
        if model not in GGUF_VARIANTS:
            logger.warning("[skip] %s — no GGUF mapping", model)
            continue
        repo, variants = GGUF_VARIANTS[model]
        for fname, tag in variants:
            if tag not in args.variants:
                continue
            path = maybe_download(repo, fname)
            if path is None:
                continue
            for n_gpu in args.n_gpu_layers:
                key = f"{model}::{tag}::ngl_{n_gpu}"
                if key in all_results and "error" not in all_results[key]:
                    logger.info("[skip] %s already benched", key)
                    continue
                logger.info("=== %s ===", key)
                r = run_one(
                    path,
                    n_gpu_layers=n_gpu,
                    n_threads=args.n_threads,
                    input_len=args.input_len,
                    output_len=args.output_len,
                    num_requests=args.num_requests,
                    num_warmup=args.num_warmup,
                    n_ctx=args.n_ctx,
                    n_batch=args.n_batch,
                )
                r.variant = tag
                rec = {**asdict(r), "summary": r.summary()}
                all_results[key] = rec
                results_path.write_text(json.dumps(all_results, indent=2))
                logger.info("[saved] %s → %s", key, r.summary())

    logger.info("Done. Results at %s", results_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
