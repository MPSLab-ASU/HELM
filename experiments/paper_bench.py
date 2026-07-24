"""
experiments/paper_bench.py
==========================
Comprehensive benchmark for HELM paper experiments.

Covers:
  Backends   : vLLM, Accelerate (HF), DeepSpeed-Inference (CPU offload), HELM
  LLM metrics: TTFT, per-token decode latency, E2E latency, tok/s
  Statistics : p50 / p95 / p99 across --num-requests independent requests
  Compiler   : compilation time, stage plan, cost-model prediction vs actual
  Runtime    : peak GPU MB, peak CPU MB
  Throughput : concurrent requests/sec and aggregate tok/s
  Feasibility: which backends can load the model (fits_in_memory), peak memory at load
  Max decode : longest decode sequence each backend sustains before OOM/timeout
  Ablations  : configurable via --ablation flag

Usage examples
--------------
# Full paper experiment on one machine
uv run python experiments/paper_bench.py --model Qwen/Qwen2.5-7B-Instruct

# Specific backends only
uv run python experiments/paper_bench.py --backends helm accelerate

# Ablation: batch sizes
uv run python experiments/paper_bench.py --ablation batch_size --backends helm

# Skip quality benchmarks (faster)
uv run python experiments/paper_bench.py --no-lm-eval
"""

from __future__ import annotations

import argparse
import functools
import gc
import json
import math
import os
import re
import signal
import statistics
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

# ─── Memory helpers ──────────────────────────────────────────────────────────

def _free_memory():
    # Aggressive cleanup at backend boundaries: HELM compile artifacts and
    # Accelerate device hooks hold circular references that need multiple
    # GC passes to break, and the CUDA caching allocator only releases back
    # to the OS after both empty_cache and ipc_collect on top of a sync.
    for _ in range(3):
        gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _gpu_free_gb() -> float:
    """Free VRAM in GiB on cuda:0, or 0 if no GPU."""
    if not torch.cuda.is_available():
        return 0.0
    free_bytes, _ = torch.cuda.mem_get_info(0)
    return free_bytes / (1024 ** 3)


def _gpu_allocated_gb() -> float:
    """PyTorch-allocated VRAM in GiB on cuda:0."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated(0) / (1024 ** 3)


def force_clean_gpu(
    target_free_gb: float,
    label: str = "",
    max_attempts: int = 6,
    sleep_s: float = 1.5,
) -> bool:
    """Loop cleanup until free VRAM >= target_free_gb, or give up.

    Backends that ran before us may have left allocator state (~8 GB has
    been observed on L40S after HELM+Accelerate). vLLM's startup memory
    check rejects if free < 0.9 × total, so we must drain residue before
    handing the GPU to the next backend. Returns True on success, False
    if we couldn't reach the target (caller should pre-fail the cell).
    """
    if not torch.cuda.is_available():
        return True
    import time as _t
    for attempt in range(max_attempts):
        _free_memory()
        free_gb = _gpu_free_gb()
        alloc_gb = _gpu_allocated_gb()
        if free_gb >= target_free_gb:
            if label:
                print(f"[mem-clean {label}] attempt {attempt+1}: free={free_gb:.2f}GB "
                      f"alloc={alloc_gb:.2f}GB target={target_free_gb:.2f}GB OK")
            return True
        if label:
            print(f"[mem-clean {label}] attempt {attempt+1}: free={free_gb:.2f}GB "
                  f"alloc={alloc_gb:.2f}GB target={target_free_gb:.2f}GB — retrying after {sleep_s}s")
        _t.sleep(sleep_s)
    free_gb = _gpu_free_gb()
    alloc_gb = _gpu_allocated_gb()
    if label:
        print(f"[mem-clean {label}] FAILED to reach target after {max_attempts} attempts; "
              f"free={free_gb:.2f}GB alloc={alloc_gb:.2f}GB target={target_free_gb:.2f}GB")
    return False

def _peak_gpu_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 ** 2
    return 0.0


def _total_gpu_used_mb() -> float:
    """Total VRAM used on cuda:0 across ALL processes (includes vLLM, which
    uses its own allocator outside PyTorch's view). Falls back to 0.0 if the
    NVIDIA query fails. Used by VLLMBackend whose memory is invisible to
    torch.cuda.max_memory_allocated."""
    if not torch.cuda.is_available():
        return 0.0
    # NVML / nvidia-smi index physical GPUs and ignore CUDA_VISIBLE_DEVICES,
    # so map cuda:0 back to the physical device it refers to.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    # Prefer pynvml (zero subprocess overhead, ships with torch on Linux)
    try:
        import pynvml
        try:
            pynvml.nvmlInit()
            if visible.startswith(("GPU-", "MIG-")):
                handle = pynvml.nvmlDeviceGetHandleByUUID(visible)
            else:
                handle = pynvml.nvmlDeviceGetHandleByIndex(int(visible) if visible.isdigit() else 0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return info.used / 1024 ** 2
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
    except Exception:
        pass
    # Fallback: shell to nvidia-smi
    try:
        import subprocess as _sp
        out = _sp.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
             "-i", visible or "0"],
            capture_output=True, text=True, timeout=5,
        )
        return float(out.stdout.strip().splitlines()[0])
    except Exception:
        return 0.0

def _reset_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def _cpu_mb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 ** 2
    except Exception:
        return 0.0

def _ram_used_pct() -> float:
    try:
        import psutil
        return psutil.virtual_memory().percent
    except Exception:
        return 0.0


def _load_time_cpu_budget_mb(reserve_mb: int = 2048) -> tuple:
    """CPU max_memory budget for model loading, in MiB.

    Capped by TOTAL system RAM minus a reserve, deliberately not by
    'available' RAM: psutil's 'available' undercounts reclaimable page cache,
    and an available-based budget made larger-than-VRAM models unloadable on
    small-RAM hosts (Qwen3-8B on the 16 GB RTX 4060 laptop) while changing
    nothing on large-RAM hosts. Returns (budget_mb, overcommit_mb)
    where overcommit_mb > 0 means the budget exceeds currently available RAM
    and the caller should warn that the load may swap.
    """
    import psutil
    vm = psutil.virtual_memory()
    budget_mb = int(vm.total / 1024 ** 2 - reserve_mb)
    available_mb = int(vm.available / 1024 ** 2)
    overcommit_mb = max(0, budget_mb - available_mb)
    return budget_mb, overcommit_mb


_GEMMA2_DECODER_ORIGINAL_FORWARD = None
_GEMMA2_DECODER_PATCH_COUNT = 0


def _is_gemma2_model_name(model_name: str) -> bool:
    name = model_name.lower()
    return "gemma2" in name or "gemma-2" in name


def _wrap_single_output_tuple_forward(original_forward):
    @functools.wraps(original_forward)
    def wrapped(self, *args, **kwargs):
        outputs = original_forward(self, *args, **kwargs)
        if kwargs.get("output_attentions", False):
            return outputs
        if isinstance(outputs, (tuple, list)) and len(outputs) == 1:
            return outputs[0]
        return outputs

    return wrapped


def _patch_gemma2_decoder_outputs() -> None:
    global _GEMMA2_DECODER_ORIGINAL_FORWARD, _GEMMA2_DECODER_PATCH_COUNT
    import transformers.models.gemma2.modeling_gemma2 as gemma2_modeling

    if _GEMMA2_DECODER_PATCH_COUNT == 0:
        _GEMMA2_DECODER_ORIGINAL_FORWARD = gemma2_modeling.Gemma2DecoderLayer.forward
        gemma2_modeling.Gemma2DecoderLayer.forward = _wrap_single_output_tuple_forward(
            gemma2_modeling.Gemma2DecoderLayer.forward
        )
    _GEMMA2_DECODER_PATCH_COUNT += 1


def _restore_gemma2_decoder_outputs() -> None:
    global _GEMMA2_DECODER_ORIGINAL_FORWARD, _GEMMA2_DECODER_PATCH_COUNT
    if _GEMMA2_DECODER_PATCH_COUNT == 0:
        return

    _GEMMA2_DECODER_PATCH_COUNT -= 1
    if _GEMMA2_DECODER_PATCH_COUNT == 0 and _GEMMA2_DECODER_ORIGINAL_FORWARD is not None:
        import transformers.models.gemma2.modeling_gemma2 as gemma2_modeling

        gemma2_modeling.Gemma2DecoderLayer.forward = _GEMMA2_DECODER_ORIGINAL_FORWARD
        _GEMMA2_DECODER_ORIGINAL_FORWARD = None


# ─── Timeout helper ──────────────────────────────────────────────────────────

class _BenchTimeout(Exception):
    pass

def _set_alarm(seconds: int):
    if hasattr(signal, "SIGALRM"):
        def _handler(sig, frame):
            raise _BenchTimeout(f"run timed out after {seconds}s")
        signal.signal(signal.SIGALRM, _handler)
        signal.alarm(seconds)

def _cancel_alarm():
    if hasattr(signal, "SIGALRM"):
        signal.alarm(0)


# ─── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class RequestResult:
    """Per-request measurement."""
    ttft_s:          float = 0.0    # time-to-first-token (= prefill time)
    decode_s:        float = 0.0    # total decode phase time
    e2e_s:           float = 0.0    # ttft + decode
    input_tokens:    int   = 0
    output_tokens:   int   = 0
    tok_per_s:       float = 0.0    # output_tokens / e2e_s
    decode_tok_per_s: float = 0.0   # output_tokens / decode_s
    peak_gpu_mb:     float = 0.0
    peak_cpu_mb:     float = 0.0
    status:          str   = "success"
    error_msg:       str   = ""


@dataclass
class BenchStats:
    """Aggregated statistics over N requests."""
    backend:      str   = ""
    ablation_tag: str   = ""          # e.g. "batch=4", "ctx=512"
    n_requests:   int   = 0
    n_success:    int   = 0

    # Latency statistics (seconds)
    ttft_p50:     float = 0.0
    ttft_p95:     float = 0.0
    ttft_p99:     float = 0.0
    ttft_mean:    float = 0.0

    decode_lat_p50: float = 0.0     # per-token decode latency (ms)
    decode_lat_p95: float = 0.0
    decode_lat_p99: float = 0.0
    decode_lat_mean: float = 0.0

    e2e_p50:      float = 0.0
    e2e_p95:      float = 0.0
    e2e_p99:      float = 0.0

    # Throughput
    tok_per_s_mean:       float = 0.0
    decode_tok_per_s_mean: float = 0.0
    throughput_req_per_s: float = 0.0  # filled by throughput_sweep()
    throughput_tok_per_s: float = 0.0
    input_tokens_mean:    float = 0.0
    input_tokens_p50:     float = 0.0
    output_tokens_mean:   float = 0.0

    # Memory
    peak_gpu_mb_mean:  float = 0.0
    peak_cpu_mb_mean:  float = 0.0

    # HELM compiler metrics (only set for helm backend)
    compile_time_s:        float = 0.0
    stage_plan:            str   = ""   # e.g. "stage0@cpu(14u), stage1@cuda(14u)"
    cost_model_decode_ms:  float = 0.0  # predicted per-token decode latency
    cost_model_prefill_ms: float = 0.0  # predicted prefill latency

    # KV paging counters (helm only): totals over all requests in this sweep
    # point. Surface how eviction/prefetch scale with context.
    kv_evict_calls:        int = 0
    kv_pages_evicted:      int = 0
    kv_bytes_evicted:      int = 0
    kv_prefetch_calls:     int = 0
    kv_pages_prefetched:   int = 0
    kv_bytes_prefetched:   int = 0

    first_error:  str = ""
    raw_requests: List[dict] = field(default_factory=list)


def _percentile(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    data_sorted = sorted(data)
    idx = (p / 100) * (len(data_sorted) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(data_sorted) - 1)
    return data_sorted[lo] + (idx - lo) * (data_sorted[hi] - data_sorted[lo])


def _aggregate(results: List[RequestResult], backend: str, ablation_tag: str = "") -> BenchStats:
    ok = [r for r in results if r.status == "success"]
    stats = BenchStats(backend=backend, ablation_tag=ablation_tag,
                       n_requests=len(results), n_success=len(ok))
    stats.first_error = next((r.error_msg for r in results if r.status != "success" and r.error_msg), "")
    if not ok:
        return stats

    ttft   = [r.ttft_s for r in ok]
    e2e    = [r.e2e_s for r in ok]
    decs   = [r.decode_s for r in ok]
    dtoks  = [r.decode_tok_per_s for r in ok]
    toks   = [r.tok_per_s for r in ok]
    gpus   = [r.peak_gpu_mb for r in ok]
    cpus   = [r.peak_cpu_mb for r in ok]
    input_tokens = [r.input_tokens for r in ok]

    # TTFT
    stats.ttft_p50  = _percentile(ttft, 50) * 1000   # ms
    stats.ttft_p95  = _percentile(ttft, 95) * 1000
    stats.ttft_p99  = _percentile(ttft, 99) * 1000
    stats.ttft_mean = statistics.mean(ttft) * 1000

    # Per-token decode latency (ms per token)
    out_tokens = [r.output_tokens for r in ok]
    per_tok = [d / t * 1000 for d, t in zip(decs, out_tokens) if t > 0]
    stats.decode_lat_p50  = _percentile(per_tok, 50)
    stats.decode_lat_p95  = _percentile(per_tok, 95)
    stats.decode_lat_p99  = _percentile(per_tok, 99)
    stats.decode_lat_mean = statistics.mean(per_tok) if per_tok else 0.0

    # E2E
    stats.e2e_p50 = _percentile(e2e, 50) * 1000
    stats.e2e_p95 = _percentile(e2e, 95) * 1000
    stats.e2e_p99 = _percentile(e2e, 99) * 1000

    # Throughput
    stats.tok_per_s_mean        = statistics.mean(toks)
    stats.decode_tok_per_s_mean = statistics.mean(dtoks)
    stats.input_tokens_mean     = statistics.mean(input_tokens)
    stats.input_tokens_p50      = _percentile(input_tokens, 50)
    stats.output_tokens_mean    = statistics.mean(out_tokens)

    # Memory
    stats.peak_gpu_mb_mean = statistics.mean(gpus)
    stats.peak_cpu_mb_mean = statistics.mean(cpus)

    stats.raw_requests = [asdict(r) for r in ok]
    return stats


# ─── Base backend interface ───────────────────────────────────────────────────

class Backend:
    name: str = "base"
    # Whether this backend performs native continuous batching. Gates the
    # throughput sweep (see throughput_sweep): only batching backends can
    # measure concurrent throughput safely — others share one mutable model
    # across threads and deadlock on a pthread futex — and meaningfully —
    # others serialize requests through the model's forward loop anyway.
    supports_continuous_batching: bool = False

    def setup(self) -> bool:
        raise NotImplementedError

    def run_one(self, prompt: str, output_len: int, input_len: int) -> RequestResult:
        raise NotImplementedError

    def teardown(self):
        pass

    def run_n(
        self,
        prompts: List[str],
        output_len: int,
        input_len: int,
        timeout_s: int = 300,
    ) -> List[RequestResult]:
        results = []
        for i, p in enumerate(prompts):
            _set_alarm(timeout_s)
            try:
                r = self.run_one(p, output_len, input_len)
            except _BenchTimeout as e:
                r = RequestResult(status="timeout", error_msg=str(e))
            except Exception as e:
                r = RequestResult(status="error", error_msg=str(e)[:300])
            finally:
                _cancel_alarm()
            results.append(r)
        return results


# ─── vLLM backend ────────────────────────────────────────────────────────────

class VLLMBackend(Backend):
    name = "vllm"
    # vLLM does native continuous batching via generate(list_of_prompts); it is
    # the only backend that can run the concurrent throughput sweep.
    supports_continuous_batching = True

    def __init__(self, model_name: str, dtype_str: str,
                 gpu_memory_utilization: float = 0.90):
        self.model_name = model_name
        self.dtype_str = dtype_str
        self.gpu_util = gpu_memory_utilization
        self._llm = None
        self._sampling_params = None

    def setup(self) -> bool:
        try:
            from vllm import LLM, SamplingParams
            print("[vLLM] Initialising engine …")
            self._llm = LLM(
                model=self.model_name,
                dtype=self.dtype_str,
                gpu_memory_utilization=self.gpu_util,
                max_model_len=4096,
            )
            self._sampling_params = SamplingParams
            print("[vLLM] Ready.")
            return True
        except Exception as e:
            print(f"[vLLM] Setup failed: {e}")
            return False

    def run_one(self, prompt: str, output_len: int, input_len: int) -> RequestResult:
        from vllm import SamplingParams
        # Mirror Accelerate / DeepSpeed measurement protocol: two separate
        # generate() calls so prefill (TTFT) and full decode are timed
        # independently. vLLM's `res.metrics.first_token_latency` is not
        # reliable for short outputs (often returns 0), so we time wallclock.
        _reset_peak()
        _sync()

        t_pre0 = time.perf_counter()
        _ = self._llm.generate([prompt],
                               SamplingParams(max_tokens=1, temperature=0.0))
        _sync()
        t_pre1 = time.perf_counter()
        ttft = t_pre1 - t_pre0

        t_dec0 = time.perf_counter()
        out = self._llm.generate([prompt],
                                 SamplingParams(max_tokens=output_len, temperature=0.0))
        _sync()
        t_dec1 = time.perf_counter()
        e2e = t_dec1 - t_dec0

        res = out[0]
        in_toks  = len(res.prompt_token_ids)
        out_toks = len(res.outputs[0].token_ids)
        decode_s = max(e2e - ttft, 1e-6)

        # vLLM uses its own CUDA allocator; torch.cuda.max_memory_allocated
        # returns near-zero. Query the driver directly for honest peak memory.
        peak_mb = _total_gpu_used_mb()

        return RequestResult(
            ttft_s=ttft,
            decode_s=decode_s,
            e2e_s=ttft + e2e,
            input_tokens=in_toks,
            output_tokens=out_toks,
            tok_per_s=out_toks / (ttft + e2e) if (ttft + e2e) > 0 else 0.0,
            decode_tok_per_s=out_toks / decode_s if decode_s > 0 else 0.0,
            peak_gpu_mb=peak_mb,
            peak_cpu_mb=_cpu_mb(),
        )

    def teardown(self):
        # vLLM 0.10+ runs EngineCore as a multiprocessing subprocess. del alone
        # does not always reap the worker or release its CUDA context before
        # the next backend tries to start. Best-effort: call the public
        # shutdown if exposed, then drop the reference, then force clean.
        if self._llm is not None:
            for method in ("shutdown", "stop_remote_worker_execution_loop", "__del__"):
                fn = getattr(self._llm, method, None)
                if callable(fn):
                    try:
                        fn()
                        break
                    except Exception:
                        continue
        del self._llm
        self._llm = None
        self._sampling_params = None
        _free_memory()


# ─── Accelerate backend ───────────────────────────────────────────────────────

class AccelerateBackend(Backend):
    name = "accelerate"

    def __init__(self, model_name: str, dtype_str: str):
        self.model_name = model_name
        self.dtype_str  = dtype_str
        self.dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                      "float32": torch.float32}.get(dtype_str, torch.float16)
        self._model = None
        self._tokenizer = None

    def setup(self) -> bool:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            print("[Accelerate] Loading model …")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            # Explicit max_memory: accelerate's internal auto-map budgets the
            # CPU by 'available' RAM, which fluctuates with desktop load and
            # makes 16 GB-host loads fail nondeterministically. Cap by total
            # RAM (same policy as the HELM backend); non-binding on large
            # hosts, required on the 16 GB RTX 4060 laptop.
            acc_max_memory: Dict = {}
            acc_cpu_budget_mb, acc_overcommit_mb = _load_time_cpu_budget_mb()
            if acc_overcommit_mb > 0:
                print(f"[Accelerate] WARNING: CPU load budget {acc_cpu_budget_mb}MiB "
                      f"exceeds currently available RAM by {acc_overcommit_mb}MiB; "
                      f"the load may swap.")
            acc_max_memory["cpu"] = f"{acc_cpu_budget_mb}MiB"
            if torch.cuda.is_available():
                _gpu_total_mb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 2
                acc_max_memory[0] = f"{int(_gpu_total_mb * 0.80)}MiB"
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                dtype=self.dtype,
                device_map="auto",
                max_memory=acc_max_memory,
                low_cpu_mem_usage=True,
            )
            self._model.eval()
            print("[Accelerate] Ready.")
            return True
        except Exception as e:
            print(f"[Accelerate] Setup failed: {e}")
            return False

    def run_one(self, prompt: str, output_len: int, input_len: int) -> RequestResult:
        tok = self._tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=input_len,
        )
        inp = tok["input_ids"]
        device = next(self._model.parameters()).device
        inp = inp.to(device)

        _reset_peak()

        # Time prefill (TTFT) separately via one forward pass
        with torch.no_grad():
            t_pre0 = time.perf_counter()
            out_ids = self._model.generate(
                inp,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
                use_cache=True,
            )
            _sync()
            t_pre1 = time.perf_counter()
            ttft = t_pre1 - t_pre0

            # Full decode
            t_dec0 = time.perf_counter()
            full_ids = self._model.generate(
                inp,
                max_new_tokens=output_len,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
                use_cache=True,
            )
            _sync()
            t_dec1 = time.perf_counter()

        in_toks  = inp.shape[1]
        out_toks = full_ids.shape[1] - in_toks
        e2e = t_dec1 - t_dec0

        return RequestResult(
            ttft_s=ttft,
            decode_s=e2e,
            e2e_s=ttft + e2e,
            input_tokens=in_toks,
            output_tokens=out_toks,
            tok_per_s=out_toks / (ttft + e2e) if (ttft + e2e) > 0 else 0.0,
            decode_tok_per_s=out_toks / e2e if e2e > 0 else 0.0,
            peak_gpu_mb=_peak_gpu_mb(),
            peak_cpu_mb=_cpu_mb(),
        )

    def teardown(self):
        # Accelerate attaches pre-forward hooks for device-map offload. del on
        # the model is normally enough (hooks die with their submodule refs),
        # but if anything else (FX trace, lm_eval) holds a sub-tensor reference,
        # the hooks keep the device map alive. Strip them explicitly first.
        if self._model is not None:
            try:
                from accelerate.hooks import remove_hook_from_submodules
                remove_hook_from_submodules(self._model)
            except Exception:
                pass
        del self._model
        self._model = None
        self._tokenizer = None
        _free_memory()


# ─── DeepSpeed backend ────────────────────────────────────────────────────────

class DeepSpeedBackend(Backend):
    """DeepSpeed-Inference with CPU offloading (ZeRO-Inference)."""
    name = "deepspeed"

    def __init__(self, model_name: str, dtype_str: str, cpu_offload: bool = True):
        self.model_name  = model_name
        self.dtype_str   = dtype_str
        self.dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                      "float32": torch.float32}.get(dtype_str, torch.float16)
        self.cpu_offload = cpu_offload
        self._model      = None
        self._tokenizer  = None

    def setup(self) -> bool:
        try:
            import deepspeed
            from transformers import AutoModelForCausalLM, AutoTokenizer
            print("[DeepSpeed] Loading model …")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            # Load to CPU first for ZeRO-Inference offload
            base = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                dtype=self.dtype,
                low_cpu_mem_usage=True,
            )
            base.eval()

            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[DeepSpeed] Initialising inference engine (cpu_offload={self.cpu_offload}) …")

            if self.cpu_offload and torch.cuda.is_available():
                # ZeRO-Inference: weights stay on CPU, only active layers on GPU
                self._model = deepspeed.init_inference(
                    base,
                    dtype=self.dtype,
                    enable_cuda_graph=False,
                    replace_with_kernel_inject=False,  # safer for Qwen
                    injection_policy=None,
                    mp_size=1,
                )
            else:
                self._model = deepspeed.init_inference(
                    base,
                    dtype=self.dtype,
                    enable_cuda_graph=False,
                )
            print("[DeepSpeed] Ready.")
            return True
        except ImportError:
            print("[DeepSpeed] deepspeed not installed — skipping")
            return False
        except Exception as e:
            print(f"[DeepSpeed] Setup failed: {e}\n{traceback.format_exc()}")
            return False

    def run_one(self, prompt: str, output_len: int, input_len: int) -> RequestResult:
        tok = self._tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=input_len,
        )
        inp = tok["input_ids"]
        if torch.cuda.is_available():
            inp = inp.cuda()

        _reset_peak()

        with torch.no_grad():
            # TTFT: generate 1 token
            t_pre0 = time.perf_counter()
            _ = self._model.generate(
                inp, max_new_tokens=1, do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id)
            _sync()
            t_pre1 = time.perf_counter()
            ttft = t_pre1 - t_pre0

            # Full generation
            t0 = time.perf_counter()
            out_ids = self._model.generate(
                inp, max_new_tokens=output_len, do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id)
            _sync()
            t1 = time.perf_counter()

        in_toks  = inp.shape[1]
        out_toks = out_ids.shape[1] - in_toks
        e2e = t1 - t0

        return RequestResult(
            ttft_s=ttft,
            decode_s=e2e,
            e2e_s=ttft + e2e,
            input_tokens=in_toks,
            output_tokens=out_toks,
            tok_per_s=out_toks / (ttft + e2e) if (ttft + e2e) > 0 else 0.0,
            decode_tok_per_s=out_toks / e2e if e2e > 0 else 0.0,
            peak_gpu_mb=_peak_gpu_mb(),
            peak_cpu_mb=_cpu_mb(),
        )

    def teardown(self):
        # DeepSpeed wraps the HF model in an InferenceEngine (`self._model.module`
        # is the original HF model). Drop both wrappers + the tokenizer; the
        # InferenceEngine's destructor releases the CUDA workspace, but only
        # if all Python refs to it are gone first.
        if self._model is not None:
            wrapped = getattr(self._model, "module", None)
            try:
                del wrapped
            except Exception:
                pass
        del self._model
        self._model = None
        self._tokenizer = None
        _free_memory()


# ─── HELM-router backend ──────────────────────────────────────────────────────

class HelmRouterBackend(Backend):
    """HELM with the router enabled: fitting models → vLLM, overflowing → HELM PipelineRuntime.

    Uses the HelmInference facade in helm/runtime/inference.py. The HELM cell
    in paper_results.json becomes whichever backend the router chose; the
    `routed_to_vllm` and `partition_plan` flags below identify which.
    """
    name = "helm-router"

    def __init__(
        self,
        model_name: str,
        dtype_str: str,
        kv_offload: bool = True,
        cpu_threads: int = 8,
        input_len: int = 64,
        batch_size: int = 1,
    ):
        self.model_name   = model_name
        self.dtype_str    = dtype_str
        self.dtype        = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                             "float32": torch.float32}.get(dtype_str, torch.float16)
        self.kv_offload   = kv_offload
        self.cpu_threads  = cpu_threads
        self.input_len    = input_len
        self.batch_size   = batch_size
        self._inference: Optional[Any] = None  # HelmInference instance
        self._routed_to_vllm = False
        self._partition_plan_str = "-"

    def setup(self) -> bool:
        try:
            from transformers import AutoTokenizer

            from helm.runtime.inference import HelmInference, HelmInferenceConfig
            print(f"[HELM-router] Initialising HelmInference for {self.model_name} …")
            # Used only to truncate prompts to input_len tokens, exactly like
            # the other backends do (HelmInference rejects over-long prompts
            # instead of truncating them).
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            cfg = HelmInferenceConfig(
                model_id=self.model_name,
                dtype=self.dtype,
                # Small headroom: decode->re-encode of a truncated prompt can
                # shift the token count by a token or two.
                max_input_tokens=int(self.input_len) + 16,
                max_new_tokens=128,    # per-call override is fine; this is the default cap
                kv_offload=self.kv_offload,
                plan_mode="auto",
                cpu_threads=self.cpu_threads,
                route_to_vllm_when_all_gpu=True,
            )
            self._inference = HelmInference(cfg)
            self._inference.setup()
            self._routed_to_vllm = bool(self._inference.routed_to_vllm)
            plan = self._inference.partition_plan
            if plan is not None and plan.stages:
                self._partition_plan_str = ", ".join(
                    f"stage{s.stage_id}@{s.device_id}({len(s.units)}u)"
                    for s in plan.stages
                )
            print(f"[HELM-router] Ready. routed_to_vllm={self._routed_to_vllm}  plan=[{self._partition_plan_str}]")
            return True
        except Exception as exc:
            print(f"[HELM-router] Setup failed: {exc}\n{traceback.format_exc()}")
            return False

    def run_one(self, prompt: str, output_len: int, input_len: int) -> RequestResult:
        # Mirror the two-phase measurement protocol used by VLLM/Accelerate/DeepSpeed
        # so HELM-router is apples-to-apples with them. HelmInference returns
        # generated text; we use output_len as the token count under the
        # standard paper_bench assumption of greedy decoding without early EOS.
        ids = self._tokenizer(prompt, truncation=True, max_length=input_len)["input_ids"]
        prompt = self._tokenizer.decode(ids, skip_special_tokens=True)
        _sync()

        t_pre0 = time.perf_counter()
        _ = self._inference.generate([prompt], max_new_tokens=1)
        _sync()
        t_pre1 = time.perf_counter()
        ttft = t_pre1 - t_pre0

        t_dec0 = time.perf_counter()
        outs = self._inference.generate([prompt], max_new_tokens=output_len)
        _sync()
        t_dec1 = time.perf_counter()
        e2e = t_dec1 - t_dec0

        decode_s = max(e2e - ttft, 1e-6)
        in_toks  = int(input_len)
        out_toks = int(output_len)  # greedy, no EOS — same convention as VLLMBackend
        peak_mb  = _total_gpu_used_mb()
        return RequestResult(
            ttft_s=ttft,
            decode_s=decode_s,
            e2e_s=ttft + e2e,
            input_tokens=in_toks,
            output_tokens=out_toks,
            tok_per_s=out_toks / (ttft + e2e) if (ttft + e2e) > 0 else 0.0,
            decode_tok_per_s=out_toks / decode_s if decode_s > 0 else 0.0,
            peak_gpu_mb=peak_mb,
            peak_cpu_mb=_cpu_mb(),
        )

    def teardown(self):
        if self._inference is not None:
            try:
                self._inference.teardown()
            except Exception:
                pass
            self._inference = None
        _free_memory()


# ─── HELM backend ─────────────────────────────────────────────────────────────

class HelmBackend(Backend):
    """HELM heterogeneous inference — auto partition mode."""
    name = "helm"

    def __init__(
        self,
        model_name:    str,
        dtype_str:     str,
        kv_offload:    bool = True,
        cpu_threads:   int  = 8,
        input_len:     int  = 64,
        batch_size:    int  = 1,
        plan_mode:     str  = "auto",
        cpu_layers:    Optional[str] = None,
        gpu_layers:    Optional[str] = None,
        disable_avx:   bool = False,   # ablation: disable AVX kernel
        disable_async: bool = False,   # ablation: disable async KV prefetch
    ):
        self.model_name   = model_name
        self.dtype_str    = dtype_str
        self.dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                      "float32": torch.float32}.get(dtype_str, torch.float16)
        self.kv_offload   = kv_offload
        self.cpu_threads  = cpu_threads
        self.input_len    = input_len
        self.batch_size   = batch_size
        self.plan_mode    = plan_mode
        self.cpu_layers   = cpu_layers
        self.gpu_layers   = gpu_layers
        self.disable_avx  = disable_avx
        self.disable_async = disable_async

        self._model           = None
        self._tokenizer       = None
        self._runtime         = None
        self._kv_offload_mgr  = None
        self._run_lock        = threading.Lock()  # HELM is single-request; serialize concurrent calls
        self._decode_wrapper  = None
        self._gemma2_decoder_patch_active = False

        # Compiler metrics captured during setup
        self.compile_time_s        = 0.0
        self.stage_plan            = ""
        self.cost_model_decode_ms  = 0.0
        self.cost_model_prefill_ms = 0.0

    def _compile_options(self, *, graph_kind: str, workload: Dict[str, int]):
        from helm.compiler.compiler import HelmCompileOptions

        return HelmCompileOptions(
            mode="both",
            objective="decode_latency",
            plan_mode=self.plan_mode,
            cpu_layers=self.cpu_layers,
            gpu_layers=self.gpu_layers,
            lower_stages=True,
            graph_kind=graph_kind,
            model_name=self.model_name,
            workload=workload,
            kv_offload=self.kv_offload,
        )

    def setup(self) -> bool:
        sys.path.insert(0, os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..')))
        try:
            from experiments.dev_pipeline import (
                load_model_and_tokenizer,
                capture_fx_graph,
                capture_decode_fx_graph,
                configure_cpu_threads,
            )
            from helm.compiler.compiler import compile_graph
            from helm.compiler.importers.decode_tracer import DecodeTracer
            from helm.compiler.importers.patch_fx import apply_fx_patch
            from helm.runtime.executor import StageRuntimeExecutor
            from helm.runtime.pipeline_runtime import PipelineRuntime
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import psutil

            configure_cpu_threads(self.cpu_threads)

            if self.disable_avx:
                os.environ["HELM_DISABLE_AVX"] = "1"
                print("[HELM] AVX kernel disabled (ablation)")
            if self.disable_async:
                os.environ["HELM_DISABLE_ASYNC_KV"] = "1"
                print("[HELM] Async KV prefetch disabled (ablation)")

            print("[HELM] Loading model …")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            gpu_budget = None
            if torch.cuda.is_available():
                gpu_total_mb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 2
                gpu_budget = f"{int(gpu_total_mb * 0.80)}MiB"
            # Load-time CPU budget is capped by TOTAL RAM, not 'available':
            # 'available' undercounts reclaimable page cache and breaks
            # larger-than-VRAM loads on small-RAM hosts (e.g. Qwen3-8B on the
            # 16 GB RTX 4060 laptop). The weights only need to land within
            # physical limits at load; StageRuntimeExecutor repositions them
            # per the partition plan afterwards.
            cpu_budget_mb, cpu_overcommit_mb = _load_time_cpu_budget_mb()
            if cpu_overcommit_mb > 0:
                print(f"[HELM] WARNING: CPU load budget {cpu_budget_mb}MiB exceeds "
                      f"currently available RAM by {cpu_overcommit_mb}MiB; the load "
                      f"may swap. Close other processes if the OS kills this run.")
            cpu_budget = f"{cpu_budget_mb}MiB"

            max_memory: Dict = {"cpu": cpu_budget}
            if gpu_budget and torch.cuda.is_available():
                max_memory[0] = gpu_budget

            try:
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    dtype=self.dtype,
                    device_map="auto",
                    max_memory=max_memory,
                    low_cpu_mem_usage=True,
                    use_cache=False,
                )
            except TypeError as e:
                # Gemma-3 IT etc. resolve to a multimodal class whose __init__
                # rejects use_cache. Fall back to the text-only causal-LM class.
                if "use_cache" not in str(e):
                    raise
                from transformers import Gemma3ForCausalLM
                self._model = Gemma3ForCausalLM.from_pretrained(
                    self.model_name,
                    dtype=self.dtype,
                    device_map="auto",
                    max_memory=max_memory,
                    low_cpu_mem_usage=True,
                )
                self._model.config.use_cache = False
            self._model.eval()

            if _is_gemma2_model_name(self.model_name):
                _patch_gemma2_decoder_outputs()
                self._gemma2_decoder_patch_active = True

            try:
                from accelerate.hooks import remove_hook_from_submodules
                remove_hook_from_submodules(self._model)
            except ImportError:
                pass

            prompt_tok = self._tokenizer(
                "Benchmark prompt for compilation.",
                return_tensors="pt",
                truncation=True,
                max_length=self.input_len,
            )
            input_ids = prompt_tok["input_ids"]
            seq_len = input_ids.shape[1]

            position_ids  = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
            cache_position = torch.arange(seq_len, dtype=torch.long)
            min_val = torch.finfo(torch.float16).min
            causal  = torch.triu(
                torch.full((seq_len, seq_len), min_val, dtype=torch.float16), diagonal=1)
            inv_mask = (1.0 - prompt_tok["attention_mask"].float()) * min_val
            attn_mask = causal[None, None, :, :] + inv_mask[:, None, None, :]
            dummy_inputs = (input_ids, attn_mask, position_ids, cache_position)

            _workload = {
                "batch_size": self.batch_size,
                "prefill_seq_len": seq_len,
                "decode_context_len": seq_len,
                "decode_tokens": 128,
                "dtype_size": 2,
            }

            print("[HELM] Tracing and compiling (auto partition mode) …")
            t_compile0 = time.perf_counter()

            prefill_gm = capture_fx_graph(
                self._model,
                dummy_inputs,
                run_dir=None,
                allow_fallback=True,
                trace_policy="explicit_leaf",
            )
            prefill_opts = self._compile_options(
                graph_kind="prefill",
                workload=_workload,
            )
            prefill_artifact = compile_graph(
                gm=prefill_gm, example_inputs=dummy_inputs,
                model=self._model, tokenizer=self._tokenizer,
                options=prefill_opts, artifacts_dir=None,
            )

            apply_fx_patch()
            dec_dummy = DecodeTracer.build_dummy_inputs(
                device="cpu", batch_size=1, dtype=torch.float16)
            decode_gm = capture_decode_fx_graph(self._model, dec_dummy, run_dir=None)
            self._decode_wrapper = getattr(self._model, "_helm_decode_wrapper", None)

            decode_opts = self._compile_options(
                graph_kind="decode",
                workload=_workload,
            )
            decode_artifact = compile_graph(
                gm=decode_gm, example_inputs=dec_dummy,
                model=self._model, tokenizer=self._tokenizer,
                options=decode_opts,
                partition_plan_override=prefill_artifact.partition_plan,
                artifacts_dir=None,
            )

            t_compile1 = time.perf_counter()
            self.compile_time_s = t_compile1 - t_compile0

            # Capture stage plan and cost model predictions
            plan = prefill_artifact.partition_plan
            if plan is not None:
                self.stage_plan = ", ".join(
                    f"stage{s.stage_id}@{s.device_id}({len(s.units)}u)"
                    for s in plan.stages
                )
            if hasattr(prefill_artifact, "plan_cost") and prefill_artifact.plan_cost is not None:
                cost = prefill_artifact.plan_cost
                self.cost_model_decode_ms  = getattr(cost, "decode_token_latency_s", 0.0) * 1000
                self.cost_model_prefill_ms = getattr(cost, "prefill_latency_s", 0.0) * 1000

            print(f"[HELM] Compilation done in {self.compile_time_s:.1f}s")
            print(f"[HELM] Stage plan: {self.stage_plan}")
            print(f"[HELM] Cost model: decode={self.cost_model_decode_ms:.1f}ms "
                  f"prefill={self.cost_model_prefill_ms:.0f}ms")

            prefill_exec = StageRuntimeExecutor(prefill_artifact.stage_graphs)
            decode_exec  = StageRuntimeExecutor(decode_artifact.stage_graphs)
            if self.kv_offload:
                from helm.runtime.kv_offload import KVOffloadManager, KVOffloadConfig
                # Allow a KV page-size sweep via env var (HELM_KV_PAGE_SIZE)
                _ps_env = os.environ.get("HELM_KV_PAGE_SIZE")
                # Allow a contiguous-KV ablation via env var
                # (HELM_CONT_CAPACITY=0 disables the fast path; HELM_CONT_CAPACITY=N
                # forces capacity N tokens; unset = auto-detect from partition plan).
                _cc_env = os.environ.get("HELM_CONT_CAPACITY")
                # Allow forcing the GPU KV residency budget (MB). Setting this
                # below the context KV footprint guarantees GPU->CPU eviction and
                # CPU->GPU prefetch, so the paging regime can be exercised and the
                # PCIe KV-streaming crossover measured.
                _wm_env = os.environ.get("HELM_GPU_KV_WATERMARK_MB")
                kv_kwargs: dict = {}
                if _ps_env:
                    kv_kwargs["page_size"] = int(_ps_env)
                if _wm_env is not None:
                    kv_kwargs["gpu_watermark_bytes"] = int(float(_wm_env) * 1024 * 1024)
                if _cc_env is not None:
                    kv_kwargs["cont_capacity"] = int(_cc_env)
                else:
                    # Auto-enable the contiguous fast path when the partition plan
                    # puts every stage on GPU: the paged KV machinery is needed only
                    # when at least one stage lives on CPU. Without this, decode
                    # tok/s on fit-in-VRAM models trails Accelerate by 10-25% because
                    # _decode_batched goes through the paged Path 2 even though no
                    # offload activity ever happens.
                    plan = prefill_artifact.partition_plan
                    if plan is not None and plan.stages and all(
                        "cuda" in (s.device_id or "") for s in plan.stages
                    ):
                        # Pre-allocate enough capacity for the whole benchmark
                        # sequence (input + max output_len observed in paper_bench
                        # plus slack). 512 covers every (input_len, output_len)
                        # configuration the runners actually use; longer sequences
                        # will fall through to the paged path automatically when
                        # cont_capacity is exceeded.
                        auto_cap = int(self.input_len) + 512
                        kv_kwargs["cont_capacity"] = auto_cap
                        print(f"[HELM] All-GPU partition detected; enabling "
                              f"contiguous KV fast path (cont_capacity={auto_cap})")
                kv_cfg = KVOffloadConfig.from_model(self._model, **kv_kwargs)
                _wm_mb = kv_cfg.gpu_watermark_bytes / (1024 * 1024)
                print(f"[HELM] KV offload: gpu_watermark={_wm_mb:.0f} MB  "
                      f"page_size={kv_cfg.page_size}  cont_capacity={kv_cfg.cont_capacity}")
                self._kv_offload_mgr = KVOffloadManager(
                    self._model, kv_cfg, batch_size=self.batch_size)
                self._kv_offload_mgr.patch_module_roots(
                    *(stage.module for stage in prefill_exec.stages),
                    *(stage.module for stage in decode_exec.stages),
                )
            self._runtime = PipelineRuntime(
                prefill_exec, decode_exec,
                tokenizer=self._tokenizer,
                dtype=self.dtype,
                kv_offload_mgr=self._kv_offload_mgr,
                decode_wrapper=self._decode_wrapper,
            )
            print("[HELM] Runtime ready.")
            return True

        except Exception as e:
            if self._gemma2_decoder_patch_active:
                _restore_gemma2_decoder_outputs()
                self._gemma2_decoder_patch_active = False
            print(f"[HELM] Setup failed:\n{traceback.format_exc()}")
            return False

    def _reset_kv(self):
        if self._kv_offload_mgr is not None:
            self._kv_offload_mgr.reset()
        from transformers.cache_utils import DynamicCache as _DC
        if self._decode_wrapper is not None:
            self._decode_wrapper.past_key_values = _DC()
        elif self._runtime is not None:
            self._runtime._reset_decode_cache()

    def run_one(self, prompt: str, output_len: int, input_len: int) -> RequestResult:
        with self._run_lock:
            return self._run_one_locked(prompt, output_len, input_len)

    def _run_one_locked(self, prompt: str, output_len: int, input_len: int) -> RequestResult:
        if self._runtime is None:
            return RequestResult(status="not_available", error_msg="not compiled")

        tok_out = self._tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=input_len,
        )
        input_ids = tok_out["input_ids"]
        attention_mask = tok_out.get("attention_mask")
        if self.batch_size > 1:
            input_ids = input_ids.expand(self.batch_size, -1).contiguous()
            if attention_mask is not None:
                attention_mask = attention_mask.expand(self.batch_size, -1).contiguous()

        self._reset_kv()
        _reset_peak()
        _sync()

        token_times = []

        def _record_token(_step, _token, _finished):
            token_times.append(time.perf_counter())

        t_start = time.perf_counter()
        generations = self._runtime.generate(
            input_ids.clone(),
            max_new_tokens=output_len,
            attention_mask=None if attention_mask is None else attention_mask.clone(),
            token_callback=_record_token,
        )
        _sync()
        t_end = time.perf_counter()

        if token_times:
            ttft = token_times[0] - t_start
            decode_s = max(t_end - token_times[0], 0.0)
        else:
            ttft = t_end - t_start
            decode_s = 0.0
        out_toks = int(generations.numel())

        return RequestResult(
            ttft_s=ttft,
            decode_s=decode_s,
            e2e_s=ttft + decode_s,
            input_tokens=input_ids.shape[1],
            output_tokens=out_toks,
            tok_per_s=out_toks / (ttft + decode_s) if (ttft + decode_s) > 0 else 0.0,
            decode_tok_per_s=out_toks / decode_s if decode_s > 0 else 0.0,
            peak_gpu_mb=_peak_gpu_mb(),
            peak_cpu_mb=_cpu_mb(),
        )

    def teardown(self):
        if self._gemma2_decoder_patch_active:
            _restore_gemma2_decoder_outputs()
            self._gemma2_decoder_patch_active = False
        kv_offload_mgr = getattr(self, "_kv_offload_mgr", None)
        if kv_offload_mgr is not None:
            kv_offload_mgr.restore()
        del self._runtime, self._model, self._kv_offload_mgr
        self._runtime = self._model = self._kv_offload_mgr = None
        # _decode_wrapper holds a direct reference to model CUDA layers and must
        # be explicitly cleared; forgetting this keeps 7+ GB pinned after teardown.
        if hasattr(self, '_decode_wrapper'):
            self._decode_wrapper = None
        # Clear ablation env vars so they don't bleed into subsequent conditions.
        os.environ.pop("HELM_DISABLE_AVX", None)
        os.environ.pop("HELM_DISABLE_ASYNC_KV", None)
        _free_memory()

    def kv_report(self) -> Optional[Dict[str, Any]]:
        if self._kv_offload_mgr is None:
            return None
        return self._kv_offload_mgr.report()


# ─── Throughput sweep ─────────────────────────────────────────────────────────

def _vllm_throughput_batch(
    backend: "VLLMBackend",
    prompts: List[str],
    output_len: int,
) -> Tuple[float, List[int], float, float]:
    """vLLM-native concurrent throughput: pass all C prompts in one
    generate() call so vLLM's internal scheduler does continuous batching.
    Returns (wall_s, output_tokens_per_request, peak_gpu_mb, mean_e2e_ms).

    Threaded concurrency against vLLM's synchronous generate() deadlocks
    (vLLM 0.10+ serializes via an internal lock), so this path is required
    for any vllm cell where concurrency > 1.
    """
    from vllm import SamplingParams
    sp = SamplingParams(max_tokens=output_len, temperature=0.0)

    _sync()
    t0 = time.perf_counter()
    out = backend._llm.generate(list(prompts), sp)
    _sync()
    t1 = time.perf_counter()

    wall_s = t1 - t0
    output_tokens = [len(r.outputs[0].token_ids) for r in out]
    peak_mb = _total_gpu_used_mb()
    # Per-request e2e is not directly observable when prompts are batched
    # by vLLM. Approximate as wall_s (the whole batch completed in this
    # window). This is the same convention vLLM benchmark scripts use.
    mean_e2e_ms = wall_s * 1000.0
    return wall_s, output_tokens, peak_mb, mean_e2e_ms


def throughput_sweep(
    backend: Backend,
    prompts: List[str],
    output_len: int,
    input_len: int,
    concurrency_levels: List[int],
) -> Dict:
    """
    Measure concurrent throughput at different concurrency levels.

    Capability gate: only backends with `supports_continuous_batching = True`
    run the sweep. For every other backend the old ThreadPoolExecutor path is
    BOTH unsafe and uninformative:

      - Unsafe: C worker threads call generate() on a single shared, mutable
        model. That model's internal state is not thread-safe — accelerate's
        device-map offload hooks, HELM's stage scheduler, and vLLM's
        synchronous engine lock all serialize through a pthread mutex — so the
        threads deadlock on a futex. Observed twice in one session
        (Qwen3-8B/helm-router and gemma-2-27b/accelerate), both presenting as
        state=S + wchan=futex_wait_queue with no forward progress.
      - Uninformative: even without the deadlock, the requests serialize
        through the model's single forward loop, so "concurrency=8" would just
        measure ~8x the sequential latency rather than real batched throughput.
        The `batch_size` ablation already reports native-batch scaling for
        those backends.

    Only vLLM (native continuous batching via generate(list)) clears the gate.

    Returns: dict[concurrency] -> per-level metrics for batching backends, or
             {"_skipped": <reason>} for backends that cannot batch. The skip
             marker is a non-numeric key with a string value, so downstream
             consumers that iterate numeric concurrency entries ignore it.
    """
    if not getattr(backend, "supports_continuous_batching", False):
        reason = ("non-batching backend: threaded concurrency deadlocks on "
                  "shared model state and measures no real throughput; "
                  "see the batch_size ablation")
        print(f"  [{backend.name}] throughput sweep skipped ({reason})")
        return {"_skipped": reason}

    results: Dict[int, Dict] = {}
    for c in concurrency_levels:
        batch = prompts[:c]
        if len(batch) < c:
            # Pad by repeating prompts so concurrency level is honored even
            # when the prompt pool is smaller than the highest C requested.
            batch = (batch * ((c // max(len(batch), 1)) + 1))[:c]

        wall_s, out_toks, peak_mb, mean_e2e = _vllm_throughput_batch(
            backend, batch, output_len)
        n_ok = len(out_toks)
        total_tok = sum(out_toks)
        results[c] = {
            "concurrency":     c,
            "wall_s":          wall_s,
            "n_success":       n_ok,
            "req_per_s":       n_ok / wall_s if wall_s > 0 else 0.0,
            "total_tok_per_s": total_tok / wall_s if wall_s > 0 else 0.0,
            "mean_e2e_ms":     mean_e2e,
            "peak_gpu_mb":     peak_mb,
            "method":          "vllm_native_batch",
        }
        print(f"  concurrency={c}: {results[c]['req_per_s']:.2f} req/s, "
              f"{results[c]['total_tok_per_s']:.1f} tok/s "
              f"({results[c]['method']})")
    return results


# ─── LM quality benchmarks ───────────────────────────────────────────────────

def run_lm_eval(
    model_name: str,
    tasks: List[str],
    dtype_str: str,
    output_dir: Path,
    limit: Optional[int] = 200,
) -> Dict:
    """
    Run lm-evaluation-harness on the specified tasks.
    Returns dict with per-task accuracy metrics.
    """
    try:
        import lm_eval
        from lm_eval import evaluator, tasks as lm_tasks
    except ImportError:
        print("[lm-eval] lm-evaluation-harness not installed — skipping quality benchmarks")
        print("  Install: uv pip install lm-eval")
        return {"error": "lm-eval not installed"}

    print(f"\n[lm-eval] Running: {tasks}  (limit={limit})")
    try:
        results = evaluator.simple_evaluate(
            model="hf",
            model_args=f"pretrained={model_name},dtype={dtype_str}",
            tasks=tasks,
            num_fewshot=None,  # use default per task
            limit=limit,
            device="cuda" if torch.cuda.is_available() else "cpu",
            batch_size="auto",
            log_samples=False,
        )
        # Extract summary metrics
        summary = {}
        for task, task_res in results["results"].items():
            summary[task] = {k: v for k, v in task_res.items()
                             if isinstance(v, (int, float))}
            print(f"  {task}: {task_res}")
        return summary
    except Exception as e:
        print(f"[lm-eval] Failed: {e}\n{traceback.format_exc()}")
        return {"error": str(e)}


# ─── Feasibility probe ───────────────────────────────────────────────────────

def probe_feasibility(backend_name: str, args) -> Dict:
    """
    Try to load the model for one backend and report whether it fits in memory.

    Returns a dict with:
      fits_in_memory   bool   — True if setup() succeeded
      peak_gpu_mb      float  — peak GPU VRAM used after load (0 on CPU-only)
      peak_cpu_mb      float  — RSS after load (MB)
      gpu_free_mb      float  — GPU VRAM free after load (useful for KV headroom)
      status           str    — "ok" | "oom" | "error"
      error_msg        str
    """
    result: Dict = {
        "backend":        backend_name,
        "fits_in_memory": False,
        "peak_gpu_mb":    0.0,
        "peak_cpu_mb":    0.0,
        "gpu_free_mb":    0.0,
        "status":         "error",
        "error_msg":      "",
    }

    _reset_peak()
    _free_memory()

    # Pre-flight: vLLM and DeepSpeed both reject if startup-free-VRAM is too
    # low. paper_bench runs all backends sequentially in one Python process,
    # so allocator residue from earlier backends (HELM compile artifacts,
    # Accelerate hooks) can starve later ones. Wait briefly for cleanup to
    # land; if we still can't reach the per-backend target, record a clear
    # PRE_FAILED status so the JSON tells the truth instead of vLLM crashing
    # with a generic "Engine core initialization failed".
    _MIN_FREE_FRACTION = {
        "vllm":       0.85,   # vLLM uses 0.9 of total by default; needs ~0.85 free
        "deepspeed":  0.30,   # init_inference needs a small headroom buffer
        "helm":       0.30,
        "accelerate": 0.30,
    }
    if torch.cuda.is_available():
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        min_free_gb = _MIN_FREE_FRACTION.get(backend_name, 0.30) * total_gb
        ok_mem = force_clean_gpu(min_free_gb, label=f"pre-{backend_name}")
        if not ok_mem:
            result["status"]    = "pre_failed"
            result["error_msg"] = (
                f"pre-flight: free_gpu_gb={_gpu_free_gb():.2f} < required {min_free_gb:.2f} "
                f"({_MIN_FREE_FRACTION.get(backend_name, 0.30):.0%} of {total_gb:.1f}GB). "
                f"Previous backend's allocator state could not be released."
            )
            result["gpu_free_mb"] = _gpu_free_gb() * 1024
            print(f"  [{backend_name}] PRE_FAILED — {result['error_msg']}")
            return result

    backend = _make_backend(backend_name, args)
    try:
        ok = backend.setup()
        if ok:
            result["fits_in_memory"] = True
            result["peak_gpu_mb"]    = _peak_gpu_mb()
            result["peak_cpu_mb"]    = _cpu_mb()
            result["status"]         = "ok"
            if torch.cuda.is_available():
                free_bytes = torch.cuda.mem_get_info(0)[0]
                result["gpu_free_mb"] = free_bytes / 1024 ** 2
            print(f"  [{backend_name}] FITS — GPU={result['peak_gpu_mb']:.0f}MB  "
                  f"free={result['gpu_free_mb']:.0f}MB  CPU_RSS={result['peak_cpu_mb']:.0f}MB")
        else:
            result["status"]    = "error"
            result["error_msg"] = "setup() returned False"
            print(f"  [{backend_name}] FAILED (setup returned False)")
    except RuntimeError as e:
        msg = str(e)
        result["status"]    = "oom" if "out of memory" in msg.lower() else "error"
        result["error_msg"] = msg[:300]
        print(f"  [{backend_name}] OOM — {msg[:120]}")
    except Exception as e:
        result["status"]    = "error"
        result["error_msg"] = str(e)[:300]
        print(f"  [{backend_name}] ERROR — {str(e)[:120]}")
    finally:
        try:
            backend.teardown()
        except Exception:
            pass
        _free_memory()

    return result


# ─── Max decode length probe ──────────────────────────────────────────────────

def _run_one_probe(backend, prompt, cand, input_len, timeout_s) -> tuple:
    """Run a single probe candidate. Returns (RequestResult, entry dict)."""
    print(f"    output_len={cand} …", end=" ", flush=True)
    _set_alarm(timeout_s)
    try:
        r = backend.run_one(prompt, cand, input_len)
        _cancel_alarm()
    except _BenchTimeout as e:
        _cancel_alarm()
        r = RequestResult(status="timeout", error_msg=str(e))
    except RuntimeError as e:
        _cancel_alarm()
        msg = str(e)
        status = "oom" if "out of memory" in msg.lower() else "error"
        r = RequestResult(status=status, error_msg=msg[:200])
    except Exception as e:
        _cancel_alarm()
        r = RequestResult(status="error", error_msg=str(e)[:200])

    entry: Dict = {
        "output_len":        cand,
        "status":            r.status,
        "ttft_ms":           r.ttft_s * 1000,
        "decode_tok_per_s":  r.decode_tok_per_s,
        "e2e_s":             r.e2e_s,
        "peak_gpu_mb":       r.peak_gpu_mb,
        "peak_cpu_mb":       r.peak_cpu_mb,
        "error_msg":         r.error_msg,
    }

    if r.status == "success":
        print(f"OK  ({r.e2e_s:.1f}s, TTFT={r.ttft_s*1000:.0f}ms, "
              f"{r.decode_tok_per_s:.1f} tok/s, GPU={r.peak_gpu_mb:.0f}MB)")
    else:
        print(f"FAILED ({r.status}) — {r.error_msg[:80]}")

    return r, entry


def probe_max_decode_length(
    backend:    Backend,
    prompt:     str,
    input_len:  int,
    candidates: List[int],
    timeout_s:  int = 120,
) -> Dict:
    """
    Binary search over candidates to find the longest output length that succeeds.

    OOM/timeout is monotone: if length L fails, all L' > L also fail.
    Binary search finds the boundary in O(log N) probes instead of O(N).

    Returns:
      max_output_len  int   — largest successful output length (0 = all failed)
      results         list  — per-length dicts for every probe attempted
      oom_at          int   — first length that triggered OOM (None if never)
    """
    candidates = sorted(candidates)
    print(f"  [{backend.name}] max-decode probe (binary search): {candidates}")

    per_len: List[Dict] = []
    oom_at = None

    lo, hi = 0, len(candidates) - 1
    best_ok = -1  # index of highest confirmed success

    while lo <= hi:
        mid = (lo + hi) // 2
        cand = candidates[mid]

        r, entry = _run_one_probe(backend, prompt, cand, input_len, timeout_s)
        per_len.append(entry)

        if r.status == "success":
            best_ok = mid
            lo = mid + 1  # try longer
        else:
            if r.status == "oom" and oom_at is None:
                oom_at = cand
            hi = mid - 1  # try shorter
            _free_memory()

        if r.status == "success":
            _free_memory()

    max_len = candidates[best_ok] if best_ok >= 0 else 0

    return {
        "max_output_len": max_len,
        "oom_at":         oom_at,
        "results":        sorted(per_len, key=lambda e: e["output_len"]),
    }


# ─── Main experiment runner ───────────────────────────────────────────────────

def _hw_info() -> Dict:
    info: Dict = {"cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        info["gpu_name"]            = torch.cuda.get_device_name(0)
        info["gpu_count"]           = torch.cuda.device_count()
        info["gpu_memory_total_mb"] = torch.cuda.get_device_properties(0).total_memory / 1024 ** 2
    try:
        import psutil, platform
        info["cpu_count_physical"] = psutil.cpu_count(logical=False)
        info["cpu_count_logical"]  = psutil.cpu_count(logical=True)
        info["ram_total_gb"]       = psutil.virtual_memory().total / 1024 ** 3
        info["platform"]           = platform.platform()
    except Exception:
        pass
    try:
        import cpuinfo
        info["cpu_brand"] = cpuinfo.get_cpu_info().get("brand_raw", "unknown")
    except Exception:
        pass
    return info


def _build_prompts(n: int, base_prompt: str,
                   target_len: Optional[int] = None) -> List[str]:
    """Generate N slightly varied prompts for statistical measurement.

    When ``target_len`` is set, each prompt is padded (by repeating the base
    text) to contain at least ``target_len`` whitespace words, so that the
    per-backend ``truncation=True, max_length=input_len`` tokenization yields a
    genuine ``input_len``-token prefill.

    Without this, ``--input-len`` only ever *caps* a short prompt: the default
    base prompt is ~20 tokens, so every "context" in a long-context sweep runs
    the same tiny prompt — the context axis is a no-op, KV never grows, and KV
    paging never activates. A BPE-tokenised whitespace word is always >= 1
    token, so >= target_len words guarantees the tokenizer can fill target_len
    tokens before truncation trims the excess.
    """
    seeds = [
        "Explain the key differences between",
        "Describe the historical significance of",
        "What are the main advantages and disadvantages of",
        "Provide a detailed overview of",
        "Compare and contrast the approaches to",
        "Summarise the core principles behind",
        "What is the relationship between",
        "How does one typically approach",
        "Describe in detail the process of",
        "What are the most important aspects of",
    ]
    prompts = []
    for i in range(n):
        seed = seeds[i % len(seeds)]
        text = f"{seed} {base_prompt}"
        if target_len is not None and target_len > 0:
            words = text.split()
            needed = target_len + 32  # margin; truncation trims to input_len
            if len(words) < needed:
                reps = (needed // max(len(words), 1)) + 1
                words = (words * reps)[:needed]
                text = " ".join(words)
        prompts.append(text)
    return prompts


def run_experiment(
    backend:       Backend,
    prompts:       List[str],
    output_len:    int,
    input_len:     int,
    ablation_tag:  str = "",
    timeout_s:     int = 300,
) -> BenchStats:
    """Run N requests through backend and return aggregated stats."""
    # Reset KV paging counters before HELM runs so the totals reflect only this
    # sweep point. No-op import guard for non-HELM backends.
    if isinstance(backend, HelmBackend):
        from helm.runtime.kv_cache import reset_paging_stats
        reset_paging_stats()

    results = backend.run_n(prompts, output_len, input_len, timeout_s=timeout_s)
    stats = _aggregate(results, backend.name, ablation_tag)

    # Attach HELM compiler metrics if available
    if isinstance(backend, HelmBackend):
        stats.compile_time_s        = backend.compile_time_s
        stats.stage_plan            = backend.stage_plan
        stats.cost_model_decode_ms  = backend.cost_model_decode_ms
        stats.cost_model_prefill_ms = backend.cost_model_prefill_ms

        from helm.runtime.kv_cache import get_paging_stats
        ps = get_paging_stats()
        stats.kv_evict_calls      = ps["evict_calls"]
        stats.kv_pages_evicted    = ps["pages_evicted"]
        stats.kv_bytes_evicted    = ps["bytes_evicted"]
        stats.kv_prefetch_calls   = ps["prefetch_calls"]
        stats.kv_pages_prefetched = ps["pages_prefetched"]
        stats.kv_bytes_prefetched = ps["bytes_prefetched"]

    return stats


def _make_backend(name: str, args) -> Backend:
    if name == "vllm":
        return VLLMBackend(args.model, args.dtype,
                           gpu_memory_utilization=args.vllm_gpu_util)
    if name == "accelerate":
        return AccelerateBackend(args.model, args.dtype)
    if name == "deepspeed":
        return DeepSpeedBackend(args.model, args.dtype, cpu_offload=True)
    if name == "helm":
        # An explicit --cpu-layers pins the partition: switch the planner to
        # manual mode (auto-partition ignores cpu_layers). Holding the split
        # constant while sweeping input length gives a controlled TTFT study.
        _cpu_layers = getattr(args, "cpu_layers", None)
        return HelmBackend(
            args.model, args.dtype,
            kv_offload=not args.no_kv_offload,
            cpu_threads=args.cpu_threads,
            input_len=args.input_len,
            batch_size=args.batch_size,
            cpu_layers=_cpu_layers,
            plan_mode="manual" if _cpu_layers else "auto",
        )
    if name == "helm-router":
        return HelmRouterBackend(
            args.model, args.dtype,
            kv_offload=not args.no_kv_offload,
            cpu_threads=args.cpu_threads,
            input_len=args.input_len,
            batch_size=args.batch_size,
        )
    raise ValueError(f"Unknown backend: {name}")


def _incremental_save(results: Dict, path: Path):
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)


def run_all(args) -> Dict:
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "paper_results.json"

    results: Dict[str, Any] = {
        "config":            vars(args),
        "hardware":          _hw_info(),
        "feasibility":       {},
        "max_decode_length": {},
        "latency_sweep":     {},
        "throughput":        {},
        "ablations":         {},
        "lm_eval":           {},
    }
    _incremental_save(results, json_path)

    output_lens: List[int] = sorted(set(args.output_lens))
    _target_len = args.input_len if getattr(args, "pad_to_input_len", False) else None
    prompts = _build_prompts(args.num_requests, args.base_prompt, target_len=_target_len)
    if _target_len:
        print(f"[bench] pad-to-input-len ON: prompts padded to >= {_target_len} "
              f"words so --input-len={args.input_len} yields a real prefill.")

    # ── 0. Feasibility probe ──────────────────────────────────────────────────
    if not args.skip_feasibility:
        print(f"\n{'='*70}")
        print(f"  SECTION 0: Feasibility Probe  backends={args.backends}")
        print(f"  (Does the model load? What is the memory footprint?)")
        print(f"{'='*70}")
        for bname in args.backends:
            print(f"\n  [{bname}] loading model …")
            results["feasibility"][bname] = probe_feasibility(bname, args)
            _incremental_save(results, json_path)

    # ── 0b. Max decode length probe ───────────────────────────────────────────
    if not args.skip_max_decode:
        print(f"\n{'='*70}")
        print(f"  SECTION 0b: Max Decode Length  candidates={args.max_decode_candidates}")
        print(f"  (Longest sequence each backend can produce before OOM)")
        print(f"{'='*70}")
        probe_prompt = prompts[0]
        for bname in args.backends:
            print(f"\n  ── {bname.upper()} ──")
            backend = _make_backend(bname, args)
            ok = backend.setup()
            if not ok:
                results["max_decode_length"][bname] = {
                    "max_output_len": 0, "oom_at": None,
                    "results": [], "error": "setup failed",
                }
                _incremental_save(results, json_path)
                continue
            results["max_decode_length"][bname] = probe_max_decode_length(
                backend, probe_prompt, args.input_len,
                args.max_decode_candidates, timeout_s=args.timeout,
            )
            backend.teardown()
            _free_memory()
            _incremental_save(results, json_path)

        # Print comparison table
        print(f"\n  ── Max decode length summary ──")
        for bname, mres in results["max_decode_length"].items():
            max_l = mres.get("max_output_len", 0)
            oom   = mres.get("oom_at", "—")
            print(f"    {bname:15s}: max={max_l:6d}  oom_at={oom}")

    # ── 1. Latency sweep (all backends × all output lengths) ─────────────────
    if args.skip_latency_sweep:
        print("  [skip] Latency sweep disabled via --skip-latency-sweep")
    else:
      print(f"\n{'='*70}")
      print(f"  SECTION 1: Latency Sweep  backends={args.backends}  "
            f"output_lens={output_lens}  N={args.num_requests}")
      print(f"{'='*70}")

    for bname in args.backends if not args.skip_latency_sweep else []:
        print(f"\n{'─'*60}")
        print(f"  Backend: {bname.upper()}")
        print(f"{'─'*60}")

        # Pre-flight memory clean before each backend setup. vLLM specifically
        # rejects startup if free VRAM < ~0.85 of total (gpu_memory_utilization
        # default = 0.9). Previous backends leave 5-10 GB of allocator residue;
        # this loop drains it before we hand the GPU over.
        if torch.cuda.is_available():
            _MIN_FREE_FRACTION_LS = {
                "vllm": 0.85, "deepspeed": 0.30, "helm": 0.30, "accelerate": 0.30,
            }
            total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            min_free_gb = _MIN_FREE_FRACTION_LS.get(bname, 0.30) * total_gb
            ok_mem = force_clean_gpu(min_free_gb, label=f"latsweep-pre-{bname}")
            if not ok_mem:
                results["latency_sweep"][bname] = {
                    "error": (
                        f"pre_failed: free_gpu_gb={_gpu_free_gb():.2f} < "
                        f"required {min_free_gb:.2f}GB. Previous backend's "
                        f"allocator state could not be released."
                    )
                }
                _incremental_save(results, json_path)
                continue

        backend = _make_backend(bname, args)
        ok = backend.setup()
        if not ok:
            results["latency_sweep"][bname] = {"error": "setup failed"}
            _incremental_save(results, json_path)
            continue

        if args.num_warmup > 0:
            print(f"  [{bname}] Warming up ({args.num_warmup} request(s)) …")
            for _ in range(args.num_warmup):
                try:
                    backend.run_one(prompts[0], output_lens[0], args.input_len)
                except Exception:
                    pass

        results["latency_sweep"][bname] = {}
        for ol in output_lens:
            print(f"\n  output_len={ol} …")
            stats = run_experiment(
                backend, prompts[:args.num_requests], ol,
                args.input_len, ablation_tag=f"out={ol}",
                timeout_s=args.timeout,
            )
            d = asdict(stats)
            d.pop("raw_requests", None)   # keep JSON compact
            results["latency_sweep"][bname][str(ol)] = d
            _incremental_save(results, json_path)
            _print_stats(stats, ol)

        # Throughput sweep for this backend
        if args.throughput_concurrency:
            print(f"\n  [{bname}] Throughput sweep: {args.throughput_concurrency}")
            try:
                results["throughput"][bname] = throughput_sweep(
                    backend, prompts, output_lens[0],
                    args.input_len, args.throughput_concurrency)
            except Exception as exc:
                print(f"  [{bname}] Throughput sweep failed: {exc}")
                results["throughput"][bname] = {"error": str(exc)}
            _incremental_save(results, json_path)

        backend.teardown()
        # Post-teardown drain so the next backend's pre-flight has a head start.
        if torch.cuda.is_available():
            force_clean_gpu(0.0, label=f"latsweep-post-{bname}", max_attempts=3, sleep_s=1.0)
        else:
            _free_memory()

    # ── 2. Ablations (HELM only) ──────────────────────────────────────────────
    if "helm" in args.backends and args.ablations:
        print(f"\n{'='*70}")
        print(f"  SECTION 2: Ablations  {args.ablations}")
        print(f"{'='*70}")

        ablation_output_len = output_lens[len(output_lens) // 2]  # mid-range

        for ablation in args.ablations:
            print(f"\n  ── Ablation: {ablation} ──")
            try:
                ablation_results = _run_ablation(ablation, args, prompts,
                                                 ablation_output_len, args.input_len)
            except Exception as exc:
                import traceback
                print(f"  [ERROR] Ablation '{ablation}' crashed: {exc}")
                traceback.print_exc()
                ablation_results = {"error": str(exc)}
            results["ablations"][ablation] = ablation_results
            _incremental_save(results, json_path)

    # ── 3. LM quality benchmarks ──────────────────────────────────────────────
    if not args.no_lm_eval and args.lm_eval_tasks:
        print(f"\n{'='*70}")
        print(f"  SECTION 3: LM Quality Benchmarks  tasks={args.lm_eval_tasks}")
        print(f"{'='*70}")
        results["lm_eval"] = run_lm_eval(
            args.model, args.lm_eval_tasks, args.dtype,
            out_dir, limit=args.lm_eval_limit,
        )
        _incremental_save(results, json_path)

    print(f"\n{'='*70}")
    print(f"  All experiments done. Results: {json_path}")
    print(f"{'='*70}")
    return results


# Approximate parameter count (billions) for the paper's model set. Used ONLY
# to trim the slow batch_size / context_length sweeps on models that must
# CPU-offload on a 24 GB GPU — large batches / long contexts on an offloaded
# model run at <1 tok/s and blow past the per-cell timeout (Qwen3-14B/helm hit
# the 7200s wall mid batch=4). Trimming changes which ablation points we
# collect, never the measured values, so the JSON stays honest: fewer keys for
# big models, no fabricated numbers.
_MODEL_SIZE_B = {
    "qwen3-4b": 4, "qwen3-8b": 8, "qwen3-14b": 14, "qwen3-32b": 32,
    "llama-2-13b": 13, "llama-3.1-8b": 8, "mistral-nemo": 12, "mistral-7b": 7,
    "olmo-2-1124-13b": 13, "gemma-2-2b": 2, "gemma-2-9b": 9, "gemma-2-27b": 27,
}
# A model needs CPU offload on a 24 GB 3090 at ~2 bytes/param above ~11-12 B.
# Everything at or above this trims the slow sweeps.
_OFFLOAD_THRESHOLD_B = 12


def _estimate_params_b(model_name: str) -> float:
    """Best-effort parameter count in billions for ablation trimming only.
    Checks the known-model map first, then a regex on the id, else 0
    (treated as small → full sweep)."""
    ml = model_name.lower()
    for key, size in _MODEL_SIZE_B.items():
        if key in ml:
            return float(size)
    m = re.search(r"(\d+(?:\.\d+)?)\s*b\b", ml)
    return float(m.group(1)) if m else 0.0


def _run_ablation(
    ablation:   str,
    args,
    prompts:    List[str],
    output_len: int,
    input_len:  int,
) -> Dict:
    """Run one ablation study and return results dict."""
    ab_results = {}
    # On CPU-offloaded models the high-batch / long-context points are
    # catastrophically slow (<1 tok/s) and cause the cell to time out before
    # finishing. Trim those sweeps for models past the offload threshold.
    is_offloaded = _estimate_params_b(args.model) >= _OFFLOAD_THRESHOLD_B

    if ablation == "batch_size":
        batch_sizes = [1, 2] if is_offloaded else [1, 2, 4, 8]
        for bs in batch_sizes:
            tag = f"batch={bs}"
            print(f"  {tag} …")
            b = HelmBackend(args.model, args.dtype,
                            kv_offload=not args.no_kv_offload,
                            cpu_threads=args.cpu_threads,
                            input_len=input_len, batch_size=bs)
            if b.setup():
                stats = run_experiment(b, prompts[:args.num_requests], output_len,
                                       input_len, ablation_tag=tag, timeout_s=args.timeout)
                d = asdict(stats); d.pop("raw_requests", None)
                ab_results[tag] = d
                _print_stats(stats, output_len)
            else:
                ab_results[tag] = {"error": "setup failed"}
            b.teardown(); _free_memory()

    elif ablation == "context_length":
        ctx_lens = [128, 256, 512] if is_offloaded else [128, 256, 512, 1024, 2048]
        for ctx in ctx_lens:
            tag = f"ctx={ctx}"
            print(f"  {tag} …")
            b = HelmBackend(args.model, args.dtype,
                            kv_offload=not args.no_kv_offload,
                            cpu_threads=args.cpu_threads,
                            input_len=ctx, batch_size=args.batch_size)
            if b.setup():
                stats = run_experiment(b, prompts[:args.num_requests], output_len,
                                       ctx, ablation_tag=tag, timeout_s=args.timeout)
                d = asdict(stats); d.pop("raw_requests", None)
                ab_results[tag] = d
                _print_stats(stats, output_len)
            else:
                ab_results[tag] = {"error": "setup failed"}
            b.teardown(); _free_memory()

    elif ablation == "no_avx":
        for disable_avx in [False, True]:
            tag = "with_avx" if not disable_avx else "no_avx"
            print(f"  {tag} …")
            b = HelmBackend(args.model, args.dtype,
                            kv_offload=not args.no_kv_offload,
                            cpu_threads=args.cpu_threads,
                            input_len=input_len, batch_size=args.batch_size,
                            disable_avx=disable_avx)
            if b.setup():
                stats = run_experiment(b, prompts[:args.num_requests], output_len,
                                       input_len, ablation_tag=tag, timeout_s=args.timeout)
                d = asdict(stats); d.pop("raw_requests", None)
                ab_results[tag] = d
                _print_stats(stats, output_len)
            else:
                ab_results[tag] = {"error": "setup failed"}
            b.teardown(); _free_memory()

    elif ablation == "no_kv_offload":
        for kv_off in [True, False]:
            tag = "kv_offload" if kv_off else "no_kv_offload"
            print(f"  {tag} …")
            b = HelmBackend(args.model, args.dtype,
                            kv_offload=kv_off,
                            cpu_threads=args.cpu_threads,
                            input_len=input_len, batch_size=args.batch_size)
            if b.setup():
                stats = run_experiment(b, prompts[:args.num_requests], output_len,
                                       input_len, ablation_tag=tag, timeout_s=args.timeout)
                d = asdict(stats); d.pop("raw_requests", None)
                ab_results[tag] = d
                _print_stats(stats, output_len)
            else:
                ab_results[tag] = {"error": "setup failed"}
            b.teardown(); _free_memory()

    elif ablation == "cpu_threads":
        for threads in [1, 2, 4, 8, 16]:
            tag = f"threads={threads}"
            print(f"  {tag} …")
            b = HelmBackend(args.model, args.dtype,
                            kv_offload=not args.no_kv_offload,
                            cpu_threads=threads,
                            input_len=input_len, batch_size=args.batch_size)
            if b.setup():
                stats = run_experiment(b, prompts[:args.num_requests], output_len,
                                       input_len, ablation_tag=tag, timeout_s=args.timeout)
                d = asdict(stats); d.pop("raw_requests", None)
                ab_results[tag] = d
                _print_stats(stats, output_len)
            else:
                ab_results[tag] = {"error": "setup failed"}
            b.teardown(); _free_memory()

    elif ablation == "vs_baselines":
        # Direct comparison at fixed output_len: all 4 backends side by side
        for bname in ["accelerate", "deepspeed", "vllm", "helm"]:
            tag = bname
            print(f"  {tag} …")
            b = _make_backend(bname, args)
            if b.setup():
                stats = run_experiment(b, prompts[:args.num_requests], output_len,
                                       input_len, ablation_tag=tag, timeout_s=args.timeout)
                d = asdict(stats); d.pop("raw_requests", None)
                ab_results[tag] = d
                _print_stats(stats, output_len)
            else:
                ab_results[tag] = {"error": "setup failed"}
            b.teardown(); _free_memory()

    else:
        print(f"  Unknown ablation: {ablation}")

    return ab_results


def _print_stats(stats: BenchStats, output_len: int):
    ok_frac = f"{stats.n_success}/{stats.n_requests}"
    print(f"  [{stats.backend}] output={output_len}  ({ok_frac} succeeded)")
    if stats.n_success < stats.n_requests:
        if stats.first_error:
            print(f"    first error: {stats.first_error}")
    if stats.n_success == 0:
        return
    print(f"    TTFT    : p50={stats.ttft_p50:.1f}ms  p95={stats.ttft_p95:.1f}ms  "
          f"p99={stats.ttft_p99:.1f}ms  mean={stats.ttft_mean:.1f}ms")
    print(f"    Decode  : p50={stats.decode_lat_p50:.1f}ms/tok  "
          f"p95={stats.decode_lat_p95:.1f}ms/tok  "
          f"p99={stats.decode_lat_p99:.1f}ms/tok")
    print(f"    E2E     : p50={stats.e2e_p50:.0f}ms  p95={stats.e2e_p95:.0f}ms  "
          f"p99={stats.e2e_p99:.0f}ms")
    print(f"    Tok/s   : {stats.tok_per_s_mean:.1f}  |  "
          f"decode tok/s: {stats.decode_tok_per_s_mean:.1f}")
    print(f"    Memory  : GPU={stats.peak_gpu_mb_mean:.0f}MB  CPU={stats.peak_cpu_mb_mean:.0f}MB")
    kv_counts = (
        stats.kv_evict_calls,
        stats.kv_pages_evicted,
        stats.kv_bytes_evicted,
        stats.kv_prefetch_calls,
        stats.kv_pages_prefetched,
        stats.kv_bytes_prefetched,
    )
    if stats.backend.startswith("helm") or any(kv_counts):
        evicted_mb = stats.kv_bytes_evicted / 1e6
        prefetched_mb = stats.kv_bytes_prefetched / 1e6
        print(
            "    KV cache: "
            f"evict_calls={stats.kv_evict_calls}  "
            f"pages_evicted={stats.kv_pages_evicted}  "
            f"bytes_evicted={stats.kv_bytes_evicted}  "
            f"evicted_mb={evicted_mb:.1f}  "
            f"prefetch_calls={stats.kv_prefetch_calls}  "
            f"pages_prefetched={stats.kv_pages_prefetched}  "
            f"bytes_prefetched={stats.kv_bytes_prefetched}  "
            f"prefetched_mb={prefetched_mb:.1f}"
        )
    if stats.compile_time_s > 0:
        print(f"    Compiler: {stats.compile_time_s:.1f}s  plan=[{stats.stage_plan}]")
        print(f"    CostModel: decode={stats.cost_model_decode_ms:.1f}ms  "
              f"prefill={stats.cost_model_prefill_ms:.0f}ms")
        if stats.decode_lat_mean > 0:
            err_pct = abs(stats.cost_model_decode_ms - stats.decode_lat_mean) / stats.decode_lat_mean * 100
            print(f"    CostModel error: {err_pct:.1f}% vs measured {stats.decode_lat_mean:.1f}ms/tok")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct",
                   help="HuggingFace model name or local path")
    p.add_argument("--backends", nargs="+",
                   choices=["helm", "helm-router", "accelerate", "vllm", "deepspeed"],
                   default=["vllm", "accelerate", "deepspeed", "helm"],
                   help="Which backends to benchmark")
    p.add_argument("--base-prompt", default=(
        "transformer-based language models and their applications in natural language processing"),
                   help="Base text appended to each varied prompt")
    p.add_argument("--input-len", type=int, default=64,
                   help="Max input tokens (truncates prompt if needed)")
    p.add_argument("--pad-to-input-len", action="store_true",
                   help="Pad each prompt to >= --input-len tokens so the "
                        "context axis is real (required for long-context / KV "
                        "paging sweeps; off by default to preserve the existing "
                        "short-context prompt semantics).")
    p.add_argument("--output-lens", type=int, nargs="+",
                   default=[64, 128, 256, 512],
                   help="Output lengths for the latency sweep")
    p.add_argument("--num-requests", type=int, default=20,
                   help="Number of independent requests for statistical measurement")
    p.add_argument("--num-warmup", type=int, default=1,
                   help="Warmup requests before each timed section (discarded)")
    p.add_argument("--batch-size", type=int, default=1,
                   help="Request batch size (HELM and Accelerate)")
    p.add_argument("--dtype", default="float16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--timeout", type=int, default=3600,
                   help="Per-request wall-clock timeout in seconds")

    # Backend-specific
    p.add_argument("--no-kv-offload", action="store_true",
                   help="Disable KV offload in HELM")
    p.add_argument("--cpu-threads", type=int, default=8,
                   help="OMP/MKL thread count for CPU stages")
    p.add_argument("--cpu-layers", default=None,
                   help="Pin the HELM CPU stage to a fixed layer range (e.g. "
                        "'0:23' for layers 0-23 inclusive), bypassing the "
                        "auto-partition planner. Used to hold the partition "
                        "constant while sweeping input length (Fig. 4 "
                        "TTFT-vs-prompt-length study).")
    p.add_argument("--vllm-gpu-util", type=float, default=0.90)

    # Throughput. Defaults to [1, 2, 4, 8] when the flag is omitted OR given
    # with no values (wrappers commonly pass the bare flag to opt-in). To skip
    # the sweep, pass `--throughput-concurrency 0` or use `--no-throughput`.
    p.add_argument("--throughput-concurrency", type=int, nargs="*",
                   default=[1, 2, 4, 8],
                   help="Concurrency levels for throughput sweep ([0] to skip)")

    # Ablations. Same nargs convention: bare `--ablations` flag means "run a
    # sensible default set" rather than "skip silently".
    _ABLATION_CHOICES = ["batch_size", "context_length", "no_avx",
                         "no_kv_offload", "cpu_threads", "vs_baselines"]
    p.add_argument("--ablations", nargs="*",
                   choices=_ABLATION_CHOICES,
                   default=[],
                   help="Ablation studies to run (all HELM unless noted). Bare "
                        "flag --ablations expands to ['batch_size', "
                        "'context_length', 'no_kv_offload'].")

    # Section skip flags (used by run_paper_experiments.sh to isolate one section per subprocess)
    p.add_argument("--skip-latency-sweep", action="store_true",
                   help="Skip the latency sweep (Section 1)")
    p.add_argument("--skip-feasibility", action="store_true",
                   help="Skip the memory feasibility probe (Section 0)")
    p.add_argument("--skip-max-decode", action="store_true",
                   help="Skip the max decode length probe (Section 0b)")
    p.add_argument("--max-decode-candidates", type=int, nargs="+",
                   default=[128, 512, 1024, 2048,
                            3072, 4096, 5120, 6144, 7168, 8192, 9216, 10240,
                            11264, 12288, 13312, 14336, 15360, 16384, 17408, 18432,
                            19456, 20480, 21504, 22528, 23552, 24576, 25600, 26624,
                            27648, 28672, 29696, 30720, 31744, 32768],
                   help="Output lengths to try in the max-decode probe (stops at first OOM)")

    # LM quality
    p.add_argument("--no-lm-eval", action="store_true", default=True,
                   help="Skip lm-evaluation-harness quality benchmarks (default: True)")
    p.add_argument("--lm-eval-tasks", nargs="+",
                   default=["mmlu", "hellaswag", "arc_easy"],
                   help="lm-eval tasks to evaluate")
    p.add_argument("--lm-eval-limit", type=int, default=200,
                   help="Max samples per task (None = all; use small value for quick checks)")

    # Output
    p.add_argument("--output-dir", default="experiments/results",
                   help="Directory to write JSON results")

    args = p.parse_args()

    # Promote bare flag invocations to sensible defaults. argparse's nargs="*"
    # converts `--throughput-concurrency` (no values) to [] which is falsy and
    # silently skips the sweep — almost always the opposite of what the user
    # meant when passing the flag. Use [0] to skip explicitly.
    import sys as _sys
    raw = _sys.argv
    if "--throughput-concurrency" in raw and args.throughput_concurrency == []:
        args.throughput_concurrency = [1, 2, 4, 8]
    if "--ablations" in raw and args.ablations == []:
        args.ablations = ["batch_size", "context_length", "no_kv_offload"]
    # Explicit-skip sentinel: [0] means user wants to skip the sweep entirely.
    if args.throughput_concurrency == [0]:
        args.throughput_concurrency = []
    return args


def _latency_cell_ok(cell: dict) -> Tuple[bool, str]:
    """The single definition of a VALID core-grid cell.

    A latency-sweep cell is valid iff: no 'error' key, n_success == n_requests
    > 0, ttft_p50 > 0, and peak_gpu_mb_mean >= 1 GB (a multi-B model that
    reports < 1 GB never actually loaded). Returns (ok, reason_if_not)."""
    if not isinstance(cell, dict) or not cell:
        return False, "no data"
    if "error" in cell:
        return False, str(cell["error"])[:120]
    row = next(iter(cell.values()), {})
    if not row:
        return False, "empty row"
    n = row.get("n_success", 0) or 0
    nr = row.get("n_requests", 0) or 0
    ttft = row.get("ttft_p50", 0) or 0
    peak = row.get("peak_gpu_mb_mean", 0) or 0
    if n == 0 or n != nr:
        return False, f"n_success={n}/{nr}"
    if ttft <= 0:
        return False, f"ttft_p50={ttft}"
    if peak < 1024:
        return False, f"peak_gpu_mb={peak:.0f}MB (<1GB)"
    return True, ""


def main():
    args = _parse_args()

    print(f"\n{'#'*70}")
    print(f"  HELM Paper Experiments")
    print(f"  Model      : {args.model}")
    print(f"  Backends   : {args.backends}")
    print(f"  Output lens: {args.output_lens}")
    print(f"  N requests : {args.num_requests}")
    print(f"  Feasibility: {'skip' if args.skip_feasibility else 'yes'}")
    print(f"  Max decode : {'skip' if args.skip_max_decode else str(args.max_decode_candidates)}")
    print(f"  Ablations  : {args.ablations or 'none'}")
    print(f"  LM eval    : {'disabled' if args.no_lm_eval else args.lm_eval_tasks}")
    hw = _hw_info()
    print(f"  Hardware   : {hw.get('cpu_brand','?')}  "
          f"RAM={hw.get('ram_total_gb',0):.0f}GB  "
          f"GPU={hw.get('gpu_name','none')}")
    print(f"{'#'*70}\n")

    results = run_all(args)

    # ── Exit code MUST reflect data validity, not merely "the process ran". ──
    # Wrapper scripts key their .done sentinels and STATUS=ok off this exit
    # code. If we exit 0 with a failed cell — e.g. a pre_failed allocator-leak
    # or a setup OOM that we recorded in the JSON but swallowed — the wrapper
    # records a false 'ok' and the bad cell is never re-run. So: exit non-zero
    # unless every requested backend produced a valid latency-sweep cell.
    if args.skip_latency_sweep:
        return  # caller opted out of the core measurement; nothing to gate on
    ls = (results or {}).get("latency_sweep", {})
    failures = []
    for bname in args.backends:
        ok, why = _latency_cell_ok(ls.get(bname, {}))
        if not ok:
            failures.append(f"{bname}: {why}")
    if failures:
        print(f"\n{'!'*70}")
        print(f"  CELL FAILED — {len(failures)}/{len(args.backends)} backend(s) "
              f"produced no valid latency data:")
        for f in failures:
            print(f"    - {f}")
        print(f"  Exiting non-zero so the wrapper records FAILED (not a false ok).")
        print(f"{'!'*70}")
        sys.exit(1)
    print(f"\n[validation] all {len(args.backends)} requested backend(s) "
          f"produced valid latency data.")


if __name__ == "__main__":
    main()
