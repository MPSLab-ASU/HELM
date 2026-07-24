"""
experiments/verify_e2e.py
=========================
End-to-end correctness check for the HELM pipeline.

Samples prompts from experiments/data/verify_prompts.json, generates a greedy
Hugging Face reference for each (plain `model.generate`, same dtype), then
deploys the model with auto CPU/GPU partitioning, runs each prompt through the
full HELM pipeline and compares the generated token IDs with the reference.
Results are saved to artifacts/verify_<timestamp>/results.json.

A prompt passes when HELM's tokens are identical to the reference, or when the
first divergence is a near-tie: HELM's token scores within --near-tie-tol
logits of the reference's top token at that step (fp16 kernels on different
devices legitimately break such ties differently). Exits with code 1 if any
prompt fails or diverges beyond that tolerance.

Usage:
    python experiments/verify_e2e.py --model Qwen/Qwen3-0.6B
    python experiments/verify_e2e.py --model Qwen/Qwen3-0.6B --kv-offload
    python experiments/verify_e2e.py --gpu-id 0 --seed 42 --num-prompts 5
    python experiments/verify_e2e.py --no-reference     # smoke only, no comparison
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.fx as fx
import torch.nn as nn

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent))

# ─── ShareGPT loading ────────────────────────────────────────────────────────

# Default prompts ship with the repo so verify_e2e is reproducible across
# environments (laptop, GPU server, CI). A ShareGPT file can still be passed
# explicitly via --sharegpt-data when broader prompt diversity is wanted.
_DEFAULT_PROMPTS_PATH = str(_HERE / "data" / "verify_prompts.json")
_DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
_DEFAULT_NEAR_TIE_TOL = 0.1


def _load_sharegpt_prompts(path: str, n: int, rng: random.Random) -> list[str]:
    """Return n prompts sampled from the prompts file at `path`.

    Supports two file shapes:
      1) ShareGPT-style:  [{"conversations": [{"from": "human", "value": "..."}, ...]}, ...]
      2) Flat list-of-strings:  ["prompt 1", "prompt 2", ...]

    Raises FileNotFoundError / RuntimeError on any failure. There is no silent
    fallback: a missing or empty prompts file must surface, because the
    benchmark audit trail depends on knowing exactly which prompts were used.
    """
    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"verify_e2e prompts file not found at '{path}'. "
            f"Pass --sharegpt-data <path> or commit a prompts file at the default location."
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"verify_e2e prompts file at '{path}' is not valid JSON: {exc}."
        ) from exc

    if isinstance(data, list) and data and all(isinstance(item, str) for item in data):
        candidates: list[str] = [s.strip() for s in data if s and s.strip()]
    else:
        candidates = []
        for conv in data:
            for turn in conv.get("conversations", []):
                if turn.get("from") in ("human", "user"):
                    text = turn.get("value", "").strip()
                    if text:
                        candidates.append(text)

    if not candidates:
        raise RuntimeError(
            f"verify_e2e prompts file at '{path}' produced zero usable prompts. "
            f"Expected either a list of strings or ShareGPT-style conversations."
        )

    return rng.sample(candidates, min(n, len(candidates)))


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _make_output_dir(base: str = "artifacts") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(base, f"verify_{ts}")
    os.makedirs(out, exist_ok=True)
    return out


def _format_prompt_as_chat(tokenizer, prompt: str) -> str:
    """Format a raw user prompt through the tokenizer's chat template when available."""
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    chat_template = getattr(tokenizer, "chat_template", None)
    if not callable(apply_chat_template) or not chat_template:
        return prompt

    try:
        return apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except (TypeError, ValueError):
        return prompt


def _build_prefill_inputs(tokenizer, prompt: str, max_input_tokens: int):
    """Tokenise prompt and build the 4-tuple (input_ids, attn_mask, position_ids, cache_position)."""
    prompt_text = _format_prompt_as_chat(tokenizer, prompt)
    tok = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_input_tokens)
    input_ids = tok["input_ids"]
    attn = tok["attention_mask"]
    seq_len = input_ids.shape[1]

    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    cache_position = torch.arange(seq_len, dtype=torch.long)

    min_val = torch.finfo(torch.float16).min
    causal = torch.triu(torch.full((seq_len, seq_len), min_val, dtype=torch.float16), diagonal=1)
    inv_mask = (1.0 - attn.float()) * min_val
    attn_mask = causal[None, None, :, :] + inv_mask[:, None, None, :]

    return input_ids, attn_mask, position_ids, cache_position


def _build_compile_options_common(model_name: str, workload: dict) -> dict:
    return {
        "mode": "both",
        "objective": "decode_latency",
        "plan_mode": "auto",
        "lower_stages": True,
        "model_name": model_name,
        "workload": workload,
        # verify_e2e intentionally loads with device_map="auto" and removes
        # Accelerate hooks before HELM stage lowering. The unpartitioned graph
        # is not dynamically executable across CPU/GPU, so this harness opts
        # into the compiler's explicit static analysis fallback.
        "allow_static_analysis_fallback": True,
    }


# ─── FX tracer for the prefill graph ─────────────────────────────────────────

class _HelmPrefillTracer(fx.Tracer):
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


class _PrefillWrapper(nn.Module):
    """
    Wraps a causal LM for FX prefill tracing.

    Bakes attention_mask / position_ids / cache_position in as constants so
    the traced graph's only input is input_ids.  Applies the Gemma embedding
    normalisation (sqrt(hidden_size)) that model.forward() normally provides
    but which we bypass by calling embed_tokens directly.
    """

    def __init__(self, model, attention_mask, position_ids, cache_position):
        super().__init__()
        self.inner = model
        self.attention_mask = attention_mask
        self.position_ids = position_ids
        self.cache_position = cache_position

        first_layer = model.model.layers[0]
        layer_sig = inspect.signature(first_layer.forward)
        self._use_pos_embed = "position_embeddings" in layer_sig.parameters

        # Probe first layer to detect tuple vs plain-tensor output
        first_param = next(first_layer.parameters(), None)
        first_dtype = first_param.dtype if first_param is not None else torch.float32
        first_device = (
            first_param.device
            if first_param is not None and not getattr(first_param, "is_meta", False)
            else torch.device("cpu")
        )
        hidden_size = int(
            getattr(model.config, "hidden_size", 0)
            or getattr(model.model.embed_tokens, "embedding_dim", 0)
            or 1
        )
        _h = torch.zeros(1, 1, hidden_size, dtype=first_dtype, device=first_device)
        _p = torch.zeros(1, 1, dtype=torch.long, device=first_device)
        _cp = torch.zeros(1, dtype=torch.long, device=first_device)
        _m = torch.zeros(1, 1, 1, 1, dtype=first_dtype, device=first_device)
        _kw = {"attention_mask": _m, "position_ids": _p, "cache_position": _cp, "use_cache": False}
        if self._use_pos_embed and hasattr(model.model, "rotary_emb"):
            _kw["position_embeddings"] = model.model.rotary_emb(_h, position_ids=_p)
        with torch.no_grad():
            _out = first_layer(_h, **_kw)
        self._layer_returns_tuple = isinstance(_out, (tuple, list))

        # Gemma models scale embeddings by sqrt(hidden_size) in model.forward() before
        # the first decoder layer. We bypass model.forward() so we must apply it here.
        model_type = getattr(model.config, "model_type", "").lower()
        self._embed_scale = float(hidden_size ** 0.5) if "gemma" in model_type else None

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
            hidden_states = layer_out[0] if self._layer_returns_tuple else layer_out

        if hasattr(self.inner.model, "norm"):
            hidden_states = self.inner.model.norm(hidden_states)

        logits = self.inner.lm_head(hidden_states)
        softcap = getattr(self.inner.config, "final_logit_softcapping", None)
        if softcap is not None:
            logits = torch.tanh(logits / softcap) * softcap
        return logits


# ─── Build HELM runtime ───────────────────────────────────────────────────────

def _load_hf_model(model_name: str, dtype):
    """Load a Hugging Face model the way build_helm_runtime does (GPU first,
    spilling to CPU RAM), so reference and HELM see identical weights."""
    from transformers import AutoModelForCausalLM

    max_memory = None
    if torch.cuda.is_available():
        gpu_total_mb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 2
        try:
            import psutil
            cpu_budget = f"{int(psutil.virtual_memory().total / 1024**2 - 2048)}MiB"
        except ImportError:
            cpu_budget = "24GiB"
        max_memory = {0: f"{int(gpu_total_mb * 0.80)}MiB", "cpu": cpu_budget}
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map="auto",
        max_memory=max_memory,
        low_cpu_mem_usage=True,
    ).eval()


@torch.inference_mode()
def compute_hf_references(*, model_name, dtype_str, prompts, max_input_tokens, max_new_tokens):
    """Greedy Hugging Face reference per prompt: token IDs + per-step logits."""
    from transformers import AutoTokenizer

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(dtype_str, torch.float16)
    print(f"[verify] Building Hugging Face greedy reference ({model_name}) ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = _load_hf_model(model_name, dtype)
    device = next(model.parameters()).device
    refs = []
    for prompt in prompts:
        input_ids = _build_prefill_inputs(tokenizer, prompt, max_input_tokens)[0].to(device)
        out = model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            output_logits=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.pad_token_id,
        )
        refs.append({
            "token_ids": out.sequences[0, input_ids.shape[1]:].tolist(),
            "logits": [step[0].float().cpu() for step in out.logits],
        })
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return refs


def compare_to_reference(helm_ids, ref, near_tie_tol: float = _DEFAULT_NEAR_TIE_TOL) -> dict:
    """Compare HELM token IDs with a reference from compute_hf_references."""
    ref_ids = ref["token_ids"]
    n = min(len(helm_ids), len(ref_ids))
    for k in range(n):
        if helm_ids[k] != ref_ids[k]:
            logits = ref["logits"][k]
            gap = float(logits.max() - logits[helm_ids[k]])
            return {
                "match": "near_tie" if gap <= near_tie_tol else "diverged",
                "compared_tokens": n,
                "first_divergence": k,
                "logit_gap": round(gap, 4),
            }
    return {"match": "identical", "compared_tokens": n}


def build_helm_runtime(
    model_name: str,
    dtype_str: str,
    cpu_threads: int,
    max_input_tokens: int,
    max_new_tokens: int,
    kv_offload: bool = False,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from helm.compiler.compiler import HelmCompileOptions, compile_graph
    from helm.compiler.importers.decode_tracer import DecodeTracer
    from helm.compiler.importers.patch_fx import apply_fx_patch
    from helm.runtime.executor import StageRuntimeExecutor
    from helm.runtime.kv_offload import KVOffloadConfig, KVOffloadManager
    from helm.runtime.pipeline_runtime import PipelineRuntime

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(dtype_str, torch.float16)
    torch.set_num_threads(cpu_threads)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"[verify] Loading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    gpu_budget = None
    if torch.cuda.is_available():
        gpu_total_mb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 2
        gpu_budget = f"{int(gpu_total_mb * 0.80)}MiB"

    try:
        import psutil
        cpu_budget = f"{int(psutil.virtual_memory().total / 1024**2 - 2048)}MiB"
    except ImportError:
        cpu_budget = "24GiB"

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

    apply_fx_patch()

    # ── Trace & compile prefill ───────────────────────────────────────────────
    print("[verify] Tracing prefill graph ...")
    # Use a short fixed sentence for tracing — shape matters, not content.
    dummy_inputs = _build_prefill_inputs(tokenizer, "Hello world.", max_input_tokens)
    _, attn_mask, position_ids, cache_position = dummy_inputs

    wrapper = _PrefillWrapper(model, attn_mask, position_ids, cache_position)
    tracer = _HelmPrefillTracer()
    prefill_gm = fx.GraphModule(wrapper, tracer.trace(wrapper))
    print(f"[verify]   Prefill graph: {len(prefill_gm.graph.nodes)} nodes")

    workload = {
        "batch_size": 1,
        "prefill_seq_len": max_input_tokens,
        "decode_context_len": max_input_tokens,
        "decode_tokens": max_new_tokens,
        "dtype_size": 2,
    }
    opts_common = _build_compile_options_common(model_name, workload)

    print("[verify] Compiling prefill ...")
    prefill_art = compile_graph(
        gm=prefill_gm,
        example_inputs=dummy_inputs,
        model=model,
        tokenizer=tokenizer,
        options=HelmCompileOptions(**opts_common, graph_kind="prefill"),
        artifacts_dir=None,
    )

    # ── Trace & compile decode ────────────────────────────────────────────────
    print("[verify] Tracing decode graph ...")
    dec_dummy = DecodeTracer.build_dummy_inputs(device="cpu", batch_size=1, dtype=torch.float16)
    dec_tracer = DecodeTracer(model)
    decode_gm = dec_tracer.trace()
    decode_wrapper = getattr(model, "_helm_decode_wrapper", None)

    print("[verify] Compiling decode ...")
    decode_art = compile_graph(
        gm=decode_gm,
        example_inputs=dec_dummy,
        model=model,
        tokenizer=tokenizer,
        options=HelmCompileOptions(**opts_common, graph_kind="decode"),
        partition_plan_override=prefill_art.partition_plan,
        artifacts_dir=None,
    )

    # ── Assemble runtime ──────────────────────────────────────────────────────
    prefill_exec = StageRuntimeExecutor(prefill_art.stage_graphs)
    decode_exec = StageRuntimeExecutor(decode_art.stage_graphs)

    kv_mgr = None
    if kv_offload:
        kv_cfg = KVOffloadConfig.from_model(model)
        kv_mgr = KVOffloadManager(model, kv_cfg, batch_size=1)
        kv_mgr.patch_module_roots(
            *(stage.module for stage in prefill_exec.stages),
            *(stage.module for stage in decode_exec.stages),
        )

    runtime = PipelineRuntime(
        prefill_exec,
        decode_exec,
        tokenizer=tokenizer,
        dtype=dtype,
        kv_offload_mgr=kv_mgr,
        decode_wrapper=decode_wrapper,
    )
    return runtime, tokenizer


# ─── Run a single prompt ──────────────────────────────────────────────────────

def run_prompt(runtime, tokenizer, prompt: str, max_input_tokens: int, max_new_tokens: int) -> dict:
    input_ids, *_ = _build_prefill_inputs(tokenizer, prompt, max_input_tokens)

    _sync()
    t0 = time.perf_counter()
    generated = runtime.generate(input_ids, max_new_tokens)
    _sync()
    elapsed_s = time.perf_counter() - t0

    response = tokenizer.batch_decode(generated, skip_special_tokens=True)
    return {
        "prompt": prompt,
        "response": response[0] if response else "",
        "token_ids": generated[0].tolist() if isinstance(generated, torch.Tensor) else [],
        "tokens_generated": generated.shape[-1] if isinstance(generated, torch.Tensor) else max_new_tokens,
        "elapsed_s": round(elapsed_s, 4),
        "status": "ok",
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="HELM end-to-end verification")
    parser.add_argument("--model", default=_DEFAULT_MODEL)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--max-input-tokens", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--gpu-id", type=int, default=0,
                        help="Physical GPU index to expose as device 0 (sets CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for prompt sampling; generated randomly if omitted")
    parser.add_argument("--num-prompts", type=int, default=5,
                        help="Number of ShareGPT prompts to sample")
    parser.add_argument("--sharegpt-data", default=_DEFAULT_PROMPTS_PATH,
                        help="Path to ShareGPT JSON dataset")
    parser.add_argument(
        "--kv-offload",
        action="store_true",
        help="Enable the paged KV cache with CPU offload (off by default here).",
    )
    parser.add_argument("--no-reference", action="store_true",
                        help="Skip the Hugging Face reference comparison (smoke run only)")
    parser.add_argument("--near-tie-tol", type=float, default=_DEFAULT_NEAR_TIE_TOL,
                        help="Max reference logit gap for a divergence to count as a "
                             "near-tie rather than a failure (default: %(default)s)")
    args = parser.parse_args()

    # Must be set before any CUDA call
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    seed = args.seed if args.seed is not None else random.randint(0, 2**31 - 1)
    rng = random.Random(seed)
    prompts = _load_sharegpt_prompts(args.sharegpt_data, args.num_prompts, rng)

    out_dir = _make_output_dir(args.artifacts_dir)
    print(f"[verify] Output dir    : {out_dir}")
    print(f"[verify] Model         : {args.model}")
    print(f"[verify] GPU id        : {args.gpu_id}")
    print(f"[verify] Seed          : {seed}")
    print(f"[verify] Num prompts   : {len(prompts)}")
    print(f"[verify] Max new tokens: {args.max_new_tokens}")
    print(f"[verify] KV offload    : {args.kv_offload}")

    references = None
    if not args.no_reference:
        references = compute_hf_references(
            model_name=args.model,
            dtype_str=args.dtype,
            prompts=prompts,
            max_input_tokens=args.max_input_tokens,
            max_new_tokens=args.max_new_tokens,
        )

    runtime, tokenizer = build_helm_runtime(
        model_name=args.model,
        dtype_str=args.dtype,
        cpu_threads=args.cpu_threads,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        kv_offload=args.kv_offload,
    )

    results = []
    any_failed = False

    print(f"\n[verify] Running {len(prompts)} prompts ...\n")
    for i, prompt in enumerate(prompts, start=1):
        print(f"  [{i}/{len(prompts)}] {prompt[:70]}...")
        try:
            r = run_prompt(runtime, tokenizer, prompt, args.max_input_tokens, args.max_new_tokens)
            if references is not None:
                r["reference"] = compare_to_reference(
                    r["token_ids"], references[i - 1], args.near_tie_tol)
                ref = r["reference"]
                if ref["match"] == "identical":
                    verdict = f"token-identical to HF ({ref['compared_tokens']} tokens)"
                elif ref["match"] == "near_tie":
                    verdict = (f"near-tie divergence at token {ref['first_divergence']} "
                               f"(HF logit gap {ref['logit_gap']})")
                else:
                    verdict = (f"DIVERGED from HF at token {ref['first_divergence']} "
                               f"(HF logit gap {ref['logit_gap']})")
                    r["status"] = "diverged"
                    any_failed = True
            else:
                verdict = "no reference comparison"
            label = "OK " if r["status"] == "ok" else "FAIL"
            print(f"         -> {label} ({r['elapsed_s']:.2f}s, {r['tokens_generated']} tokens; {verdict})")
            print(f"         -> \"{r['response'][:80]}\"")
        except Exception as exc:
            print(f"         -> FAILED: {exc}")
            r = {
                "prompt": prompt, "response": None,
                "tokens_generated": 0, "elapsed_s": 0.0,
                "status": "failed", "error": str(exc),
            }
            any_failed = True
        results.append(r)

    summary = {
        "model": args.model,
        "dtype": args.dtype,
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
        "gpu_id": args.gpu_id,
        "kv_offload": args.kv_offload,
        "reference": "none" if args.no_reference else "hf_greedy",
        "near_tie_tol": args.near_tie_tol,
        "seed": seed,
        "num_prompts": len(prompts),
        "num_failed": sum(1 for r in results if r["status"] != "ok"),
        "results": results,
    }
    results_path = os.path.join(out_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[verify] Results saved to {results_path}")
    if any_failed:
        n = summary["num_failed"]
        print(f"\n[verify] FAILED: {n}/{len(prompts)} prompts failed.")
        sys.exit(1)
    else:
        print(f"\n[verify] All {len(prompts)} prompts passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
