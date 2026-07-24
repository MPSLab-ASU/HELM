"""Local greedy inference through HELM, with automatic vLLM routing.

HELM's own CPU+GPU pipeline is the only path that can serve a model larger
than VRAM, and it is what runs whenever the partition plan places at least
one stage on CPU. When the plan places every stage on GPU (the model fits in
VRAM), vLLM is faster, so the backend is selected by
``HelmInferenceConfig.route_to_vllm_when_all_gpu``:

* ``None`` (default, "auto"): route fit-in-VRAM models to vLLM when the
  ``vllm`` package is installed; otherwise run HELM's native executor and
  log how to enable vLLM. Never fails because vLLM is missing.
* ``True``: always route fit-in-VRAM models to vLLM; setup raises an
  actionable error if vLLM is not installed.
* ``False``: always use HELM's native executor.

The ``helm`` CLI maps ``--backend auto|router|helm`` onto these values.
"""

from __future__ import annotations

import gc
import logging
import os
import threading
from dataclasses import dataclass
from typing import List, Optional, Protocol

import torch

from helm.compiler.partition.partition_plan import PartitionPlan

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class HelmInferenceConfig:
    """Configuration for HelmInference.

    ``model_id`` accepts a local save_pretrained directory or model ID.
    Input limits reject oversized prompts; they never silently truncate.
    ``route_to_vllm_when_all_gpu``: None (auto) routes models that fit in
    VRAM to vLLM when it is installed; True requires vLLM; False always
    uses HELM's native executor.
    """

    model_id: str
    dtype: torch.dtype = torch.float16
    max_input_tokens: int = 64
    max_new_tokens: int = 64
    kv_offload: bool = True
    plan_mode: str = "auto"
    cpu_layers: Optional[str] = None
    gpu_layers: Optional[str] = None
    cpu_threads: int = 8
    route_to_vllm_when_all_gpu: Optional[bool] = None
    vllm_gpu_memory_utilization: float = 0.85
    vllm_max_model_len: int = 4096

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("model_id must be a non-empty string")
        if self.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("dtype must be torch.float16, torch.bfloat16, or torch.float32")
        for name in ("max_input_tokens", "cpu_threads", "vllm_max_model_len"):
            _validate_token_count(getattr(self, name), name, minimum=1)
        _validate_token_count(self.max_new_tokens, "max_new_tokens")
        if self.plan_mode not in ("auto", "manual"):
            raise ValueError("plan_mode must be 'auto' or 'manual'")
        if self.route_to_vllm_when_all_gpu not in (None, True, False):
            raise ValueError("route_to_vllm_when_all_gpu must be None, True, or False")
        if not 0 < self.vllm_gpu_memory_utilization <= 1:
            raise ValueError("vllm_gpu_memory_utilization must be in (0, 1]")


def _validate_token_count(value, name: str, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _encode_prompt(tokenizer, prompt: str, config: HelmInferenceConfig,
                   max_new_tokens: int, context_limit=None):
    encoded = tokenizer(prompt, return_tensors="pt", truncation=False)
    length = encoded["input_ids"].shape[1]
    if length == 0:
        raise ValueError("prompt must encode to at least one token")
    if length > config.max_input_tokens:
        raise ValueError(
            f"prompt has {length} tokens, exceeding max_input_tokens={config.max_input_tokens}; "
            "shorten the prompt or increase the limit"
        )
    if context_limit is not None and length + max_new_tokens > context_limit:
        raise ValueError(
            f"prompt plus max_new_tokens exceeds model context limit {context_limit}"
        )
    return encoded


class _Handle(Protocol):
    """Common surface for the two backend handles."""

    def generate(self, prompts: List[str], max_new_tokens: int) -> List[str]: ...

    def teardown(self) -> None: ...


def _is_all_gpu(plan: Optional[PartitionPlan]) -> bool:
    """True iff every partition stage targets a CUDA device.

    Mirrors the detection used by paper_bench (commit cc1bf42) for the
    contiguous-KV fast-path heuristic; kept in one place so router and
    benchmark agree.
    """
    if plan is None or not plan.stages:
        return False
    return all("cuda" in (stage.device_id or "") for stage in plan.stages)


def _vllm_available() -> bool:
    """True iff the optional ``vllm`` package can be imported."""
    import importlib.util

    return importlib.util.find_spec("vllm") is not None


def _free_gpu() -> None:
    """Drop torch's caching allocator and force a GC pass.

    HELM's compile artifacts and HF model objects hold circular refs that
    need a few GC sweeps to break; vLLM cannot share the GPU until they
    are released.
    """
    for _ in range(3):
        gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


class HelmInference:
    """User-facing inference handle that routes to vLLM or HELM by plan.

    Usage::

        inference = HelmInference(HelmInferenceConfig(model_id="Qwen/Qwen2.5-1.5B-Instruct"))
        inference.setup()
        outputs = inference.generate(["Hello, world!"], max_new_tokens=64)
        print(inference.routed_to_vllm)  # True if the model fit in VRAM and vLLM is installed
        inference.teardown()

    Construction does not allocate; call ``setup()`` to load the model and
    compile the partition plan.
    """

    def __init__(self, config: HelmInferenceConfig) -> None:
        self.config = config
        self._lock = threading.RLock()
        self._handle: Optional[_Handle] = None
        self._routed_to_vllm = False
        self._partition_plan: Optional[PartitionPlan] = None
        self._gemma2_patched = False

    @property
    def routed_to_vllm(self) -> bool:
        """Whether the most recent ``setup()`` chose the vLLM backend."""
        return self._routed_to_vllm

    @property
    def partition_plan(self) -> Optional[PartitionPlan]:
        """The HELM partition plan observed during ``setup()`` (None before setup)."""
        return self._partition_plan

    def __enter__(self) -> "HelmInference":
        self.setup()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.teardown()

    def setup(self) -> None:
        """Load once; failed setup releases partially constructed resources."""
        with self._lock:
            if self._handle is not None:
                return
            try:
                self._setup()
            except BaseException:
                self.teardown()
                raise

    def _setup(self) -> None:
        """Compile the partition plan and bind a backend handle.

        On the all-GPU path with routing enabled, this loads the HF model
        once to compile the plan, then frees it and lets vLLM reload —
        unavoidable since vLLM owns its model state. The overhead is a
        one-time setup cost amortised over many requests.
        """
        # gemma-2's Gemma2DecoderLayer.forward returns a single-element
        # tuple that breaks HELM's FX trace assumptions. Patch it before
        # the model loads so the traced graph has the right shape.
        if _is_gemma2_model_name(self.config.model_id):
            _patch_gemma2_decoder_outputs()
            self._gemma2_patched = True

        plan, prefill_artifact, model, tokenizer, traced_seq_len = _compile_partition_plan(self.config)
        self._partition_plan = plan

        route = self.config.route_to_vllm_when_all_gpu
        if route is None:
            route = _vllm_available()
            if _is_all_gpu(plan) and not route:
                logger.warning(
                    "HelmInference: the model fits in VRAM but vLLM is not installed; "
                    "running HELM's native executor. Install vLLM (see research/README.md) "
                    "to route fit-in-VRAM models to it automatically."
                )

        if _is_all_gpu(plan) and route:
            # Drop the HF model + HELM artifacts so vLLM has the GPU.
            del prefill_artifact, model
            _free_gpu()
            self._handle = _VLLMHandle(self.config, tokenizer)
            self._handle.setup()
            self._routed_to_vllm = True
            logger.info(
                "HelmInference: all-GPU partition (%d stages); routed to vLLM.",
                len(plan.stages) if plan else 0,
            )
        else:
            # Hybrid (at least one CPU stage), or routing disabled. Keep
            # the loaded model and finish building PipelineRuntime.
            self._handle = _HelmHandle(self.config, plan, prefill_artifact, model, tokenizer, traced_seq_len)
            self._handle.setup()
            self._routed_to_vllm = False
            logger.info(
                "HelmInference: %s partition; running HELM PipelineRuntime.",
                "all-GPU (router disabled)" if _is_all_gpu(plan) else "hybrid CPU+GPU",
            )

    def generate(self, prompts: List[str], max_new_tokens: Optional[int] = None) -> List[str]:
        """Generate completions for a batch of prompts."""
        with self._lock:
            if self._handle is None:
                raise RuntimeError("HelmInference.setup() must be called before generate().")
            if isinstance(prompts, str):
                prompts = [prompts]
            if not isinstance(prompts, (list, tuple)) or any(not isinstance(p, str) for p in prompts):
                raise ValueError("prompts must be a string or a sequence of strings")
            n = self.config.max_new_tokens if max_new_tokens is None else max_new_tokens
            _validate_token_count(n, "max_new_tokens")
            return self._handle.generate(list(prompts), n)

    def teardown(self) -> None:
        """Release resources. Safe after failed setup and repeated calls."""
        with self._lock:
            handle, self._handle = self._handle, None
            try:
                if handle is not None:
                    handle.teardown()
            finally:
                if self._gemma2_patched:
                    _restore_gemma2_decoder_outputs()
                    self._gemma2_patched = False
                self._partition_plan = None
                self._routed_to_vllm = False
                _free_gpu()

    close = teardown


def _is_gemma2_model_name(model_name: str) -> bool:
    name = model_name.lower()
    return "gemma2" in name or "gemma-2" in name


_GEMMA2_DECODER_ORIGINAL_FORWARD = None
_GEMMA2_DECODER_PATCH_COUNT = 0


def _wrap_single_output_tuple_forward(original_forward):
    """Make Gemma2DecoderLayer return a bare tensor (not a 1-tuple).

    HELM's FX trace assumes the decoder layer returns ``hidden_states``
    directly; the upstream class wraps it in a single-element tuple,
    which produces a malformed graph and downstream attention shape
    mismatches. Mirror of paper_bench's same-named helper.
    """
    import functools

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


def _load_model_with_auto_device_map(config: HelmInferenceConfig):
    """Load the model split across CPU and GPU by memory budget.

    HELM exists to serve models that do not fit on one GPU. Two load
    strategies are possible:

    1. ``device_map="cpu"`` (HELM's canonical pattern, used by verify_e2e
       and dev_pipeline). Clean — accelerate is not involved beyond the
       initial weight materialisation. Requires SYSTEM RAM >= full model
       weights plus HELM's transient executor buffers.

    2. ``device_map="auto"`` with explicit CPU/GPU budgets (paper_bench's
       pattern). Pre-splits weights across CPU+GPU during ``from_pretrained``
       so peak system RAM stays bounded. Works on tight-RAM machines.

    We prefer (1) when system RAM > 1.5x model size, else fall back to
    (2). Trade-off: (1) is cleaner but needs RAM headroom; (2) is more
    forgiving but introduces accelerate into HELM's flow.
    """
    import psutil
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Estimate model size from HF config to choose load strategy.
    cfg = AutoConfig.from_pretrained(config.model_id)
    approx_params = _estimate_param_count(cfg)
    bytes_per_param = 2 if config.dtype in (torch.float16, torch.bfloat16) else 4
    model_bytes = approx_params * bytes_per_param
    ram_available = int(psutil.virtual_memory().available)

    if model_bytes * 1.5 <= ram_available:
        # Path 1: CPU-only single-device load (HELM canonical).
        device_map = "cpu"
        max_memory = None
        logger.info(
            "HelmInference: loading on CPU only (model=%.1f GB, RAM avail=%.1f GB).",
            model_bytes / 1024**3, ram_available / 1024**3,
        )
    else:
        # Path 2: accelerate auto-split (paper_bench pattern). Necessary
        # when system RAM cannot hold the full model + HELM transients.
        device_map = "auto"
        ram_mb = ram_available / (1024 * 1024)
        max_memory = {"cpu": f"{int(max(ram_mb - 2048, 1024))}MiB"}
        if torch.cuda.is_available():
            gpu_total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
            max_memory[0] = f"{int(gpu_total_mb * 0.80)}MiB"
        logger.info(
            "HelmInference: model %.1f GB exceeds RAM headroom (%.1f GB available); "
            "using device_map=auto + budgets to avoid system OOM.",
            model_bytes / 1024**3, ram_available / 1024**3,
        )

    common_kwargs = dict(
        dtype=config.dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    if max_memory is not None:
        common_kwargs["max_memory"] = max_memory

    try:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            use_cache=False,
            **common_kwargs,
        )
    except TypeError as exc:
        # Gemma-3 IT resolves to a multimodal class whose __init__ rejects
        # use_cache. Fall back to the text-only causal-LM class.
        if "use_cache" not in str(exc):
            raise
        from transformers import Gemma3ForCausalLM
        model = Gemma3ForCausalLM.from_pretrained(
            config.model_id,
            **common_kwargs,
        )
        model.config.use_cache = False
    model.eval()

    # When device_map="auto" is used, accelerate installs pre-forward hooks
    # that reshape inputs in ways HELM's FX trace cannot follow. Remove them.
    if device_map == "auto":
        try:
            from accelerate.hooks import remove_hook_from_submodules
            remove_hook_from_submodules(model)
        except ImportError:
            pass

    return model, tokenizer


def _estimate_param_count(cfg) -> int:
    """Rough parameter count from HF config — close enough for memory sizing."""
    h = int(getattr(cfg, "hidden_size", 0) or 0)
    ff = int(getattr(cfg, "intermediate_size", 0) or 4 * h)
    n_layers = int(getattr(cfg, "num_hidden_layers", 0) or 0)
    vocab = int(getattr(cfg, "vocab_size", 0) or 0)
    if not h or not n_layers:
        # Unknown architecture — be conservative.
        return 10 * 10**9
    # Per-layer: 4*h*h (attn) + 3*h*ff (mlp) ≈ 12*h^2 if ff=4h
    per_layer = 4 * h * h + 3 * h * ff
    embed = vocab * h  # tied lm_head counted via embedding
    return n_layers * per_layer + embed


def _gpu_memory_reserve_for_model(model, config: HelmInferenceConfig) -> Optional[int]:
    """Return this model's GPU reserve in MiB, without changing process state.

    The planner's default 512 MB safety margin does not cover:
      1. The tied-weight clone (HELM detaches lm_head from embed_tokens at
         executor build; the detached copy lives on the GPU stage).
      2. Activation / KV / FX-trace scratch during the first forward pass.

    Pass this per-model value to the compiler rather than altering process-wide
    state. Respect a caller's environment override when one is set.
    """
    if os.environ.get("HELM_GPU_MEMORY_RESERVE_MB"):
        return None

    cfg = getattr(model, "config", None)
    if cfg is None:
        return None

    dtype_size = 2 if config.dtype in (torch.float16, torch.bfloat16) else 4
    vocab = int(getattr(cfg, "vocab_size", 0) or 0)
    hidden = int(getattr(cfg, "hidden_size", 0) or 0)
    intermediate = int(getattr(cfg, "intermediate_size", 0) or 0)
    n_layers = int(getattr(cfg, "num_hidden_layers", 32) or 32)
    n_heads = int(getattr(cfg, "num_attention_heads", 0) or 0)
    kv_heads = int(getattr(cfg, "num_key_value_heads", 0) or n_heads)
    head_dim = int(getattr(cfg, "head_dim", 0) or (hidden // n_heads if n_heads else 0))

    # Tied-embedding clone: HELM detaches lm_head from embed_tokens on the GPU
    # stage, costing one extra embedding matrix — only when the weights are
    # actually tied (HF's default when the config does not say).
    tied = bool(getattr(cfg, "tie_word_embeddings", True))
    tied_clone_bytes = vocab * hidden * dtype_size if tied else 0

    # Forward-pass scratch: activations are transient per layer, so budget the
    # widest layer's buffers (x4 for Q/K/V + MLP intermediates, x1.5 slack for
    # fragmentation), plus the logits and the KV cache for the whole sequence.
    seq_budget = max(config.max_input_tokens + config.max_new_tokens, 256)
    width = max(hidden, intermediate)
    activation_bytes = int(seq_budget * width * dtype_size * 4 * 1.5)
    activation_bytes += seq_budget * vocab * 4  # fp32 logits
    activation_bytes += 2 * n_layers * kv_heads * head_dim * seq_budget * dtype_size

    # CUDA context + caching-allocator slack. Measured on Qwen3-8B / 16 GB GPU:
    # a 512 MB margin (total reserve ~750 MB) peaks at 14.4 of 15.5 GB with
    # no OOM, while a 1.5 GB margin moved 3 more layers to CPU (-21% tok/s).
    fixed_overhead_bytes = 512 * 1024 * 1024

    reserve_mb = int((tied_clone_bytes + activation_bytes + fixed_overhead_bytes)
                     / (1024 * 1024))
    logger.info(
        "HelmInference: reserving %d MiB of GPU memory "
        "(tied_clone=%.2f GB, activation=%.2f GB) for model %s.",
        reserve_mb,
        tied_clone_bytes / (1024 ** 3),
        activation_bytes / (1024 ** 3),
        config.model_id,
    )
    return reserve_mb


def _compile_partition_plan(config: HelmInferenceConfig):
    """Run HELM through the partition-plan stage of compilation.

    Returns ``(plan, prefill_artifact, model, tokenizer)``. ``prefill_artifact``
    has the stage graphs needed to build PipelineRuntime on the HELM path;
    on the vLLM path the caller frees it.
    """
    from helm._pipeline import (
        capture_fx_graph,
        configure_cpu_threads,
    )
    from helm.compiler.compiler import HelmCompileOptions, compile_graph

    configure_cpu_threads(config.cpu_threads)
    model, tokenizer = _load_model_with_auto_device_map(config)
    gpu_memory_reserve_mb = _gpu_memory_reserve_for_model(model, config)

    # Build prefill dummy inputs the way paper_bench does. Use a short seed
    # prompt: for gemma-2 (sliding-window attention) HELM's FX trace bakes
    # the dummy seq_len in subtly, so the trace length must be ≤ any
    # actual generate prompt's length. A 4-token seed is short enough.
    seed_prompt = "Benchmark prompt for compilation."
    # Match paper_bench: tokenize without padding. The FX trace is shape-
    # polymorphic for the standard attention path; padding to max_length
    # specialises the graph in a way that breaks decode steps for
    # sliding-window models like gemma-2 (cache_len=traced+1 fails).
    prompt_tok = tokenizer(
        seed_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=config.max_input_tokens,
    )
    input_ids = prompt_tok["input_ids"]
    attention_mask = prompt_tok["attention_mask"]
    seq_len = input_ids.shape[1]

    min_val = torch.finfo(config.dtype).min
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    cache_position = torch.arange(seq_len, dtype=torch.long)
    causal = torch.triu(
        torch.full((seq_len, seq_len), min_val, dtype=config.dtype), diagonal=1
    )
    inv_mask = (1.0 - attention_mask.to(config.dtype)) * min_val
    attn_mask = causal[None, None, :, :] + inv_mask[:, None, None, :]
    dummy_inputs = (input_ids, attn_mask, position_ids, cache_position)

    gm = capture_fx_graph(
        model, dummy_inputs, run_dir=None,
        allow_fallback=True, trace_policy="explicit_leaf",
    )

    options = HelmCompileOptions(
        mode="both",
        objective="decode_latency",
        plan_mode=config.plan_mode,
        cpu_layers=config.cpu_layers,
        gpu_layers=config.gpu_layers,
        lower_stages=True,
        graph_kind="prefill",
        model_name=config.model_id,
        kv_offload=config.kv_offload,
        gpu_memory_reserve_mb=gpu_memory_reserve_mb,
        # Some models (gemma-2 with sliding-window attention) defeat dynamic
        # shape propagation in HybridAnalyzer; fall back to static analysis
        # rather than failing the compile. Same flag verify_e2e uses.
        allow_static_analysis_fallback=True,
        workload={
            "batch_size": 1,
            # Use the ACTUAL traced seq_len, not max_input_tokens. The graph
            # was traced at this length; telling the cost model a different
            # number causes incorrect planning and (for gemma-2) decode-time
            # mask shape mismatches.
            "prefill_seq_len": seq_len,
            "decode_context_len": seq_len,
            "decode_tokens": config.max_new_tokens,
            "dtype_size": 2 if config.dtype in (torch.float16, torch.bfloat16) else 4,
        },
    )

    artifact = compile_graph(
        gm=gm,
        example_inputs=dummy_inputs,
        model=model,
        tokenizer=tokenizer,
        options=options,
        artifacts_dir=None,
    )
    return artifact.partition_plan, artifact, model, tokenizer, seq_len


class _VLLMHandle:
    """Wraps a vLLM engine behind the _Handle protocol.

    Per HelmInferenceConfig contract: if vLLM is unavailable or fails to
    initialise, we re-raise rather than silently fall back to HELM.
    """

    def __init__(self, config: HelmInferenceConfig, tokenizer) -> None:
        self._config = config
        self._tokenizer = tokenizer  # kept for parity with _HelmHandle; vLLM owns its own
        self._llm = None

    def setup(self) -> None:
        try:
            from vllm import LLM
        except ImportError as exc:
            raise RuntimeError(
                "HelmInference routed to vLLM but the vllm package is not installed. "
                "Use the separate research environment (research/README.md) or "
                "set route_to_vllm_when_all_gpu=False."
            ) from exc

        dtype_str = {
            torch.float16: "float16",
            torch.bfloat16: "bfloat16",
            torch.float32: "float32",
        }[self._config.dtype]

        self._llm = LLM(
            model=self._config.model_id,
            dtype=dtype_str,
            gpu_memory_utilization=self._config.vllm_gpu_memory_utilization,
            max_model_len=self._config.vllm_max_model_len,
        )

    def generate(self, prompts: List[str], max_new_tokens: int) -> List[str]:
        if self._llm is None:
            raise RuntimeError("_VLLMHandle.setup() not called.")
        for prompt in prompts:
            _encode_prompt(self._tokenizer, prompt, self._config, max_new_tokens,
                           self._config.vllm_max_model_len)
        if max_new_tokens == 0 or not prompts:
            return ["" for _ in prompts]
        from vllm import SamplingParams

        params = SamplingParams(max_tokens=max_new_tokens, temperature=0.0)
        outputs = self._llm.generate(prompts, params)
        # vLLM returns RequestOutput list ordered by input.
        return [out.outputs[0].text for out in outputs]

    def teardown(self) -> None:
        if self._llm is not None:
            del self._llm
            self._llm = None


class _HelmHandle:
    """Wraps HELM's PipelineRuntime behind the _Handle protocol.

    This path is what makes HELM unique: it serves models that do not fit
    in GPU VRAM by partitioning the model across CPU and GPU.
    """

    def __init__(
        self,
        config: HelmInferenceConfig,
        plan: PartitionPlan,
        prefill_artifact,
        model,
        tokenizer,
        traced_seq_len: int,
    ) -> None:
        self._config = config
        self._plan = plan
        self._prefill_artifact = prefill_artifact
        self._model = model
        self._tokenizer = tokenizer
        self._traced_seq_len = traced_seq_len
        self._runtime = None
        self._kv_offload_mgr = None

    def setup(self) -> None:
        from helm._pipeline import capture_decode_fx_graph
        from helm.compiler.compiler import HelmCompileOptions, compile_graph
        from helm.compiler.importers.decode_tracer import DecodeTracer
        from helm.compiler.importers.patch_fx import apply_fx_patch
        from helm.runtime.executor import StageRuntimeExecutor
        from helm.runtime.pipeline_runtime import PipelineRuntime

        # paper_bench re-applies the FX patch between prefill and decode
        # traces. The patch is idempotent but ensures the decode trace runs
        # with the same monkey-patches the prefill trace had.
        apply_fx_patch()

        # Decode dummy on CPU per paper_bench convention (the dummy is small
        # and the runtime moves stages to their target devices anyway).
        decode_dummy = DecodeTracer.build_dummy_inputs(
            device="cpu", batch_size=1, dtype=self._config.dtype
        )
        decode_gm = capture_decode_fx_graph(self._model, decode_dummy, run_dir=None)
        decode_wrapper = getattr(self._model, "_helm_decode_wrapper", None)

        decode_opts = HelmCompileOptions(
            mode="both",
            objective="decode_latency",
            plan_mode=self._config.plan_mode,
            cpu_layers=self._config.cpu_layers,
            gpu_layers=self._config.gpu_layers,
            lower_stages=True,
            graph_kind="decode",
            model_name=self._config.model_id,
            kv_offload=self._config.kv_offload,
            gpu_memory_reserve_mb=_gpu_memory_reserve_for_model(self._model, self._config),
            allow_static_analysis_fallback=True,
            workload={
                "batch_size": 1,
                # Use the prefill's traced seq_len, not max_input_tokens
                # (see _compile_partition_plan for rationale).
                "prefill_seq_len": self._traced_seq_len,
                "decode_context_len": self._traced_seq_len,
                "decode_tokens": self._config.max_new_tokens,
                "dtype_size": 2 if self._config.dtype in (torch.float16, torch.bfloat16) else 4,
            },
        )

        # paper_bench does NOT pass partition_plan_override here. It re-plans
        # the decode graph from scratch and gets a consistent placement. Doing
        # the same avoids subtle prefill/decode plan mismatches.
        decode_artifact = compile_graph(
            gm=decode_gm,
            example_inputs=decode_dummy,
            model=self._model,
            tokenizer=self._tokenizer,
            options=decode_opts,
            artifacts_dir=None,
        )

        prefill_exec = StageRuntimeExecutor(self._prefill_artifact.stage_graphs)
        decode_exec = StageRuntimeExecutor(decode_artifact.stage_graphs)

        if self._config.kv_offload:
            from helm.runtime.kv_offload import KVOffloadConfig, KVOffloadManager

            kv_kwargs: dict = {}
            # Auto-enable contiguous KV fast path when all-GPU (matches cc1bf42).
            if _is_all_gpu(self._plan):
                kv_kwargs["cont_capacity"] = int(self._config.max_input_tokens) + 512
            kv_cfg = KVOffloadConfig.from_model(self._model, **kv_kwargs)
            self._kv_offload_mgr = KVOffloadManager(self._model, kv_cfg, batch_size=1)
            self._kv_offload_mgr.patch_module_roots(
                *(stage.module for stage in prefill_exec.stages),
                *(stage.module for stage in decode_exec.stages),
            )

        self._runtime = PipelineRuntime(
            prefill_exec, decode_exec,
            tokenizer=self._tokenizer,
            dtype=self._config.dtype,
            kv_offload_mgr=self._kv_offload_mgr,
            decode_wrapper=decode_wrapper,
        )

    @torch.inference_mode()
    def generate(self, prompts: List[str], max_new_tokens: int) -> List[str]:
        if self._runtime is None:
            raise RuntimeError("_HelmHandle.setup() not called.")

        device = next(self._model.parameters()).device
        outputs: List[str] = []
        eos_id = self._tokenizer.eos_token_id
        pad_id = self._tokenizer.pad_token_id
        if pad_id is None:
            pad_id = eos_id
        context_limit = getattr(self._model.config, "max_position_embeddings", None)
        # Validate the whole batch before executing any request.
        encoded_prompts = [
            _encode_prompt(self._tokenizer, prompt, self._config, max_new_tokens, context_limit)
            for prompt in prompts
        ]
        if max_new_tokens == 0:
            return ["" for _ in prompts]

        for encoded in encoded_prompts:
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            collected: List[int] = []

            def _capture(_step: int, token: torch.Tensor, _finished: torch.Tensor) -> None:
                collected.append(int(token.item()))

            self._runtime.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_id,
                pad_token_id=pad_id,
                token_callback=_capture,
            )
            outputs.append(self._tokenizer.decode(collected, skip_special_tokens=True))
        return outputs

    def teardown(self) -> None:
        try:
            if self._kv_offload_mgr is not None:
                self._kv_offload_mgr.restore()
        finally:
            _release_decode_wrapper(self._model)
            self._runtime = None
            self._kv_offload_mgr = None
            self._model = None
            self._prefill_artifact = None
            self._tokenizer = None


def _release_decode_wrapper(model) -> None:
    wrapper = getattr(model, "_helm_decode_wrapper", None)
    if wrapper is not None:
        for hook in wrapper._kv_hooks:
            hook.remove()
        wrapper._kv_hooks.clear()
        wrapper.reset_cache()
        delattr(model, "_helm_decode_wrapper")


__all__ = ["HelmInference", "HelmInferenceConfig"]