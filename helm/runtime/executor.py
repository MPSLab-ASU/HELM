import logging
import torch
import time
from typing import List, Dict, Any, Union
from .stage import Stage
from .tensor_transfer import move_tensor
from transformers.cache_utils import DynamicCache as HF_DynamicCache

logger = logging.getLogger(__name__)

_PROFILE = False   # set True to enable per-stage timing every decode step
_call_count = 0

class DynamicCache(HF_DynamicCache):
    """
    Compatibility wrapper so FX graphs that call DynamicCache(layers=[])
    still work even if the installed HF version does not support the
    'layers' argument.
    """

    def __init__(self, *args, **kwargs):
        kwargs.pop("layers", None)
        super().__init__(*args, **kwargs)


class StageRuntimeExecutor:
    """
    Executes a pipeline of FX stages.

    For decode stages that contain a _Wrapper with past_key_values,
    the executor sets the cache attribute before forward and reads
    it back after forward.
    """

    # Types that should not be moved between devices
    _PASSTHROUGH_TYPES = (HF_DynamicCache,)

    def __init__(self, stages: List[Stage]):
        self.stages = stages
        # Clone parameters shared across stage boundaries (e.g. tied embeddings
        # in Qwen3 where embed_tokens.weight == lm_head.weight).  Without this,
        # moving stage-0 params to CPU drags lm_head (stage 1) to CPU too, and
        # vice-versa — causing 200-300ms of PCIe transfers every token.
        self._break_cross_stage_ties()
        # Pre-position all stage weights to their target devices eagerly.
        #
        # Without this, device_map='auto' (used during model loading) may place
        # weights on devices that differ from the HELM partition.  The first
        # run() would then trigger lazy moves inside the execute loop, but by
        # that point ALL weights from every stage may already be in CPU RAM
        # simultaneously (the GPU-stage weights that device_map left on CPU +
        # the CPU-stage weights being pulled off GPU = full 16 GB model in RAM
        # at once → severe disk paging on 16 GB machines → 10-30 s / token).
        #
        # Fix: move CPU-stage weights first (freeing any GPU memory they occupy),
        # then move GPU-stage weights from CPU to CUDA.  Peak memory is bounded
        # by the larger single-stage footprint, not the full model size.
        self._prewarm_stage_devices()
        self._patch_cpu_linears()

    def _get_stage_submodule(self, stage: Stage, target: str, context: str) -> torch.nn.Module:
        try:
            return stage.module.get_submodule(target)
        except AttributeError as exc:
            raise RuntimeError(
                f"Stage {stage.stage_id} graph references missing submodule "
                f"{target!r} {context}. The stage GraphModule is inconsistent "
                "with its FX graph."
            ) from exc

    def _resolve_stage_attr(self, stage: Stage, target: str, context: str) -> Any:
        try:
            return self._resolve_attr(stage.module, target)
        except AttributeError as exc:
            raise RuntimeError(
                f"Stage {stage.stage_id} graph references missing attribute "
                f"{target!r} {context}. The stage GraphModule is inconsistent "
                "with its FX graph."
            ) from exc

    def _break_cross_stage_ties(self):
        """
        Detect parameters whose storage is shared across two different stages
        (tied weights) and clone them so each stage owns an independent copy.
        After this call, targeted per-stage .to() moves are safe.
        """
        # data_ptr -> (stage_idx, node_target, param_name)
        seen: Dict[int, tuple] = {}
        for stage_idx, stage in enumerate(self.stages):
            for node in stage.module.graph.nodes:
                if node.op != 'call_module':
                    continue
                submod = self._get_stage_submodule(
                    stage,
                    str(node.target),
                    "during cross-stage tied-weight scan",
                )
                for pname, param in list(submod.named_parameters(recurse=True)):
                    ptr = param.data_ptr()
                    if ptr not in seen:
                        seen[ptr] = (stage_idx, node.target, pname)
                        continue
                    prev_stage, prev_mod, prev_pname = seen[ptr]
                    if prev_stage == stage_idx:
                        continue
                    # Tied across stages — clone for this (later) stage.
                    # Always clone to CPU regardless of the param's current
                    # device.  With device_map='auto' the tied embedding
                    # (embed_tokens / lm_head, ~1.8 GB for large-vocab models)
                    # is already on GPU; cloning in-place would spike GPU usage
                    # by another 1.8 GB, leaving no headroom for the very first
                    # prewarm move and causing CUDA OOM before any weight is
                    # freed.  Cloning to CPU is safe: _prewarm_stage_devices
                    # will detect that the clone is on the wrong device and
                    # move it to GPU interleaved with freeing earlier layers,
                    # so peak GPU never exceeds the initial device_map budget.
                    parts = (node.target + '.' + pname).split('.')
                    parent = stage.module
                    for p in parts[:-1]:
                        parent = getattr(parent, p)
                    cloned = torch.nn.Parameter(
                        param.data.detach().cpu().clone(), requires_grad=False)
                    setattr(parent, parts[-1], cloned)
                    logger.info("[HELM] Tied weight cloned: stage %d '%s.%s' detached from stage %d '%s.%s'",
                                stage_idx, node.target, pname, prev_stage, prev_mod, prev_pname)

    def _prewarm_stage_devices(self):
        """
        Swap parameters to their target devices without exceeding either
        device's memory capacity.

        Background: device_map='auto' places early layers on GPU and later
        layers on CPU. HELM's latency-optimal partition is the opposite — early
        (embedding + first N transformer blocks) on CPU, later blocks on GPU.
        A naïve "move all CPU-stage weights first" approach would spike CPU RAM
        to ~15 GB (GPU-stage weights already in CPU RAM + CPU-stage weights
        being moved from GPU) which exceeds the 16 GB total on this machine.

        Fix: interleave the moves so peak memory never exceeds the initial
        state plus one extra submodule:
          1. Move a GPU-stage submodule from CPU → GPU  (frees CPU RAM)
          2. Move a CPU-stage submodule from GPU → CPU  (frees GPU VRAM)
          Repeat until all submodules are on their target device.
        """
        # Collect submodules that are on the wrong device, by target.
        cpu_stage_on_gpu: List[tuple] = []   # (stage, name) — need GPU→CPU
        gpu_stage_on_cpu: List[tuple] = []   # (stage, name) — need CPU→GPU

        for stage in self.stages:
            is_cuda = "cuda" in stage.device
            target = torch.device(stage.device)
            seen: set = set()
            for node in stage.module.graph.nodes:
                if node.op != 'call_module' or node.target in seen:
                    continue
                submod = self._get_stage_submodule(
                    stage,
                    str(node.target),
                    "during stage device prewarm scan",
                )
                p = next(submod.parameters(), None)
                if p is None or p.device == target:
                    continue
                seen.add(node.target)
                if is_cuda:
                    gpu_stage_on_cpu.append((stage, node.target))
                else:
                    cpu_stage_on_gpu.append((stage, node.target))

        def _move(stage, name):
            self._get_stage_submodule(
                stage,
                str(name),
                "during stage device prewarm move",
            ).to(stage.device)

        # Interleaved swap: always move a GPU-stage submod first (frees CPU RAM),
        # then a CPU-stage submod (frees GPU VRAM). Peak overhead = one submodule.
        gi, ci = 0, 0
        while gi < len(gpu_stage_on_cpu) and ci < len(cpu_stage_on_gpu):
            _move(*gpu_stage_on_cpu[gi]); gi += 1   # CPU→GPU: frees CPU, uses GPU
            _move(*cpu_stage_on_gpu[ci]); ci += 1   # GPU→CPU: frees GPU, uses CPU

        # Drain any remainder (e.g. all-GPU model where there are no CPU-stage moves)
        while gi < len(gpu_stage_on_cpu):
            _move(*gpu_stage_on_cpu[gi]); gi += 1
        while ci < len(cpu_stage_on_gpu):
            _move(*cpu_stage_on_gpu[ci]); ci += 1

        # Handle get_attr tensors (rotary embeddings, bias, etc.) — typically small
        for stage in self.stages:
            target = torch.device(stage.device)
            for node in stage.module.graph.nodes:
                if node.op == 'get_attr':
                    val = self._resolve_stage_attr(
                        stage,
                        str(node.target),
                        "during stage device prewarm",
                    )
                    if isinstance(val, torch.Tensor) and val.device != target:
                        new_val = val.to(stage.device)
                        obj = stage.module
                        parts = node.target.split(".")
                        for part in parts[:-1]:
                            obj = getattr(obj, part)
                        existing = getattr(obj, parts[-1], None)
                        if isinstance(existing, torch.nn.Parameter):
                            new_val = torch.nn.Parameter(
                                new_val, requires_grad=existing.requires_grad)
                        setattr(obj, parts[-1], new_val)
            stage.module.eval()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            logger.info("[HELM] Stage weights pre-positioned: GPU %.2f GB allocated",
                        torch.cuda.memory_allocated() / 1e9)

    def _patch_cpu_linears(self):
        """
        Replace nn.Linear on CPU stages with AVX2+F16C wrappers.
        Only patches stages whose device is CPU and whose weight dtype is fp16.
        Falls back to torch linear if the native extension failed to compile,
        but reports the exact load error so the benchmark path is not silent.
        """
        from helm.kernels import patch_cpu_linears, is_available, load_error
        if not is_available():
            logger.warning(
                "[HELM] AVX2+F16C CPU kernel unavailable; using torch.nn.Linear "
                "for CPU stages. load_error=%s",
                load_error() or "<unknown>",
            )
            return
        total = 0
        for stage in self.stages:
            if "cuda" in stage.device:
                continue
            n = patch_cpu_linears(stage.module)
            total += n
        if total:
            logger.info("[HELM] Patched %d nn.Linear(s) on CPU stage(s) with AVX2+F16C fp16 GEMV kernel", total)

    def _find_wrapper(self, module):
        """
        Find the _Wrapper submodule that holds past_key_values.
        Returns the wrapper or None.
        """
        if hasattr(module, 'past_key_values'):
            return module
        for name, child in module.named_modules():
            if hasattr(child, 'past_key_values'):
                return child
        return None

    def _resolve_attr(self, module, target):
        """
        Resolve a get_attr target string to an attribute on module.

        FX mangles dotted attribute paths (e.g. ``model.model.rotary_emb``)
        into underscore-separated names (``model_model_rotary_emb``) when the
        path contains only known module/tensor names that happen to be joined by
        underscores in the graph.  We try the dotted path first, then
        systematically try every way to split the underscored target back into
        parts, rather than blindly replacing *all* underscores with dots (which
        would misparse legitimate names like ``self_attn`` or ``lm_head``).
        """
        # 1. Try the target as-is (handles already-dotted paths).
        try:
            obj = module
            for part in target.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            pass

        # 2. If the target contains underscores, try every possible way to
        #    re-insert dots in place of underscores until one resolves.
        #    We iterate splits from left to right so we prefer longer leading
        #    components (e.g. "model_model" before "model" + "model").
        if "_" in target:
            parts = target.split("_")
            # Generate all non-empty partitions of `parts` by varying split points.
            # For N parts there are 2^(N-1) candidate dotted paths; for typical
            # FX names (2-4 components) this is at most 8 attempts.
            n = len(parts)
            for mask in range(1 << (n - 1)):
                # Build candidate path: bit=1 means insert a dot, bit=0 means keep underscore.
                candidate_parts = []
                current = parts[0]
                for i in range(n - 1):
                    if mask & (1 << i):
                        candidate_parts.append(current)
                        current = parts[i + 1]
                    else:
                        current = current + "_" + parts[i + 1]
                candidate_parts.append(current)
                try:
                    obj = module
                    for part in candidate_parts:
                        obj = getattr(obj, part)
                    return obj
                except AttributeError:
                    continue

        raise AttributeError(
            f"Cannot resolve attribute '{target}' on {type(module).__name__}"
        )

    def run(
        self,
        inputs: Union[torch.Tensor, Dict[str, Any], tuple],
        past_key_values=None,
    ):
        global _call_count
        _call_count += 1
        _t_run_start = time.perf_counter() if _PROFILE else 0
        runtime_env = {}

        # Initialize runtime_env with input
        if isinstance(inputs, dict):
            runtime_env.update(inputs)
            # Extract past_key_values from dict if present
            if past_key_values is None and "past_key_values" in runtime_env:
                past_key_values = runtime_env.pop("past_key_values")
        elif isinstance(inputs, tuple) and len(inputs) > 0:
            runtime_env["input_ids"] = inputs[0]
            if len(inputs) > 1:
                runtime_env["attention_mask"] = inputs[1]
            if len(inputs) > 2:
                runtime_env["position_ids"] = inputs[2]
            if len(inputs) > 3:
                runtime_env["cache_position"] = inputs[3]
        else:
            runtime_env["input_ids"] = inputs

        for stage_idx, stage in enumerate(self.stages):
            kwargs = {}
            for node in stage.module.graph.nodes:
                if node.op == 'placeholder':
                    if node.target in runtime_env:
                        val = runtime_env[node.target]
                    else:
                        try:
                            val = self._resolve_attr(stage.module, node.target)
                        except AttributeError:
                            raise RuntimeError(f"Missing input '{node.target}' for Stage {stage.stage_id}")

                    if isinstance(val, self._PASSTHROUGH_TYPES):
                        kwargs[node.target] = val
                    else:
                        kwargs[node.target] = move_tensor(val, stage.device)

            # Inject past_key_values into wrapper if available
            wrapper = self._find_wrapper(stage.module)
            if wrapper is not None and past_key_values is not None:
                wrapper.past_key_values = past_key_values

            # Ensure all inputs are on the stage device
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor) and v.device != torch.device(stage.device):
                    kwargs[k] = v.to(stage.device)

            # Move only this stage's submodules to their target device.
            # Targeted per-submodule moves (not stage.module.to()) so that
            # parameters belonging to other stages — including cloned
            # ex-tied-weight partners — are never touched.
            # After the first token all params are on their target device;
            # needs_move becomes False and this block is fully skipped.
            target_device = torch.device(stage.device)

            needs_move = False
            for node in stage.module.graph.nodes:
                if node.op == 'call_module':
                    submod = self._get_stage_submodule(
                        stage,
                        str(node.target),
                        "during lazy stage device check",
                    )
                    p = next(submod.parameters(), None)
                    if p is not None and p.device != target_device:
                        needs_move = True
                    break  # sample only the first call_module

            if needs_move:
                for node in stage.module.graph.nodes:
                    if node.op == 'call_module':
                        self._get_stage_submodule(
                            stage,
                            str(node.target),
                            "during lazy stage device move",
                        ).to(stage.device)

                for node in stage.module.graph.nodes:
                    if node.op == "get_attr":
                        val = self._resolve_stage_attr(
                            stage,
                            str(node.target),
                            "during lazy stage device move",
                        )
                        if isinstance(val, torch.Tensor) and val.device != target_device:
                            new_val = val.to(stage.device)
                            obj = stage.module
                            parts = node.target.split(".")
                            for part in parts[:-1]:
                                obj = getattr(obj, part)
                            # Preserve Parameter type if needed — plain .to() returns
                            # a Tensor, but nn.Module.setattr requires a Parameter.
                            existing = getattr(obj, parts[-1], None)
                            if isinstance(existing, torch.nn.Parameter):
                                new_val = torch.nn.Parameter(
                                    new_val, requires_grad=existing.requires_grad)
                            setattr(obj, parts[-1], new_val)

                stage.module.eval()

            # Ensure DynamicCache exists in FX graph globals
            stage.module.forward.__globals__["DynamicCache"] = DynamicCache

            _t_fwd_start = time.perf_counter() if _PROFILE else 0
            with torch.no_grad():
                outputs = stage.module(**kwargs)
            if _PROFILE:
                if stage.device == "cuda":
                    torch.cuda.synchronize()
                _t_fwd = (time.perf_counter() - _t_fwd_start) * 1000
                _t_move = (_t_fwd_start - _t_run_start) * 1000 if stage_idx == 0 else 0
                logger.debug("[profile call=%d stage=%d(%s)] setup=%.1fms forward=%.1fms",
                             _call_count, stage_idx, stage.device, _t_move, _t_fwd)

            # Read back updated cache from wrapper
            if wrapper is not None and hasattr(wrapper, 'past_key_values'):
                past_key_values = wrapper.past_key_values

            # Intermediate stages return dicts of vars to inject back into namespace
            if stage_idx < len(self.stages) - 1:
                if isinstance(outputs, dict):
                    runtime_env.update(outputs)
                else:
                    raise RuntimeError(f"Expected intermediate Stage {stage.stage_id} to return a dict, got {type(outputs)}")
            else:
                # Final stage output unpacking
                result = {}

                if isinstance(outputs, tuple):
                    if len(outputs) == 5:
                        result["logits"] = outputs[0]
                        result["past_key_values"] = outputs[1]
                        result["q"] = outputs[2]
                        result["k"] = outputs[3]
                        result["v"] = outputs[4]
                    elif len(outputs) >= 2:
                        result["logits"] = outputs[0]
                        result["past_key_values"] = outputs[1] if not isinstance(outputs[1], self._PASSTHROUGH_TYPES) else outputs[1]
                        # Also check if the second element is a DynamicCache
                        if isinstance(outputs[1], self._PASSTHROUGH_TYPES):
                            result["past_key_values"] = outputs[1]
                    else:
                        result["logits"] = outputs[0]
                elif isinstance(outputs, dict):
                    result = outputs
                else:
                    result["logits"] = outputs

                # Attach the latest cache
                if past_key_values is not None:
                    result["past_key_values"] = past_key_values

                return result
