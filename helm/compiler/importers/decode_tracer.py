import inspect
import logging
import threading
import weakref
from typing import Optional, Tuple

import torch
import torch.fx as fx
import torch.nn as nn
from torch.utils.hooks import RemovableHandle

from transformers.cache_utils import DynamicCache

log = logging.getLogger(__name__)

# Models that scale embeddings by sqrt(hidden_size) in their forward() before
# the first decoder layer.  _Wrapper bypasses model.forward() and calls
# embed_tokens directly, so it must apply the same scaling here.
_SCALED_EMBED_MODELS: frozenset[str] = frozenset({"gemma", "gemma2", "gemma3"})

# Exact final path-segment names that must be treated as FX leaf modules.
# Matched against the last component of module_qualified_name (e.g. "embed_tokens"),
# NOT as a substring, to avoid false positives on names like "shared_embed_tokens_proj".
_LEAF_MODULE_NAMES: frozenset[str] = frozenset(
    {"embed_tokens", "rotary_emb", "lm_head"}
)


def _validate_model_structure(model: nn.Module) -> None:
    """Raise ValueError with a clear diagnostic if model lacks the expected structure.

    DecodeTracer requires model.model.layers to be a non-empty ModuleList of
    decoder layers, as found in LlamaForCausalLM, MistralForCausalLM, etc.
    """
    inner = getattr(model, "model", None)
    if inner is None:
        raise ValueError(
            f"DecodeTracer requires a model with a .model attribute "
            f"(e.g. LlamaForCausalLM, MistralForCausalLM). "
            f"Got {type(model).__name__!r} which has no .model sub-module. "
            f"If your model uses a different attribute name (e.g. .transformer), "
            f"subclass DecodeTracer and override _Wrapper accordingly."
        )
    layers = getattr(inner, "layers", None)
    if layers is None:
        raise ValueError(
            f"DecodeTracer requires model.model.layers (a non-empty ModuleList). "
            f"{type(inner).__name__!r} has no .layers attribute."
        )
    if len(layers) == 0:
        raise ValueError(
            f"DecodeTracer requires at least one decoder layer in model.model.layers. "
            f"Got an empty ModuleList on {type(inner).__name__!r}."
        )


class DecodeTracer:
    """
    Traces a single decode step with KV cache support.

    Decoder layers are FX *leaf modules* so their real forward() (including
    DynamicCache.update()) runs natively at each step.  The DynamicCache is
    kept entirely off the FX graph - it is injected before each layer call
    via a registered forward_pre_hook, avoiding FX code-gen trying to
    repr() the cache object.

    Graph contract:
        inputs:  (input_ids, attention_mask, position_ids, cache_position)
                 attention_mask is a 4D float causal mask, shape (B, 1, 1, S)
                 where S = cache_position + 1.
        outputs: logits tensor

    Thread safety
    -------------
    ``trace()`` is NOT thread-safe.  Calling it concurrently on the same
    ``DecodeTracer`` instance raises ``RuntimeError``.  Use a separate
    ``DecodeTracer`` instance per thread.

    Cache reset
    -----------
    Call ``wrapper.reset_cache()`` (or equivalently
    ``wrapper.past_key_values = DynamicCache()``) between independent
    inference sessions to prevent KV state from leaking across requests.
    """

    # ------------------------------------------------------------------ #
    #  Inner: FX Tracer                                                   #
    # ------------------------------------------------------------------ #

    class _HelmTracer(fx.Tracer):
        def is_leaf_module(self, module: nn.Module, module_qualified_name: str) -> bool:
            # Match the *final path segment only* to avoid substring false positives.
            # e.g. "model.layers.0.embed_tokens"             -> leaf (exact match)
            #      "model.layers.0.shared_embed_tokens_proj" -> NOT a leaf
            leaf_name = module_qualified_name.rsplit(".", 1)[-1]
            if leaf_name in _LEAF_MODULE_NAMES:
                return True
            # Decoder layers are leaves so DynamicCache.update() runs natively
            # at runtime (list.append is invisible to FX symbolic tracing).
            if module.__class__.__name__.endswith("DecoderLayer"):
                return True
            return super().is_leaf_module(module, module_qualified_name)

    # ------------------------------------------------------------------ #
    #  Inner: nn.Module wrapper traced by FX                              #
    # ------------------------------------------------------------------ #

    class _Wrapper(nn.Module):
        """
        Thin nn.Module that FX traces in place of the full model.

        INVARIANT: No attribute on _Wrapper may create a path back to
        ``model._helm_decode_wrapper`` before ``DecodeTracer.trace()``
        returns.  The back-reference is stored via ``object.__setattr__``
        (bypassing nn.Module registration) AFTER tracing to avoid an
        infinite cycle in FX's collect_tensor_attrs:
            wrapper.model._helm_decode_wrapper.model._helm_decode_wrapper ...
        """

        def __init__(self, model: nn.Module) -> None:
            super().__init__()

            _validate_model_structure(model)

            # Remove stale hooks from any previous _Wrapper on this model so
            # they don't accumulate and corrupt the KV cache at runtime.
            old_wrapper = getattr(model, "_helm_decode_wrapper", None)
            if old_wrapper is not None and hasattr(old_wrapper, "_kv_hooks"):
                log.debug(
                    "Removing %d stale KV hooks from previous _Wrapper.",
                    len(old_wrapper._kv_hooks),
                )
                for h in old_wrapper._kv_hooks:
                    h.remove()

            self.model = model
            self.past_key_values: DynamicCache = DynamicCache()

            first_layer = model.model.layers[0]
            layer_sig = inspect.signature(first_layer.forward)
            self._kv_kwarg: str = (
                "past_key_value"
                if "past_key_value" in layer_sig.parameters
                else "past_key_values"
            )
            self._use_pos_embed: bool = "position_embeddings" in layer_sig.parameters

            # ── Probe: detect whether decoder layers return a tuple ───────────────
            # Must run before FX tracing because isinstance(Proxy, ...) is always
            # False during tracing, which would silently erase required getitem()
            # nodes for tuple-returning layers (e.g. Gemma-2).
            #
            # We temporarily force eval() to suppress dropout masks and batch-norm
            # running-stat updates that a training-mode probe would cause.
            first_param = next(first_layer.parameters(), None)
            if first_param is None:
                raise ValueError(
                    f"DecodeTracer: first decoder layer {type(first_layer).__name__!r} "
                    f"has no parameters. Cannot determine device/dtype for probe."
                )
            first_dtype = first_param.dtype
            first_device = (
                first_param.device
                if not getattr(first_param, "is_meta", False)
                else torch.device("cpu")
            )

            hidden_size = model.config.hidden_size
            _hidden = torch.zeros(
                1, 1, hidden_size, dtype=first_dtype, device=first_device
            )
            _pos = torch.zeros(1, 1, dtype=torch.long, device=first_device)
            _cache_pos = torch.zeros(1, dtype=torch.long, device=first_device)
            _mask = torch.zeros(1, 1, 1, 1, dtype=_hidden.dtype, device=first_device)
            _kwargs: dict = {
                "attention_mask": _mask,
                "position_ids": _pos,
                "cache_position": _cache_pos,
            }
            # Pass past_key_value=None explicitly so models with a required
            # KV-cache parameter don't raise TypeError during the probe call.
            if self._kv_kwarg in layer_sig.parameters:
                _kwargs[self._kv_kwarg] = None

            if self._use_pos_embed and hasattr(model.model, "rotary_emb"):
                # On multi-GPU models rotary_emb may be on a different device.
                # Detect and raise rather than silently producing wrong outputs.
                rotary_param = next(model.model.rotary_emb.parameters(), None)
                if rotary_param is not None and rotary_param.device != first_device:
                    raise ValueError(
                        f"DecodeTracer: rotary_emb is on {rotary_param.device} but "
                        f"first decoder layer parameters are on {first_device}. "
                        f"Cannot probe for tuple-return detection. Move all modules "
                        f"to the same device before calling DecodeTracer.trace()."
                    )
                _kwargs["position_embeddings"] = model.model.rotary_emb(
                    _hidden, position_ids=_pos
                )

            was_training = first_layer.training
            first_layer.eval()
            log.debug("Probing %r for output structure...", type(first_layer).__name__)
            try:
                with torch.no_grad():
                    _out = first_layer(_hidden, **_kwargs)
            finally:
                first_layer.train(was_training)

            self._layer_returns_tuple: bool = isinstance(_out, (tuple, list))
            log.debug(
                "Probe complete: layer returns %s.",
                "tuple" if self._layer_returns_tuple else "tensor",
            )

            # Gemma models scale embeddings by sqrt(hidden_size) in their
            # model.forward() before the first decoder layer.  _Wrapper
            # bypasses model.forward() and calls embed_tokens directly, so we
            # must apply the same scaling here or hidden states will be ~60x too
            # small, causing all-newline degenerate outputs.
            model_type = getattr(model.config, "model_type", "").lower()
            self._embed_scale: Optional[float] = (
                float(model.config.hidden_size**0.5)
                if model_type in _SCALED_EMBED_MODELS
                else None
            )

            # ── Register forward_pre_hooks ────────────────────────────────────────
            # Hooks inject the DynamicCache before each layer's forward at runtime.
            # Hooks are NOT called during FX tracing (leaf modules are opaque).
            #
            # A weakref to self breaks the reference cycle:
            #   _Wrapper -> _kv_hooks -> closure -> _Wrapper
            # Without weakref, 80+ hook closures each hold a strong ref to the
            # wrapper, delaying GPU memory reclamation in long-running processes.
            self._kv_hooks: list[RemovableHandle] = []
            wrapper_ref: weakref.ref = weakref.ref(self)
            kv_kwarg = self._kv_kwarg

            def _make_hook(ref: weakref.ref, kv_key: str):
                def _hook(module, args, kwargs):
                    w = ref()
                    if w is None:
                        raise RuntimeError(
                            "DecodeTracer._Wrapper has been garbage collected while "
                            "KV forward_pre_hooks are still active. Remove all hooks "
                            "via _Wrapper._kv_hooks before releasing the wrapper."
                        )
                    kwargs[kv_key] = w.past_key_values
                    kwargs["use_cache"] = True
                    return args, kwargs

                return _hook

            for layer in model.model.layers:
                self._kv_hooks.append(
                    layer.register_forward_pre_hook(
                        _make_hook(wrapper_ref, kv_kwarg), with_kwargs=True
                    )
                )
            log.debug("Registered %d KV forward_pre_hooks.", len(self._kv_hooks))

        def reset_cache(self) -> None:
            """Replace the KV cache with a fresh empty DynamicCache.

            Call this between independent inference sessions to prevent KV state
            from a previous session contaminating the next one.  Direct attribute
            assignment (``wrapper.past_key_values = DynamicCache()``) has the same
            effect and is supported for backward compatibility, but this method
            is the preferred API.
            """
            self.past_key_values = DynamicCache()
            log.debug("KV cache reset.")

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,  # 4D float causal mask (B, 1, 1, S)
            position_ids: torch.Tensor,
            cache_position: torch.Tensor,
        ) -> torch.Tensor:
            hidden_states = self.model.model.embed_tokens(input_ids)
            if self._embed_scale is not None:
                hidden_states = hidden_states * self._embed_scale
            attention_mask = attention_mask.to(hidden_states.dtype)

            position_embeddings = None
            if self._use_pos_embed and hasattr(self.model.model, "rotary_emb"):
                position_embeddings = self.model.model.rotary_emb(
                    hidden_states, position_ids=position_ids
                )

            for layer in self.model.model.layers:
                layer_kwargs: dict = {
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                    "cache_position": cache_position,
                }
                if self._use_pos_embed and position_embeddings is not None:
                    layer_kwargs["position_embeddings"] = position_embeddings

                # past_key_value is NOT in layer_kwargs here - injected by
                # _hook at runtime so DynamicCache never enters the FX graph.
                # _layer_returns_tuple is detected before tracing so getitem()
                # nodes are emitted for tuple-returning layers like Gemma-2.
                layer_out = layer(hidden_states, **layer_kwargs)
                if self._layer_returns_tuple:
                    hidden_states = layer_out[0]
                else:
                    hidden_states = layer_out

            if hasattr(self.model.model, "norm"):
                hidden_states = self.model.model.norm(hidden_states)

            # Only the last position's logits are ever consumed for next-token
            # prediction (prefill seeds token 0 from the final prompt position;
            # decode runs one token at a time). Materializing logits for every
            # prefill position costs seq_len x vocab and is fatal for long
            # context with a large vocab (e.g. Mistral-Nemo's ~131K vocab makes
            # a 128K prefill's logits ~34 GB). Slice to the last position before
            # lm_head so the graph only ever produces [batch, 1, vocab]. The
            # slice is a no-op at decode (seq_len==1) and during trace. This
            # assumes the last index is the last *valid* token (unpadded or
            # left-padded prompts, the standard generation convention); the
            # runtime's next-token selector is matched to this in
            # pipeline_runtime._select_next_token_logits.
            hidden_states = hidden_states[:, -1:, :]

            logits = self.model.lm_head(hidden_states)
            # Gemma-2 logit softcapping: tanh(x / cap) * cap.
            # Order matters - do NOT reorder these three lines.
            final_softcap = getattr(self.model.config, "final_logit_softcapping", None)
            if final_softcap is not None:
                logits = logits / final_softcap
                logits = torch.tanh(logits)
                logits = logits * final_softcap
            return logits  # KV cache updated in-place via forward_pre_hooks

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def __init__(self, model: nn.Module) -> None:
        _validate_model_structure(model)
        self.model = model
        self._wrapper: Optional[DecodeTracer._Wrapper] = None
        # Guards against concurrent trace() calls on the same instance.
        self._trace_lock = threading.Lock()

    def trace(self) -> fx.GraphModule:
        """
        Symbolically trace a single decode step. No concrete inputs are
        needed - FX uses proxies. Dummy inputs for runtime are provided
        separately via ``build_dummy_inputs()``.

        Not thread-safe: raises ``RuntimeError`` if called concurrently
        on the same instance.  Use a separate ``DecodeTracer`` per thread.
        """
        if not self._trace_lock.acquire(blocking=False):
            raise RuntimeError(
                "DecodeTracer.trace() is already running on this instance. "
                "DecodeTracer is not thread-safe - use a separate instance per thread."
            )
        try:
            tracer = self._HelmTracer()
            wrapper = self._Wrapper(self.model)
            try:
                log.debug("Starting FX symbolic trace...")
                graph = tracer.trace(wrapper)
            except Exception:
                log.debug(
                    "FX trace failed; removing %d KV hooks.", len(wrapper._kv_hooks)
                )
                for h in wrapper._kv_hooks:
                    h.remove()
                raise

            # Store the wrapper on DecodeTracer for callers that have a reference
            # to the tracer itself.
            self._wrapper = wrapper

            # Also store it on the model via object.__setattr__ (bypassing
            # nn.Module.__setattr__) so that getattr(model, "_helm_decode_wrapper")
            # works for callers that only hold a reference to the model.
            #
            # We MUST use object.__setattr__ here.  Normal assignment would call
            # nn.Module.__setattr__, which registers _Wrapper as a child module in
            # model._modules.  That causes model.named_modules() to recurse into
            # wrapper.model (the original model), creating an infinite cycle.
            #
            # This back-reference is set AFTER tracing to avoid FX collect_tensor_attrs
            # entering an infinite cycle through the (not-yet-existing) circular ref:
            #   wrapper.model._helm_decode_wrapper.model._helm_decode_wrapper ...
            object.__setattr__(self.model, "_helm_decode_wrapper", wrapper)

            gm = fx.GraphModule(wrapper, graph)
            gm.graph.lint()
            gm.recompile()
            log.debug("FX trace complete: %d nodes.", sum(1 for _ in gm.graph.nodes))
            return gm
        finally:
            self._trace_lock.release()

    @staticmethod
    def build_dummy_inputs(
        device: str = "cpu",
        batch_size: int = 1,
        dtype: torch.dtype = torch.float32,
        cache_position: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Construct decode dummy inputs for a single decode step.

        Returns a 4-tuple: (input_ids, causal_mask, position_ids, cache_position_tensor)

        ``causal_mask`` shape: ``(batch_size, 1, 1, cache_position + 1)``.
        Value 0.0 at every position - an additive mask that attends to all
        tokens from position 0 through ``cache_position`` inclusive.  At the
        first decode step (``cache_position=0``) this collapses to
        ``(batch_size, 1, 1, 1)``.

        All batch elements share the same ``cache_position`` scalar.  This is
        correct for uniform-length batches; variable-length batches require a
        custom mask.

        ``past_key_values`` is managed by the ``_Wrapper`` instance and is not
        returned here.

        Raises
        ------
        ValueError
            If ``cache_position`` is negative.
        """
        if cache_position < 0:
            raise ValueError(f"cache_position must be >= 0, got {cache_position}.")

        # Query attends to all past KV entries plus itself: positions 0..cache_position.
        seq_len = cache_position + 1
        input_ids = torch.zeros((batch_size, 1), dtype=torch.long, device=device)

        # Additive causal mask: 0.0 = attend, large negative = mask out.
        # At decode time there is only one query token, so no future positions
        # need masking - all seq_len key positions are attended to.
        causal_mask = torch.zeros(
            (batch_size, 1, 1, seq_len), dtype=dtype, device=device
        )

        position_ids = torch.full(
            (batch_size, 1), cache_position, dtype=torch.long, device=device
        )
        cache_position_tensor = torch.tensor(
            [cache_position], dtype=torch.long, device=device
        )

        return input_ids, causal_mask, position_ids, cache_position_tensor
