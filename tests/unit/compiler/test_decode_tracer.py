"""
Unit tests for DecodeTracer (helm/compiler/importers/decode_tracer.py).

Uses a minimal TinyLlamaLike model that mirrors the module-path structure
expected by DecodeTracer without loading a real LLM.
"""
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.fx as fx
import torch.nn as nn
from transformers.cache_utils import DynamicCache

from helm.compiler.importers.decode_tracer import (
    DecodeTracer,
    _LEAF_MODULE_NAMES,
    _validate_model_structure,
)


# ── Minimal fake model ────────────────────────────────────────────────────────

class TinyDecoderLayer(nn.Module):
    """Class name ends with 'DecoderLayer' so _HelmTracer marks it as a leaf."""

    def __init__(self, hidden: int = 8):
        super().__init__()
        self.self_attn = nn.Linear(hidden, hidden, bias=False)
        self.mlp = nn.Linear(hidden, hidden, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        cache_position=None,
        past_key_values=None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        return self.mlp(self.self_attn(hidden_states))


class _TinyInner(nn.Module):
    def __init__(self, vocab: int = 16, hidden: int = 8, n_layers: int = 2):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList([TinyDecoderLayer(hidden) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x):
        raise NotImplementedError


class TinyLlamaLike(nn.Module):
    """Minimal stand-in for LlamaForCausalLM."""

    def __init__(self, vocab: int = 16, hidden: int = 8, n_layers: int = 2):
        super().__init__()
        self.model = _TinyInner(vocab, hidden, n_layers)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self.config = SimpleNamespace(
            hidden_size=hidden,
            model_type="llama",
            final_logit_softcapping=None,
        )

    def forward(self, input_ids, **kwargs):
        raise NotImplementedError


@pytest.fixture
def tiny_model():
    """Fresh model instance per test — avoids hook state leaking between tests."""
    return TinyLlamaLike().eval()


@pytest.fixture
def tracer(tiny_model):
    return DecodeTracer(tiny_model)


# ── _validate_model_structure ─────────────────────────────────────────────────

class TestValidateModelStructure:
    def test_valid_model_passes(self, tiny_model):
        _validate_model_structure(tiny_model)  # must not raise

    def test_missing_model_attr_raises(self):
        bad = nn.Linear(4, 4)
        with pytest.raises(ValueError, match=r"\.model attribute"):
            _validate_model_structure(bad)

    def test_model_without_layers_raises(self):
        class NoLayers(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Linear(4, 4)  # .model exists, no .layers
        with pytest.raises(ValueError, match=r"\.layers"):
            _validate_model_structure(NoLayers())

    def test_empty_layers_raises(self):
        class EmptyLayers(nn.Module):
            def __init__(self):
                super().__init__()
                inner = nn.Module()
                inner.layers = nn.ModuleList()  # empty
                self.model = inner
        with pytest.raises(ValueError, match=r"empty"):
            _validate_model_structure(EmptyLayers())

    def test_decode_tracer_init_validates(self):
        """DecodeTracer.__init__ itself must run the validation."""
        bad = nn.Linear(4, 4)
        with pytest.raises(ValueError, match=r"\.model attribute"):
            DecodeTracer(bad)


# ── build_dummy_inputs ────────────────────────────────────────────────────────

class TestBuildDummyInputs:
    def test_step0_shapes(self):
        ids, mask, pos, cp = DecodeTracer.build_dummy_inputs(batch_size=1, cache_position=0)
        assert ids.shape == (1, 1)
        assert mask.shape == (1, 1, 1, 1)   # seq_len = 0 + 1 = 1
        assert pos.shape == (1, 1)
        assert cp.shape == (1,)
        assert cp.item() == 0

    def test_step5_mask_width(self):
        _, mask, pos, cp = DecodeTracer.build_dummy_inputs(batch_size=2, cache_position=5)
        assert mask.shape == (2, 1, 1, 6)   # seq_len = 5 + 1 = 6
        assert (pos == 5).all()
        assert cp.item() == 5

    def test_mask_is_all_zeros(self):
        """Additive mask must be 0.0 everywhere — attend to all positions."""
        _, mask, _, _ = DecodeTracer.build_dummy_inputs(cache_position=10)
        assert (mask == 0.0).all()

    def test_batch_size_propagates(self):
        ids, mask, pos, _ = DecodeTracer.build_dummy_inputs(batch_size=4, cache_position=3)
        assert ids.shape[0] == 4
        assert mask.shape[0] == 4
        assert pos.shape[0] == 4

    def test_negative_cache_position_raises(self):
        with pytest.raises(ValueError, match=r"cache_position must be >= 0"):
            DecodeTracer.build_dummy_inputs(cache_position=-1)

    def test_position_ids_match_cache_position(self):
        _, _, pos, cp = DecodeTracer.build_dummy_inputs(batch_size=3, cache_position=7)
        assert (pos == 7).all()
        assert cp.item() == 7


# ── _HelmTracer.is_leaf_module ────────────────────────────────────────────────

class TestHelmTracerLeafMatching:
    @pytest.fixture(scope="class")
    @classmethod
    def ht(cls):
        return DecodeTracer._HelmTracer()

    def test_embed_tokens_exact_segment_is_leaf(self, ht):
        assert ht.is_leaf_module(nn.Embedding(16, 8), "model.model.embed_tokens") is True

    def test_lm_head_exact_segment_is_leaf(self, ht):
        assert ht.is_leaf_module(nn.Linear(8, 16), "model.lm_head") is True

    def test_rotary_emb_exact_segment_is_leaf(self, ht):
        assert ht.is_leaf_module(nn.Identity(), "model.model.rotary_emb") is True

    def test_decoder_layer_class_name_is_leaf(self, ht):
        assert ht.is_leaf_module(TinyDecoderLayer(), "model.model.layers.0") is True

    def test_substring_embed_tokens_not_leaf(self, ht):
        """'shared_embed_tokens_proj' contains 'embed_tokens' as substring —
        must NOT be matched by our exact-segment rule."""
        # Verify the name is not in the exact-match set
        assert "shared_embed_tokens_proj" not in _LEAF_MODULE_NAMES
        # The segment check should not fire: rsplit gives "shared_embed_tokens_proj"
        leaf_name = "model.shared_embed_tokens_proj".rsplit(".", 1)[-1]
        assert leaf_name not in _LEAF_MODULE_NAMES

    def test_non_decoder_layer_class_not_leaf_by_name(self, ht):
        """nn.LayerNorm class name does not end with 'DecoderLayer'."""
        assert not type(nn.LayerNorm(8)).__name__.endswith("DecoderLayer")


# ── DecodeTracer.trace() ──────────────────────────────────────────────────────

class TestTrace:
    def test_returns_graph_module(self, tracer):
        gm = tracer.trace()
        assert isinstance(gm, fx.GraphModule)

    def test_wrapper_stored_on_tracer(self, tracer):
        tracer.trace()
        assert tracer._wrapper is not None
        assert isinstance(tracer._wrapper, DecodeTracer._Wrapper)

    def test_wrapper_on_model_accessible_via_getattr(self, tiny_model):
        DecodeTracer(tiny_model).trace()
        assert getattr(tiny_model, "_helm_decode_wrapper", None) is not None

    def test_wrapper_not_registered_in_model_modules(self, tiny_model):
        """object.__setattr__ must bypass nn.Module._modules registration
        to prevent named_modules() from entering an infinite cycle."""
        DecodeTracer(tiny_model).trace()
        assert "_helm_decode_wrapper" not in tiny_model._modules

    def test_hook_count_equals_num_layers(self, tiny_model):
        t = DecodeTracer(tiny_model)
        t.trace()
        assert len(t._wrapper._kv_hooks) == len(tiny_model.model.layers)

    def test_retrace_removes_stale_hooks(self, tiny_model):
        """Re-tracing must remove the previous wrapper's hooks from each layer."""
        t = DecodeTracer(tiny_model)
        t.trace()
        # Record the hook IDs registered by the first trace
        first_hook_ids = {h.id for h in t._wrapper._kv_hooks}
        t.trace()
        # Those IDs must no longer appear in any layer's pre-hook dict
        for layer in tiny_model.model.layers:
            active_ids = set(layer._forward_pre_hooks.keys())
            assert not first_hook_ids & active_ids, (
                f"Stale hook IDs {first_hook_ids & active_ids} still active after re-trace"
            )

    def test_failed_trace_removes_hooks(self, tiny_model):
        """If FX tracing raises, all hooks registered by that _Wrapper are removed."""
        hooks_before = sum(
            len(layer._forward_pre_hooks) for layer in tiny_model.model.layers
        )
        with patch.object(
            DecodeTracer._HelmTracer, "trace", side_effect=RuntimeError("simulated failure")
        ):
            with pytest.raises(RuntimeError, match="simulated failure"):
                DecodeTracer(tiny_model).trace()

        hooks_after = sum(
            len(layer._forward_pre_hooks) for layer in tiny_model.model.layers
        )
        assert hooks_after == hooks_before, (
            f"hook count changed after failed trace: {hooks_before} → {hooks_after}"
        )

    def test_graph_has_placeholder_and_output(self, tracer):
        gm = tracer.trace()
        ops = [n.op for n in gm.graph.nodes]
        assert "placeholder" in ops
        assert "output" in ops


# ── Thread safety ─────────────────────────────────────────────────────────────

class TestThreadSafety:
    def test_concurrent_trace_raises(self, tiny_model):
        """Acquiring the lock externally simulates a thread already inside trace()."""
        t = DecodeTracer(tiny_model)
        acquired = t._trace_lock.acquire(blocking=False)
        assert acquired, "lock should be free before any trace() call"
        try:
            with pytest.raises(RuntimeError, match="not thread-safe"):
                t.trace()
        finally:
            t._trace_lock.release()

    def test_lock_released_after_successful_trace(self, tracer):
        tracer.trace()
        # Lock must be free so a subsequent trace() can proceed
        assert tracer._trace_lock.acquire(blocking=False)
        tracer._trace_lock.release()

    def test_lock_released_after_failed_trace(self, tiny_model):
        t = DecodeTracer(tiny_model)
        with patch.object(
            DecodeTracer._HelmTracer, "trace", side_effect=RuntimeError("boom")
        ):
            with pytest.raises(RuntimeError):
                t.trace()
        # Lock must be free after the failure
        assert t._trace_lock.acquire(blocking=False)
        t._trace_lock.release()


# ── _Wrapper.reset_cache() ────────────────────────────────────────────────────

class TestResetCache:
    def test_reset_replaces_cache_instance(self, tiny_model):
        t = DecodeTracer(tiny_model)
        t.trace()
        wrapper = t._wrapper
        old = wrapper.past_key_values
        wrapper.reset_cache()
        assert wrapper.past_key_values is not old
        assert isinstance(wrapper.past_key_values, DynamicCache)

    def test_reset_clears_accumulated_kv(self, tiny_model):
        t = DecodeTracer(tiny_model)
        t.trace()
        wrapper = t._wrapper
        # Simulate a decode step via the public update() API
        dummy_k = torch.zeros(1, 1, 1, 8)
        dummy_v = torch.zeros(1, 1, 1, 8)
        wrapper.past_key_values.update(dummy_k, dummy_v, layer_idx=0)
        assert len(wrapper.past_key_values) > 0
        wrapper.reset_cache()
        assert len(wrapper.past_key_values) == 0

    def test_hook_injects_new_cache_after_reset(self, tiny_model):
        """Hook must read past_key_values dynamically (via weakref), not a snapshot,
        so reset_cache() is immediately visible to the hook."""
        t = DecodeTracer(tiny_model)
        t.trace()
        wrapper = t._wrapper
        old_cache = wrapper.past_key_values
        wrapper.reset_cache()

        # Invoke the first layer's hook directly to see what it injects
        first_layer = tiny_model.model.layers[0]
        injected = {}
        for hook_fn in first_layer._forward_pre_hooks.values():
            _, kw = hook_fn(first_layer, (), {})
            injected = kw
            break

        # Use explicit fallback — DynamicCache() is falsy when empty, so `or` would swallow it
        new_cache = injected.get("past_key_values", injected.get("past_key_value"))
        assert new_cache is not None, "hook must inject a cache"
        assert new_cache is not old_cache
        assert new_cache is wrapper.past_key_values

    def test_direct_assignment_also_works(self, tiny_model):
        """Direct attribute assignment (backward-compat API) must behave like reset_cache()."""
        t = DecodeTracer(tiny_model)
        t.trace()
        wrapper = t._wrapper
        new_cache = DynamicCache()
        wrapper.past_key_values = new_cache

        first_layer = tiny_model.model.layers[0]
        for hook_fn in first_layer._forward_pre_hooks.values():
            _, kw = hook_fn(first_layer, (), {})
            injected = kw.get("past_key_values", kw.get("past_key_value"))
            assert injected is new_cache
            break
