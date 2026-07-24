import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.llama import LlamaConfig, LlamaForCausalLM

from helm.runtime.kv_offload import (
    KVOffloadConfig,
    KVOffloadManager,
    _decode_batched,
    _make_gemma2_forward,
    _make_llama_forward,
)


def _build_tiny_llama():
    config = LlamaConfig(
        vocab_size=97,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    return LlamaForCausalLM(config).eval()


def _build_tiny_mistral():
    from transformers.models.mistral import MistralConfig, MistralForCausalLM
    config = MistralConfig(
        vocab_size=97,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        attention_dropout=0.0,
        sliding_window=None,
    )
    config._attn_implementation = "eager"
    return MistralForCausalLM(config).eval()


def test_mistral_patches_mistral_attention_not_llama():
    """Mistral defines its own MistralAttention (not a LlamaAttention subclass).

    Regression guard: patching LlamaAttention would silently miss every Mistral
    attention module, so KV offload (and paging) would never engage. The manager
    must patch MistralAttention directly and route decode through the cache.
    """
    import transformers.models.mistral.modeling_mistral as mm

    model = _build_tiny_mistral()
    kv_mgr = KVOffloadManager(model, KVOffloadConfig.from_model(model), batch_size=1)
    try:
        assert "mistral" in kv_mgr._patched
        assert kv_mgr._patched["mistral"][0] is mm.MistralAttention

        attn = model.model.layers[0].self_attn
        kvcm = kv_mgr.kvcms[0]
        decode_calls = 0
        orig_decode = kvcm.append_decode

        def count_decode(*args, **kwargs):
            nonlocal decode_calls
            decode_calls += 1
            return orig_decode(*args, **kwargs)

        kvcm.append_decode = count_decode
        _run_single_decode(attn, model)
        assert decode_calls == 1, "Mistral decode did not route through KVCacheManager"
    finally:
        kv_mgr.restore()


def _run_single_decode(attn, model):
    hidden = torch.randn(1, 1, model.config.hidden_size, dtype=torch.float32)
    position_ids = torch.tensor([[0]], dtype=torch.long)
    position_embeddings = model.model.rotary_emb(hidden, position_ids=position_ids)
    attention_mask = torch.zeros((1, 1, 1, 1), dtype=torch.float32)
    cache_position = torch.tensor([0], dtype=torch.long)

    with torch.no_grad():
        attn(
            hidden,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            cache_position=cache_position,
        )


def test_patch_module_roots_rebinds_shadowed_attention_forward():
    torch.manual_seed(0)
    base_model = _build_tiny_llama()
    shadow_model = copy.deepcopy(base_model)
    shadow_attn = shadow_model.model.layers[0].self_attn

    # Shadow the class method with an instance-level bound forward so later
    # class patches do not affect this copied module unless we rebind it.
    shadow_attn.forward = shadow_attn.forward

    kv_mgr = KVOffloadManager(base_model, KVOffloadConfig.from_model(base_model), batch_size=1)
    kvcm = kv_mgr.kvcms[0]
    decode_calls = 0
    orig_decode = kvcm.append_decode

    def count_decode(*args, **kwargs):
        nonlocal decode_calls
        decode_calls += 1
        return orig_decode(*args, **kwargs)

    kvcm.append_decode = count_decode

    try:
        _run_single_decode(shadow_attn, shadow_model)
        assert decode_calls == 0

        kv_mgr.patch_module_roots(shadow_model)

        _run_single_decode(shadow_attn, shadow_model)
        assert decode_calls == 1
    finally:
        kv_mgr.restore()


class _RecordingDecodeKVCache:
    use_contiguous = False

    def __init__(self):
        self.skip_residency_values = []
        self.pages = []

    def all_gpu_resident(self, layer_idx):
        return True

    def append_decode(self, layer_idx, k, v, skip_residency=False):
        self.skip_residency_values.append(skip_residency)
        self.pages.append(
            SimpleNamespace(
                used_tokens=1,
                k_tensor=k,
                v_tensor=v,
            )
        )

    def iterate_layer_pages(self, layer_idx):
        return list(self.pages)


def test_decode_batched_enforces_residency_even_when_pages_start_on_gpu():
    kvcm = _RecordingDecodeKVCache()
    q = torch.randn(1, 1, 1, 4)
    k = torch.randn(1, 1, 1, 4)
    v = torch.randn(1, 1, 1, 4)

    _decode_batched([kvcm], q, k, v, layer_idx=0, scale=1.0)

    assert kvcm.skip_residency_values == [False]


def _build_padded_decode_case():
    """
    Padded prefill of length 5: positions 0-2 are real tokens, 3-4 are pad
    positions holding "poison" K/V. The pad keys are built aligned with q
    (q . pad_k = |q|^2 * scale > 0 regardless of q's random sign) so they are
    *guaranteed* to dominate an unmasked softmax — with pad_v set far from the
    real values, an unmasked decode is guaranteed to diverge measurably from
    the real-only reference, making this a real (non-coincidental) check of
    whether attention_mask is honored.

    Returns (kvcm, q, k_dec, v_dec, mask, expected_masked_output).
    """
    from helm.runtime.kv_allocator import KVAllocator
    from helm.runtime.kv_cache import KVCacheManager

    num_kv_heads, head_dim, page_size = 1, 4, 8
    torch.manual_seed(0)
    q = torch.randn(1, num_kv_heads, 1, head_dim)
    real_k = torch.randn(1, num_kv_heads, 3, head_dim)
    real_v = torch.randn(1, num_kv_heads, 3, head_dim)
    pad_k = q.expand(1, num_kv_heads, 2, head_dim) * 50.0
    pad_v = torch.full((1, num_kv_heads, 2, head_dim), 1000.0)
    k_dec = torch.randn(1, num_kv_heads, 1, head_dim)
    v_dec = torch.randn(1, num_kv_heads, 1, head_dim)

    alloc = KVAllocator(num_layers=1, num_kv_heads=num_kv_heads, head_dim=head_dim,
                        page_size=page_size, dtype=torch.float32)
    kvcm = KVCacheManager(alloc, gpu_high_watermark_bytes=0)
    kvcm.use_contiguous = False
    kvcm.append_prefill(0, torch.cat([real_k, pad_k], dim=2), torch.cat([real_v, pad_v], dim=2))

    # Additive mask over [real0, real1, real2, pad0, pad1, decode]: 0 to keep,
    # -inf to exclude the two pad slots.
    mask = torch.zeros(1, 1, 1, 6)
    mask[..., 3:5] = float("-inf")

    # Reference: attention over only the real prefix + the new decode token,
    # with the pad K/V excluded from the computation entirely.
    k_ref = torch.cat([real_k, k_dec], dim=2)
    v_ref = torch.cat([real_v, v_dec], dim=2)
    expected = F.scaled_dot_product_attention(q, k_ref, v_ref, scale=1.0)

    return kvcm, q, k_dec, v_dec, mask, expected


def test_decode_batched_paged_path_masks_padded_prompt_tokens():
    """
    Regression: batched decode (Qwen2/Qwen3/Llama/OLMo2 — every architecture
    except Gemma2, which already builds its own mask) must forward the
    caller's attention_mask into the paged decode path. Prefill caches
    whatever K/V it is handed, including pad positions from batch padding, so
    without the mask a shorter prompt's decode step would attend to its own
    pad-token K/V garbage cached alongside the real tokens.

    Uses the real KVAllocator/KVCacheManager (CPU pages) so this exercises the
    actual mixed/CPU-residency streaming attention path in kv_cache.py, not a
    mock.
    """
    kvcm, q, k_dec, v_dec, mask, expected = _build_padded_decode_case()

    out_masked = _decode_batched([kvcm], q, k_dec, v_dec, layer_idx=0, scale=1.0,
                                 attention_mask=mask)

    assert torch.allclose(out_masked, expected, atol=1e-4), (
        "decode output was pulled toward the poison pad K/V — attention_mask "
        "is being dropped on the batched decode path"
    )


def test_decode_batched_paged_path_without_mask_is_unaffected():
    """Sanity check for the previous test: without an attention_mask, decode
    genuinely does attend over the (poison) pad tokens too, so the masked
    test above is actually exercising the mask rather than trivially passing."""
    kvcm, q, k_dec, v_dec, _mask, expected_masked = _build_padded_decode_case()

    out_unmasked = _decode_batched([kvcm], q, k_dec, v_dec, layer_idx=0, scale=1.0)

    assert not torch.allclose(out_unmasked, expected_masked, atol=1e-4)


def test_config_from_model_disables_contiguous_cache_by_default():
    model = _build_tiny_llama()

    cfg = KVOffloadConfig.from_model(model)

    assert cfg.cont_capacity == 0


def test_config_from_model_preserves_explicit_contiguous_capacity():
    model = _build_tiny_llama()

    cfg = KVOffloadConfig.from_model(model, cont_capacity=128)

    assert cfg.cont_capacity == 128


class _RecordingKVCache:
    def append_prefill(self, layer_idx, k, v):
        return None

    def append_decode(self, layer_idx, k, v):
        return None

    def iterate_layer_pages(self, layer_idx):
        return []


class _PagedPrefillKVCache:
    use_contiguous = False

    def __init__(self):
        self.pages = []

    def append_prefill(self, layer_idx, k, v):
        start_token = sum(page.used_tokens for page in self.pages)
        self.pages.append(
            SimpleNamespace(
                used_tokens=k.shape[2],
                capacity_tokens=k.shape[2],
                start_token=start_token,
                k_tensor=k.clone(),
                v_tensor=v.clone(),
                device=k.device,
                state="CPU",
            )
        )

    def iterate_layer_pages(self, layer_idx):
        return list(self.pages)


def test_llama_prefill_with_past_streams_over_all_paged_kv(monkeypatch):
    import transformers.models.llama.modeling_llama as llama_modeling

    monkeypatch.setattr(
        llama_modeling,
        "apply_rotary_pos_emb",
        lambda q, k, cos, sin: (q, k),
    )

    stream_calls = []

    def fake_streaming_attention(
        query, pages, scale=None, attention_mask=None, softcap=None
    ):
        stream_calls.append(
            {
                "query_len": query.shape[2],
                "tokens": sum(page.used_tokens for page in pages),
                "mask_shape": tuple(attention_mask.shape) if attention_mask is not None else None,
            }
        )
        return torch.zeros_like(query)

    monkeypatch.setattr(
        "helm.runtime.kv_offload.perform_streaming_attention",
        fake_streaming_attention,
    )

    attn = nn.Module()
    attn.q_proj = nn.Linear(4, 4, bias=False)
    attn.k_proj = nn.Linear(4, 4, bias=False)
    attn.v_proj = nn.Linear(4, 4, bias=False)
    attn.o_proj = nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        for proj in (attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj):
            proj.weight.copy_(torch.eye(4))
    attn.head_dim = 2
    attn.attention_dropout = 0.0
    attn.training = False
    attn.layer_idx = 0

    cache = _PagedPrefillKVCache()
    forward = _make_llama_forward([cache])

    first_chunk = torch.randn(1, 2, 4)
    forward(
        attn,
        first_chunk,
        position_embeddings=(None, None),
        attention_mask=torch.zeros(1, 1, 2, 2),
        cache_position=torch.arange(2, dtype=torch.long),
    )
    assert stream_calls == []

    second_chunk = torch.randn(1, 2, 4)
    forward(
        attn,
        second_chunk,
        position_embeddings=(None, None),
        attention_mask=torch.zeros(1, 1, 2, 4),
        cache_position=torch.arange(2, 4, dtype=torch.long),
    )

    assert stream_calls == [
        {"query_len": 2, "tokens": 4, "mask_shape": (1, 1, 2, 4)}
    ]


def test_gemma2_forward_enforces_sliding_window_mask(monkeypatch):
    import transformers.models.gemma2.modeling_gemma2 as gemma2_modeling

    monkeypatch.setattr(
        gemma2_modeling,
        "apply_rotary_pos_emb",
        lambda q, k, cos, sin: (q, k),
    )

    attn = nn.Module()
    attn.q_proj = nn.Linear(3, 1, bias=False)
    attn.k_proj = nn.Linear(3, 1, bias=False)
    attn.v_proj = nn.Linear(3, 1, bias=False)
    attn.o_proj = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        attn.q_proj.weight.copy_(torch.tensor([[1.0, 0.0, 0.0]]))
        attn.k_proj.weight.copy_(torch.tensor([[0.0, 1.0, 0.0]]))
        attn.v_proj.weight.copy_(torch.tensor([[0.0, 0.0, 1.0]]))
        attn.o_proj.weight.fill_(1.0)

    attn.head_dim = 1
    attn.num_key_value_groups = 1
    attn.attention_dropout = 0.0
    attn.training = False
    attn.layer_idx = 0
    attn.scaling = 1.0
    attn.attn_logit_softcapping = None
    attn.sliding_window = 2

    hidden_states = torch.tensor(
        [[[1.0, 10.0, 100.0], [1.0, 0.0, 1.0], [1.0, 0.0, 2.0]]],
        dtype=torch.float32,
    )
    min_val = torch.finfo(torch.float32).min
    causal_mask = torch.full((3, 3), min_val, dtype=torch.float32)
    causal_mask = torch.triu(causal_mask, diagonal=1)[None, None, :, :]

    forward = _make_gemma2_forward([_RecordingKVCache()])
    output, _ = forward(
        attn,
        hidden_states,
        position_embeddings=(None, None),
        attention_mask=causal_mask,
        cache_position=torch.arange(3, dtype=torch.long),
    )

    assert output[0, 2, 0].item() == pytest.approx(1.5, abs=1e-4)


@pytest.mark.parametrize("restore_first", [0, 1])
def test_managers_do_not_change_other_models_or_attention_class(restore_first):
    from transformers.models.llama.modeling_llama import LlamaAttention

    original = LlamaAttention.forward
    models = [_build_tiny_llama() for _ in range(3)]
    managers = []
    try:
        for model in models[:2]:
            managers.append(KVOffloadManager(model, KVOffloadConfig.from_model(model)))
        assert LlamaAttention.forward is original
        assert models[2].model.layers[0].self_attn.forward.__func__ is original
        managers[restore_first].restore()
        remaining = 1 - restore_first
        _run_single_decode(models[remaining].model.layers[0].self_attn, models[remaining])
        assert managers[remaining].kvcms[0].seq_len() == 1
        assert LlamaAttention.forward is original
    finally:
        for manager in managers:
            manager.restore()
    assert all("forward" not in model.model.layers[0].self_attn.__dict__ for model in models)
