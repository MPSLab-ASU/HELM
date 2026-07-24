"""Real mixed CPU/GPU generation with forced KV eviction; no downloads."""
import pytest
import torch

from helm.runtime.inference import HelmInference, HelmInferenceConfig
from helm.runtime.kv_cache import get_paging_stats, reset_paging_stats
from helm.runtime.kv_offload import KVOffloadConfig


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("disable_async", [False, True])
def test_hybrid_eviction_matches_reference_across_requests(tmp_path, monkeypatch, disable_async):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HELM_PREFILL_CHUNK", "128")
    monkeypatch.setenv("HELM_DISABLE_ASYNC_KV", "1" if disable_async else "0")
    vocab = {"<unk>": 0, "<pad>": 1, **{f"t{i}": i for i in range(2, 32)}}
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", pad_token="<pad>")
    tokenizer.model_input_names = ["input_ids", "attention_mask"]
    tokenizer.save_pretrained(tmp_path)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        reference = LlamaForCausalLM(LlamaConfig(
            vocab_size=32, hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=1024, pad_token_id=1, eos_token_id=None,
            attention_dropout=0.0,
        )).eval()
    reference.save_pretrained(tmp_path)
    prompts = [" ".join(f"t{2 + i % 30}" for i in range(length)) for length in [513, 17]]
    expected = []
    with torch.inference_mode():
        for prompt in prompts:
            encoded = tokenizer(prompt, return_tensors="pt")
            ids = reference.generate(**encoded, max_new_tokens=16, do_sample=False)
            expected.append(tokenizer.decode(ids[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True))
    original = KVOffloadConfig.from_model
    def small_cache(model, **kwargs):
        kwargs.update(page_size=16, gpu_watermark_bytes=1024, cont_capacity=0)
        return original(model, **kwargs)
    monkeypatch.setattr(KVOffloadConfig, "from_model", staticmethod(small_cache))
    config = HelmInferenceConfig(
        model_id=str(tmp_path), dtype=torch.float32, max_input_tokens=600,
        max_new_tokens=16, cpu_threads=2, kv_offload=True,
        plan_mode="manual", cpu_layers="0:0", gpu_layers="1:1",
        route_to_vllm_when_all_gpu=False,
    )
    with HelmInference(config) as inference:
        devices = {stage.device_id.split(":")[0] for stage in inference.partition_plan.stages}
        assert devices == {"cpu", "cuda"}
        reset_paging_stats()
        assert inference.generate(prompts[:1]) == expected[:1]
        stats = get_paging_stats()
        assert stats["pages_evicted"] > 0
        if not disable_async:
            assert stats["pages_prefetched"] > 0
        parameters = list(inference._handle._model.parameters())
        placement = [(p.device, p.data_ptr()) for p in parameters]
        assert inference.generate(prompts[1:]) == expected[1:]
        assert [(p.device, p.data_ptr()) for p in parameters] == placement
        print("verified paging counters:", stats)
