"""Offline end-to-end proof for the public native inference path."""

import os

import torch

from helm.runtime.inference import HelmInference, HelmInferenceConfig


def test_native_generate_matches_huggingface_across_requests(tmp_path, monkeypatch):
    """Trace, compile, and decode a real tiny Llama, without network or GPU needs."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    # Inference sets a model-aware reserve while planning; do not leak it to
    # other tests or the caller's environment.
    monkeypatch.delenv("HELM_GPU_MEMORY_RESERVE_MB", raising=False)

    vocab = {
        "<pad>": 0, "<eos>": 1, "<unk>": 2, "Benchmark": 3,
        "prompt": 4, "for": 5, "compilation": 6, ".": 7,
        "Hello": 8, "world": 9, "Longer": 10, "test": 11,
        "sentence": 12, "more": 13, "tokens": 14,
    }
    tokenizers_model = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizers_model.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizers_model,
        pad_token="<pad>", eos_token="<eos>", unk_token="<unk>",
    )
    tokenizer.model_input_names = ["input_ids", "attention_mask"]
    tokenizer.save_pretrained(tmp_path)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        reference = LlamaForCausalLM(LlamaConfig(
            vocab_size=len(vocab), hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=128, pad_token_id=0, eos_token_id=1,
        )).eval()
    reference.save_pretrained(tmp_path)

    prompts = ["Hello world", "Longer test sentence", "Hello",
               "Hello world more tokens"]
    expected = []
    with torch.inference_mode():
        for prompt in prompts:
            encoded = tokenizer(prompt, return_tensors="pt")
            ids = reference.generate(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                max_new_tokens=3, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            expected.append(tokenizer.decode(
                ids[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True,
            ))

    config = HelmInferenceConfig(
        model_id=str(tmp_path), dtype=torch.float32,
        max_input_tokens=16, max_new_tokens=3, cpu_threads=2,
        route_to_vllm_when_all_gpu=False,
    )
    with HelmInference(config) as inference:
        assert inference.routed_to_vllm is False
        actual = inference.generate(prompts[:2]) + inference.generate(prompts[2:])
    assert actual == expected
    assert "HELM_GPU_MEMORY_RESERVE_MB" not in os.environ
