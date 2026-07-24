import transformers
import torch

from experiments.paper_bench import BenchStats, HelmBackend, _print_stats


class _FakeTokenizer:
    def __call__(self, prompt, return_tensors="pt", truncation=True, max_length=None):
        return {
            "input_ids": torch.tensor([[7, 8, 9, 0]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1, 0]], dtype=torch.long),
        }


class _FakeRuntime:
    def __init__(self):
        self.generate_calls = []

    def _reset_decode_cache(self):
        return None

    def generate(self, input_ids, max_new_tokens=8, attention_mask=None, **kwargs):
        self.generate_calls.append(
            {
                "input_ids": input_ids.clone(),
                "max_new_tokens": max_new_tokens,
                "attention_mask": None if attention_mask is None else attention_mask.clone(),
            }
        )
        return torch.full((input_ids.shape[0], max_new_tokens), 5, dtype=torch.long)

    def prefill(self, input_ids):
        raise AssertionError("benchmark path should use runtime.generate()")


def test_helm_backend_run_one_uses_runtime_generate_with_attention_mask():
    backend = HelmBackend("dummy", "float16", kv_offload=False, batch_size=1)
    backend._tokenizer = _FakeTokenizer()
    backend._runtime = _FakeRuntime()

    result = backend._run_one_locked("prompt", output_len=3, input_len=16)

    assert result.status == "success"
    assert result.output_tokens == 3
    assert len(backend._runtime.generate_calls) == 1
    call = backend._runtime.generate_calls[0]
    assert call["max_new_tokens"] == 3
    assert call["input_ids"].tolist() == [[7, 8, 9, 0]]
    assert call["attention_mask"].tolist() == [[1, 1, 1, 0]]


def test_helm_backend_compile_options_propagate_kv_offload_to_decode():
    backend = HelmBackend("dummy", "float16", kv_offload=True, batch_size=1)

    opts = backend._compile_options(
        graph_kind="decode",
        workload={
            "batch_size": 1,
            "prefill_seq_len": 4,
            "decode_context_len": 4,
            "decode_tokens": 8,
            "dtype_size": 2,
        },
    )

    assert opts.graph_kind == "decode"
    assert opts.kv_offload is True


def test_print_stats_includes_kv_cache_counters(capsys):
    stats = BenchStats(
        backend="helm",
        n_requests=1,
        n_success=1,
        ttft_p50=11.0,
        ttft_p95=12.0,
        ttft_p99=13.0,
        ttft_mean=11.5,
        decode_lat_p50=3.0,
        decode_lat_p95=4.0,
        decode_lat_p99=5.0,
        e2e_p50=41.0,
        e2e_p95=42.0,
        e2e_p99=43.0,
        tok_per_s_mean=10.0,
        decode_tok_per_s_mean=20.0,
        peak_gpu_mb_mean=1234.0,
        peak_cpu_mb_mean=5678.0,
        kv_evict_calls=7,
        kv_pages_evicted=11,
        kv_bytes_evicted=12_582_912,
        kv_prefetch_calls=13,
        kv_pages_prefetched=17,
        kv_bytes_prefetched=25_165_824,
    )

    _print_stats(stats, output_len=8)

    out = capsys.readouterr().out
    assert "KV cache" in out
    assert "evict_calls=7" in out
    assert "pages_evicted=11" in out
    assert "bytes_evicted=12582912" in out
    assert "evicted_mb=12.6" in out
    assert "prefetch_calls=13" in out
    assert "pages_prefetched=17" in out
    assert "bytes_prefetched=25165824" in out
    assert "prefetched_mb=25.2" in out


class _SetupFailTokenizer:
    def __init__(self):
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"

    def __call__(self, *args, **kwargs):
        raise RuntimeError("boom during setup")


class _SetupFailModel:
    def eval(self):
        return self


def test_helm_backend_setup_restores_gemma2_patch_on_failure(monkeypatch):
    patch_calls = []
    restore_calls = []

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: _SetupFailTokenizer(),
    )
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        lambda *args, **kwargs: _SetupFailModel(),
    )
    monkeypatch.setattr(
        "experiments.paper_bench._patch_gemma2_decoder_outputs",
        lambda: patch_calls.append("patch"),
    )
    monkeypatch.setattr(
        "experiments.paper_bench._restore_gemma2_decoder_outputs",
        lambda: restore_calls.append("restore"),
    )

    backend = HelmBackend("google/gemma-2-2b", "float16", kv_offload=False, batch_size=1)

    assert backend.setup() is False
    assert patch_calls == ["patch"]
    assert restore_calls == ["restore"]
