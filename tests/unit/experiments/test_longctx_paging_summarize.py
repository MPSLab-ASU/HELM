import json
import sys

from experiments import longctx_paging_summarize as summarize


def test_collect_surfaces_all_cache_and_memory_metrics(tmp_path):
    result_dir = tmp_path / "ctx_16384"
    result_dir.mkdir()
    (result_dir / "paper_results.json").write_text(json.dumps({
        "config": {"input_len": 16384},
        "latency_sweep": {
            "helm": {
                "8": {
                    "stage_plan": "stage0@cpu(4u), stage1@cuda(38u)",
                    "input_tokens_p50": 16384,
                    "decode_tok_per_s_mean": 0.25,
                    "ttft_p50": 503079.0,
                    "peak_gpu_mb_mean": 20724.0,
                    "peak_cpu_mb_mean": 7560.0,
                    "kv_evict_calls": 9344,
                    "kv_pages_evicted": 9344,
                    "kv_bytes_evicted": 2449473536,
                    "kv_prefetch_calls": 4905,
                    "kv_pages_prefetched": 650784,
                    "kv_bytes_prefetched": 170599120896,
                }
            }
        },
    }))

    rows = summarize.collect(tmp_path)

    assert rows == [{
        "ctx": 16384,
        "actual_input": 16384.0,
        "tok_s": 0.25,
        "ttft_ms": 503079.0,
        "peak_gpu_mb": 20724.0,
        "peak_cpu_mb": 7560.0,
        "plan": "stage0@cpu(4u), stage1@cuda(38u)",
        "evict_calls": 9344,
        "pages_evicted": 9344,
        "bytes_evicted": 2449473536,
        "prefetch_calls": 4905,
        "pages_prefetched": 650784,
        "bytes_prefetched": 170599120896,
    }]


def test_main_prints_cache_metrics(tmp_path, monkeypatch, capsys):
    result_dir = tmp_path / "ctx_16384"
    result_dir.mkdir()
    (result_dir / "paper_results.json").write_text(json.dumps({
        "config": {"input_len": 16384},
        "latency_sweep": {
            "helm": {
                "8": {
                    "stage_plan": "stage0@cpu(4u), stage1@cuda(38u)",
                    "input_tokens_p50": 16384,
                    "decode_tok_per_s_mean": 0.25,
                    "ttft_p50": 503079.0,
                    "peak_gpu_mb_mean": 20724.0,
                    "peak_cpu_mb_mean": 7560.0,
                    "kv_evict_calls": 9344,
                    "kv_pages_evicted": 9344,
                    "kv_bytes_evicted": 2449473536,
                    "kv_prefetch_calls": 4905,
                    "kv_pages_prefetched": 650784,
                    "kv_bytes_prefetched": 170599120896,
                }
            }
        },
    }))
    monkeypatch.setattr(sys, "argv", ["summarize", str(tmp_path)])

    summarize.main()

    out = capsys.readouterr().out
    assert "GPU MB" in out
    assert "CPU MB" in out
    assert "evict_calls" in out
    assert "prefetch_calls" in out
    assert "MB evicted" in out
    assert "MB streamed" in out
    assert "9344" in out
    assert "4905" in out
