import json
from pathlib import Path

from experiments import llama_cpp_bench


def test_main_can_benchmark_local_gguf_without_hf_mapping(tmp_path, monkeypatch):
    local_model = tmp_path / "local-model.gguf"
    local_model.write_bytes(b"not a real gguf for this unit test")
    output_dir = tmp_path / "out"

    calls = []

    def fake_run_one(
        model_path: Path,
        n_gpu_layers: int,
        n_threads: int,
        input_len: int,
        output_len: int,
        num_requests: int,
        num_warmup: int = 1,
        n_ctx: int | None = None,
        n_batch: int = 512,
    ) -> llama_cpp_bench.BenchResult:
        calls.append(
            {
                "model_path": model_path,
                "n_gpu_layers": n_gpu_layers,
                "n_threads": n_threads,
                "input_len": input_len,
                "output_len": output_len,
                "num_requests": num_requests,
                "num_warmup": num_warmup,
                "n_ctx": n_ctx,
                "n_batch": n_batch,
            }
        )
        return llama_cpp_bench.BenchResult(
            model=str(model_path),
            variant="",
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            input_len=input_len,
            output_len=output_len,
            num_requests=num_requests,
            ttft_ms=[12.0],
            decode_tok_per_s=[4.0],
        )

    monkeypatch.setattr(llama_cpp_bench, "run_one", fake_run_one)
    monkeypatch.setattr(
        "sys.argv",
        [
            "llama_cpp_bench.py",
            "--local-gguf",
            str(local_model),
            "--local-tag",
            "fp16-local",
            "--output-dir",
            str(output_dir),
            "--num-requests",
            "1",
            "--input-len",
            "8",
            "--output-len",
            "1",
            "--n-gpu-layers",
            "0",
            "--n-threads",
            "2",
            "--num-warmup",
            "0",
            "--n-ctx",
            "128",
            "--n-batch",
            "64",
        ],
    )

    assert llama_cpp_bench.main() == 0
    assert calls == [
        {
            "model_path": local_model,
            "n_gpu_layers": 0,
            "n_threads": 2,
            "input_len": 8,
            "output_len": 1,
            "num_requests": 1,
            "num_warmup": 0,
            "n_ctx": 128,
            "n_batch": 64,
        }
    ]
    records = json.loads((output_dir / "results.json").read_text())
    assert records["local-model::fp16-local::ngl_0"]["variant"] == "fp16-local"
