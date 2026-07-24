import os
import shutil
import subprocess
import textwrap
from pathlib import Path


def _make_fake_repo(tmp_path: Path, script_src: Path, paper_bench_body: str) -> Path:
    repo = tmp_path / "repo"
    (repo / "experiments").mkdir(parents=True)
    shutil.copy(script_src, repo / "experiments" / "run_llama13b_helm.sh")
    (repo / "experiments" / "paper_bench.py").write_text(textwrap.dedent(paper_bench_body))

    fake_conda = tmp_path / "conda" / "etc" / "profile.d"
    fake_conda.mkdir(parents=True)
    (fake_conda / "conda.sh").write_text(
        "conda() { if [ \"$1\" = activate ]; then return 0; fi; return 0; }\n"
    )

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "nvidia-smi").write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"$*\" == *memory.free* ]]; then echo 24000; exit 0; fi\n"
        "echo '0, NVIDIA GeForce RTX 3090, 24576 MiB, 279 MiB, 23846 MiB'\n"
    )
    (fake_bin / "free").write_text("#!/usr/bin/env bash\necho 'Mem: 30Gi 1Gi 29Gi'\n")
    (fake_bin / "nvidia-smi").chmod(0o755)
    (fake_bin / "free").chmod(0o755)

    hf_hub = tmp_path / "hf" / "hub"
    (hf_hub / "models--NousResearch--Llama-2-13b-hf").mkdir(parents=True)
    (hf_hub / "models--google--gemma-2-9b-it").mkdir(parents=True)

    return repo


def _run_driver(tmp_path: Path, repo: Path, extra_env: dict[str, str]) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{tmp_path / 'bin'}:{env['PATH']}",
            "CONDA_BASE": str(tmp_path / "conda"),
            "HF_HOME": str(tmp_path / "hf"),
            "OUT_ROOT": str(tmp_path / "out"),
            "SECTIONS": "LLAMA13_HELM",
            "QUICK": "1",
            "LLAMA13_MIN_GPU_FREE_MB": "1",
        }
    )
    env.update(extra_env)
    return subprocess.run(
        ["bash", "experiments/run_llama13b_helm.sh"],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )


def test_llama13b_driver_fails_when_any_requested_output_len_has_zero_success(tmp_path: Path):
    script_src = Path(__file__).parents[3] / "experiments" / "run_llama13b_helm.sh"
    repo = _make_fake_repo(
        tmp_path,
        script_src,
        """
        import json
        import sys
        from pathlib import Path

        out_dir = Path(sys.argv[sys.argv.index("--output-dir") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "paper_results.json").write_text(json.dumps({
            "latency_sweep": {
                "helm": {
                    "16": {
                        "n_requests": 1,
                        "n_success": 1,
                        "ttft_p50": 1.0,
                        "decode_tok_per_s_mean": 2.0,
                        "stage_plan": "stage0@cpu(1u), stage1@cuda(1u)",
                    },
                    "32": {
                        "n_requests": 1,
                        "n_success": 0,
                    },
                }
            }
        }))
        """,
    )

    result = _run_driver(tmp_path, repo, {"OUTPUT_LENS": "16 32"})

    assert result.returncode != 0
    status = (tmp_path / "out" / "STATUS.tsv").read_text()
    assert "\tfailed\t" in status
    assert "output=32: 0/1 successful requests" in status


def test_llama13b_driver_succeeds_when_all_requested_output_lens_succeed(tmp_path: Path):
    script_src = Path(__file__).parents[3] / "experiments" / "run_llama13b_helm.sh"
    repo = _make_fake_repo(
        tmp_path,
        script_src,
        """
        import json
        import sys
        from pathlib import Path

        out_dir = Path(sys.argv[sys.argv.index("--output-dir") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "paper_results.json").write_text(json.dumps({
            "latency_sweep": {
                "helm": {
                    "16": {
                        "n_requests": 1,
                        "n_success": 1,
                        "ttft_p50": 1.0,
                        "decode_tok_per_s_mean": 2.0,
                        "stage_plan": "stage0@cpu(1u), stage1@cuda(1u)",
                    },
                }
            }
        }))
        """,
    )

    result = _run_driver(tmp_path, repo, {"OUTPUT_LENS": "16"})

    assert result.returncode == 0
    status = (tmp_path / "out" / "STATUS.tsv").read_text()
    assert "\tsuccess\t0\t" in status
