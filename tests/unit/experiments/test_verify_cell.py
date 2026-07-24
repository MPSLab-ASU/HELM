import json
import os
import subprocess
import textwrap
from pathlib import Path

REPO = Path(__file__).parents[3]


def _stub_paper_bench(tmp_path: Path, decode_lat_p50_ms: float, ttft_p50_ms: float) -> Path:
    """A stand-in paper_bench.py that emits one valid helm latency entry."""
    stub = tmp_path / "stub_paper_bench.py"
    stub.write_text(textwrap.dedent(f"""
        import json, sys
        from pathlib import Path

        out_dir = Path(sys.argv[sys.argv.index("--output-dir") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "paper_results.json").write_text(json.dumps({{
            "latency_sweep": {{
                "helm": {{
                    "128": {{
                        "n_requests": 3,
                        "n_success": 3,
                        "decode_lat_p50": {decode_lat_p50_ms},
                        "ttft_p50": {ttft_p50_ms},
                    }}
                }}
            }}
        }}))
    """))
    return stub


def _run(tmp_path: Path, stub: Path) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.update(
        {
            "PAPER_BENCH": str(stub),
            "OUT_DIR": str(tmp_path / "out"),
            "NUM_REQUESTS": "3",
            # verify_cell.sh refuses to run unpinned on multi-GPU hosts; the
            # stub never touches a GPU, so pin index 0 to stay hermetic.
            "GPU_ID": "0",
        }
    )
    return subprocess.run(
        ["bash", "experiments/verify_cell.sh", "Qwen/Qwen3-4B", "helm", "RTX 4060", "Qwen3-4B"],
        cwd=REPO,
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
    )


def test_verify_cell_passes_on_paper_matching_numbers(tmp_path: Path):
    # Table II RTX 4060 / Qwen3-4B: 18.8 tok/s (53.19 ms/tok), TTFT 0.340 s.
    stub = _stub_paper_bench(tmp_path, decode_lat_p50_ms=53.19, ttft_p50_ms=340.0)

    result = _run(tmp_path, stub)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "RUN CHECK PASSED" in result.stdout
    assert json.loads((tmp_path / "out" / "paper_results.json").read_text())


def test_verify_cell_fails_beyond_tolerance(tmp_path: Path):
    # 2x slower than the paper cell: outside the default +/-25% tolerance.
    stub = _stub_paper_bench(tmp_path, decode_lat_p50_ms=106.4, ttft_p50_ms=340.0)

    result = _run(tmp_path, stub)

    assert result.returncode != 0
    assert "RUN CHECK FAILED" in result.stdout
