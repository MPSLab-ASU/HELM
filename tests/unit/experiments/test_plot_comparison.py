import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parents[3]

matplotlib = pytest.importorskip("matplotlib")


def test_plot_comparison_renders_png_and_ratios(tmp_path: Path):
    out_png = tmp_path / "cmp.png"
    result = subprocess.run(
        [
            sys.executable, "results/plot_comparison.py",
            "--gpu", "RTX 4060", "--model", "Qwen3-4B",
            "--run", "helm=experiments/results/rtx4060_in128_rerun/Qwen-Qwen3-4B/helm/A_latency/paper_results.json",
            "--run", "accelerate=experiments/results/rtx4060_in128_rerun/Qwen-Qwen3-4B/helm/A_latency/paper_results.json",
            "--out", str(out_png),
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    # The committed RTX 4060 raw run reproduces Table II, so the HELM-vs-
    # Accelerate ratio must match the paper's 3.3x on both sides.
    assert "measured 3.3x on this host (paper: 3.3x)" in result.stdout
    assert out_png.stat().st_size > 10_000


def test_plot_comparison_rejects_unknown_backend(tmp_path: Path):
    result = subprocess.run(
        [
            sys.executable, "results/plot_comparison.py",
            "--gpu", "RTX 4060", "--model", "Qwen3-4B",
            "--run", "nonsense=whatever.json",
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert "unknown backend" in result.stderr
