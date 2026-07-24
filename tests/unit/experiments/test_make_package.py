"""Tests for make_package.sh's non-git fallback path.

In a git checkout the tarball is `git archive HEAD`. Outside git it falls back
to `tar` with an exclude list. A previous version of that fallback blanket-
excluded `experiments/results`, silently dropping the committed reference
results (canonical + measured scientific evidence the validator re-derives)
from the artifact tarball. These tests build a synthetic non-git tree and
assert the fallback now (a) preserves committed reference results and (b) still
excludes credentials, assistant/editor config, model weights, caches, and the
.git dir.
"""

import subprocess
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
MAKE_PACKAGE = REPO / "make_package.sh"


def _build_fake_tree(root: Path) -> None:
    """A tree that is deliberately NOT a git repo to exercise the tar fallback."""
    def w(rel: str, content: str = "x") -> None:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)

    # --- must be PRESERVED: source + committed reference results ---
    w("helm/__init__.py", "# helm package")
    w("results/paper/table2_decode_throughput.csv", "model,helm\nQwen3-4B,87.1\n")
    w("experiments/results/HELM_ROUTER_VS_VLLM_p50.md", "# router\n")
    w("experiments/results/20260316_160248/SUMMARY.md", "# summary\n")
    w("experiments/results/20260316_160248/Qwen-Qwen3-4B/helm/C_ablations/"
      "paper_results.json", '{"latency_sweep": {}}')
    w("README.md", "# HELM\n")

    # --- must be EXCLUDED: credentials, assistant/editor, models, caches, git ---
    w(".git/config", "[core]\n")
    w(".env", "# local configuration must not ship\n")
    w(".env.production", "# deployment configuration must not ship\n")
    w("secrets.env", "# local credentials must not ship\n")
    w("CLAUDE.md", "assistant memory\n")
    w("AGENTS.md", "assistant rules\n")
    w(".cursor/rules", "cursor rules\n")
    w(".cursorrules", "local editor rules\n")
    w(".claude/settings.json", "{}\n")
    w("model.safetensors", "BINARYWEIGHTS")
    w("weights.bin", "BINARYWEIGHTS")
    w("adapter.gguf", "BINARYWEIGHTS")
    w(".venv/pyvenv.cfg", "home = /usr\n")
    w(".pytest_cache/CACHEDIR.TAG", "cache\n")
    w("helm/__pycache__/mod.cpython-311.pyc", "bytecode")
    w("experiments/results/rtx_3090_local_sweep/paper_results.json", "{}")
    w("experiments/results/run.log", "local log line\n")


def _run_package(tree: Path, tmp_path: Path) -> Path:
    # make_package.sh resolves its ROOT from BASH_SOURCE, so copy it into the
    # fake tree and run it there -- that makes ROOT == the (non-git) fake tree
    # and exercises the tar fallback rather than the git-archive path.
    (tree / "make_package.sh").write_text(MAKE_PACKAGE.read_text())
    out = tmp_path / "helm_artifact_test.tar.gz"
    result = subprocess.run(
        ["bash", "make_package.sh"],
        cwd=tree,
        env={"OUT": str(out), "PATH": "/usr/bin:/bin:/usr/local/bin"},
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert out.exists(), result.stdout + result.stderr
    # Confirm we actually took the non-git fallback (no .git in the fake tree).
    assert "git archive" not in result.stdout
    return out


def _members(tarball: Path) -> set[str]:
    with tarfile.open(tarball, "r:gz") as tf:
        # Strip the HELM/ prefix the packager adds.
        return {m.name[len("HELM/"):] for m in tf.getmembers()
                if m.name.startswith("HELM/") and not m.isdir()}


def test_fallback_preserves_committed_reference_results(tmp_path: Path):
    tree = tmp_path / "HELM"
    tree.mkdir()
    _build_fake_tree(tree)
    tarball = _run_package(tree, tmp_path)
    names = _members(tarball)

    # Canonical + measured scientific evidence must ship.
    assert "results/paper/table2_decode_throughput.csv" in names
    assert "experiments/results/HELM_ROUTER_VS_VLLM_p50.md" in names
    assert "experiments/results/20260316_160248/SUMMARY.md" in names
    assert ("experiments/results/20260316_160248/Qwen-Qwen3-4B/helm/"
            "C_ablations/paper_results.json") in names
    # Source ships too.
    assert "helm/__init__.py" in names
    assert "README.md" in names


def test_fallback_excludes_secrets_assistant_and_weights(tmp_path: Path):
    tree = tmp_path / "HELM"
    tree.mkdir()
    _build_fake_tree(tree)
    tarball = _run_package(tree, tmp_path)
    names = _members(tarball)

    forbidden = {
        ".git/config",
        ".env",
        ".env.production",
        "secrets.env",
        "CLAUDE.md",
        "AGENTS.md",
        ".cursor/rules",
        ".cursorrules",
        ".claude/settings.json",
        "model.safetensors",
        "weights.bin",
        "adapter.gguf",
        ".venv/pyvenv.cfg",
        ".pytest_cache/CACHEDIR.TAG",
        "helm/__pycache__/mod.cpython-311.pyc",
        # Untracked local sweep outputs and logs are byproducts, not evidence.
        "experiments/results/rtx_3090_local_sweep/paper_results.json",
        "experiments/results/run.log",
    }
    leaked = forbidden & names
    assert not leaked, f"artifact tarball leaked: {sorted(leaked)}"


def test_git_package_refuses_dirty_tree_without_writing_archive(tmp_path):
    tree = tmp_path / "HELM"
    tree.mkdir()
    (tree / "make_package.sh").write_text(MAKE_PACKAGE.read_text())
    subprocess.run(["git", "init", "-q", str(tree)], check=True)
    out = tmp_path / "release.tar.gz"
    result = subprocess.run(
        ["bash", "make_package.sh"], cwd=tree,
        env={"OUT": str(out), "PATH": "/usr/bin:/bin:/usr/local/bin"},
        text=True, capture_output=True,
    )
    assert result.returncode != 0
    assert "refusing to archive stale HEAD" in result.stderr
    assert not out.exists()
