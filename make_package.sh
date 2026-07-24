#!/usr/bin/env bash
# =============================================================================
# make_package.sh - build the HELM artifact tarball.
#
# In a git checkout the tarball is `git archive HEAD` (exactly the tracked
# tree, prefixed HELM/): deterministic, and no local venvs, caches, model
# weights, or untracked sweep outputs can leak in. Outside git it falls back
# to tar with an exclude list that mirrors that intent: it preserves the
# committed reference results under experiments/results/ and results/paper/
# (canonical + measured scientific evidence the validator depends on) while
# excluding credentials, assistant/editor config, model weights, caches, the
# .git dir, and untracked local sweep outputs. This is the tarball attached
# by the "Artifact package" CI workflow and archived on Zenodo (see
# ARTIFACT.md).
#
# Usage:  bash make_package.sh
# Output: ../helm_artifact_<date>.tar.gz  (override with OUT=/path/file.tar.gz)
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIRNAME="$(basename "$ROOT")"
OUT="${OUT:-$ROOT/../helm_artifact_$(date +%Y%m%d).tar.gz}"

echo "[package] building $OUT"

if git -C "$ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    if [[ -n "$(git -C "$ROOT" status --porcelain)" ]]; then
        echo "[package] ERROR: refusing to archive stale HEAD with uncommitted changes." >&2
        echo "Commit the intended release or package a clean exported source directory." >&2
        exit 1
    fi
    git -C "$ROOT" archive --format=tar.gz --prefix=HELM/ -o "$OUT" HEAD
else
    # No git: preserve committed reference results (experiments/results/ and
    # results/paper/ hold the canonical + measured scientific evidence the
    # validator re-derives) but drop credentials, assistant/editor config,
    # model weights, caches, and untracked local sweep byproducts. This mirrors
    # what `git archive HEAD` would ship; only generated OUTPUTS inside
    # experiments/results/ (logs, local-only sweep dirs) are pruned, not the
    # committed reference data.
    tar -czf "$OUT" -C "$ROOT/.." \
      --transform "s,^$DIRNAME/,HELM/," \
      --exclude="$DIRNAME/.git" \
      --exclude="$DIRNAME/.venv" \
      --exclude="$DIRNAME/.venv-research" \
      --exclude="$DIRNAME/venv" \
      --exclude="$DIRNAME/paper" \
      --exclude="$DIRNAME/artifacts" \
      --exclude="$DIRNAME/dump" \
      --exclude="$DIRNAME/logs" \
      --exclude="$DIRNAME/checkpoints" \
      --exclude="$DIRNAME/experiments/llama_cpp_src" \
      --exclude="$DIRNAME/experiments/results/llama_cpp_src" \
      --exclude="$DIRNAME/experiments/results/rtx_3090_"'*' \
      --exclude="$DIRNAME/experiments/results/vllm_rtx3090_SUMMARY.md" \
      --exclude="$DIRNAME/experiments/"'*.log' \
      --exclude="$DIRNAME/experiments/results/"'*.log' \
      --exclude="$DIRNAME/.mypy_cache" \
      --exclude="$DIRNAME/.ruff_cache" \
      --exclude="$DIRNAME/.pytest_cache" \
      --exclude="$DIRNAME/.coverage" \
      --exclude="$DIRNAME/.vscode" \
      --exclude="$DIRNAME/.idea" \
      --exclude="$DIRNAME/.cursor" \
      --exclude="$DIRNAME/.cursorrules" \
      --exclude="$DIRNAME/.claude" \
      --exclude="$DIRNAME/helm.egg-info" \
      --exclude="$DIRNAME/AGENTS.md" \
      --exclude="$DIRNAME/CLAUDE.md" \
      --exclude="$DIRNAME/.env" \
      --exclude="$DIRNAME/.env.*" \
      --exclude='*.env' \
      --exclude='*/__pycache__' \
      --exclude='*.pyc' \
      --exclude='*.gguf' \
      --exclude='*.safetensors' \
      --exclude='*.bin' \
      --exclude='*.tar.gz' \
      "$DIRNAME"
fi

echo "[package] done:"
ls -lh "$OUT"
echo "[package] contents (top level):"
tar -tzf "$OUT" | sed 's,^HELM/,,' | awk -F/ 'NF<=2' | sort -u | head -40
echo "[package] sha256:"
sha256sum "$OUT"
