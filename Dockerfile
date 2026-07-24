# syntax=docker/dockerfile:1
# =============================================================================
# HELM artifact container.  Build with BuildKit (default since Docker 23; on
# older installs: DOCKER_BUILDKIT=1 docker build ...).
# Third-party dependencies are installed at the exact versions pinned in
# uv.lock, so the container matches the native `uv sync` environment.
#
# CPU-only (small image; unit tests, cost model, planner, results validation).
# The default command runs the artifact functional check (pytest tests/ -q
# followed by python results/validate_results.py, via `bash reproduce.sh test`):
#   docker build --build-arg TORCH_INDEX=cpu -t helm:cpu .
#   docker run --rm helm:cpu
#
# GPU (default, CUDA 13.0 wheels - the wheels bundle the CUDA runtime, so the
# host only needs an NVIDIA driver + nvidia-container-toolkit). Mount the HF
# cache so model downloads persist across runs:
#   docker build -t helm .
#   docker run --rm --gpus all \
#       -v $HOME/.cache/huggingface:/root/.cache/huggingface \
#       helm bash reproduce.sh smoke        # kick-the-tires end-to-end run
#   docker run --rm -it --gpus all \
#       -v $HOME/.cache/huggingface:/root/.cache/huggingface helm bash   # shell
#
# Optional baselines/evaluation are isolated in research/; never add them here.
# =============================================================================
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim

# build-essential: the AVX2+F16C GEMV kernel is JIT-compiled via
# torch.utils.cpp_extension on first use; ninja speeds that build up.
# procps: free(1), used by the RAM preflights in the experiment drivers.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ninja-build git procps \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/helm

# Torch pin: keep in sync with pyproject.toml [project.dependencies].
ARG TORCH_INDEX=cu130
RUN pip install --no-cache-dir torch==2.13.0 torchvision==0.28.0 \
    --index-url https://download.pytorch.org/whl/${TORCH_INDEX}

# Install third-party deps at the exact versions pinned in uv.lock (the same
# set `uv sync` installs, dev group included — the default command runs
# pytest). torch/torchvision and the CUDA-runtime wheels they drag in
# (nvidia-*, triton, cuda-*) are filtered out: those come from the
# TORCH_INDEX-specific layer above. Only pyproject.toml/uv.lock edits
# invalidate this (large) layer; source edits re-run just the cheap editable
# install below.
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv \
    && uv export --frozen --no-emit-project --no-hashes \
        --output-file /tmp/requirements-full.txt \
    && grep -Ev "^(torch|torchvision)==|^(nvidia-|triton==|cuda-)" \
        /tmp/requirements-full.txt > /tmp/requirements.txt \
    && test -s /tmp/requirements.txt \
    && grep -q "^transformers==" /tmp/requirements.txt \
    && pip install --no-cache-dir -r /tmp/requirements.txt \
    && pip uninstall -y uv

COPY . .

RUN pip install --no-cache-dir --no-deps -e .

ARG INSTALL_GPU_EXTRAS=0
RUN test "$INSTALL_GPU_EXTRAS" = "0" || (echo "Research extras are not supported in the production image; see research/README.md" >&2; exit 1)

# Default command = the artifact functional check: pytest tests/ -q followed
# by the canonical-results self-check (results/validate_results.py). Override
# with `... helm bash` for an interactive shell, or `... helm bash
# reproduce.sh <stage>` for the paper-reproduction stages (see reproduce.sh).
CMD ["bash", "reproduce.sh", "test"]
