# Optional paper research environment — NOT production-safe

Native HELM is installed from the root `pyproject.toml` and `uv.lock`. The former
`gpu` and `eval` extras are intentionally separated: evaluation pulls in NLTK
(GHSA-8mgp-746c-j5xp) and sqlitedict (GHSA-g4r7-86gm-pgqc) with unresolved upstream
advisories. vLLM 0.30.0 also requires setuptools <81 on Python >=3.12, conflicting
with the native security floor. Isolation does not fix those vulnerabilities.

Do not expose this environment as a service. Use trusted inputs only, no production
credentials, private cache directories, and a separate disposable OS/container
account when processing research data. A Python virtual environment is not a
security sandbox. These dependencies are not included in the production wheel or
installed by the native Docker image. Historical paper measurements are retained;
they are not re-measured claims for the updated dependencies.

From the repository root, explicitly opt in (Python 3.11 avoids the vLLM/setuptools
metadata conflict). These instructions install research software but do not claim
that all backends have been validated together:

```bash
uv venv --python 3.11 .venv-research
source .venv-research/bin/activate
uv pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130
uv pip install -e '.[dev]' -r research/requirements-gpu.txt -r research/requirements-eval.txt
uv pip check
QUICK=1 bash experiments/run_paper_experiments.sh
```

Only install the requirements file(s) needed for your experiment. Do not run
`uv sync` in this environment: the native lock deliberately excludes the research
packages. The paper runner uses the active Python without installing packages.
With vLLM installed here, HELM automatically routes models that fit in VRAM to
it (`--backend auto`, the default); `--backend router` /
`route_to_vllm_when_all_gpu=True` makes vLLM mandatory for such models, and
`--backend helm` / `False` disables routing.

Rollback: discard this environment. The native environment and lock remain intact.
