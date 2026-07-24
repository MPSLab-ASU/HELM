# HELM: Heterogeneous Execution for Large Models

[![CI](https://github.com/MPSLab-ASU/HELM/actions/workflows/ci.yml/badge.svg)](https://github.com/MPSLab-ASU/HELM/actions/workflows/ci.yml)
[![Artifact package](https://github.com/MPSLab-ASU/HELM/actions/workflows/artifact.yml/badge.svg)](https://github.com/MPSLab-ASU/HELM/actions/workflows/artifact.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21548044.svg)](https://doi.org/10.5281/zenodo.21548044)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**HELM** is a compiler and runtime for running large language models on consumer hardware — machines with a single GPU and CPU RAM. It microbenchmarks your hardware, compiles the model into static per-device FX subgraphs, and executes heterogeneously across CPU and GPU with a paged KV cache that offloads cold pages to CPU RAM during long-context decode.

---

## Quick start

```bash
python3 -m venv helm-env && source helm-env/bin/activate
pip install "git+https://github.com/MPSLab-ASU/HELM.git"
helm --model Qwen/Qwen3-0.6B --mode generate --prompt "Explain what a compiler does." --max-new-tokens 64
```

The first run downloads the model from Hugging Face and prints the partition
HELM chose, e.g. `[helm] partition plan: stage0@cuda(30u); running on: HELM pipeline`,
followed by the generated text.

---

## Installation

### Requirements

| | |
|---|---|
| OS | Linux x86-64 |
| Python | 3.10+ (use 3.11 if you also want vLLM) |
| GPU (optional) | NVIDIA GPU with driver **≥ 580**. The PyTorch wheels bundle CUDA 13.0, so no CUDA toolkit install is needed. Check with `nvidia-smi`. |
| CPU kernel (optional) | GCC ≥ 9 and an AVX2 CPU (Intel Haswell+ / AMD Zen+) for the fast CPU kernel, compiled automatically on first use (`sudo apt install build-essential`). Without it HELM warns and falls back to plain PyTorch. |
| Disk / RAM | Model weights (~2 GB per billion parameters in fp16) must fit in GPU memory + CPU RAM combined. |

### Option 1 — pip (recommended for users)

```bash
python3 -m venv helm-env && source helm-env/bin/activate
pip install "git+https://github.com/MPSLab-ASU/HELM.git"
helm --help
```

### Option 2 — pip + vLLM (fastest for models that fit in VRAM)

When a model fits entirely on the GPU, HELM automatically hands it to
[vLLM](https://github.com/vllm-project/vllm) if vLLM is installed; models
that don't fit always run on HELM's CPU+GPU pipeline. vLLM is optional; use
Python 3.11 for it (on 3.12 its setuptools pin conflicts with HELM's):

```bash
python3.11 -m venv helm-vllm && source helm-vllm/bin/activate
pip install "git+https://github.com/MPSLab-ASU/HELM.git" vllm==0.30.0
helm --model Qwen/Qwen3-4B --mode generate --max-new-tokens 64
# [helm] partition plan: stage0@cuda(38u); running on: vLLM
```

Without vLLM, models that fit run on HELM's native executor and HELM prints
a one-line notice. Use `--backend helm` to always run HELM's own executor.

### Option 3 — from source (development)

Uses the pinned lockfile via [`uv`](https://docs.astral.sh/uv/)
(`curl -LsSf https://astral.sh/uv/install.sh | sh`):

```bash
git clone https://github.com/MPSLab-ASU/HELM.git && cd HELM
uv sync                      # exact pinned environment incl. test tools
uv run pytest tests/ -q      # GPU tests skip automatically without a GPU
uv run helm --help
```

### CPU-only machines

```bash
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu
pip install "git+https://github.com/MPSLab-ASU/HELM.git"
```

### Docker

```bash
git clone https://github.com/MPSLab-ASU/HELM.git && cd HELM
docker build -t helm .                                  # GPU image (CUDA 13.0 wheels)
docker run --rm -it --gpus all \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface helm bash
# CPU-only image: docker build --build-arg TORCH_INDEX=cpu -t helm:cpu .
```

### Troubleshooting

- **`CUDA driver version is insufficient`** — update the NVIDIA driver to ≥ 580, or use the CPU-only install.
- **`AVX2 CPU kernel unavailable` warning / slow CPU layers** — install a C++ compiler (`sudo apt install build-essential`); the kernel needs an AVX2 CPU.
- **401 / gated model** — run `huggingface-cli login` and accept the model's license on Hugging Face.
- **Prompt rejected as too long** — raise `--max-input-tokens` (HELM never truncates silently).

Load only model files you trust; HELM never enables `trust_remote_code`.
The DeepSpeed / lm-eval baselines used in the paper are covered in
[research/README.md](research/README.md).

---

## Usage

### Generate text

```bash
uv run helm --model Qwen/Qwen3-4B --mode generate \
    --prompt "Explain what a compiler does." --max-new-tokens 64
# [helm] partition plan: stage0@cuda(38u); running on: vLLM          (fits in VRAM, vLLM installed)
# [helm] partition plan: stage0@cpu(2u) + stage1@cuda(36u); running on: HELM pipeline   (Qwen3-8B, 16 GB GPU)
```

### Compile + run with diagnostics (auto partition)

`execute_stagewise` prints the selected partition plan (e.g.
`stage0@cpu(9u) + stage1@cuda(33u)`), the generated text and throughput, and
saves the compiler artifacts (FX graphs, IR, plan) under `artifacts/run_NNN/`.

```bash
uv run helm \
    --model Qwen/Qwen3-4B \
    --mode execute_stagewise \
    --compiler-plan auto \
    --max-new-tokens 64 \
    --kv-offload
```

### Manual partition

Specify exactly which layers run on CPU and which on GPU. Ranges are **inclusive**.

```bash
uv run helm \
    --model Qwen/Qwen3-14B \
    --mode execute_stagewise \
    --compiler-plan manual \
    --compiler-cpu-layers 0:7 \
    --compiler-gpu-layers 8:47 \
    --max-new-tokens 128 \
    --kv-offload
```

### Inspect the partition plan without executing

```bash
uv run helm --model Qwen/Qwen3-8B --mode plan --compiler-plan auto --print-plan
```

### Python API

```python
import torch
from helm import HelmInference, HelmInferenceConfig

config = HelmInferenceConfig(
    model_id="Qwen/Qwen3-4B",
    dtype=torch.float16,
    max_input_tokens=256,
    max_new_tokens=64,
    kv_offload=True,        # paged KV cache with CPU offload
    plan_mode="auto",       # or "manual" with cpu_layers="0:7", gpu_layers="8:35"
    # route_to_vllm_when_all_gpu=None (default): use vLLM for models that fit in
    # VRAM when it is installed; True requires vLLM; False always uses HELM.
)
with HelmInference(config) as helm:
    print("vLLM" if helm.routed_to_vllm else "HELM", helm.partition_plan)
    print(helm.generate(["Explain what a compiler does."])[0])
```

Prompts longer than `max_input_tokens` are rejected, never silently truncated.

### CLI reference

| Flag | Default | Description |
|---|---|---|
| `--model` | required | Hugging Face model ID or local path |
| `--mode` | `plan` | `generate` · `execute_stagewise` · `plan` · `lower` · `units` · `import` · `baseline` · `dry_run` |
| `--prompt` | `"Explain what a compiler does."` | Prompt text |
| `--max-input-tokens` | `64` | Maximum prompt length in tokens |
| `--max-new-tokens` | `8` | Tokens to generate |
| `--dtype` | `float16` | `float16` · `bfloat16` · `float32` |
| `--compiler-plan` | `auto` | `auto` (profiled) or `manual` |
| `--compiler-cpu-layers` | — | Layer range for CPU, e.g. `0:7` |
| `--compiler-gpu-layers` | — | Layer range for GPU, e.g. `8:47` |
| `--kv-offload` | off | Paged KV cache with CPU offloading |
| `--backend` | `auto` | `generate` mode: `auto` (vLLM for models that fit in VRAM when installed, else HELM), `router` (require vLLM for models that fit), `helm` (always native HELM) |
| `--cpu-threads` | `6` | PyTorch CPU threads for CPU stages |
| `--print-plan` | off | Print the full partition plan JSON |
| `--save-artifacts-dir` | `artifacts` | Where per-run compiler artifacts are written |

Run `helm --help` for the remaining diagnostic flags.

---

## How It Works

```
Model (nn.Module)
      │
      ▼
 [1] FX Tracing          — Capture a single decode graph (seq_len=1); the same
      │                     compiled graph is reused for both prefill and decode
      ▼
 [2] HelmGraph IR         — Lift FX nodes into typed IR with shape, FLOP, and
      │                     byte-cost metadata (shape propagation + fallback estimator)
      ▼
 [3] HybridAnalyzer       — Annotate each node with param bytes, activation bytes,
      │                     and KV bytes per token; aggregate per transformer block
      ▼
 [4] PartitionUnitBuilder — Group nodes into coarse units: one embedding unit,
      │                     L transformer-block units, one output-projection unit
      ▼
 [5] DeviceProfiler       — Microbenchmark actual GPU/CPU FLOPS (decode GEMV regime
      │                     + prefill GEMM regime), DRAM bandwidth, and PCIe H2D/D2H;
      │                     results cached across compile() calls
      ▼
 [6] StrategySelector     — Score all O(L) feasible CPU→GPU split points under the
      │                     roofline cost model (separate projection + attention
      │                     rooflines; L3 cache hierarchy for short-context KV;
      │                     GPU flash-attention activation traffic = 0);
      │                     return the split minimising the configured objective
      ▼
 [7] StageFXBuilder       — Fragment the FX graph into per-device standalone
      │                     GraphModules at the chosen split boundary
      ▼
 [8] PipelineRuntime      — Prefill pass → autoregressive decode loop (batch ≥ 1);
      │                     both phases reuse the same compiled stage subgraphs
      ▼
 [9] StageRuntimeExecutor — Execute stages in sequence; transfer activation tensor
      │                     across device boundary; pre-position weights one submodule
      │                     at a time to avoid peak-RAM spikes during load
      ▼
[10] KVOffloadManager     — Paged KV cache with GPU watermark; evict oldest pages
                            to CPU pinned RAM; stream back page-by-page during decode
                            via async H2D + online-softmax streaming attention
```

**StrategySelector** evaluates three plan classes: (1) all-GPU, (2) all feasible CPU→GPU 2-stage splits, (3) all-CPU. The exhaustive search runs in O(L) time and takes <1 ms.

**KVOffloadManager** patches attention `forward` at the class level (no FX graph changes needed). Supports `batch_size > 1`: one `KVCacheManager` per batch item sharing a single `KVAllocator` pool. Eviction policy: oldest page (by `start_token`) first, protecting the active tail page.

---

## vs. Existing Tools

**vs. Accelerate** — Accelerate uses `device_map='auto'` which fills GPU from layer 0 upward with no runtime cost model. HELM microbenchmarks your actual hardware (FLOPS at decode/prefill regimes, memory bandwidth, PCIe throughput) and finds the partition that minimises decode latency under a roofline model. It compiles to isolated FX subgraphs rather than attaching per-layer forward hooks, eliminating Python-level synchronisation overhead at every boundary.

**vs. vLLM** — vLLM requires all model weights to fit in GPU VRAM and OOMs on models larger than VRAM capacity. HELM targets exactly this regime: models that cannot fit on a single GPU but need to run on one anyway.

**vs. FlexGen** — FlexGen overlaps weight streaming with compute, which only helps at batch sizes in the hundreds. HELM keeps all weights resident in CPU RAM + GPU VRAM statically; the only cross-device traffic is a single activation tensor at the stage boundary, once per decode step.

**vs. DeepSpeed ZeRO-Inference** — DeepSpeed streams weights from CPU per layer during inference, incurring O(L) PCIe round-trips per generated token. HELM places weights statically and only transfers activations at the stage boundary — no per-layer weight movement during decode.

---

## Results (paper Table II excerpt — RTX 3090, 24 GB VRAM, batch=1, fp16)

Decode throughput (tok/s) / TTFT p50 (s) at output length 128:

| Model | HELM | Accelerate | DeepSpeed | vLLM |
|---|---|---|---|---|
| Qwen3-4B  | **87.1** / 0.019 | 25.7 / 0.041 | 26.7 / 0.040 | 87.0 / 0.018 |
| Qwen3-8B  | **50.3** / 0.028 | 25.8 / 0.042 | 26.5 / 0.040 | 50.3 / 0.027 |
| Qwen3-14B | **8.1** / 4.7    | 1.3 / 0.782  | ✗            | OOM |
| Qwen3-32B | **1.8** / 11.7   | 0.2 / 5.3    | ✗            | OOM |

Key observations (full 3-GPU × 10-model grid in `results/paper/`):
- On **models that fit in VRAM** (Qwen3-4B/8B above), the reported numbers are from HELM's **router path**, which routes the all-GPU plan to vLLM and matches vLLM's throughput (1.0×) — the compile-time partitioning adds no overhead when offloading is unnecessary. This routing is the default whenever vLLM is installed.
- On **models that exceed VRAM** (14B/32B on the 3090; everything above 4B on an 8 GB GPU), which cannot use the router path (vLLM OOMs), HELM's own CPU+GPU pipeline is the fastest feasible backend: up to **10.4×** higher decode throughput than the strongest feasible baseline, **6.2×** geometric mean across the nine overflow configurations.
- HELM's **paged KV offload** extends the maximum context length by a geometric mean of **5.9×** and up to **256×** (Qwen3-14B: 32K vs. Accelerate's 128), with token-identical greedy outputs with and without offload.

Every number above is pinned in machine-readable form under
[`results/paper/`](results/paper) and re-derived in CI by
[`results/validate_results.py`](results/validate_results.py).

---

## Reproducing the paper

Docker-first, one obvious path (no GPU needed for the first step):

```bash
git clone https://github.com/MPSLab-ASU/HELM && cd HELM

# 1. CPU-only functional check: builds the CPU image and runs the CPU test
#    suite + the canonical-results self-check (the image's default command).
#    Expect the GPU-gated tests to skip here (they run on CUDA hosts; see
#    REPRODUCING.md for the per-host pass/skip split).
docker build --build-arg TORCH_INDEX=cpu -t helm:cpu . && docker run --rm helm:cpu

# 2. Native GPU image (CUDA 13.0 wheels). Research baselines are separate.
docker build -t helm .
# See research/README.md for optional paper baselines.

# 3. One-command reproduction (native or in-container): each target runs
#    its experiments AND then renders the corresponding paper-vs-measured
#    comparison (PNG + markdown in <run_dir>/figures/) automatically.
#    Run with no args for the full target list + time estimates:
bash reproduce.sh test        # pytest + results validator (CPU-only, ~5 min)
bash reproduce.sh smoke       # small end-to-end GPU run + correctness (~30 min)
bash reproduce.sh table2      # Table II grid -> renders Table II (hours)
bash reproduce.sh fig4        # also: fig5, fig6, table3 — each renders its figure
bash reproduce.sh all         # everything above in sequence + PASS/FAIL/SKIPPED summary
# Utilities:
bash reproduce.sh figures [run_dir]   # re-render only (no experiments)
bash reproduce.sh validate <run_dir> "<GPU>" <Model>   # compare a run to the paper
```

Docker runs that download models should mount the Hugging Face cache, e.g.:

```bash
docker run --rm --gpus all -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    helm bash reproduce.sh smoke
```

Per-claim commands, expected outputs and tolerances: [`REPRODUCING.md`](REPRODUCING.md).

---

## Paper Experiments

The full experiment suite runs with a single command:

```bash
bash experiments/run_paper_experiments.sh
```

No arguments required. It detects hardware, selects the backends installed in the active environment (HELM and Accelerate always; vLLM and DeepSpeed when the [research environment](research/README.md) is active), and runs **4 models × up to 4 backends** over sections E/A/B/C by default (add `SECTIONS=...,F` for the slow max-decode probe behind Fig. 5, and `NO_LM_EVAL=0` for section D):

| Section | What it measures |
|---|---|
| **E — Feasibility** | Does the model load? Peak GPU/CPU memory at load time |
| **F — Max decode length** | Longest output before OOM |
| **A — Latency sweep** | TTFT, decode tok/s — p50/p95/p99 across 10 requests (5 with `QUICK=1`) |
| **B — Throughput** | Aggregate tok/s at batch sizes 1/2/4/8 |
| **C — Ablations** | batch size, context length, AVX on/off, KV offload on/off, CPU threads |
| **D — LM quality** | MMLU, HellaSwag, ARC-Easy via lm-evaluation-harness |

Models: `Qwen/Qwen3-4B`, `Qwen/Qwen3-8B`, `Qwen/Qwen3-14B`, `Qwen/Qwen3-32B`

**Crash recovery:** each section writes a `.done` sentinel on completion. Re-running skips completed sections.

**Environment overrides:**

```bash
QUICK=1              # 5 requests, shorter sequences — smoke-test on any machine
NO_LM_EVAL=1         # skip Section D
SECTIONS=E,F,A       # run only specific sections
MODELS_OVERRIDE=Qwen/Qwen3-4B      # comma-separated model list
BACKENDS_OVERRIDE=helm,accelerate  # comma-separated backend list
OUT_ROOT=/scratch/results
DELETE_MODEL_CACHE=1 # delete each model's HF cache after its runs (off by default)
```

Results land in `experiments/results/<timestamp>/` with a `SUMMARY.md` containing paper-ready tables.

---

## Artifact Evaluation

This repository is packaged as an ACM artifact for the paper (ESWEEK CASES call for
artifacts, badges *Available* / *Functional-Reusable* / *Results Reproduced*):

- **[`ARTIFACT.md`](ARTIFACT.md)** — artifact appendix: check-list, hardware/
  software requirements, claims↔experiments mapping and badge justification.
- **[`REPRODUCING.md`](REPRODUCING.md)** — command-by-command reproduction of
  every table and figure, with expected outputs, tolerances and time
  estimates (kick-the-tires path: ~30 min).
- **[`results/paper/`](results/paper)** — the paper's numbers in
  machine-readable form; [`results/validate_results.py`](results/validate_results.py)
  self-checks them and compares fresh runs against Table II.
- **CI** (`.github/workflows/`) — CPU-only test suite on every
  push, canonical-results validation, artifact-tarball build
  (`make_package.sh`) and a Docker image build.
- **`Dockerfile`** — pinned CUDA-13.0 / CPU-only container environments;
  `CITATION.cff` / `.zenodo.json` — citation and archival (DOI) metadata.

---

## Citation

If you use HELM in your research, please cite the paper
([IEEE Xplore](https://ieeexplore.ieee.org/document/11673279),
[doi:10.1109/TCAD.2026.3729371](https://doi.org/10.1109/TCAD.2026.3729371)):

```bibtex
@ARTICLE{11673279,
  author={Khedkar, Atharva and Pandey, Shashwat and Shrivastava, Aviral},
  journal={IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems},
  title={HELM: Compiler-Guided Heterogeneous Execution for Large Language Models on Memory-Constrained Systems},
  year={2026},
  pages={1-1},
  doi={10.1109/TCAD.2026.3729371}}
```

The archived research artifact is
[doi:10.5281/zenodo.21548044](https://doi.org/10.5281/zenodo.21548044);
citation metadata is also in [`CITATION.cff`](CITATION.cff).

---

## License

[Apache License 2.0](LICENSE)
