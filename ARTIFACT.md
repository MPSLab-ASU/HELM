# Artifact Appendix

**Paper:** *HELM: Compiler-Guided Heterogeneous Execution for Large Language
Models on Memory-Constrained Systems* (IEEE TCAD 2026, ESWEEK CASES;
[doi:10.1109/TCAD.2026.3729371](https://doi.org/10.1109/TCAD.2026.3729371))

This appendix follows the ACM ["Artifact Review and Badging v1.1"](https://www.acm.org/publications/policies/artifact-review-and-badging-current)
terminology used by the [ESWEEK CASES Call for Artifacts](https://esweek.org/call-for-artifacts-cases/).
It covers all three badges: **Artifacts Available**, **Artifacts Evaluated —
Functional/Reusable**, and **Results Validated — Reproduced**.

---

## A.1 Abstract

HELM is a compiler and runtime for single-node CPU+GPU LLM inference. It
profiles the target hardware, uses a hardware-calibrated roofline cost model
to exhaustively search all feasible contiguous layer-to-device partitions,
compiles the chosen partition into static per-device execution graphs (one
activation transfer per decode step instead of per-layer weight streaming),
executes CPU-resident layers with a JIT-compiled AVX2+F16C GEMV kernel, and
extends context length beyond VRAM with a paged KV-cache manager plus
streaming (online-softmax) paged attention.

This artifact contains the full source code (compiler, runtime, kernels), the
benchmark harness for all baselines (vLLM, Hugging Face Accelerate, DeepSpeed
ZeRO-Inference, llama.cpp), single-command experiment drivers, a
CPU-only test suite, the canonical paper numbers in machine-readable form
(`results/paper/`), and a validator that checks fresh runs against the paper
(`results/validate_results.py`).

## A.2 Artifact check-list (meta-information)

| Item | Description |
|---|---|
| **Algorithm** | Roofline-based exhaustive contiguous-partition search (O(L)); static per-device FX graph compilation; paged KV cache with streaming online-softmax attention; AVX2+F16C GEMV kernel |
| **Program** | Python 3.10+ package `helm` (PyTorch 2.13 / torch.fx); C++ kernel JIT-built via `torch.utils.cpp_extension` |
| **Models** | Qwen3-4B/8B/14B/32B, Llama-2-13B, Llama-3.1-8B, Mistral-Nemo-12B, OLMo-2-13B, Gemma-2-9B/27B (Hugging Face Hub, fp16/bf16) |
| **Data set** | Fixed 128-token instruction prompt (benchmarks); `experiments/data/verify_prompts.json` (correctness) |
| **Run-time environment** | Ubuntu 22.04+, Python >= 3.10, CUDA-13.0 PyTorch wheels (NVIDIA driver >= 580), gcc for the JIT kernel; Docker image provided |
| **Hardware** | Paper testbeds: RTX 4060 Laptop 8 GB + i7 16 GB RAM; RTX 3090 24 GB + EPYC 7H12 125 GB RAM; L40S 48 GB + 60 GB RAM. Minimum for the functional path: any AVX2 x86 CPU (no GPU needed for the test suite); any >= 8 GB CUDA GPU for end-to-end inference |
| **Metrics** | Decode throughput (tok/s, p50 over 10 requests), TTFT (p50), max output tokens before OOM, peak GPU/CPU memory |
| **Output** | `paper_results.json` per run + `STATUS.tsv`/`SUMMARY.md`; validated against `results/paper/` by `results/validate_results.py` |
| **Experiments** | `experiments/run_paper_experiments.sh` (Table II, Fig. 5), `experiments/ttft_prompt_scaling.sh` (Fig. 4), `experiments/longctx_paging.sh` (Fig. 6), `experiments/llama_cpp_bench.py` (Table III), `experiments/verify_e2e.py` (correctness) |
| **Disk space** | ~10 GB (code + Qwen3-4B kick-the-tires); ~350 GB for the full 10-model grid |
| **Time (kick-the-tires)** | ~30 min: unit tests ~5 min (CPU-only) + one small-model end-to-end run |
| **Time (full)** | Several hours per GPU class; see `REPRODUCING.md` per-experiment estimates |
| **Publicly available** | GitHub + Zenodo DOI 10.5281/zenodo.21548044 (see A.3) |
| **License** | Apache-2.0 |

## A.3 Description

### How delivered

- **Git repository (development + issues):** https://github.com/MPSLab-ASU/HELM
- **Archival snapshot (Artifacts Available badge):** the v1.0.0 tarball
  (built by `make_package.sh`) is deposited on Zenodo as
  **DOI [10.5281/zenodo.21548044](https://doi.org/10.5281/zenodo.21548044)**
  (sha256 `bb278606bf9b7a4ecc595b211c7a75a14da6e4e5ed9ff63cff4ec95139be0985`);
  `.zenodo.json` in this repo pins the deposition metadata. *Evaluators
  should prefer the DOI snapshot; the git repo is the living copy.*

### Hardware dependencies

- **Functional evaluation:** any x86-64 machine with AVX2 (the kernel
  self-disables on non-AVX2 hardware); no GPU required — the CPU-only test
  suite, cost model, partition search and results validation run without a GPU.
- **End-to-end inference:** one CUDA GPU (>= 8 GB VRAM) + >= 16 GB host RAM.
- **Reproducing the paper's absolute numbers:** the three testbeds in Section
  V-A (RTX 4060 Laptop 8 GB, RTX 3090 24 GB, L40S 48 GB). On other hardware
  the *relative* claims (HELM fastest feasible backend; baselines OOM where
  reported; context extension) still hold, but absolute tok/s tracks the host
  CPU/DRAM (Section VI-E) — use `results/validate_results.py --check-run
  ... --tolerance 0.25` and compare backend *ratios* on identical hardware.

### Software dependencies

Python >= 3.10; pinned dependency set in `uv.lock` (PyTorch 2.13.0 cu130,
transformers, accelerate). Optional paper baselines and lm-eval have moved to
the separate research environment documented in `research/README.md`. gcc/build-essential for the JIT AVX2 kernel. Hugging Face
Hub access for model weights (Llama-3.1 and Gemma-2 are license-gated; the
Llama-2-13B comparison uses the ungated `NousResearch/Llama-2-13b-hf` mirror).

## A.4 Installation

Three equivalent paths (details in `README.md` and `REPRODUCING.md`):

```bash
# 1. uv (pinned lockfile - recommended on a CUDA machine)
git clone https://github.com/MPSLab-ASU/HELM && cd HELM
uv sync && uv run pytest tests/ -q

# 2. pip, CPU-only (functional evaluation without a GPU)
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
pytest tests/ -q

# 3. Docker
docker build -t helm .                          # CUDA 13.0 wheels
docker build --build-arg TORCH_INDEX=cpu -t helm:cpu .   # CPU-only
docker run --rm helm:cpu pytest tests/ -q
```

## A.5 Experiment workflow

1. **Kick the tires (~30 min, badge: Functional).**
   `pytest tests/ -q` (CPU-only; expected pass/skip split per
   host class in `REPRODUCING.md` §1) ->
   `python results/validate_results.py` (canonical-results self-check) ->
   one small end-to-end run on any CUDA GPU:
   `uv run helm --model Qwen/Qwen3-4B --mode execute_stagewise --compiler-plan auto --max-new-tokens 64 --kv-offload`
   -> correctness: `python experiments/verify_e2e.py --model Qwen/Qwen3-0.6B`.
2. **Main results (badge: Reproduced).** Per-GPU single-command sweeps:
   `bash experiments/run_paper_experiments.sh` (sections E/A/B/C by default ->
   Table II; add `SECTIONS=...,F` for the Fig. 5 max-decode data), then compare with
   `python results/validate_results.py --check-run <run>/paper_results.json --gpu <GPU> --model <model>`.
3. **Figures & ablations.** Fig. 4: `experiments/ttft_prompt_scaling.sh`;
   Fig. 6: `experiments/longctx_paging.sh`; Table III:
   `experiments/llama_cpp_bench.py`; ablation section C of
   `run_paper_experiments.sh` (AVX2 kernel / KV offload / threads).

Step-by-step commands, expected outputs and time estimates per claim:
**`REPRODUCING.md`**.

## A.6 Evaluation and expected results

The paper's four experimental claims map to artifacts as follows:

| # | Claim (paper section) | Experiment | Expected result |
|---|---|---|---|
| C1 | HELM sustains inference wherever weights fit combined CPU+GPU memory; baselines OOM/fail on overflow configs (VI-A, Table II) | `run_paper_experiments.sh` section E per model/backend | Feasibility grid matches `results/paper/table2_decode_throughput.csv` cells (`OOM`/`FAIL_INIT`) |
| C2 | Up to 10.4x higher decode throughput, geomean 6.2x over strongest feasible baseline on 9 overflow configs (VI-B) | sections A (+ helm vs accelerate/deepspeed/vllm) | p50 tok/s within tolerance of Table II; ratios preserved on same host |
| C3 | Max context extension up to 256x, geomean 5.9x (VI-C, Fig. 5) | section F | Max output tokens per backend match `results/paper/fig5_max_output_tokens.json` |
| C4 | Design-choice ablations: AVX2 kernel + KV offload contribute meaningfully; paging is correctness-preserving (VII) | section C + `verify_e2e.py` + `experiments/longctx_paging.sh` | Ablation deltas within `results/paper/*.json` bounds; token-identical output with/without offload |

`python results/validate_results.py` re-derives every headline number from
the committed tables (6.2x geomean, 5.9x/256x context, per-row speedups) and
fails non-zero on any inconsistency; CI runs it on every push.

## A.7 Badge justification

- **Artifacts Available:** the exact evaluated version is archived on Zenodo
  with a DOI (metadata in `.zenodo.json`, citation in `CITATION.cff`);
  the repository is public under Apache-2.0.
- **Artifacts Evaluated — Functional/Reusable:** documented (README,
  CODEBASE_GUIDE, this appendix), complete (compiler + runtime + kernels +
  all benchmark harnesses + baselines), exercisable (single-command drivers,
  CPU-only tests, CI on every push, Docker image), consistent (canonical
  results self-check in CI).
- **Results Validated — Reproduced:** `REPRODUCING.md` maps every table and
  figure to a driver script with expected outputs and tolerances;
  `results/validate_results.py --check-run` mechanically compares fresh runs
  to the paper; committed raw runs under `experiments/results/` demonstrate
  the pipeline end-to-end (e.g. the RTX 4060 rows of Table II).
