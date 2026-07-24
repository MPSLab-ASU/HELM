# Reproducing the paper's results

This guide maps every table and figure of the paper to a concrete command,
its expected output, and a time estimate. Read `ARTIFACT.md` first for the
badge-level overview; see `results/README.md` for the canonical numbers and
the validator.

> **One-command path:** every stage below is wrapped by `reproduce.sh` —
> `bash reproduce.sh <target>` (`test` / `smoke` / `table2` / `table3` /
> `fig4` / `fig5` / `fig6` / `all`) runs the experiments and then
> automatically renders the corresponding paper-vs-measured table/figure
> (PNG + markdown in `<run_root>/figures/`). `bash reproduce.sh` prints the
> target list with time estimates; `figures [run_root]` (re-render only)
> and `validate` remain as utilities.

Throughout: `<GPU>` is one of the paper's Table II names (`RTX 4060`,
`RTX 3090`, `L40S`) and runs are compared with

```bash
python results/validate_results.py --check-run <out>/paper_results.json \
    --gpu "<GPU>" --model <Model> [--tolerance 0.25]
```

> **Hardware note.** Absolute decode throughput on overflow models tracks the
> host CPU/DRAM bandwidth of the machine (paper §VI-E). On hardware other
> than the paper's testbeds, verify the *relative* claims — HELM is the
> fastest feasible backend, the baselines OOM/fail where Table II says they
> do, and the context-extension ratios hold — rather than absolute tok/s.

---

## 0. Setup

```bash
git clone https://github.com/MPSLab-ASU/HELM && cd HELM
uv sync                        # pinned deps (torch 2.13 cu130) from uv.lock
# For baselines/evaluation use research/README.md (separate environment).
# or:  docker build -t helm .  # see Dockerfile header for run flags
```

Model weights come from the Hugging Face Hub on first use
(`huggingface-cli login` needed for the gated `meta-llama/Llama-3.1-8B` and
`google/gemma-2-*`; the Llama-2-13B comparison intentionally uses the ungated
`NousResearch/Llama-2-13b-hf` mirror). Disk: ~10 GB for the kick-the-tires
model, ~350 GB for the full 10-model grid.

## 1. Kick the tires (~30 min)

```bash
# 1a. Test suite - CPU-only, no model downloads, ~5 min:
uv run pytest tests/ -q

# 1b. Canonical-results self-check - stdlib only, seconds:
python results/validate_results.py

# 1c. One end-to-end partitioned run (any CUDA GPU, downloads Qwen3-4B ~8 GB):
uv run helm --model Qwen/Qwen3-4B --mode execute_stagewise \
    --compiler-plan auto --max-new-tokens 64 --kv-offload

# 1d. Correctness vs. HuggingFace baseline (greedy outputs must match):
uv run python experiments/verify_e2e.py --model Qwen/Qwen3-0.6B
```

Expected: 1a passes on a CPU-only x86-64 (AVX2) host — the documented
functional path — with the GPU-gated tests skipped. The
pass/skip split varies with hardware: on a CUDA machine the GPU-gated tests
run instead of skipping, and on non-AVX2/ARM hosts 4 additional AVX2-gated
kernel tests skip. 1c prints the selected partition plan
(`cpu(Xu)+cuda(Yu)`), then generates text with per-token timings; 1d reports
token-identical outputs.

## 1.5 Fast path: verify one Table II cell in minutes

`experiments/verify_cell.sh` runs a single model x backend cell at the
paper's operating point (padded 128-token input, 128 output tokens) with
**n=3 requests** and pipes the result straight into the validator. Three
requests suffice for the +/-25% tolerance because per-token decode variation
is below 1% (paper, Table III note); the full 10-request protocol is only
needed for publication-grade p95/p99 tails.

```bash
bash experiments/verify_cell.sh Qwen/Qwen3-14B helm       "RTX 3090" Qwen3-14B
bash experiments/verify_cell.sh Qwen/Qwen3-14B accelerate "RTX 3090" Qwen3-14B
```

On multi-GPU hosts the script lists the GPUs and asks you to pick one with
`GPU_ID=<index>` (the paper's setup is strictly single-GPU; an unpinned run
lets baselines spread across devices and invalidates the OOM comparison).

After verifying one or more backends, render a visual side-by-side against
the paper's cell (PNG + console ratio table; the HELM-vs-baseline ratio is
the host-independent form of the speedup claim):

```bash
python results/plot_comparison.py --gpu "RTX 3090" --model Qwen3-14B \
    --run helm=<out>/paper_results.json --run accelerate=<out>/paper_results.json
```

Approximate wall-clock per cell (weights already cached; add one-time
download otherwise): HELM on an overflow model ~5-10 min (dominated by model
load), Accelerate ~10-15 min (its decode is the slow part being measured),
in-VRAM cells ~3-5 min. Baseline OOM cells (vLLM/DeepSpeed on overflow
models) need no timing at all — the feasibility probe reaches its verdict in
~1-2 min: `SECTIONS=E BACKENDS_OVERRIDE=vllm,deepspeed MODELS_OVERRIDE=<id>
bash experiments/run_paper_experiments.sh`.

The full-sweep driver also honors `NUM_REQUESTS_OVERRIDE` and
`OUTPUT_LENS_OVERRIDE` for the same purpose across a whole grid.

## 2. Table II — decode throughput / TTFT (paper §VI-A/B) 

One command per machine; sections E (feasibility), A (latency) and F
(max-decode, also feeds Fig. 5) matter for Table II claims:

```bash
# RTX 4060 class (8 GB): the two feasible models, all backends.
# INPUT_LEN=128 PAD_INPUT=1 matches the paper's padded 128-token prompt.
MODELS_OVERRIDE="Qwen/Qwen3-4B,Qwen/Qwen3-8B" SECTIONS=E,F,A INPUT_LEN=128 PAD_INPUT=1 \
    bash experiments/run_paper_experiments.sh

# RTX 3090 / L40S class: full grid (hours; resumable via .done sentinels)
INPUT_LEN=128 PAD_INPUT=1 bash experiments/run_paper_experiments.sh   # defaults: SECTIONS=E,A,B,C
# the architecture-diversity rows use the same driver with other models, e.g.:
MODELS_OVERRIDE="mistralai/Mistral-Nemo-Instruct-2407,allenai/OLMo-2-1124-13B-Instruct" \
    INPUT_LEN=128 PAD_INPUT=1 bash experiments/run_paper_experiments.sh
```

Outputs land in `experiments/results/<timestamp>/<model>/<backend>/A_latency/paper_results.json`.
Validate each cell, e.g.:

```bash
python results/validate_results.py \
    --check-run "experiments/results/<ts>/Qwen-Qwen3-8B/helm/A_latency/paper_results.json" \
    --gpu "RTX 4060" --model Qwen3-8B
```

> **Baselines:** the vLLM and DeepSpeed baselines (and the `helm-router`
> backend) need the separate research environment in `research/README.md`;
> without it the driver skips them with a warning and runs HELM + Accelerate.
>
> **In-VRAM rows:** Table II's HELM column on rows marked 1.0x is the vLLM
> fast path — HELM's router dispatches all-GPU plans to vLLM (paper §IV). To
> reproduce those cells run `BACKENDS_OVERRIDE=helm-router` in the research
> environment; the plain `helm` backend bypasses the router and reads lower
> there by design. The overflow
> rows (the paper's headline claims) use the plain `helm` compiled pipeline.

Expected (paper Table II, tok/s / TTFT-s): HELM is bold-best on every
overflow row — e.g. RTX 4060 Qwen3-4B `18.8 / 0.340` vs Accelerate
`5.7 / 0.201` (3.3x); RTX 3090 Qwen3-14B `8.1 / 4.7` vs `1.3 / 0.782`
(6.1x); L40S Qwen3-32B `4.8 / 6.5` vs `0.5 / 2.2` (10.4x). vLLM/DeepSpeed
print `OOM`/`x` exactly where Table II does. In-VRAM rows dispatch to the
vLLM fast path and match vLLM within noise (1.0x). Time: minutes per
in-VRAM cell; up to ~1 h per 32B overflow cell.

## 3. Fig. 5 — maximum context extension (paper §VI-C)

Section F (off by default — it OOM-probes one output-length candidate at a
time and is slow) produces the max-output-tokens-per-backend grid:

```bash
SECTIONS=F INPUT_LEN=128 PAD_INPUT=1 bash experiments/run_paper_experiments.sh
```

Expected (`results/paper/fig5_max_output_tokens.json`): RTX 4060 — HELM 8K
vs Accelerate 1K (Qwen3-4B), 4K vs 128 (Qwen3-8B); RTX 3090 — HELM 32K on
Qwen3-14B vs Accelerate 128 (the 256x cell), 4K on Qwen3-32B where every
baseline fails; L40S — HELM matches vLLM at 32K for 4B-14B, 8K vs 1K on
32B. Geometric mean over feasible Qwen3 pairs: 5.9x.

## 4. Fig. 4 — TTFT vs. prompt length (Qwen3-32B, RTX 3090)

```bash
bash experiments/ttft_prompt_scaling.sh                  # sweeps 128..4096-token prompts
python experiments/ttft_prompt_scaling_summarize.py <out_dir>
```

Expected: TTFT p50 rises from ~11.7 s (128 tokens) to ~243 s (4096), i.e.
~20.8x over the 32x input range at the auto-selected `cpu(43u)+cuda(23u)`
plan. Time: ~1-2 h (32B prefill on CPU is slow — that is the point).

## 5. Fig. 6 — long-context paging (Mistral-Nemo-12B, RTX 3090)

```bash
# MODEL pins the paper's Fig. 6 config (the script's own default is the
# Llama-2-13B paging setup); `bash reproduce.sh fig6` does this for you.
MODEL=mistralai/Mistral-Nemo-Instruct-2407 bash experiments/longctx_paging.sh
python experiments/longctx_paging_summarize.py <out_dir>
```

Expected: ~13.5 / 9.5 / 5.7 tok/s at 1K/2K/4K with zero eviction, then
paging activates: ~1.46 tok/s at 8K and ~0.37 at 16K with ~8.1 GB evicted
and ~584 GB cumulatively prefetched, peak GPU memory bounded ~23.4 GB; vLLM
fails KV allocation at 16K. Correctness: `verify_e2e.py` greedy outputs are
token-identical with and without offload.

## 6. Table III — llama.cpp comparison (Llama-2-13B, RTX 3090)

```bash
# llama.cpp side: needs llama-cpp-python (CUDA build) and a Llama-2-13B GGUF
# (e.g. huggingface-cli download TheBloke/Llama-2-13B-GGUF llama-2-13b.Q4_K_M.gguf).
# `GGUF=<path> bash reproduce.sh table3` wraps this ngl sweep:
uv run python experiments/llama_cpp_bench.py \
    --local-gguf <path/to/llama-2-13b.Q4_K_M.gguf> --local-tag q4_k_m \
    --n-gpu-layers -1 0 20 30 34 --output-dir experiments/results/table3_llamacpp
# HELM side (auto plan). The driver defaults to offline loading for benchmark
# hygiene - pre-download once, or set HF_HUB_OFFLINE=0 to allow downloads:
HF_HUB_OFFLINE=0 bash experiments/run_llama13b_helm.sh
```

Expected: HELM auto fp16 ~10.6 tok/s beats every manual fp16 `--n-gpu-layers`
setting (best hand-tuned: 9.78 at ngl=34); 4-bit Q4_K_M is faster (74.5
all-GPU) but is a different precision/accuracy target (paper §VII-1).

## 6.5 Rendering the tables and figures

`results/make_figures.py` replicates the paper's tables and figures as PNGs
plus markdown companions (printed to the console too, so no display is
needed), directly comparable with the paper's versions:

```bash
bash reproduce.sh figures                    # paper-reference rendition from
                                             # results/paper/ alone (no GPU,
                                             # needs matplotlib: dev extras)
bash reproduce.sh figures <run_root>         # overlay the measured outputs a
                                             # stage above wrote under <run_root>,
                                             # labeled "paper" vs "measured"
# equivalently, per table/figure:
python results/make_figures.py all|table2|table3|fig4|fig5|fig6 \
    [--run-root <dir>] [--out-dir results/figures]
```

Outputs: Table II heatmap grid with OOM/init failures as hatched cells,
Table III bar comparison, Fig. 4-6 curves/bars. The paper-reference
rendition is committed in `results/figures/` (the target output of a fresh
clone); `reproduce.sh` writes paper-vs-measured overlays to
`<run_root>/figures/` so the committed files stay untouched. `--run-root` understands the exact layouts the
stages produce: `run_paper_experiments.sh` trees for table2/fig5,
`ttft_prompt_scaling.sh` / `longctx_paging.sh` output roots for
fig4/fig6, and `llama_cpp_bench.py` `results.json` (plus any Llama-2-13B
latency run) for table3; measured runs are attributed to their Table II GPU
row / Fig. 5 panel via the run's recorded `gpu_name`.

## 7. Ablations (paper §VII)

```bash
SECTIONS=C bash experiments/run_paper_experiments.sh   # no_avx / no_kv_offload / threads / ctx
```

Expected (committed references: `experiments/results/rtx_3090_helm_{14b,32b}_ablations.json`,
`experiments/results/avx_ablation_result.json`, `kv_off_8b_result.json`):
disabling the AVX2 kernel drops decode throughput 39.6-87.6% on CPU-heavy
partitions and <=0.4% on all-GPU plans; disabling KV offload costs 51-79% in
the all-GPU regime (DynamicCache fallback) and -9.7%..+36% on mixed
partitions; paged vs. contiguous attention differ <= 2.3%.

## 8. Compile-time (paper §VI-D)

```bash
uv run python experiments/compile_breakdown.py --model Qwen/Qwen3-32B
```

Expected: < 60 s total on all evaluated configs; on Qwen3-32B/RTX 3090
~15 s = 13.2 s device profiling + 1.8 s graph analysis + <0.01 s partition
search + 0.73 s warm kernel JIT. Profiles are cached and reused across runs.

## Troubleshooting

- **AVX2 kernel build fails or its 4 tests skip** — install
  `build-essential`/gcc >= 9 *and* make sure `ninja` is on PATH
  (`uv sync` / `pip install -e ".[dev]"` provide it; torch's cpp_extension
  refuses to build without it). The runtime falls back to the (slow)
  PyTorch fp16 path and says so.
- **Gated model 401** — `huggingface-cli login` and accept the Llama-3.1 /
  Gemma licenses, or restrict runs to the ungated models.
- **A baseline OOMs where the paper shows a number** (or vice versa) — check
  GPU VRAM matches the paper's class and that nothing else occupies the GPU
  (`nvidia-smi`); drivers `>= 580` for cu130 wheels.
- **Throughput far from Table II on matching GPU** — pin CPU threads
  (`CPU_THREADS=16`), close background load, and re-run; first request pays
  one-time compile/warm-up cost by design (p50 over 10 requests excludes it).
