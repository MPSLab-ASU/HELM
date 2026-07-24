# helm-router vs vLLM — same-box L40S validation

Date: 2026-05-27
Host: L40S (NVIDIA L40S 46 GB, Intel Xeon Platinum 8462Y+, 70 GB RAM)

## Question

Does the HelmInference router (commit `68fa577`, `helm/runtime/inference.py`)
match vLLM throughput when the partition planner picks all-GPU?

## Method

- N=10 requests per cell, output_len=128, dtype=float16, input_len=128
- paper_bench's `helm-router` backend uses `HelmInference` which routes to
  vLLM when `_is_all_gpu(plan)` is true (default), else `_HelmHandle`
  (HELM's PipelineRuntime).
- Same paper_bench invocation flags for both (`--skip-feasibility
  --skip-max-decode --throughput-concurrency 0`).
- Same-box: both backends measured on the same L40S host to control for
  hardware/state variance.
- Sanity-gated (`peak_gpu_mb_mean ≥ 1 GB`, `ttft_p50 > 0`,
  `n_success == n_requests`, no `error` key). All cells reported below
  pass the gate.

## Results (paper-consistent p50 decode tok/s = 1000 / decode_lat_p50_ms)

| Model | vLLM | helm-router | gap (hr vs vllm) | verdict |
|---|---|---|---|---|
| Qwen3-4B | 83.6 | 83.6 | +0.0% | **PASS** |
| Qwen3-8B | 46.7 | 46.7 | −0.0% | **PASS** |
| Qwen3-14B | 26.0 | 26.0 | −0.0% | **PASS** |
| Llama-2-13B | 28.5 | 28.5 | −0.0% | **PASS** |
| Mistral-Nemo-Instruct-2407 | 31.2 | 31.2 | +0.0% | **PASS** |
| OLMo-2-13B | 27.7 | 27.7 | +0.0% | **PASS** |
| **Qwen3-32B** | **FAILED (OOM)** | **2.2** | — | **HELM-only** |

Pass criterion: `|gap| ≤ 5%`. **6/6 fitting models PASS at ±0.0%.**

## What the router did

- Fitting models (Qwen3 ≤14B, Llama-2-13B, Mistral-Nemo, OLMo-2-13B):
  partition plan = single `stage0@cuda(Nu)`; `HelmInference._is_all_gpu`
  returns True; router delegates to `_VLLMHandle`. Throughput is vLLM's.
- Qwen3-32B (47.4 GB fp16 weights > 46 GB usable VRAM): partition plan =
  `stage0@cpu(25u), stage1@cuda(41u)`; router keeps the loaded HF model
  and uses `_HelmHandle` (HELM PipelineRuntime). vLLM OOMs in
  init_inference and DeepSpeed similarly fails. helm-router 2.2 tok/s is
  the only valid 32B number on this box.

## Reproducing

In the research environment (research/README.md), run one cell per backend
with the same flags, e.g.:

```bash
python experiments/paper_bench.py --model Qwen/Qwen3-8B --backends vllm helm-router \
    --dtype float16 --input-len 128 --output-lens 128 --num-requests 10 \
    --no-lm-eval --throughput-concurrency 0 --skip-feasibility --skip-max-decode \
    --output-dir experiments/results/router_check
```
