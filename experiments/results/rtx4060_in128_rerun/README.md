# RTX 4060 Laptop (8 GB) — in=128 padded rerun (2026-06-11)

Paper-grade source files for the RTX 4060 rows of Table II and the latency
panels in the paper. Both runs use `experiments/paper_bench.py`
with `pad_to_input_len=true`, input=output=128, fp16, 10 requests, batch 1.

| File | Model | HELM decode tok/s | Accelerate decode tok/s | Ratio |
|------|-------|------------------:|------------------------:|------:|
| `Qwen-Qwen3-4B/helm/A_latency/paper_results.json` | Qwen3-4B | 18.66 (mean) | 5.67 (mean) | 3.3× |
| `Qwen-Qwen3-8B/helm/A_latency/paper_results.json` | Qwen3-8B | 3.51 (mean) | 0.56 (mean) | 6.3× |

Both cells: `n_success=10/10`, `peak_gpu_mb` ≈ 6.6–7.3 GB (sane for an 8 GB
card), TTFT > 0. The files follow the standard run layout produced by
`experiments/run_paper_experiments.sh`
(`<model>/<backend>/A_latency/paper_results.json`), so directory-level
validation works as documented:

```bash
bash reproduce.sh validate experiments/results/rtx4060_in128_rerun/Qwen-Qwen3-4B \
    "RTX 4060" Qwen3-4B
bash reproduce.sh validate experiments/results/rtx4060_in128_rerun/Qwen-Qwen3-8B \
    "RTX 4060" Qwen3-8B
```

Provenance: the JSON contents are byte-identical to the files as produced on
the bench machine; only the filenames were normalized into the standard
layout (`4bfinal.json` → `Qwen-Qwen3-4B/...`, `4060_8b_paper_results.json` →
`Qwen-Qwen3-8B/...`). Each file carries both the `helm` and `accelerate`
latency sweeps in its `latency_sweep` section.

A third file from the same effort, `4bpaper_results.json` (Qwen3-4B,
old schema without `pad_to_input_len`), was discarded as stale and is NOT
included here.
