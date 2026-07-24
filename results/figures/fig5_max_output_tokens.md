## Fig. 5 — maximum supported output length per backend (log2 scale)

Source: results/paper/fig5_max_output_tokens.json (W=weight-load OOM, I=init failure, KV=KV-allocation OOM, F=footprint > CPU+GPU memory, NR=reported only graphically in the paper)

### (a) RTX 4060 (8 GB)

| Model | HELM | Accelerate | DeepSpeed | vLLM |
|---|---|---|---|---|
| Qwen3-4B | 8K | 1K | I | I |
| Qwen3-8B | 4K | 128 | I | I |
| Qwen3-14B | F | F | F | F |
| Qwen3-32B | F | F | F | F |

### (b) RTX 3090 (24 GB)

| Model | HELM | Accelerate | DeepSpeed | vLLM |
|---|---|---|---|---|
| Qwen3-4B | 32K | NR | NR | 8K |
| Qwen3-8B | 32K | NR | NR | 8K |
| Qwen3-14B | 32K | 128 | I | KV |
| Qwen3-32B | 4K | KV | I | W |

### (c) L40S (48 GB)

| Model | HELM | Accelerate | DeepSpeed | vLLM |
|---|---|---|---|---|
| Qwen3-4B | 32K | NR | NR | 32K |
| Qwen3-8B | 32K | NR | NR | 32K |
| Qwen3-14B | 32K | NR | NR | 32K |
| Qwen3-32B | 8K | 1K | I | W |

Headline: geometric-mean context extension 5.9x over feasible Qwen3 pairs, max 256x (Qwen3-14B on RTX 3090: HELM 32K vs Accelerate 128).
