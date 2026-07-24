## Fig. 6 — decode throughput vs context length, KV paging active (Mistral-Nemo-12B, RTX 3090, cpu(1u)+cuda(41u))

Source: results/paper/fig6_paging_throughput.json

| context (tokens) | paper tok/s |
|---|---|
| 1024 | 13.53 |
| 2048 | 9.52 |
| 4096 | 5.69 |
| 8192 | 1.46 |
| 16384 | 0.37 |

Paging onset: past 4K context (KV GPU-resident with zero eviction up to 4K); at 16K the run evicts 8.1 GB and cumulatively prefetches 584 GB of KV pages while peak GPU memory stays bounded (22.3 -> 23.4 GB); vLLM fails KV allocation at 16K.
