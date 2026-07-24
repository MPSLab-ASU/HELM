## Fig. 4 — TTFT vs prompt length (Qwen3-32B, RTX 3090, cpu(43u)+cuda(23u), n=5)

Source: results/paper/fig4_ttft_prompt_scaling.json (canonical anchors: the 128- and 4096-token points behind the paper's 20.8x claim)

| prompt tokens | paper TTFT p50 (s) | |
|---|---|
| 128 | 11.7 |
| 256 | — |
| 512 | — |
| 1024 | — |
| 2048 | — |
| 4096 | 243.1 |

Paper claim: TTFT scales 20.8x over the 32x input range (prefill on a heavy CPU partition grows with prompt length, not flat).
