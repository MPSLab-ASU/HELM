## Table III — HELM vs llama.cpp (Llama-2-13B, RTX 3090, in=out=128)

Source: results/paper/table3_llamacpp_comparison.csv (paper reference, no measured run)

| Config | Precision | paper tok/s | measured tok/s |
|---|---|---|---|
| llama.cpp ngl=-1 (all GPU) | Q4_K_M | 74.5 | — |
| llama.cpp ngl=0 (all CPU) | Q4_K_M | 7.45 | — |
| llama.cpp ngl=20 | Q4_K_M | 13.1 | — |
| llama.cpp ngl=20 | fp16 | 4.29 | — |
| llama.cpp ngl=30 | fp16 | 7.16 | — |
| llama.cpp ngl=34 (best fp16) | fp16 | 9.78 | — |
| HELM (auto) | fp16 | 10.6 | — |
