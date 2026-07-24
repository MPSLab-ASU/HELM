# Independent RTX 3090 reproduction runs (artifact evaluation)

Raw outputs from reproducing Table II cells on an RTX 3090 host **different
from the paper's testbed**, as part of preparing the ACM artifact:

- **This host:** RTX 3090 24 GB (`CUDA_VISIBLE_DEVICES=0`; a 2080 Ti is also
  present but excluded) + AMD Ryzen 9 9900X (12C/24T) + 30 GB RAM,
  Linux 6.17, vLLM 0.17.1, fp16, `INPUT_LEN=128 PAD_INPUT=1`.
- **Paper's 3090 testbed (Section V-A):** RTX 3090 24 GB + AMD EPYC 7H12
  (128C) + 125 GB RAM.

Because decode throughput on overflow models tracks host CPU/DRAM bandwidth
(paper Section VI-E), absolute tok/s on this host is not expected to equal
Table II; feasibility verdicts and cross-backend ratios are the
host-independent claims being checked.

## Contents

| Directory | Table II cell | Result |
|---|---|---|
| `qwen3-14b_vllm_E_feasibility/` | RTX 3090 / Qwen3-14B / vLLM = `OOM` | **Reproduced.** vLLM 0.17.1 fails with `torch.OutOfMemoryError` while allocating weights (`gpu_model_runner.load_model`, "Failed to load model - not enough GPU memory", 22.88 GiB in use of 23.56 GiB) - the paper's weight-load OOM failure mode. See `run.log`; the JSON records `fits_in_memory: false`. |

Additional observed HELM data point from the same host (ablation config,
output=64, n=1, batch=1): auto partition `cpu(9u)+cuda(33u)`, decode p50
132.6 ms/tok = 7.5 tok/s vs. the paper's 8.1 tok/s on the EPYC host (-7%).

Logs are sanitized: local usernames and private IP addresses are replaced
with placeholders; benchmark-relevant content is unmodified.
