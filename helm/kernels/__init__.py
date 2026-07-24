"""
helm.kernels — native CPU acceleration for HELM CPU stages.

Exports
-------
fp16_linear(input, weight, bias=None) -> Tensor
    AVX2+F16C fp16 GEMV replacement for nn.Linear on CPU.
    Fast path activates when batch×seq_len == 1 (decode step).
    Falls back to torch.nn.functional.linear otherwise (prefill).

patch_cpu_linears(module) -> None
    Walk an nn.Module and replace every nn.Linear with a wrapper that
    calls fp16_linear when running on CPU with fp16 weights.
"""

from __future__ import annotations

import logging
import os
import pathlib

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Build / load the C++ extension (JIT-compiled, cached in torch's build dir)
# ---------------------------------------------------------------------------
_ext = None
_load_error: str | None = None
_warned_unavailable = False


def _load_ext():
    global _ext, _load_error
    if _ext is not None or _load_error is not None:
        return

    src = pathlib.Path(__file__).parent / "fp16_gemv.cpp"
    try:
        from torch.utils.cpp_extension import load
        _ext = load(
            name="helm_fp16_gemv",
            sources=[str(src)],
            extra_cflags=["-O3", "-march=native", "-ffast-math", "-fopenmp"],
            extra_ldflags=["-fopenmp"],
            verbose=False,
        )
    except Exception as e:
        _load_error = str(e)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fp16_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Drop-in replacement for F.linear when weight is fp16 on CPU.

    Two fast paths:
    - decode (batch×seq == 1): AVX2+F16C GEMV kernel — reads weight matrix
      once row-by-row at near-peak memory bandwidth.
    - prefill (batch×seq > 1): cast to float32 and use MKL SGEMM via
      torch.linear — float32 GEMM is compute-bound and orders of magnitude
      faster than PyTorch's unoptimised fp16 GEMM path on CPU.
    """
    _load_ext()

    if (weight.dtype == torch.float16
            and weight.device.type == "cpu"
            and input.dtype == torch.float16):
        # Count tokens (all leading dims × seq)
        b = 1
        for s in input.shape[:-1]:
            b *= s

        if b == 1 and _ext is not None and not os.environ.get("HELM_DISABLE_AVX"):
            # Decode: AVX2+F16C GEMV
            return _ext.helm_fp16_linear(input, weight, bias)
        else:
            # Prefill: float32 SGEMM via MKL, then cast back
            bias_f32 = bias.float() if bias is not None else None
            return F.linear(input.float(), weight.float(), bias_f32).half()

    return F.linear(input, weight, bias)


def is_available() -> bool:
    """Return True if the AVX2+F16C extension compiled successfully."""
    _load_ext()
    return _ext is not None


def load_error() -> str | None:
    """Return the compilation error string, or None if extension is OK."""
    _load_ext()
    return _load_error


# ---------------------------------------------------------------------------
# Module patching
# ---------------------------------------------------------------------------

class _AVXLinear(nn.Module):
    """
    Thin wrapper around nn.Linear that routes the forward call through
    helm_fp16_linear when the weight is fp16 on CPU (decode fast path).
    Falls back to F.linear for all other cases (prefill, GPU, fp32).
    """

    def __init__(self, linear: nn.Linear):
        super().__init__()
        # Keep the original weight/bias as-is (no copy)
        self.weight = linear.weight
        self.bias   = linear.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (self.weight.dtype == torch.float16
                and self.weight.device.type == "cpu"
                and x.dtype == torch.float16):
            return fp16_linear(x, self.weight, self.bias)
        return F.linear(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        return (f"in_features={self.weight.shape[1]}, "
                f"out_features={self.weight.shape[0]}, "
                f"bias={self.bias is not None}, avx=True")


def patch_cpu_linears(module: nn.Module, verbose: bool = False) -> int:
    """
    Walk *module* recursively and replace every nn.Linear whose weight is
    fp16 on CPU with _AVXLinear.  Returns the number of layers patched.

    Safe to call on a full stage GraphModule — only leaf nn.Linear modules
    are replaced; the graph structure is not changed.
    """
    global _warned_unavailable
    _load_ext()
    if _ext is None:
        if not _warned_unavailable:
            # Visible once per process: CPU stages run much slower without it.
            logger.warning(
                "[HELM kernels] AVX2 CPU kernel unavailable; CPU layers fall back to "
                "plain PyTorch (slower). Needs an AVX2 CPU, a C++ compiler (GCC >= 9) "
                "and ninja. Cause: %s", _load_error)
            _warned_unavailable = True
        elif verbose:
            logger.info("[HELM kernels] AVX extension unavailable (%s); skipping patch", _load_error)
        return 0

    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            if (child.weight.dtype == torch.float16
                    and child.weight.device.type == "cpu"):
                setattr(module, name, _AVXLinear(child))
                count += 1
        else:
            count += patch_cpu_linears(child, verbose=verbose)
    return count
