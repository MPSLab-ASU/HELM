"""
Numeric-correctness tests for the AVX2+F16C fp16 GEMV kernel
(helm/kernels/fp16_gemv.cpp via helm.kernels.fp16_linear).

The kernel is a core CPU-decode performance claim of the paper; before these
tests only its failure-to-compile warning path was asserted. The extension is
JIT-compiled on first use; on hardware/toolchains where it cannot build,
is_available() is False and the numeric tests skip (the fallback path test
still runs everywhere).
"""
import pytest
import torch
import torch.nn.functional as F

from helm import kernels

needs_kernel = pytest.mark.skipif(
    not kernels.is_available(),
    reason=f"AVX2+F16C extension unavailable: {kernels.load_error()}",
)


def _ref(x: torch.Tensor, w: torch.Tensor, b=None) -> torch.Tensor:
    """fp32 reference of F.linear, cast back to fp16."""
    return F.linear(x.float(), w.float(), b.float() if b is not None else None).half()


@needs_kernel
def test_gemv_decode_path_matches_fp32_linear():
    """b==1 decode: the AVX2 GEMV kernel vs an fp32 reference."""
    torch.manual_seed(0)
    x = torch.randn(1, 1, 256, dtype=torch.float16)
    w = torch.randn(512, 256, dtype=torch.float16)

    out = kernels.fp16_linear(x, w)

    assert out.dtype == torch.float16
    assert out.shape == (1, 1, 512)
    assert torch.allclose(out.float(), _ref(x, w).float(), atol=2e-2, rtol=1e-2)


@needs_kernel
def test_gemv_with_bias_matches_fp32_linear():
    torch.manual_seed(1)
    x = torch.randn(1, 1, 128, dtype=torch.float16)
    w = torch.randn(64, 128, dtype=torch.float16)
    b = torch.randn(64, dtype=torch.float16)

    out = kernels.fp16_linear(x, w, b)

    assert torch.allclose(out.float(), _ref(x, w, b).float(), atol=2e-2, rtol=1e-2)


@needs_kernel
def test_gemv_non_multiple_of_simd_width():
    """Odd K/N exercise the kernel's scalar tail handling."""
    torch.manual_seed(2)
    x = torch.randn(1, 1, 131, dtype=torch.float16)  # 131 % 8 != 0
    w = torch.randn(77, 131, dtype=torch.float16)

    out = kernels.fp16_linear(x, w)

    assert torch.allclose(out.float(), _ref(x, w).float(), atol=2e-2, rtol=1e-2)


def test_prefill_path_matches_fp32_linear():
    """b>1 prefill takes the MKL SGEMM path — runs with or without the ext."""
    torch.manual_seed(3)
    x = torch.randn(1, 6, 64, dtype=torch.float16)
    w = torch.randn(32, 64, dtype=torch.float16)

    out = kernels.fp16_linear(x, w)

    assert out.shape == (1, 6, 32)
    assert torch.allclose(out.float(), _ref(x, w).float(), atol=2e-2, rtol=1e-2)


def test_non_fp16_input_falls_through_to_torch():
    x = torch.randn(1, 1, 16)
    w = torch.randn(8, 16)
    out = kernels.fp16_linear(x, w)
    assert torch.allclose(out, F.linear(x, w), atol=1e-6)


@needs_kernel
def test_patch_cpu_linears_preserves_module_output():
    """patch_cpu_linears swaps nn.Linear for the AVX wrapper; outputs must
    match the unpatched module on the decode-shaped input it accelerates."""
    torch.manual_seed(4)
    m = torch.nn.Sequential(torch.nn.Linear(64, 32), torch.nn.Linear(32, 16)).half()
    x = torch.randn(1, 1, 64, dtype=torch.float16)
    with torch.no_grad():
        before = m(x)

    n = kernels.patch_cpu_linears(m)

    assert n == 2
    with torch.no_grad():
        after = m(x)
    assert torch.allclose(before.float(), after.float(), atol=2e-2, rtol=1e-2)
