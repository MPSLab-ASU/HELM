/*
 * fp16_gemv.cpp — AVX2 + F16C fp16 GEMV kernel for HELM CPU stages
 *
 * Replaces PyTorch's slow fp16 linear path on CPU (which has no optimised
 * MKL/OpenBLAS backend for fp16).  Uses F16C to convert fp16→fp32 in
 * registers and AVX2 FMA to accumulate, achieving near-memory-bandwidth-
 * limited throughput.
 *
 * Interface (mirrors F.linear):
 *   helm_fp16_linear(input, weight, bias=None) → Tensor
 *
 * Shapes:
 *   input  : [..., K]  fp16 or fp32 (any leading batch dims)
 *   weight : [N, K]    fp16
 *   bias   : [N]       fp16 or fp32, optional
 *   output : [..., N]  same dtype as input
 *
 * Fast path: when the batch dimension collapses to 1 (seq_len=1 decode) the
 * inner loop is a pure GEMV and is memory-bandwidth limited by reading the
 * weight matrix row-by-row from DRAM.  Prefill (batch>1) falls back to
 * torch::linear to avoid writing a full GEMM.
 */

#include <torch/extension.h>
#include <immintrin.h>   // AVX2 + F16C
#include <cstdint>
#include <cstring>

// ---------------------------------------------------------------------------
// Horizontal sum of an __m256 into a single float
// ---------------------------------------------------------------------------
static inline float hsum256(__m256 v) {
    __m128 hi  = _mm256_extractf128_ps(v, 1);
    __m128 lo  = _mm256_castps256_ps128(v);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}

// ---------------------------------------------------------------------------
// Single-row dot product: result = sum_j(W_row[j] * x[j])
// W_row and x are fp16 (uint16_t*), n must be a multiple of 8.
// ---------------------------------------------------------------------------
static inline float dot_fp16_row(const uint16_t* __restrict__ W_row,
                                  const uint16_t* __restrict__ x,
                                  int64_t n) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    int64_t j = 0;
    // Unroll 4×8 = 32 elements per iteration for ILP
    for (; j <= n - 32; j += 32) {
        __m256 w0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(W_row + j)));
        __m256 w1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(W_row + j +  8)));
        __m256 w2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(W_row + j + 16)));
        __m256 w3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(W_row + j + 24)));

        __m256 x0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(x + j)));
        __m256 x1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(x + j +  8)));
        __m256 x2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(x + j + 16)));
        __m256 x3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(x + j + 24)));

        acc0 = _mm256_fmadd_ps(w0, x0, acc0);
        acc1 = _mm256_fmadd_ps(w1, x1, acc1);
        acc2 = _mm256_fmadd_ps(w2, x2, acc2);
        acc3 = _mm256_fmadd_ps(w3, x3, acc3);
    }
    // Tail: 8 elements at a time
    for (; j <= n - 8; j += 8) {
        __m256 w0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(W_row + j)));
        __m256 x0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(x + j)));
        acc0 = _mm256_fmadd_ps(w0, x0, acc0);
    }

    // Scalar tail for n not a multiple of 8 (rare for transformer dims)
    acc0 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    float result = hsum256(acc0);
    for (; j < n; j++) {
        // _cvtsh_ss: single fp16→fp32 scalar (F16C)
        result += _cvtsh_ss(W_row[j]) * _cvtsh_ss(x[j]);
    }
    return result;
}

// ---------------------------------------------------------------------------
// Core GEMV: y[i] = dot(W[i,:], x)   for i in [0, m)
// Parallelised over rows with OpenMP.
// ---------------------------------------------------------------------------
static void gemv_fp16(const uint16_t* __restrict__ W,
                       const uint16_t* __restrict__ x,
                       float*          __restrict__ y,
                       int64_t m, int64_t n) {
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < m; i++) {
        y[i] = dot_fp16_row(W + i * n, x, n);
    }
}

// ---------------------------------------------------------------------------
// PyTorch-facing entry point
// ---------------------------------------------------------------------------
torch::Tensor helm_fp16_linear(
        torch::Tensor input,                        // [..., K] fp16
        torch::Tensor weight,                       // [N, K]   fp16
        c10::optional<torch::Tensor> bias_opt) {    // [N]      fp16/fp32, optional

    TORCH_CHECK(input.device().is_cpu(),  "helm_fp16_linear: input must be on CPU");
    TORCH_CHECK(weight.device().is_cpu(), "helm_fp16_linear: weight must be on CPU");
    TORCH_CHECK(weight.dim() == 2,        "helm_fp16_linear: weight must be 2-D");
    TORCH_CHECK(weight.dtype() == torch::kFloat16,
                "helm_fp16_linear: weight must be fp16");
    TORCH_CHECK(weight.is_contiguous(),   "helm_fp16_linear: weight must be contiguous");

    const int64_t N = weight.size(0);
    const int64_t K = weight.size(1);

    // Reshape input to [B, K]
    auto in_sizes  = input.sizes().vec();
    int64_t B = 1;
    for (int i = 0; i < (int)in_sizes.size() - 1; i++) B *= in_sizes[i];
    int64_t K_in = in_sizes.back();
    TORCH_CHECK(K_in == K, "helm_fp16_linear: input last dim (", K_in,
                ") != weight K (", K, ")");

    // For decode (B=1) use our AVX2 GEMV; otherwise fall back to torch::linear
    // (prefill GEMM performance is less critical than decode)
    if (B != 1 || input.dtype() != torch::kFloat16) {
        return torch::linear(input, weight,
                             bias_opt.has_value() ? bias_opt.value()
                                                  : torch::Tensor());
    }

    auto x_flat = input.reshape({K}).contiguous();
    auto out_fp32 = torch::empty({N}, torch::kFloat32);

    gemv_fp16(
        reinterpret_cast<const uint16_t*>(weight.data_ptr()),
        reinterpret_cast<const uint16_t*>(x_flat.data_ptr()),
        out_fp32.data_ptr<float>(),
        N, K
    );

    // Add bias in fp32 then convert to fp16 to match F.linear output dtype
    if (bias_opt.has_value()) {
        out_fp32 += bias_opt.value().to(torch::kFloat32);
    }

    auto out_fp16 = out_fp32.to(torch::kFloat16);

    // Restore leading batch dimensions: [..., N]
    auto out_sizes = std::vector<int64_t>(in_sizes.begin(), in_sizes.end() - 1);
    out_sizes.push_back(N);
    return out_fp16.reshape(out_sizes);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("helm_fp16_linear", &helm_fp16_linear,
          "AVX2+F16C fp16 GEMV/linear for CPU stages (decode fast path)",
          py::arg("input"), py::arg("weight"), py::arg("bias") = py::none());
}
