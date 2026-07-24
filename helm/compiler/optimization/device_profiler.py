"""
helm/compiler/optimization/device_profiler.py
==============================================
Runtime microbenchmarks that replace all hardcoded constants in
_estimate_device_profiles().

Three benchmarks per device:
  1. stream_bandwidth  – large copy, measures peak memory bandwidth (GB/s)
  2. matmul_flops_decode  – (1, H)×(H, H) matmul, decode regime (seq=1)
  3. matmul_flops_prefill – (S, H)×(H, H) matmul, prefill regime (seq=128)

Plus two cross-device benchmarks:
  4. pcie_h2d  – CPU pinned → GPU, gives CPU→GPU PCIe bandwidth
  5. pcie_d2h  – GPU → CPU pinned, gives GPU→CPU PCIe bandwidth

Results are module-level cached so repeated compile() calls within the same
process pay the cost only once.
"""

from __future__ import annotations

import os
import time
import logging
from typing import Dict, Optional, Tuple

import torch

from helm.compiler.optimization.cost_model import DeviceProfile, LinkProfile

logger = logging.getLogger(__name__)

# ── tuneable constants ────────────────────────────────────────────────────────

_STREAM_BYTES      = 256 * 1024 * 1024  # 256 MB: sized to exceed GPU cache so the copy measures DRAM,
                                        # not cache. Verified on the RTX 3090: bandwidth plateaus from
                                        # 128 MB up (838/842/842 GB/s at 128/256/512 MB); 64 and 256 MB
                                        # agree within 1.5% (830 vs 842 GB/s). See
                                        # experiments/results/ac1_gpu_bw_probe_*.json. Larger-cache GPUs
                                        # were not re-profiled.
_PCIE_BYTES        = 64 * 1024 * 1024   # 64 MB — same for PCIe
_PREFILL_SEQ       = 128                 # seq-len for prefill FLOPS calibration
_WARMUP_ITERS      = 2
_TIMED_ITERS       = 8                   # default timed iterations
_BUDGET_S_PER_OP   = 0.4                 # max wall-clock seconds to spend per benchmark op
                                          # (adaptive: fewer iters when a single call is slow)

# Module-level result cache: key = (hidden_size, dtype_str, n_cpu_units,
# selected CUDA device or None). CPU-only measurements must not shadow GPU
# calibration, and measurements from one GPU must not be reused for another.
_cache: Dict[Tuple[int, str, int, Optional[int]], "_ProfilerResult"] = {}


# ── result container ─────────────────────────────────────────────────────────

class _ProfilerResult:
    """Raw numbers from one calibration run."""

    def __init__(self):
        self.cpu_mem_bw:        float = 0.0   # bytes / s
        self.cpu_flops_decode:  float = 0.0   # effective FLOPS/s at seq=1
        self.cpu_flops_prefill: float = 0.0   # effective FLOPS/s at seq=128
        self.gpu_mem_bw:        float = 0.0
        self.gpu_flops_decode:  float = 0.0
        self.gpu_flops_prefill: float = 0.0
        self.pcie_h2d_bw:       float = 0.0   # bytes / s
        self.pcie_d2h_bw:       float = 0.0


# ── timing helpers ────────────────────────────────────────────────────────────

def _time_cpu(fn, warmup: int = _WARMUP_ITERS, iters: int = _TIMED_ITERS) -> float:
    """Return seconds per call for a CPU function.

    Adaptive: one probe call after warmup determines how many timed iterations
    fit within _BUDGET_S_PER_OP.  This keeps calibration fast even when a
    single call is expensive (e.g. large CPU matmul at seq=128).
    """
    for _ in range(warmup):
        fn()
    # Probe one call to estimate per-call time.
    t_probe = time.perf_counter()
    fn()
    probe_s = time.perf_counter() - t_probe
    # Adaptive iteration count: fit inside budget, capped at iters.
    n = max(1, min(iters, int(_BUDGET_S_PER_OP / max(probe_s, 1e-9))))
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


def _time_gpu(fn, warmup: int = _WARMUP_ITERS, iters: int = _TIMED_ITERS) -> float:
    """Return seconds per call for a CUDA function (synchronised timing).

    Adaptive: one synchronised probe call sets the iteration budget.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    # Probe one call.
    t_probe = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    probe_s = time.perf_counter() - t_probe
    n = max(1, min(iters, int(_BUDGET_S_PER_OP / max(probe_s, 1e-9))))
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


# ── per-device benchmarks ────────────────────────────────────────────────────

def _bench_cpu(hidden_size: int, dtype: torch.dtype, result: _ProfilerResult,
               n_cpu_units: int = 0) -> None:
    """Fill CPU fields of result in-place.

    Memory bandwidth is measured using multiple separately-allocated fp16 buffers
    sized to match actual weight tensors (hidden_size × hidden_size).  This captures
    TLB pressure and prefetch misses that arise when reading many non-contiguous
    parameter blocks during inference — the dominant effect for large CPU stages.

    n_cpu_units: number of transformer blocks expected on the CPU stage.  Used to
    set the number of probe buffers; more blocks → more allocations → more TLB
    pressure.  When 0, falls back to a small fixed count.
    """

    # Choose number of probe buffers: 4 weight-like tensors per CPU unit,
    # capped at 256 to keep profiling fast.  Minimum 8 so even all-GPU runs
    # see some allocation overhead (avoids over-optimistic estimates).
    n_bufs = max(8, min(n_cpu_units * 4, 256))

    # Total profiling memory cap: 512 MB regardless of model size.
    # TLB pressure comes from the NUMBER of separate allocations, not total size;
    # each buffer only needs to be large enough to span many pages (≥1 MB).
    MAX_TOTAL_BYTES = 512 * 1024 * 1024  # 512 MB hard cap
    bytes_per_buf = max(
        1 * 1024 * 1024,                          # minimum 1 MB per buffer
        min(hidden_size * hidden_size * 2,         # ideal: one weight-matrix
            MAX_TOTAL_BYTES // n_bufs),            # but capped by total budget
    )
    elem_per_buf = bytes_per_buf // 2  # fp16

    # Allocate separately — NOT as views of one tensor — to replicate the
    # non-contiguous virtual-address layout of real model weights.
    srcs = [torch.randn(elem_per_buf, dtype=torch.float16) for _ in range(n_bufs)]
    dst  = torch.empty(elem_per_buf, dtype=torch.float16)
    # Each dst.copy_(s) reads bytes_per_buf AND writes bytes_per_buf.
    # Total traffic = n_bufs reads + n_bufs writes = 2 * n_bufs * bytes_per_buf.
    # This matches the GPU benchmark which uses 2 * _STREAM_BYTES / elapsed.
    total_bytes = 2 * n_bufs * bytes_per_buf

    def _read_all():
        for s in srcs:
            dst.copy_(s)

    elapsed = _time_cpu(_read_all)
    result.cpu_mem_bw = total_bytes / elapsed
    logger.debug("CPU mem_bw = %.1f GB/s  (n_bufs=%d, buf_size=%.1f MB)",
                 result.cpu_mem_bw / 1e9, n_bufs, bytes_per_buf / 1e6)

    # 2. Decode FLOPS — HELM's AVX2+F16C kernel makes CPU decode memory-BW-limited.
    #    A GEMV (seq=1) has arithmetic intensity ≈ 1 FLOP/byte (2 FLOPs per 2-byte fp16
    #    weight element), so effective FLOPS/s ≈ mem_bw × 1.  Using the raw PyTorch
    #    fp16 matmul here gives a wildly pessimistic 0.1–0.2 GFLOPS (PyTorch has no
    #    AVX2+F16C GEMV path), which would make the roofline model 100–200× too slow.
    #    Setting cpu_flops_decode = cpu_mem_bw is analytically equivalent to saying
    #    "this device is always memory-BW-limited at decode" which is correct for
    #    HELM's actual execution path.
    result.cpu_flops_decode = result.cpu_mem_bw  # 1 FLOP/byte ≡ BW-limited GEMV
    logger.debug("CPU flops_decode (AVX BW-equiv) = %.3f TFLOPS", result.cpu_flops_decode / 1e12)

    # 3. Prefill FLOPS — seq=128, GEMM in fp32.
    #    On x86 CPUs, PyTorch fp16 GEMM is extremely slow (0.2 GFLOPS) because
    #    x86 has no native fp16 BLAS.  During actual model forward passes, fp16
    #    weights are upcast to fp32 for computation (PyTorch CPU dispatch), so
    #    benchmarking fp32 GEMM better represents real prefill throughput.
    W     = torch.randn(hidden_size, hidden_size, dtype=torch.float32)
    x_pre = torch.randn(_PREFILL_SEQ, hidden_size, dtype=torch.float32)
    elapsed = _time_cpu(lambda: torch.matmul(x_pre, W))
    flops_pre = 2 * _PREFILL_SEQ * hidden_size * hidden_size
    result.cpu_flops_prefill = flops_pre / elapsed
    logger.debug("CPU flops_prefill (fp32) = %.3f TFLOPS", result.cpu_flops_prefill / 1e12)


def _bench_gpu(hidden_size: int, dtype: torch.dtype, result: _ProfilerResult) -> None:
    """Fill GPU fields of result in-place. No-op if CUDA unavailable."""
    if not torch.cuda.is_available():
        return

    # 1. Memory bandwidth — stream copy of a large GPU buffer.
    n_elems = _STREAM_BYTES // 2  # float16 (2 bytes)
    src_gpu = torch.randn(n_elems, dtype=torch.float16, device="cuda")
    dst_gpu = torch.empty_like(src_gpu)
    elapsed = _time_gpu(lambda: dst_gpu.copy_(src_gpu))
    result.gpu_mem_bw = 2 * _STREAM_BYTES / elapsed
    logger.debug("GPU mem_bw = %.1f GB/s", result.gpu_mem_bw / 1e9)

    # 2. Decode FLOPS — seq=1.
    x_dec = torch.randn(1, hidden_size, dtype=dtype, device="cuda")
    W_gpu = torch.randn(hidden_size, hidden_size, dtype=dtype, device="cuda")
    elapsed = _time_gpu(lambda: torch.matmul(x_dec, W_gpu))
    flops_dec = 2 * 1 * hidden_size * hidden_size
    result.gpu_flops_decode = flops_dec / elapsed
    logger.debug("GPU flops_decode = %.3f TFLOPS", result.gpu_flops_decode / 1e12)

    # 3. Prefill FLOPS — seq=128.
    x_pre = torch.randn(_PREFILL_SEQ, hidden_size, dtype=dtype, device="cuda")
    elapsed = _time_gpu(lambda: torch.matmul(x_pre, W_gpu))
    flops_pre = 2 * _PREFILL_SEQ * hidden_size * hidden_size
    result.gpu_flops_prefill = flops_pre / elapsed
    logger.debug("GPU flops_prefill = %.3f TFLOPS", result.gpu_flops_prefill / 1e12)


def _bench_pcie(result: _ProfilerResult) -> None:
    """Measure PCIe bandwidth in both directions using pinned memory."""
    if not torch.cuda.is_available():
        result.pcie_h2d_bw = 0.0
        result.pcie_d2h_bw = 0.0
        return

    n_elems = _PCIE_BYTES // 2  # float16

    # Host-to-device (H2D) using pinned (page-locked) host memory.
    # Pinned memory is required for DMA — non-pinned forces a staging copy
    # that roughly halves the measured throughput.
    src_pin = torch.randn(n_elems, dtype=torch.float16).pin_memory()
    dst_gpu = torch.empty(n_elems, dtype=torch.float16, device="cuda")
    elapsed = _time_gpu(lambda: dst_gpu.copy_(src_pin, non_blocking=False))
    result.pcie_h2d_bw = _PCIE_BYTES / elapsed
    logger.debug("PCIe H2D = %.1f GB/s", result.pcie_h2d_bw / 1e9)

    # Device-to-host (D2H).
    src_gpu2  = torch.randn(n_elems, dtype=torch.float16, device="cuda")
    dst_pin   = torch.empty(n_elems, dtype=torch.float16).pin_memory()
    elapsed   = _time_gpu(lambda: dst_pin.copy_(src_gpu2, non_blocking=False))
    result.pcie_d2h_bw = _PCIE_BYTES / elapsed
    logger.debug("PCIe D2H = %.1f GB/s", result.pcie_d2h_bw / 1e9)


# ── public API ────────────────────────────────────────────────────────────────

def profile_devices(
    hidden_size: int,
    dtype: torch.dtype,
    allow_gpu: bool = True,
    force: bool = False,
    n_cpu_units: int = 0,
    gpu_memory_reserve_mb: Optional[int] = None,
) -> Tuple[Dict[str, DeviceProfile], Dict[Tuple[str, str], LinkProfile]]:
    """
    Run microbenchmarks (once per process unless *force* is True) and return
    measured DeviceProfile / LinkProfile objects for use by the cost model.

    Parameters
    ----------
    hidden_size : int
        The model's hidden dimension.  Matmul benchmarks run at this size so
        the effective-FLOPS estimate matches the model's actual operand shapes.
    dtype : torch.dtype
        The dtype used for matmul benchmarks (should match model weights).
    allow_gpu : bool
        If False, GPU benchmarks are skipped and no GPU device is returned.
    force : bool
        Re-run benchmarks even if a cached result exists.
    gpu_memory_reserve_mb : int, optional
        Per-compile GPU headroom. An explicit HELM_GPU_MEMORY_RESERVE_MB
        environment setting takes precedence.

    Returns
    -------
    devices : dict[str, DeviceProfile]
    links   : dict[tuple[str,str], LinkProfile]
    """
    gpu_device = torch.cuda.current_device() if allow_gpu and torch.cuda.is_available() else None
    cache_key = (hidden_size, str(dtype), n_cpu_units, gpu_device)
    if not force and cache_key in _cache:
        result = _cache[cache_key]
        logger.debug("device_profiler: using cached calibration for %s", cache_key)
    else:
        result = _ProfilerResult()
        t_start = time.perf_counter()

        logger.info("device_profiler: calibrating CPU (hidden=%d, dtype=%s, n_cpu_units=%d) …",
                    hidden_size, dtype, n_cpu_units)
        _bench_cpu(hidden_size, dtype, result, n_cpu_units=n_cpu_units)

        if gpu_device is not None:
            logger.info("device_profiler: calibrating GPU …")
            _bench_gpu(hidden_size, dtype, result)
            logger.info("device_profiler: measuring PCIe bandwidth …")
            _bench_pcie(result)

        elapsed = time.perf_counter() - t_start
        logger.info(
            "device_profiler: calibration done in %.2f s  "
            "(CPU bw=%.1f GB/s, decode=%.3f TFLOPS | "
            "GPU bw=%.1f GB/s, decode=%.3f TFLOPS | "
            "PCIe H2D=%.1f GB/s)",
            elapsed,
            result.cpu_mem_bw / 1e9,
            result.cpu_flops_decode / 1e12,
            result.gpu_mem_bw / 1e9,
            result.gpu_flops_decode / 1e12,
            result.pcie_h2d_bw / 1e9,
        )
        _cache[cache_key] = result

    return _build_profiles(result, allow_gpu, gpu_memory_reserve_mb)


def _build_profiles(
    r: _ProfilerResult,
    allow_gpu: bool,
    gpu_memory_reserve_mb: Optional[int] = None,
) -> Tuple[Dict[str, DeviceProfile], Dict[Tuple[str, str], LinkProfile]]:
    """Convert raw benchmark numbers into DeviceProfile / LinkProfile objects."""
    import psutil

    devices: Dict[str, DeviceProfile] = {}
    links:   Dict[Tuple[str, str], LinkProfile] = {}

    # ── CPU ──────────────────────────────────────────────────────────────────
    # Estimate L3 size from cpuinfo if available; fall back to 8 MB.
    default_l3_bytes = 8 * 1024 * 1024
    try:
        import cpuinfo
        l3_bytes = int(cpuinfo.get_cpu_info().get("l3_cache_size", default_l3_bytes))
    except Exception as exc:
        l3_bytes = default_l3_bytes
        logger.warning(
            "device_profiler: failed to read CPU L3 cache size via cpuinfo "
            "(%s); using conservative default %d bytes",
            exc,
            default_l3_bytes,
        )
    # L3 bandwidth is ~4× DRAM BW on modern x86 (Intel/AMD).
    # Using a conservative 3× factor to avoid over-optimism.
    l3_bw = r.cpu_mem_bw * 3.0

    devices["cpu"] = DeviceProfile(
        device_id="cpu",
        device_type="cpu",
        peak_flops_prefill=r.cpu_flops_prefill,
        peak_flops_decode=r.cpu_flops_decode,
        mem_bandwidth=r.cpu_mem_bw,
        memory_capacity=int(psutil.virtual_memory().total),
        l3_size_bytes=l3_bytes,
        l3_bandwidth=l3_bw,
    )

    # ── GPU ──────────────────────────────────────────────────────────────────
    if allow_gpu and torch.cuda.is_available() and r.gpu_mem_bw > 0:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        # Reserve 512 MiB by default, a model-specific per-compile amount
        # when supplied, or a manual environment override. Raw calibration is
        # cached, but capacity is rebuilt for each compile with its own reserve.
        reserve_bytes = 512 * 1024 * 1024
        if gpu_memory_reserve_mb is not None:
            if isinstance(gpu_memory_reserve_mb, bool) or not isinstance(gpu_memory_reserve_mb, int) or gpu_memory_reserve_mb < 0:
                raise ValueError("gpu_memory_reserve_mb must be a non-negative integer")
            reserve_bytes = gpu_memory_reserve_mb * 1024 * 1024
        env_override = os.environ.get("HELM_GPU_MEMORY_RESERVE_MB")
        if env_override:
            try:
                reserve_mb = int(env_override)
                if reserve_mb < 0:
                    raise ValueError
            except ValueError:
                raise ValueError("HELM_GPU_MEMORY_RESERVE_MB must be a non-negative integer") from None
            reserve_bytes = reserve_mb * 1024 * 1024
        devices["cuda"] = DeviceProfile(
            device_id="cuda",
            device_type="cuda",
            peak_flops_prefill=r.gpu_flops_prefill,
            peak_flops_decode=r.gpu_flops_decode,
            mem_bandwidth=r.gpu_mem_bw,
            memory_capacity=max(0, int(props.total_memory) - reserve_bytes),
        )

    # ── PCIe links ───────────────────────────────────────────────────────────
    if allow_gpu and torch.cuda.is_available() and r.pcie_h2d_bw > 0:
        links[("cpu", "cuda")] = LinkProfile(
            src="cpu", dst="cuda",
            bandwidth_bytes_per_s=r.pcie_h2d_bw,
            latency_s=10e-6,   # PCIe latency is ~5-15 µs; 10 µs is a safe estimate
        )
        links[("cuda", "cpu")] = LinkProfile(
            src="cuda", dst="cpu",
            bandwidth_bytes_per_s=r.pcie_d2h_bw,
            latency_s=10e-6,
        )

    return devices, links


def clear_cache() -> None:
    """Invalidate the module-level profiling cache (useful in tests)."""
    _cache.clear()
