"""Per-compile GPU reserve must not leak across models or compilations."""

import os
from types import SimpleNamespace

import pytest
import torch

from helm.compiler.optimization import device_profiler
from helm.runtime.inference import HelmInferenceConfig, _gpu_memory_reserve_for_model


def test_model_reserve_is_per_model_without_mutating_environment(monkeypatch):
    monkeypatch.delenv("HELM_GPU_MEMORY_RESERVE_MB", raising=False)
    config = HelmInferenceConfig(model_id="local/model")
    small = SimpleNamespace(config=SimpleNamespace(
        vocab_size=16, hidden_size=64, num_hidden_layers=2,
    ))
    large = SimpleNamespace(config=SimpleNamespace(
        vocab_size=300000, hidden_size=4096, num_hidden_layers=40,
    ))

    small_reserve = _gpu_memory_reserve_for_model(small, config)
    large_reserve = _gpu_memory_reserve_for_model(large, config)

    assert small_reserve is not None
    assert large_reserve is not None
    assert large_reserve > small_reserve
    assert "HELM_GPU_MEMORY_RESERVE_MB" not in os.environ

    # Untied models (e.g. Qwen3-8B) must not reserve an embedding clone.
    untied = SimpleNamespace(config=SimpleNamespace(
        vocab_size=300000, hidden_size=4096, num_hidden_layers=40,
        tie_word_embeddings=False,
    ))
    embed_mb = 300000 * 4096 * 2 // (1024 * 1024)
    assert large_reserve - _gpu_memory_reserve_for_model(untied, config) >= embed_mb - 1

    monkeypatch.setenv("HELM_GPU_MEMORY_RESERVE_MB", "2048")
    assert _gpu_memory_reserve_for_model(large, config) is None
    assert os.environ["HELM_GPU_MEMORY_RESERVE_MB"] == "2048"


def test_cached_calibration_rebuilds_capacity_per_compile(monkeypatch):
    monkeypatch.delenv("HELM_GPU_MEMORY_RESERVE_MB", raising=False)
    result = device_profiler._ProfilerResult()
    result.gpu_mem_bw = 1.0
    key = (64, str(torch.float32), 0, 0)
    monkeypatch.setattr(device_profiler, "_cache", {key: result})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties",
        lambda _device: SimpleNamespace(total_memory=16 * 1024**3),
    )

    def capacity(reserve):
        devices, _ = device_profiler.profile_devices(
            64, torch.float32, gpu_memory_reserve_mb=reserve,
        )
        return devices["cuda"].memory_capacity

    assert capacity(1536) == 16 * 1024**3 - 1536 * 1024**2
    assert capacity(4096) == 16 * 1024**3 - 4096 * 1024**2

    monkeypatch.setenv("HELM_GPU_MEMORY_RESERVE_MB", "2048")
    assert capacity(4096) == 16 * 1024**3 - 2048 * 1024**2

    for invalid in ("-1", "not-a-number"):
        monkeypatch.setenv("HELM_GPU_MEMORY_RESERVE_MB", invalid)
        with pytest.raises(ValueError, match="HELM_GPU_MEMORY_RESERVE_MB"):
            capacity(4096)


def test_cache_distinguishes_cpu_only_and_each_gpu(monkeypatch):
    monkeypatch.setattr(device_profiler, "_cache", {})
    device = [0]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: device[0])
    monkeypatch.setattr(
        torch.cuda, "get_device_properties",
        lambda _device: SimpleNamespace(total_memory=16 * 1024**3),
    )
    monkeypatch.setattr(device_profiler, "_bench_cpu", lambda _h, _d, r, **kw: setattr(r, "cpu_mem_bw", 1.0))
    monkeypatch.setattr(device_profiler, "_bench_gpu", lambda _h, _d, r: setattr(r, "gpu_mem_bw", 100.0 + device[0]))
    monkeypatch.setattr(device_profiler, "_bench_pcie", lambda _r: None)

    cpu_devices, _ = device_profiler.profile_devices(64, torch.float32, allow_gpu=False)
    gpu_devices, _ = device_profiler.profile_devices(64, torch.float32, allow_gpu=True)

    assert "cuda" not in cpu_devices
    assert gpu_devices["cuda"].mem_bandwidth == 100.0

    device[0] = 1
    other_gpu_devices, _ = device_profiler.profile_devices(64, torch.float32, allow_gpu=True)
    assert other_gpu_devices["cuda"].mem_bandwidth == 101.0
