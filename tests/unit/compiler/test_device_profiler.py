import logging
import sys
from types import SimpleNamespace

from helm.compiler.optimization.device_profiler import _ProfilerResult, _build_profiles


def test_build_profiles_logs_l3_cache_default_when_cpuinfo_fails(monkeypatch, caplog):
    def fail_cpuinfo():
        raise RuntimeError("cpuinfo broken")

    monkeypatch.setitem(
        sys.modules,
        "cpuinfo",
        SimpleNamespace(get_cpu_info=fail_cpuinfo),
    )
    result = _ProfilerResult()
    result.cpu_mem_bw = 50e9
    result.cpu_flops_decode = 50e9
    result.cpu_flops_prefill = 100e9

    with caplog.at_level(logging.WARNING):
        devices, _ = _build_profiles(result, allow_gpu=False)

    assert devices["cpu"].l3_size_bytes == 8 * 1024 * 1024
    assert "failed to read CPU L3 cache size" in caplog.text
    assert "cpuinfo broken" in caplog.text
