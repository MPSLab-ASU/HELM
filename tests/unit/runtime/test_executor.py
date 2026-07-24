from transformers.cache_utils import DynamicCache as HF_DynamicCache
import logging
import pytest
import torch
import torch.fx as fx
import torch.nn as nn

from helm.runtime.executor import DynamicCache, StageRuntimeExecutor
from helm.runtime.stage import Stage


class _LinearModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class _BufferModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("scale", torch.ones(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.scale


def test_executor_passthrough_types_include_hf_dynamic_cache():
    assert issubclass(HF_DynamicCache, StageRuntimeExecutor._PASSTHROUGH_TYPES)
    assert issubclass(DynamicCache, StageRuntimeExecutor._PASSTHROUGH_TYPES)


def test_executor_rejects_missing_call_module_target(monkeypatch):
    monkeypatch.setattr(StageRuntimeExecutor, "_patch_cpu_linears", lambda self: None)

    gm = fx.symbolic_trace(_LinearModel().eval())
    delattr(gm, "linear")
    stage = Stage(stage_id=3, device="cpu", module=gm)

    with pytest.raises(
        RuntimeError,
        match="Stage 3 graph references missing submodule 'linear'",
    ):
        StageRuntimeExecutor([stage])


def test_executor_rejects_missing_get_attr_target(monkeypatch):
    monkeypatch.setattr(StageRuntimeExecutor, "_patch_cpu_linears", lambda self: None)

    gm = fx.symbolic_trace(_BufferModel().eval())
    delattr(gm, "scale")
    stage = Stage(stage_id=4, device="cpu", module=gm)

    with pytest.raises(
        RuntimeError,
        match="Stage 4 graph references missing attribute 'scale'",
    ):
        StageRuntimeExecutor([stage])


def test_patch_cpu_linears_reports_unavailable_native_kernel(monkeypatch, caplog):
    import helm.kernels as kernels

    monkeypatch.setattr(kernels, "is_available", lambda: False)
    monkeypatch.setattr(kernels, "load_error", lambda: "native compile failed")

    executor = StageRuntimeExecutor.__new__(StageRuntimeExecutor)
    executor.stages = []

    with caplog.at_level(logging.WARNING):
        executor._patch_cpu_linears()

    assert "AVX2+F16C CPU kernel unavailable" in caplog.text
    assert "native compile failed" in caplog.text
