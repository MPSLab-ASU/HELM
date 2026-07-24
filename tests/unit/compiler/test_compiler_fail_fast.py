from types import SimpleNamespace

import pytest
import torch

from helm.compiler import compiler as compiler_module
from helm.compiler.compiler import HelmCompileOptions, _auto_partition_plan, compile_graph
from helm.compiler.partition.partition_units import PartitionUnit


def _unit(unit_id: int = 0) -> PartitionUnit:
    return PartitionUnit(
        unit_id=unit_id,
        unit_type="transformer_block",
        layer_start=unit_id,
        layer_end=unit_id,
        node_ids=[unit_id],
        param_bytes=1024,
        activation_bytes=256,
        flops_prefill=1024,
        flops_decode=128,
    )


def test_compile_graph_raises_exact_error_when_dynamic_analysis_fails(
    monkeypatch,
    tiny_gm_with_leaves,
    tiny_transformer,
) -> None:
    def fail_analysis(*args, **kwargs):
        raise RuntimeError("shape propagation exploded")

    monkeypatch.setattr(compiler_module, "_run_analysis", fail_analysis)

    options = HelmCompileOptions(
        plan_mode="manual",
        cpu_layers="0:1",
        model_name="tiny-model",
        graph_kind="decode",
    )

    with pytest.raises(
        RuntimeError,
        match="HELM dynamic analysis failed.*tiny-model.*shape propagation exploded",
    ):
        compile_graph(
            tiny_gm_with_leaves,
            torch.tensor([[1, 2, 3]], dtype=torch.long),
            tiny_transformer,
            options=options,
        )


def test_compile_graph_static_fallback_requires_explicit_option(
    monkeypatch,
    tiny_gm_with_leaves,
    tiny_transformer,
) -> None:
    def fail_analysis(*args, **kwargs):
        raise RuntimeError("shape propagation exploded")

    monkeypatch.setattr(compiler_module, "_run_analysis", fail_analysis)

    options = HelmCompileOptions(
        plan_mode="manual",
        cpu_layers="0:1",
        model_name="tiny-model",
        graph_kind="decode",
        allow_static_analysis_fallback=True,
    )

    artifact = compile_graph(
        tiny_gm_with_leaves,
        torch.tensor([[1, 2, 3]], dtype=torch.long),
        tiny_transformer,
        options=options,
    )

    analysis = artifact.helmir.metadata["analysis"]
    assert analysis["analysis_mode"] == "static_fallback"
    assert analysis["analysis_error"] == "shape propagation exploded"


def test_auto_partition_plan_rejects_invalid_hf_config(monkeypatch, cpu_profile) -> None:
    monkeypatch.setattr(
        compiler_module,
        "profile_devices",
        lambda **kwargs: ({"cpu": cpu_profile}, {}),
    )

    bad_config = SimpleNamespace(
        hidden_size=0,
        intermediate_size=16,
        num_attention_heads=0,
        num_key_value_heads=0,
    )

    with pytest.raises(
        RuntimeError,
        match="failed to build ModelConfig.*broken-model.*hidden_size",
    ):
        _auto_partition_plan(
            units=[_unit()],
            options=HelmCompileOptions(allow_gpu=False, model_name="broken-model"),
            model_name="broken-model",
            hf_config=bad_config,
        )
