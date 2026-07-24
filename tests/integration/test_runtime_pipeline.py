"""
Integration tests for the HELM runtime pipeline:
  Compiled Stage objects → StageRuntimeExecutor → output logits

These tests verify that the full compile + execute path produces correct,
deterministic output that matches the original model's forward pass.
"""
import pytest
import torch
import torch.fx as fx

from helm.compiler.IR.graph import HelmGraph
from helm.compiler.importers.fx_importer import FXImporter
from helm.compiler.partition.partition_units import PartitionUnitBuilder
from helm.compiler.optimization.cost_model import HelmCostModel, ModelConfig, WorkloadSpec
from helm.compiler.optimization.strategy_selector import StrategySelector, StrategySelectorConfig
from helm.compiler.lowering.stage_fx_builder import StageFXBuilder
from helm.runtime.executor import StageRuntimeExecutor


# ──────────────────────────────────────────────────────────────────────────────
# Shared compile helper
# ──────────────────────────────────────────────────────────────────────────────

_TINY_MC = ModelConfig(hidden_size=8, intermediate_size=16,
                       num_attention_heads=2, num_kv_heads=2, head_dim=4)


def _build_executor(gm, cpu_profile):
    helm_graph = HelmGraph(gm.graph)
    FXImporter(gm, helm_graph).run()
    units = PartitionUnitBuilder(helm_graph).build()
    for u in units:
        u.param_bytes = 1 * 1024 ** 2
    workload = WorkloadSpec(1, 4, 8, 4)
    model = HelmCostModel({"cpu": cpu_profile}, {}, model_config=_TINY_MC)
    plan, _ = StrategySelector(model, ["cpu"],
                               StrategySelectorConfig(allow_cpu=True)).select(units, workload)
    stages = StageFXBuilder(gm, helm_graph, units, plan).build()
    return StageRuntimeExecutor(stages)


# ──────────────────────────────────────────────────────────────────────────────
# Basic output correctness
# ──────────────────────────────────────────────────────────────────────────────

def test_executor_produces_logits(tiny_gm_with_leaves, cpu_profile):
    executor = _build_executor(tiny_gm_with_leaves, cpu_profile)
    input_ids = torch.randint(0, 64, (1, 4))
    result = executor.run({"input_ids": input_ids})
    assert "logits" in result
    assert isinstance(result["logits"], torch.Tensor)


def test_executor_logits_shape(tiny_gm_with_leaves, cpu_profile):
    """TinyTransformer: batch=1, seq=4, vocab=64."""
    executor = _build_executor(tiny_gm_with_leaves, cpu_profile)
    input_ids = torch.randint(0, 64, (1, 4))
    result = executor.run({"input_ids": input_ids})
    assert result["logits"].shape == (1, 4, 64)


def test_executor_result_is_tensor(tiny_gm_with_leaves, cpu_profile):
    executor = _build_executor(tiny_gm_with_leaves, cpu_profile)
    result = executor.run({"input_ids": torch.randint(0, 64, (1, 3))})
    assert result["logits"].ndim == 3


def test_executor_no_nans(tiny_gm_with_leaves, cpu_profile):
    executor = _build_executor(tiny_gm_with_leaves, cpu_profile)
    result = executor.run({"input_ids": torch.randint(0, 64, (1, 4))})
    assert not torch.isnan(result["logits"]).any()


def test_executor_no_infs(tiny_gm_with_leaves, cpu_profile):
    executor = _build_executor(tiny_gm_with_leaves, cpu_profile)
    result = executor.run({"input_ids": torch.randint(0, 64, (1, 4))})
    assert not torch.isinf(result["logits"]).any()


# ──────────────────────────────────────────────────────────────────────────────
# Determinism
# ──────────────────────────────────────────────────────────────────────────────

def test_executor_deterministic(tiny_gm_with_leaves, cpu_profile):
    """Same input → same output (no stochasticity in eval mode)."""
    executor = _build_executor(tiny_gm_with_leaves, cpu_profile)
    ids = torch.tensor([[5, 10, 15, 20]])
    r1 = executor.run({"input_ids": ids.clone()})
    r2 = executor.run({"input_ids": ids.clone()})
    assert torch.allclose(r1["logits"], r2["logits"])


def test_executor_different_inputs_different_outputs(tiny_gm_with_leaves, cpu_profile):
    executor = _build_executor(tiny_gm_with_leaves, cpu_profile)
    ids1 = torch.tensor([[1, 2, 3, 4]])
    ids2 = torch.tensor([[50, 51, 52, 53]])
    r1 = executor.run({"input_ids": ids1})
    r2 = executor.run({"input_ids": ids2})
    assert not torch.allclose(r1["logits"], r2["logits"])


# ──────────────────────────────────────────────────────────────────────────────
# Output matches original model
# ──────────────────────────────────────────────────────────────────────────────

def test_executor_matches_reference_model(tiny_gm_with_leaves, tiny_transformer, cpu_profile):
    """
    The HELM-compiled pipeline must produce the same logits as a direct
    model.forward() call.
    """
    executor = _build_executor(tiny_gm_with_leaves, cpu_profile)
    input_ids = torch.tensor([[5, 10, 15, 20]])

    with torch.no_grad():
        helm_logits = executor.run({"input_ids": input_ids})["logits"]
        ref_logits = tiny_transformer(input_ids)

    assert helm_logits.shape == ref_logits.shape
    assert torch.allclose(helm_logits, ref_logits, atol=1e-5), (
        f"Max absolute difference: {(helm_logits - ref_logits).abs().max().item():.2e}"
    )


def test_executor_matches_reference_various_lengths(tiny_gm_with_leaves, tiny_transformer, cpu_profile):
    """Test with different sequence lengths."""
    executor = _build_executor(tiny_gm_with_leaves, cpu_profile)
    for seq_len in [1, 2, 4, 8]:
        ids = torch.randint(0, 64, (1, seq_len))
        with torch.no_grad():
            helm = executor.run({"input_ids": ids})["logits"]
            ref = tiny_transformer(ids)
        assert torch.allclose(helm, ref, atol=1e-5), \
            f"Mismatch at seq_len={seq_len}: max_diff={( helm - ref).abs().max():.2e}"


# ──────────────────────────────────────────────────────────────────────────────
# Input format variants
# ──────────────────────────────────────────────────────────────────────────────

def test_executor_accepts_tuple_input(tiny_gm_with_leaves, cpu_profile):
    executor = _build_executor(tiny_gm_with_leaves, cpu_profile)
    ids = torch.randint(0, 64, (1, 4))
    result = executor.run((ids,))
    assert "logits" in result


def test_executor_accepts_tensor_input(tiny_gm_with_leaves, cpu_profile):
    executor = _build_executor(tiny_gm_with_leaves, cpu_profile)
    ids = torch.randint(0, 64, (1, 4))
    result = executor.run(ids)
    assert "logits" in result


# ──────────────────────────────────────────────────────────────────────────────
# Output device
# ──────────────────────────────────────────────────────────────────────────────

def test_output_on_cpu(tiny_gm_with_leaves, cpu_profile):
    executor = _build_executor(tiny_gm_with_leaves, cpu_profile)
    result = executor.run({"input_ids": torch.randint(0, 64, (1, 4))})
    assert result["logits"].device.type == "cpu"


# ──────────────────────────────────────────────────────────────────────────────
# Batch inference
# ──────────────────────────────────────────────────────────────────────────────

def test_executor_batch_output_shape(tiny_gm_with_leaves, cpu_profile):
    """Batch size B should produce logits of shape (B, seq, vocab)."""
    executor = _build_executor(tiny_gm_with_leaves, cpu_profile)
    for batch_size in [1, 2, 4]:
        ids = torch.randint(0, 64, (batch_size, 4))
        result = executor.run({"input_ids": ids})
        assert result["logits"].shape == (batch_size, 4, 64), \
            f"Wrong shape for batch_size={batch_size}: {result['logits'].shape}"


def test_executor_batch_matches_single(tiny_gm_with_leaves, tiny_transformer, cpu_profile):
    """
    Running the same prompt at batch_size=4 must produce the same logits
    as running it at batch_size=1 (all items are identical, so outputs must match).
    """
    executor = _build_executor(tiny_gm_with_leaves, cpu_profile)
    ids = torch.tensor([[5, 10, 15, 20]])
    ids_batched = ids.expand(4, -1)

    with torch.no_grad():
        single = executor.run({"input_ids": ids})["logits"]       # (1, 4, 64)
        batched = executor.run({"input_ids": ids_batched})["logits"]  # (4, 4, 64)

    for i in range(4):
        assert torch.allclose(batched[i], single[0], atol=1e-5), \
            f"Batch item {i} differs from single: max_diff={( batched[i] - single[0]).abs().max():.2e}"


def test_executor_batch_different_items(tiny_gm_with_leaves, cpu_profile):
    """Different prompts in the same batch must produce different logits."""
    executor = _build_executor(tiny_gm_with_leaves, cpu_profile)
    ids = torch.tensor([[1, 2, 3, 4], [50, 51, 52, 53]])
    result = executor.run({"input_ids": ids})
    assert not torch.allclose(result["logits"][0], result["logits"][1])


# ──────────────────────────────────────────────────────────────────────────────
# Tied-weight detection and breaking
# ──────────────────────────────────────────────────────────────────────────────

def _build_tied_executor(tiny_transformer_tied, cpu_profile):
    """Build a 2-stage executor from the tied-weight model."""
    import torch.fx as fx
    from helm.compiler.IR.graph import HelmGraph
    from helm.compiler.importers.fx_importer import FXImporter
    from helm.compiler.partition.partition_units import PartitionUnitBuilder
    from helm.compiler.optimization.cost_model import HelmCostModel, ModelConfig, WorkloadSpec
    from helm.compiler.optimization.strategy_selector import StrategySelector, StrategySelectorConfig
    from helm.compiler.lowering.stage_fx_builder import StageFXBuilder

    class _LeafTracer(fx.Tracer):
        def is_leaf_module(self, m, module_qualified_name):
            from tests.conftest import TinyLayer
            return isinstance(m, TinyLayer) or super().is_leaf_module(m, module_qualified_name)

    graph = _LeafTracer().trace(tiny_transformer_tied)
    gm = fx.GraphModule(tiny_transformer_tied, graph)

    helm_graph = HelmGraph(gm.graph)
    FXImporter(gm, helm_graph).run()
    units = PartitionUnitBuilder(helm_graph).build()
    for u in units:
        u.param_bytes = 1 * 1024 ** 2
    workload = WorkloadSpec(1, 4, 8, 4)
    model = HelmCostModel({"cpu": cpu_profile}, {}, model_config=_TINY_MC)
    plan, _ = StrategySelector(model, ["cpu"],
                               StrategySelectorConfig(allow_cpu=True)).select(units, workload)
    stages = StageFXBuilder(gm, helm_graph, units, plan).build()
    return StageRuntimeExecutor(stages)


def test_tied_weights_broken_across_stages(tiny_transformer_tied, cpu_profile):
    """
    After executor __init__, lm_head.weight must no longer share storage
    with embed_tokens.weight (cross-stage tied weight was cloned).
    """
    executor = _build_tied_executor(tiny_transformer_tied, cpu_profile)

    # Collect data_ptrs for all parameters across all stages
    all_ptrs: list[list[int]] = []
    for stage in executor.stages:
        ptrs = set()
        for node in stage.module.graph.nodes:
            if node.op == 'call_module':
                try:
                    submod = stage.module.get_submodule(node.target)
                    for p in submod.parameters():
                        ptrs.add(p.data_ptr())
                except AttributeError:
                    pass
        all_ptrs.append(ptrs)

    if len(all_ptrs) >= 2:
        # No data_ptr should appear in two different stages
        for i in range(len(all_ptrs)):
            for j in range(i + 1, len(all_ptrs)):
                overlap = all_ptrs[i] & all_ptrs[j]
                assert len(overlap) == 0, \
                    f"Stages {i} and {j} still share {len(overlap)} parameter(s) after tie-breaking"


def test_tied_model_produces_correct_logits(tiny_transformer_tied, cpu_profile):
    """The tied model must still produce correct logits after weight cloning."""
    executor = _build_tied_executor(tiny_transformer_tied, cpu_profile)
    ids = torch.tensor([[5, 10, 15, 20]])
    with torch.no_grad():
        result = executor.run({"input_ids": ids})
        ref = tiny_transformer_tied(ids)
    assert result["logits"].shape == ref.shape
    assert torch.allclose(result["logits"], ref, atol=1e-5)
