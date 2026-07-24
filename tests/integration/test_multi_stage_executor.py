"""
Integration tests for MULTI-STAGE execution: a manually forced 2-stage
partition plan run through StageFXBuilder + StageRuntimeExecutor.

The auto-selected plan for the tiny test models always collapses to a single
stage, so without these tests no partition boundary is ever executed: the
cross-stage activation handoff, per-stage submodule resolution, and
cross-stage tied-weight breaking would be regression-invisible. These tests
force the split explicitly and pin the multi-stage output to the reference
model's logits.
"""
import pytest
import torch

from helm.compiler.IR.graph import HelmGraph
from helm.compiler.importers.fx_importer import FXImporter
from helm.compiler.lowering.stage_fx_builder import StageFXBuilder
from helm.compiler.partition.partition_plan import PartitionPlan, StageSpec
from helm.compiler.partition.partition_units import PartitionUnitBuilder
from helm.runtime.executor import StageRuntimeExecutor


def _two_stage_executor(gm):
    """Compile gm into a forced cpu|cpu 2-stage plan (split after unit 1)."""
    helm_graph = HelmGraph(gm.graph)
    FXImporter(gm, helm_graph).run()
    units = PartitionUnitBuilder(helm_graph).build()
    assert len(units) >= 3, "tiny model must yield embed + layers + head units"
    split = 2  # stage 0: embedding + first layer; stage 1: rest + lm_head
    plan = PartitionPlan(stages=[
        StageSpec(stage_id=0, device_id="cpu", units=units[:split]),
        StageSpec(stage_id=1, device_id="cpu", units=units[split:]),
    ])
    stages = StageFXBuilder(gm, helm_graph, units, plan).build()
    assert len(stages) == 2, f"expected a forced 2-stage build, got {len(stages)}"
    return StageRuntimeExecutor(stages), stages


def test_two_stage_plan_matches_reference_model(tiny_gm_with_leaves, tiny_transformer):
    """Cross-stage activation handoff must be logit-exact vs the reference."""
    executor, _ = _two_stage_executor(tiny_gm_with_leaves)
    input_ids = torch.randint(0, 64, (1, 4), generator=torch.Generator().manual_seed(0))

    result = executor.run({"input_ids": input_ids})
    with torch.no_grad():
        expected = tiny_transformer(input_ids)

    assert result["logits"].shape == expected.shape
    assert torch.allclose(result["logits"], expected, atol=1e-5)


def test_two_stage_plan_deterministic(tiny_gm_with_leaves):
    executor, _ = _two_stage_executor(tiny_gm_with_leaves)
    input_ids = torch.randint(0, 64, (1, 5), generator=torch.Generator().manual_seed(1))

    first = executor.run({"input_ids": input_ids})["logits"]
    second = executor.run({"input_ids": input_ids})["logits"]

    assert torch.equal(first, second)


def test_two_stage_batch_matches_reference(tiny_gm_with_leaves, tiny_transformer):
    executor, _ = _two_stage_executor(tiny_gm_with_leaves)
    input_ids = torch.randint(0, 64, (3, 4), generator=torch.Generator().manual_seed(2))

    result = executor.run({"input_ids": input_ids})
    with torch.no_grad():
        expected = tiny_transformer(input_ids)

    assert torch.allclose(result["logits"], expected, atol=1e-5)


def test_tied_weights_cloned_across_stage_boundary(tiny_gm_with_leaves_tied, tiny_transformer_tied):
    """TinyTransformerTied shares lm_head.weight with embed_tokens.weight
    (Qwen3-style). With the embedding in stage 0 and the head in stage 1 the
    executor must break the tie (clone storage) yet stay logit-exact."""
    executor, stages = _two_stage_executor(tiny_gm_with_leaves_tied)

    embed_ptrs = {p.data_ptr() for n, p in stages[0].module.named_parameters()
                  if "embed_tokens" in n}
    head_ptrs = {p.data_ptr() for n, p in stages[1].module.named_parameters()
                 if "lm_head" in n}
    assert embed_ptrs and head_ptrs, "expected embed in stage 0 and lm_head in stage 1"
    assert embed_ptrs.isdisjoint(head_ptrs), (
        "cross-stage tied weights must not share storage after executor init"
    )

    input_ids = torch.randint(0, 64, (1, 4), generator=torch.Generator().manual_seed(3))
    result = executor.run({"input_ids": input_ids})
    with torch.no_grad():
        expected = tiny_transformer_tied(input_ids)
    assert torch.allclose(result["logits"], expected, atol=1e-5)
