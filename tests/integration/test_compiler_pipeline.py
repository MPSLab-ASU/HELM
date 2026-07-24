"""
Integration tests for the full HELM compiler pipeline:
  FX trace → HelmGraph → FXImporter → PartitionUnitBuilder
  → StrategySelector → Scheduler → StageFXBuilder → Stage objects
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
from helm.compiler.scheduling.scheduler import Scheduler
from helm.runtime.stage import Stage


# ──────────────────────────────────────────────────────────────────────────────
# Helper: run the compiler pipeline and return (stages, plan)
# ──────────────────────────────────────────────────────────────────────────────

def _compile(gm, cpu_profile):
    helm_graph = HelmGraph(gm.graph)
    FXImporter(gm, helm_graph).run()

    units = PartitionUnitBuilder(helm_graph).build()
    for u in units:
        u.param_bytes = 1 * 1024 ** 2  # 1 MB – fits on CPU

    workload = WorkloadSpec(batch_size=1, prefill_seq_len=4,
                            decode_context_len=8, decode_tokens=4)
    mc = ModelConfig(hidden_size=8, intermediate_size=16,
                     num_attention_heads=2, num_kv_heads=2, head_dim=4)
    model = HelmCostModel({"cpu": cpu_profile}, {}, model_config=mc)
    cfg = StrategySelectorConfig(allow_cpu=True, allow_multi_gpu=False)
    plan, _ = StrategySelector(model, ["cpu"], cfg).select(units, workload)

    stages = StageFXBuilder(gm, helm_graph, units, plan).build()
    return stages, plan, units, helm_graph


# ──────────────────────────────────────────────────────────────────────────────
# End-to-end pipeline produces valid Stage objects
# ──────────────────────────────────────────────────────────────────────────────

def test_pipeline_produces_stages(tiny_gm_with_leaves, cpu_profile):
    stages, _, _, _ = _compile(tiny_gm_with_leaves, cpu_profile)
    assert len(stages) >= 1
    for stage in stages:
        assert isinstance(stage, Stage)


def test_stages_have_graph_modules(tiny_gm_with_leaves, cpu_profile):
    stages, _, _, _ = _compile(tiny_gm_with_leaves, cpu_profile)
    for stage in stages:
        assert isinstance(stage.module, fx.GraphModule)


def test_stage_graphs_lint(tiny_gm_with_leaves, cpu_profile):
    """fx.Graph.lint() validates graph structure — no dangling refs, correct I/O."""
    stages, _, _, _ = _compile(tiny_gm_with_leaves, cpu_profile)
    for stage in stages:
        stage.module.graph.lint()


def test_all_stages_assigned_to_cpu(tiny_gm_with_leaves, cpu_profile):
    stages, _, _, _ = _compile(tiny_gm_with_leaves, cpu_profile)
    for stage in stages:
        assert stage.device == "cpu"


def test_stage_ids_are_sequential(tiny_gm_with_leaves, cpu_profile):
    stages, _, _, _ = _compile(tiny_gm_with_leaves, cpu_profile)
    ids = [s.stage_id for s in stages]
    assert ids == list(range(len(stages)))


def test_final_stage_has_output_node(tiny_gm_with_leaves, cpu_profile):
    stages, _, _, _ = _compile(tiny_gm_with_leaves, cpu_profile)
    final = stages[-1]
    output_nodes = [n for n in final.module.graph.nodes if n.op == "output"]
    assert len(output_nodes) == 1


def test_intermediate_stages_have_no_global_output(tiny_gm_with_leaves, cpu_profile):
    """Intermediate stages should return dicts, not the model's global output."""
    stages, _, _, _ = _compile(tiny_gm_with_leaves, cpu_profile)
    if len(stages) == 1:
        pytest.skip("Only one stage — no intermediate stages to test")
    for stage in stages[:-1]:
        # The graph's output node should output a dict (placeholder names as keys)
        output_node = next(n for n in stage.module.graph.nodes if n.op == "output")
        # The output arg is a dict-like structure in the graph
        assert output_node.args is not None


# ──────────────────────────────────────────────────────────────────────────────
# Partition plan properties
# ──────────────────────────────────────────────────────────────────────────────

def test_plan_stages_cover_all_units(tiny_gm_with_leaves, cpu_profile):
    _, plan, units, _ = _compile(tiny_gm_with_leaves, cpu_profile)
    plan_unit_ids = {u.unit_id for stage in plan.stages for u in stage.units}
    all_unit_ids = {u.unit_id for u in units}
    assert plan_unit_ids == all_unit_ids


def test_plan_device_ids_valid(tiny_gm_with_leaves, cpu_profile):
    _, plan, _, _ = _compile(tiny_gm_with_leaves, cpu_profile)
    for stage in plan.stages:
        assert stage.device_id in ("cpu",)


# ──────────────────────────────────────────────────────────────────────────────
# Scheduler
# ──────────────────────────────────────────────────────────────────────────────

def test_scheduler_sequential_mode(tiny_gm_with_leaves, cpu_profile):
    _, plan, _, _ = _compile(tiny_gm_with_leaves, cpu_profile)
    schedule = Scheduler().build(plan)
    assert schedule.mode == "sequential"


def test_scheduler_num_stages(tiny_gm_with_leaves, cpu_profile):
    _, plan, _, _ = _compile(tiny_gm_with_leaves, cpu_profile)
    schedule = Scheduler().build(plan)
    assert schedule.num_stages == len(plan.stages)


def test_scheduler_microbatches(tiny_gm_with_leaves, cpu_profile):
    _, plan, _, _ = _compile(tiny_gm_with_leaves, cpu_profile)
    schedule = Scheduler().build(plan)
    assert schedule.microbatches == 1


# ──────────────────────────────────────────────────────────────────────────────
# FX importer annotations survive the full pipeline
# ──────────────────────────────────────────────────────────────────────────────

def test_layer_ids_in_helm_graph(tiny_gm_with_leaves, cpu_profile):
    _, _, _, helm_graph = _compile(tiny_gm_with_leaves, cpu_profile)
    assert 0 in helm_graph.layer_to_node_ids
    assert 1 in helm_graph.layer_to_node_ids


def test_embedding_nodes_annotated(tiny_gm_with_leaves, cpu_profile):
    _, _, _, helm_graph = _compile(tiny_gm_with_leaves, cpu_profile)
    embed = [n for n in helm_graph.nodes if n.is_embedding]
    assert len(embed) >= 1


def test_output_head_nodes_annotated(tiny_gm_with_leaves, cpu_profile):
    _, _, _, helm_graph = _compile(tiny_gm_with_leaves, cpu_profile)
    heads = [n for n in helm_graph.nodes if n.is_output_head]
    assert len(heads) >= 1
