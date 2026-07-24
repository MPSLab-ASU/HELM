import copy

import pytest
import torch

from helm.compiler.importers.fx_importer import FXImporter
from helm.compiler.IR.graph import HelmGraph
from helm.compiler.lowering.stage_fx_builder import StageFXBuilder
from helm.compiler.partition.partition_plan import PartitionPlan, StageSpec
from helm.compiler.partition.partition_units import PartitionUnitBuilder


def test_stage_fx_builder_uses_partition_plan_stage_ids(tiny_gm_with_leaves):
    helm_graph = HelmGraph(tiny_gm_with_leaves.graph)
    FXImporter(tiny_gm_with_leaves, helm_graph).run()
    units = PartitionUnitBuilder(helm_graph).build()

    plan = PartitionPlan(stages=[
        StageSpec(stage_id=2, device_id="cpu", units=units[:-1]),
        StageSpec(stage_id=5, device_id="cpu", units=units[-1:]),
    ])

    stages = StageFXBuilder(tiny_gm_with_leaves, helm_graph, units, plan).build()

    assert [stage.stage_id for stage in stages] == [2, 5]
    final_stage = stages[-1]
    assert any(node.op == "output" for node in final_stage.module.graph.nodes)


def test_stage_fx_builder_rejects_unassigned_fx_nodes(tiny_gm_with_leaves):
    gm = copy.deepcopy(tiny_gm_with_leaves)
    gm.register_buffer("_unused_buffer", torch.ones(1))
    output_node = next(node for node in gm.graph.nodes if node.op == "output")
    with gm.graph.inserting_before(output_node):
        gm.graph.get_attr("_unused_buffer")
    gm.recompile()

    helm_graph = HelmGraph(gm.graph)
    FXImporter(gm, helm_graph).run()
    units = PartitionUnitBuilder(helm_graph).build()
    plan = PartitionPlan(stages=[StageSpec(stage_id=7, device_id="cpu", units=units)])

    with pytest.raises(RuntimeError, match="could not infer stage.*_unused_buffer"):
        StageFXBuilder(gm, helm_graph, units, plan).build()
