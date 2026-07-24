from helm.compiler.compiler import _manual_partition_plan
from helm.compiler.partition.partition_units import PartitionUnit


def _unit(unit_id, unit_type, layer_id=None):
    return PartitionUnit(
        unit_id=unit_id,
        unit_type=unit_type,
        layer_start=layer_id,
        layer_end=layer_id,
        node_ids=[unit_id],
        param_bytes=1024,
    )


def test_manual_cpu_layers_keep_unassigned_output_on_cpu():
    units = [
        _unit(0, "embedding"),
        _unit(1, "transformer_block", 0),
        _unit(2, "transformer_block", 1),
        _unit(3, "output"),
    ]

    plan, plan_data = _manual_partition_plan(
        units=units,
        cpu_layers="0:1",
        gpu_layers=None,
        model_name="tiny",
    )

    assert set(plan_data["assignments"].values()) == {"cpu"}
    assert [stage.device_id for stage in plan.stages] == ["cpu"]
