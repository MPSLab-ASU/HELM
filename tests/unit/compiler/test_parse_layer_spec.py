"""Tests for parse_layer_spec, including the multi-range (interleaved) form.

Multi-range support is what lets us express non-contiguous CPU placements
(e.g. cpu(0:5,20:25)) to study how expanding the partition design space beyond a
single contiguous split affects performance.
"""
import pytest

from helm.compiler.compiler import _manual_partition_plan, parse_layer_spec
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


# --- single-range / single-int (must keep working) -------------------------
def test_none_and_empty():
    assert parse_layer_spec(None) == set()
    assert parse_layer_spec("") == set()


def test_single_int():
    assert parse_layer_spec("43") == {43}


def test_single_range_inclusive():
    assert parse_layer_spec("0:42") == set(range(0, 43))


# --- multi-range (new) ------------------------------------------------------
def test_two_ranges():
    assert parse_layer_spec("0:5,20:25") == set(range(0, 6)) | set(range(20, 26))


def test_mixed_ranges_and_ints_with_whitespace():
    assert parse_layer_spec("0:1, 3 , 5:6") == {0, 1, 3, 5, 6}


def test_invalid_range_raises():
    with pytest.raises(ValueError):
        parse_layer_spec("9:0")


# --- interleaved placement produces alternating stages ----------------------
def test_interleaved_cpu_set_yields_four_stages():
    # 8 transformer blocks (layers 0..7); CPU = {0,1, 4,5} in two blocks =>
    # cpu, cpu, gpu, gpu, cpu, cpu, gpu, gpu => 4 alternating stages.
    units = [_unit(0, "embedding")]
    units += [_unit(i + 1, "transformer_block", i) for i in range(8)]
    units += [_unit(9, "output")]

    plan, plan_data = _manual_partition_plan(
        units=units,
        cpu_layers="0:1,4:5",
        gpu_layers=None,
        model_name="tiny",
    )

    transformer_devices = [
        plan_data["assignments"][u.unit_id]
        for u in units
        if u.unit_type == "transformer_block"
    ]
    assert transformer_devices == ["cpu", "cpu", "cuda", "cuda",
                                   "cpu", "cpu", "cuda", "cuda"]
    # Transformer region alone crosses the CPU/GPU boundary 3 times; the
    # contiguous split would cross it once. More crossings = more PCIe transfers.
    crossings = sum(
        1 for a, b in zip(transformer_devices, transformer_devices[1:]) if a != b
    )
    assert crossings == 3
