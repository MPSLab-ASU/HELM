"""
Unit tests for PartitionUnitBuilder (helm/compiler/partition/partition_units.py).
"""
import pytest

from helm.compiler.partition.partition_units import PartitionUnit, PartitionUnitBuilder


# ──────────────────────────────────────────────────────────────────────────────
# Unit types produced for TinyTransformer (2 layers)
# ──────────────────────────────────────────────────────────────────────────────

def test_embedding_unit_created(annotated_tiny_helm_graph):
    units = PartitionUnitBuilder(annotated_tiny_helm_graph).build()
    types = [u.unit_type for u in units]
    assert "embedding" in types


def test_transformer_block_unit_count(annotated_tiny_helm_graph):
    units = PartitionUnitBuilder(annotated_tiny_helm_graph).build()
    blocks = [u for u in units if u.unit_type == "transformer_block"]
    assert len(blocks) == 2


def test_output_unit_created(annotated_tiny_helm_graph):
    units = PartitionUnitBuilder(annotated_tiny_helm_graph).build()
    types = [u.unit_type for u in units]
    assert "output" in types


def test_unit_ids_are_unique(annotated_tiny_helm_graph):
    units = PartitionUnitBuilder(annotated_tiny_helm_graph).build()
    ids = [u.unit_id for u in units]
    assert len(ids) == len(set(ids))


def test_every_unit_has_nodes(annotated_tiny_helm_graph):
    units = PartitionUnitBuilder(annotated_tiny_helm_graph).build()
    for unit in units:
        assert len(unit.node_ids) > 0, f"Unit {unit.unit_id} ({unit.unit_type}) has no nodes"


# ──────────────────────────────────────────────────────────────────────────────
# Layer range on transformer_block units
# ──────────────────────────────────────────────────────────────────────────────

def test_block_units_layer_range(annotated_tiny_helm_graph):
    units = PartitionUnitBuilder(annotated_tiny_helm_graph).build()
    blocks = sorted(
        [u for u in units if u.unit_type == "transformer_block"],
        key=lambda u: u.layer_start,
    )
    assert blocks[0].layer_start == 0
    assert blocks[1].layer_start == 1


def test_block_unit_layer_start_equals_end(annotated_tiny_helm_graph):
    units = PartitionUnitBuilder(annotated_tiny_helm_graph).build()
    for u in units:
        if u.unit_type == "transformer_block":
            assert u.layer_start == u.layer_end


def test_embedding_unit_has_no_layer_range(annotated_tiny_helm_graph):
    units = PartitionUnitBuilder(annotated_tiny_helm_graph).build()
    for u in units:
        if u.unit_type == "embedding":
            assert u.layer_start is None
            assert u.layer_end is None


def test_output_unit_has_no_layer_range(annotated_tiny_helm_graph):
    units = PartitionUnitBuilder(annotated_tiny_helm_graph).build()
    for u in units:
        if u.unit_type == "output":
            assert u.layer_start is None
            assert u.layer_end is None


# ──────────────────────────────────────────────────────────────────────────────
# Cost aggregation
# ──────────────────────────────────────────────────────────────────────────────

def test_param_bytes_aggregated_correctly(annotated_tiny_helm_graph):
    units = PartitionUnitBuilder(annotated_tiny_helm_graph).build()
    for unit in units:
        expected = sum(
            annotated_tiny_helm_graph.helm_id_to_node[nid].param_bytes
            for nid in unit.node_ids
        )
        assert unit.param_bytes == expected


def test_activation_bytes_aggregated_correctly(annotated_tiny_helm_graph):
    units = PartitionUnitBuilder(annotated_tiny_helm_graph).build()
    for unit in units:
        expected = sum(
            annotated_tiny_helm_graph.helm_id_to_node[nid].activation_bytes
            for nid in unit.node_ids
        )
        assert unit.activation_bytes == expected


def test_flops_prefill_aggregated_correctly(annotated_tiny_helm_graph):
    units = PartitionUnitBuilder(annotated_tiny_helm_graph).build()
    for unit in units:
        expected = sum(
            annotated_tiny_helm_graph.helm_id_to_node[nid].flops_prefill
            for nid in unit.node_ids
        )
        assert unit.flops_prefill == expected


def test_kv_bytes_aggregated_correctly(annotated_tiny_helm_graph):
    units = PartitionUnitBuilder(annotated_tiny_helm_graph).build()
    for unit in units:
        expected = sum(
            annotated_tiny_helm_graph.helm_id_to_node[nid].kv_bytes_per_token
            for nid in unit.node_ids
        )
        assert unit.kv_bytes_per_token == expected


# ──────────────────────────────────────────────────────────────────────────────
# Semantic flags
# ──────────────────────────────────────────────────────────────────────────────

def test_block_unit_contains_attention(annotated_tiny_helm_graph):
    units = PartitionUnitBuilder(annotated_tiny_helm_graph).build()
    blocks = [u for u in units if u.unit_type == "transformer_block"]
    has_attn = any(u.contains_attention for u in blocks)
    assert has_attn


def test_block_unit_contains_mlp(annotated_tiny_helm_graph):
    units = PartitionUnitBuilder(annotated_tiny_helm_graph).build()
    blocks = [u for u in units if u.unit_type == "transformer_block"]
    has_mlp = any(u.contains_mlp for u in blocks)
    assert has_mlp


def test_semantic_flag_propagation(annotated_tiny_helm_graph):
    """contains_* flags are OR'd over all nodes in the unit."""
    units = PartitionUnitBuilder(annotated_tiny_helm_graph).build()
    for unit in units:
        has_attn = any(
            annotated_tiny_helm_graph.helm_id_to_node[nid].is_attention
            for nid in unit.node_ids
        )
        has_mlp = any(
            annotated_tiny_helm_graph.helm_id_to_node[nid].is_mlp
            for nid in unit.node_ids
        )
        assert unit.contains_attention == has_attn
        assert unit.contains_mlp == has_mlp


# ──────────────────────────────────────────────────────────────────────────────
# Edge case: graph with no embedding or output nodes
# ──────────────────────────────────────────────────────────────────────────────

def test_build_is_idempotent(annotated_tiny_helm_graph):
    """Calling build() twice must return the same result, not doubled units."""
    builder = PartitionUnitBuilder(annotated_tiny_helm_graph)
    first = builder.build()
    second = builder.build()
    assert len(first) == len(second)
    assert [u.unit_id for u in first] == [u.unit_id for u in second]


def test_aggregate_cost_raises_on_missing_node(annotated_tiny_helm_graph):
    """_aggregate_cost must raise RuntimeError if a node id is not in the graph."""
    unit = PartitionUnit(
        unit_id=0,
        unit_type="transformer_block",
        layer_start=0,
        layer_end=0,
        node_ids=[99999],
    )
    builder = PartitionUnitBuilder(annotated_tiny_helm_graph)
    with pytest.raises(RuntimeError, match="not found in helm_id_to_node"):
        builder._aggregate_cost(unit)


def test_contains_norm_flag_propagation(annotated_tiny_helm_graph):
    """contains_norm must be True for any unit whose nodes include a norm node."""
    units = PartitionUnitBuilder(annotated_tiny_helm_graph).build()
    for unit in units:
        expected = any(
            annotated_tiny_helm_graph.helm_id_to_node[nid].is_norm
            for nid in unit.node_ids
        )
        assert unit.contains_norm == expected


def test_contains_kv_projection_flag_propagation(annotated_tiny_helm_graph):
    """contains_kv_projection must reflect whether any node is a KV projection."""
    units = PartitionUnitBuilder(annotated_tiny_helm_graph).build()
    for unit in units:
        expected = any(
            annotated_tiny_helm_graph.helm_id_to_node[nid].is_kv_projection
            for nid in unit.node_ids
        )
        assert unit.contains_kv_projection == expected


def test_no_embedding_nodes_produces_no_embedding_unit(simple_helm_graph):
    """SimpleMLP has no embed_tokens or lm_head, so no embedding/output units."""
    units = PartitionUnitBuilder(simple_helm_graph).build()
    types = [u.unit_type for u in units]
    assert "embedding" not in types
    assert "output" not in types


def test_empty_graph_produces_no_units():
    import torch.fx as fx
    import torch.nn as nn
    # A module with no layers → no layer_to_node_ids entries
    gm = fx.symbolic_trace(nn.Identity())
    from helm.compiler.IR.graph import HelmGraph
    g = HelmGraph(gm.graph)
    units = PartitionUnitBuilder(g).build()
    # Identity has no embedding, transformer blocks, or lm_head
    assert all(u.unit_type not in ("embedding", "transformer_block", "output") for u in units)
