"""
Unit tests for HelmGraph, HelmNode, and HelmEdge (helm/compiler/IR/graph.py).
"""
import torch
import torch.nn as nn
import torch.fx as fx
import pytest

from helm.compiler.IR.graph import HelmGraph, HelmNode, HelmEdge


# ──────────────────────────────────────────────────────────────────────────────
# HelmEdge
# ──────────────────────────────────────────────────────────────────────────────

def test_edge_repr_contains_ids():
    e = HelmEdge(src_id=0, dst_id=3, tensor_shape=[2, 4], tensor_bytes=16)
    r = repr(e)
    assert "0 -> 3" in r
    assert "[2, 4]" in r


def test_edge_defaults():
    e = HelmEdge(src_id=1, dst_id=2)
    assert e.tensor_shape == []
    assert e.tensor_bytes == 0
    assert e.is_residual is False
    assert e.crosses_layer_boundary is False


def test_edge_residual_flag():
    e = HelmEdge(src_id=0, dst_id=1, is_residual=True)
    assert e.is_residual is True


# ──────────────────────────────────────────────────────────────────────────────
# HelmNode (constructed via HelmGraph)
# ──────────────────────────────────────────────────────────────────────────────

def test_node_default_semantic_flags(simple_helm_graph):
    for node in simple_helm_graph.nodes:
        if node.op_type == "placeholder":
            assert node.is_attention is False
            assert node.is_mlp is False
            assert node.is_norm is False
            assert node.is_embedding is False
            assert node.is_output_head is False
            assert node.is_kv_projection is False
            break



def test_node_mark_attention():
    gm = fx.symbolic_trace(nn.Linear(2, 2, bias=False))
    g = HelmGraph(gm.graph)
    node = g.nodes[0]
    node.mark_attention()
    assert node.is_attention is True


def test_node_mark_mlp():
    gm = fx.symbolic_trace(nn.Linear(2, 2, bias=False))
    g = HelmGraph(gm.graph)
    node = g.nodes[0]
    node.mark_mlp()
    assert node.is_mlp is True


def test_node_mark_norm():
    gm = fx.symbolic_trace(nn.Linear(2, 2, bias=False))
    g = HelmGraph(gm.graph)
    node = g.nodes[0]
    node.mark_norm()
    assert node.is_norm is True


def test_node_mark_kv_projection():
    gm = fx.symbolic_trace(nn.Linear(2, 2, bias=False))
    g = HelmGraph(gm.graph)
    node = g.nodes[0]
    node.mark_kv_projection()
    assert node.is_kv_projection is True


def test_node_repr_contains_name(simple_helm_graph):
    node = simple_helm_graph.nodes[0]
    assert node.name in repr(node)


# ──────────────────────────────────────────────────────────────────────────────
# get_output_bytes_str formatting
# ──────────────────────────────────────────────────────────────────────────────

def _fresh_node():
    gm = fx.symbolic_trace(nn.Linear(2, 2, bias=False))
    g = HelmGraph(gm.graph)
    return g.nodes[0]


def test_output_bytes_str_zero():
    node = _fresh_node()
    node.activation_bytes = 0
    assert node.get_output_bytes_str() == ""


def test_output_bytes_str_bytes():
    node = _fresh_node()
    node.activation_bytes = 512
    s = node.get_output_bytes_str()
    assert "B" in s
    assert "K" not in s


def test_output_bytes_str_kb():
    node = _fresh_node()
    node.activation_bytes = 2048
    assert "KB" in node.get_output_bytes_str()


def test_output_bytes_str_mb():
    node = _fresh_node()
    node.activation_bytes = 3 * 1024 * 1024
    assert "MB" in node.get_output_bytes_str()


def test_output_bytes_str_gb():
    node = _fresh_node()
    node.activation_bytes = 2 * 1024 ** 3
    assert "GB" in node.get_output_bytes_str()


# ──────────────────────────────────────────────────────────────────────────────
# HelmGraph construction
# ──────────────────────────────────────────────────────────────────────────────

def test_node_count_matches_fx(simple_gm, simple_helm_graph):
    fx_count = sum(1 for _ in simple_gm.graph.nodes)
    assert len(simple_helm_graph.nodes) == fx_count


def test_input_node_ids_are_placeholders(simple_helm_graph):
    assert len(simple_helm_graph.input_node_ids) > 0
    for nid in simple_helm_graph.input_node_ids:
        assert simple_helm_graph.get_node(nid).op_type == "placeholder"


def test_output_node_ids_are_outputs(simple_helm_graph):
    assert len(simple_helm_graph.output_node_ids) > 0
    for nid in simple_helm_graph.output_node_ids:
        assert simple_helm_graph.get_node(nid).op_type == "output"


def test_edges_are_built(simple_helm_graph):
    assert len(simple_helm_graph.edges) > 0


def test_users_back_populated(simple_helm_graph):
    for edge in simple_helm_graph.edges:
        src = simple_helm_graph.get_node(edge.src_id)
        assert any(e.dst_id == edge.dst_id for e in src.out_edges)


def test_dependencies_match_incoming_edges(simple_helm_graph):
    for node in simple_helm_graph.nodes:
        incoming = simple_helm_graph.get_incoming_edges(node.id)
        in_edge_src_ids = {e.src_id for e in node.in_edges}
        for e in incoming:
            assert e.src_id in in_edge_src_ids


def test_helm_id_to_node_lookup(simple_helm_graph):
    for node in simple_helm_graph.nodes:
        assert simple_helm_graph.get_node(node.id) is node


def test_fx_to_helm_all_mapped(simple_gm, simple_helm_graph):
    for fx_node in simple_gm.graph.nodes:
        assert fx_node in simple_helm_graph.fx_to_helm




def test_nodes_in_order_length(simple_helm_graph):
    assert len(simple_helm_graph.nodes_in_order()) == len(simple_helm_graph.nodes)


def test_get_outgoing_edges_correct(simple_helm_graph):
    for node in simple_helm_graph.nodes:
        for e in simple_helm_graph.get_outgoing_edges(node.id):
            assert e.src_id == node.id


def test_get_incoming_edges_correct(simple_helm_graph):
    for node in simple_helm_graph.nodes:
        for e in simple_helm_graph.get_incoming_edges(node.id):
            assert e.dst_id == node.id


def test_node_ids_are_sequential(simple_helm_graph):
    ids = [n.id for n in simple_helm_graph.nodes]
    assert ids == list(range(len(ids)))


def test_node_names_unique(simple_helm_graph):
    names = [n.name for n in simple_helm_graph.nodes]
    assert len(names) == len(set(names))
