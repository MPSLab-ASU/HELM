"""
Unit tests for FXImporter (helm/compiler/importers/fx_importer.py).
"""
import pytest
import torch.fx as fx

from helm.compiler.IR.graph import HelmGraph
from helm.compiler.importers.fx_importer import FXImporter


# ──────────────────────────────────────────────────────────────────────────────
# Layer ID extraction
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("path,expected", [
    ("model.layers.0.self_attn",  0),
    ("model.layers.12.mlp",      12),
    ("transformer.h.5.attn",      5),
    ("model.blocks.3.norm",        3),
    ("embed_tokens",            None),
    ("lm_head",                 None),
    ("",                        None),
    ("layers.0",                   0),
    ("h.10.mlp",                  10),
])
def test_extract_layer_id(tiny_gm, tiny_helm_graph, path, expected):
    importer = FXImporter(tiny_gm, tiny_helm_graph)
    assert importer._extract_layer_id_from_path(path) == expected


# ──────────────────────────────────────────────────────────────────────────────
# Semantic tag assignment (isolated)
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("keyword,flag", [
    ("self_attn",    "is_attention"),
    ("attention",    "is_attention"),
    ("attn",         "is_attention"),
    ("mlp",          "is_mlp"),
    ("feed_forward", "is_mlp"),
    ("down_proj",    "is_mlp"),
    ("up_proj",      "is_mlp"),
    ("gate_proj",    "is_mlp"),
    ("norm",         "is_norm"),
    ("layernorm",    "is_norm"),
    ("rmsnorm",      "is_norm"),
    ("embed_tokens", "is_embedding"),
    ("embedding",    "is_embedding"),
    ("wte",          "is_embedding"),
    ("lm_head",      "is_output_head"),
    ("output_head",  "is_output_head"),
    ("k_proj",       "is_kv_projection"),
    ("v_proj",       "is_kv_projection"),
    ("k_norm",       "is_kv_projection"),
    ("v_norm",       "is_kv_projection"),
])
def test_semantic_tag_keyword(tiny_gm, tiny_helm_graph, keyword, flag):
    g = HelmGraph(tiny_gm.graph)
    imp = FXImporter(tiny_gm, g)
    node = g.nodes[0]
    node.module_path = keyword
    node.fx_node_name = ""
    node.name = ""
    node.target = ""
    imp._assign_semantic_tags()
    assert getattr(node, flag) is True


def test_no_spurious_tags_on_plain_path(tiny_gm):
    g = HelmGraph(tiny_gm.graph)
    imp = FXImporter(tiny_gm, g)
    node = g.nodes[0]
    node.module_path = "some.plain.linear"
    node.fx_node_name = ""
    node.name = ""
    node.target = ""
    imp._assign_semantic_tags()
    assert node.is_attention is False
    assert node.is_mlp is False
    assert node.is_norm is False
    assert node.is_embedding is False
    assert node.is_output_head is False
    assert node.is_kv_projection is False


# ──────────────────────────────────────────────────────────────────────────────
# _longest_common_module_prefix
# ──────────────────────────────────────────────────────────────────────────────

def _make_importer():
    imp = FXImporter.__new__(FXImporter)
    return imp


def test_common_prefix_single():
    imp = _make_importer()
    assert imp._longest_common_module_prefix(["model.layers.0.attn"]) == "model.layers.0.attn"


def test_common_prefix_two_siblings():
    imp = _make_importer()
    result = imp._longest_common_module_prefix(["model.layers.0.attn", "model.layers.0.mlp"])
    assert result == "model.layers.0"


def test_common_prefix_different_layers():
    imp = _make_importer()
    result = imp._longest_common_module_prefix(["model.layers.0.attn", "model.layers.1.attn"])
    assert result == "model.layers"


def test_common_prefix_empty_list():
    imp = _make_importer()
    assert imp._longest_common_module_prefix([]) == ""


def test_common_prefix_no_overlap():
    imp = _make_importer()
    assert imp._longest_common_module_prefix(["a.b.c", "x.y.z"]) == ""


def test_common_prefix_with_empty_strings():
    imp = _make_importer()
    # Empty paths are filtered out
    assert imp._longest_common_module_prefix(["", ""]) == ""


# ──────────────────────────────────────────────────────────────────────────────
# Full importer run on TinyTransformer
# ──────────────────────────────────────────────────────────────────────────────

def test_run_assigns_layer_0_and_1(annotated_tiny_helm_graph):
    layer_ids = {n.layer_id for n in annotated_tiny_helm_graph.nodes if n.layer_id is not None}
    assert 0 in layer_ids
    assert 1 in layer_ids


def test_run_populates_layer_index(annotated_tiny_helm_graph):
    assert len(annotated_tiny_helm_graph.layer_to_node_ids) >= 2
    assert 0 in annotated_tiny_helm_graph.layer_to_node_ids
    assert 1 in annotated_tiny_helm_graph.layer_to_node_ids


def test_run_finds_embedding_nodes(annotated_tiny_helm_graph):
    embed = [n for n in annotated_tiny_helm_graph.nodes if n.is_embedding]
    assert len(embed) >= 1


def test_run_finds_lm_head_nodes(annotated_tiny_helm_graph):
    heads = [n for n in annotated_tiny_helm_graph.nodes if n.is_output_head]
    assert len(heads) >= 1


def test_run_finds_attention_nodes(annotated_tiny_helm_graph):
    attn = [n for n in annotated_tiny_helm_graph.nodes if n.is_attention]
    assert len(attn) >= 1


def test_run_finds_mlp_nodes(annotated_tiny_helm_graph):
    mlp = [n for n in annotated_tiny_helm_graph.nodes if n.is_mlp]
    assert len(mlp) >= 1


def test_run_finds_norm_nodes(annotated_tiny_helm_graph):
    norm = [n for n in annotated_tiny_helm_graph.nodes if n.is_norm]
    assert len(norm) >= 1


def test_call_module_nodes_have_module_path(annotated_tiny_helm_graph):
    for node in annotated_tiny_helm_graph.nodes:
        if node.op_type == "call_module":
            assert node.module_path != "", f"Node {node.name} (call_module) has empty module_path"
