"""
Unit tests for HybridAnalyzer (helm/compiler/analysis/hybrid_analyzer.py).
"""
import json
from types import SimpleNamespace

import pytest
import torch
import torch.fx as fx
import torch.nn as nn

from helm.compiler.IR.graph import HelmGraph
from helm.compiler.analysis.hybrid_analyzer import HybridAnalyzer
from helm.compiler.importers.fx_importer import FXImporter


class _LeafTracer(fx.Tracer):
    def __init__(self, leaf_types: tuple[type[nn.Module], ...] = ()) -> None:
        super().__init__()
        self._leaf_types = leaf_types

    def is_leaf_module(self, module: nn.Module, module_qualified_name: str) -> bool:
        return isinstance(module, self._leaf_types) or super().is_leaf_module(
            module, module_qualified_name
        )


class _ConfiguredTinyLayer(nn.Module):
    def __init__(self, hidden: int = 8):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden)
        self.self_attn = nn.Linear(hidden, hidden, bias=False)
        self.mlp = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layernorm(x)
        h = self.self_attn(h)
        return self.mlp(h)


class _ConfiguredTinyTransformer(nn.Module):
    def __init__(self, vocab: int = 64, hidden: int = 8, n_layers: int = 2):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList(
            [_ConfiguredTinyLayer(hidden) for _ in range(n_layers)]
        )
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self.config = SimpleNamespace(
            hidden_size=hidden,
            intermediate_size=hidden,
            num_attention_heads=2,
            num_key_value_heads=2,
            vocab_size=vocab,
            torch_dtype=torch.float32,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)


class _OutputHeadModel(nn.Module):
    def __init__(self, vocab: int = 32, hidden: int = 8):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.output_head = nn.Linear(hidden, vocab, bias=False)
        self.config = SimpleNamespace(
            hidden_size=hidden,
            intermediate_size=hidden,
            num_attention_heads=2,
            num_key_value_heads=2,
            vocab_size=vocab,
            torch_dtype=torch.float32,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        return self.output_head(x)


class _TiedEmbeddingHeadModel(nn.Module):
    def __init__(self, vocab: int = 64, hidden: int = 8):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self.lm_head.weight = self.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        return self.lm_head(x)


def _build_graph(
    model: nn.Module, leaf_types: tuple[type[nn.Module], ...] = ()
) -> tuple[fx.GraphModule, HelmGraph]:
    if leaf_types:
        tracer = _LeafTracer(leaf_types)
        gm = fx.GraphModule(model, tracer.trace(model))
    else:
        gm = fx.symbolic_trace(model)
    helm_graph = HelmGraph(gm.graph)
    FXImporter(gm, helm_graph).run()
    return gm, helm_graph


def test_init_rejects_unused_tokenizer_kwarg(
    tiny_gm,
    tiny_helm_graph,
    tiny_transformer,
):
    with pytest.raises(TypeError):
        HybridAnalyzer(
            gm=tiny_gm,
            helm_graph=tiny_helm_graph,
            model=tiny_transformer,
            tokenizer=None,
        )


def test_run_and_export_summary(
    tmp_path,
    tiny_gm_with_leaves,
    annotated_helm_graph_with_leaves,
    tiny_transformer,
):
    analyzer = HybridAnalyzer(
        gm=tiny_gm_with_leaves,
        helm_graph=annotated_helm_graph_with_leaves,
        model=tiny_transformer,
    )

    summary = analyzer.run(torch.tensor([[1, 2, 3]], dtype=torch.long))

    assert summary.num_nodes == len(annotated_helm_graph_with_leaves.nodes)
    assert summary.num_nodes_with_shapes > 0
    assert summary.total_activation_bytes > 0

    export_path = tmp_path / "hybrid_analysis.json"
    analyzer.export_summary(str(export_path))

    data = json.loads(export_path.read_text())
    assert len(data) == len(annotated_helm_graph_with_leaves.nodes)
    assert any(node["target"] == "layers.0" for node in data)


def test_run_annotates_root_level_transformer_blocks() -> None:
    model = _ConfiguredTinyTransformer().eval()
    gm, helm_graph = _build_graph(model, leaf_types=(_ConfiguredTinyLayer,))
    analyzer = HybridAnalyzer(gm=gm, helm_graph=helm_graph, model=model)

    summary = analyzer.run(torch.tensor([[1, 2, 3]], dtype=torch.long))

    layer_nodes = {
        str(node.target): node
        for node in helm_graph.nodes
        if node.op_type == "call_module" and str(node.target).startswith("layers.")
    }
    assert set(layer_nodes) == {"layers.0", "layers.1"}
    assert all(node.flops_prefill > 0 for node in layer_nodes.values())
    assert all(node.flops_decode > 0 for node in layer_nodes.values())
    assert all(node.kv_bytes_per_token > 0 for node in layer_nodes.values())
    assert summary.total_kv_bytes_per_token == sum(
        node.kv_bytes_per_token for node in layer_nodes.values()
    )


def test_run_annotates_output_head_aliases() -> None:
    model = _OutputHeadModel().eval()
    gm, helm_graph = _build_graph(model)
    analyzer = HybridAnalyzer(gm=gm, helm_graph=helm_graph, model=model)

    analyzer.run(torch.tensor([[1, 2, 3]], dtype=torch.long))

    output_head = next(
        node
        for node in helm_graph.nodes
        if node.op_type == "call_module" and str(node.target) == "output_head"
    )
    assert output_head.flops_prefill == 2 * 1 * 3 * 8 * 32
    assert output_head.flops_decode == 2 * 1 * 1 * 8 * 32


def test_run_counts_tied_parameters_once() -> None:
    model = _TiedEmbeddingHeadModel().eval()
    gm, helm_graph = _build_graph(model)
    analyzer = HybridAnalyzer(gm=gm, helm_graph=helm_graph, model=model)

    summary = analyzer.run(torch.tensor([[1, 2, 3]], dtype=torch.long))

    unique_param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    call_module_param_bytes = sum(
        node.param_bytes for node in helm_graph.nodes if node.op_type == "call_module"
    )
    assert call_module_param_bytes == unique_param_bytes
    assert summary.total_param_bytes == unique_param_bytes
