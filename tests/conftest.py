"""
Shared fixtures for the HELM test suite.
"""
import pytest
import torch
import torch.nn as nn
import torch.fx as fx

from helm.compiler.IR.graph import HelmGraph
from helm.compiler.importers.fx_importer import FXImporter
from helm.compiler.optimization.cost_model import DeviceProfile, LinkProfile, HelmCostModel, ModelConfig, WorkloadSpec
from helm.runtime.kv_allocator import KVAllocator


# ──────────────────────────────────────────────────────────────────────────────
# Tiny model definitions
# ──────────────────────────────────────────────────────────────────────────────

class TinyLayer(nn.Module):
    """One transformer-style layer with the module-path structure HELM expects."""

    def __init__(self, hidden: int = 8):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden)
        self.self_attn = nn.Linear(hidden, hidden, bias=False)
        self.mlp = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layernorm(x)
        h = self.self_attn(h)
        return self.mlp(h)


class TinyTransformer(nn.Module):
    """
    Minimal 2-layer transformer whose module paths mirror real models:
      embed_tokens / layers.0.{self_attn,mlp,...} / layers.1.{...} / lm_head
    This gives FXImporter the patterns it needs without loading a real LLM.
    """

    def __init__(self, vocab: int = 64, hidden: int = 8, n_layers: int = 2):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList([TinyLayer(hidden) for _ in range(n_layers)])
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        x = self.layers[0](x)
        x = self.layers[1](x)
        return self.lm_head(x)


class TinyTransformerTied(nn.Module):
    """
    Like TinyTransformer but lm_head.weight is tied to embed_tokens.weight,
    mirroring Qwen3's architecture.  Used to test that HELM's executor
    correctly detects and breaks cross-stage tied weights.
    """

    def __init__(self, vocab: int = 64, hidden: int = 8, n_layers: int = 2):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList([TinyLayer(hidden) for _ in range(n_layers)])
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        # Tie weights: lm_head.weight shares storage with embed_tokens.weight
        self.lm_head.weight = self.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        x = self.layers[0](x)
        x = self.layers[1](x)
        return self.lm_head(x)


class SimpleMLP(nn.Module):
    """Tiny two-layer MLP for basic FX graph tests."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4, 8, bias=False)
        self.linear2 = nn.Linear(8, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(torch.relu(self.linear1(x)))


# ──────────────────────────────────────────────────────────────────────────────
# FX / HelmGraph fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def simple_mlp():
    return SimpleMLP().eval()


@pytest.fixture(scope="session")
def simple_gm(simple_mlp):
    return fx.symbolic_trace(simple_mlp)


@pytest.fixture(scope="session")
def simple_helm_graph(simple_gm):
    return HelmGraph(simple_gm.graph)


@pytest.fixture(scope="session")
def tiny_transformer():
    return TinyTransformer().eval()


@pytest.fixture(scope="session")
def tiny_transformer_tied():
    return TinyTransformerTied().eval()


@pytest.fixture(scope="session")
def tiny_gm(tiny_transformer):
    """Fully-expanded FX graph (every individual op is a separate node)."""
    return fx.symbolic_trace(tiny_transformer)


@pytest.fixture(scope="session")
def tiny_gm_with_leaves(tiny_transformer):
    """
    FX graph with TinyLayer as leaf modules.
    Each layer becomes a single call_module node — the structure PartitionUnitBuilder expects.
    """
    class _LeafTracer(fx.Tracer):
        def is_leaf_module(self, m, module_qualified_name):
            return isinstance(m, TinyLayer) or super().is_leaf_module(m, module_qualified_name)

    tracer = _LeafTracer()
    graph = tracer.trace(tiny_transformer)
    return fx.GraphModule(tiny_transformer, graph)


@pytest.fixture(scope="session")
def tiny_gm_with_leaves_tied(tiny_transformer_tied):
    """Leaf-mode FX graph of the tied-weight model (for cross-stage tie tests)."""
    class _LeafTracer(fx.Tracer):
        def is_leaf_module(self, m, module_qualified_name):
            return isinstance(m, TinyLayer) or super().is_leaf_module(m, module_qualified_name)

    tracer = _LeafTracer()
    graph = tracer.trace(tiny_transformer_tied)
    return fx.GraphModule(tiny_transformer_tied, graph)


@pytest.fixture(scope="session")
def tiny_helm_graph(tiny_gm):
    return HelmGraph(tiny_gm.graph)


@pytest.fixture(scope="session")
def tiny_helm_graph_with_leaves(tiny_gm_with_leaves):
    return HelmGraph(tiny_gm_with_leaves.graph)


@pytest.fixture(scope="session")
def annotated_tiny_helm_graph(tiny_gm, tiny_helm_graph):
    """HelmGraph after FXImporter has run (fully-expanded graph)."""
    importer = FXImporter(tiny_gm, tiny_helm_graph)
    importer.run()
    return tiny_helm_graph


@pytest.fixture(scope="session")
def annotated_helm_graph_with_leaves(tiny_gm_with_leaves, tiny_helm_graph_with_leaves):
    """Leaf-mode HelmGraph after FXImporter."""
    importer = FXImporter(tiny_gm_with_leaves, tiny_helm_graph_with_leaves)
    importer.run()
    return tiny_helm_graph_with_leaves


# ──────────────────────────────────────────────────────────────────────────────
# Cost model fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def cpu_profile():
    return DeviceProfile(
        device_id="cpu",
        device_type="cpu",
        peak_flops_prefill=1e12,
        peak_flops_decode=1e12,
        mem_bandwidth=50e9,
        memory_capacity=32 * 1024 ** 3,
    )


@pytest.fixture()
def gpu_profile():
    return DeviceProfile(
        device_id="cuda:0",
        device_type="cuda",
        peak_flops_prefill=100e12,
        peak_flops_decode=100e12,
        mem_bandwidth=500e9,
        memory_capacity=8 * 1024 ** 3,
    )


@pytest.fixture()
def pcie_link():
    return LinkProfile(
        src="cpu",
        dst="cuda:0",
        bandwidth_bytes_per_s=16e9,
        latency_s=1e-4,
    )


@pytest.fixture()
def cost_model(cpu_profile, gpu_profile, pcie_link):
    # ModelConfig is required for cross-device boundary communication estimates.
    mc = ModelConfig(
        hidden_size=4096,
        intermediate_size=11008,
        num_attention_heads=32,
        num_kv_heads=32,
        head_dim=128,
    )
    devices = {"cpu": cpu_profile, "cuda:0": gpu_profile}
    links = {
        ("cpu", "cuda:0"): pcie_link,
        ("cuda:0", "cpu"): pcie_link,
    }
    return HelmCostModel(devices, links, model_config=mc)


@pytest.fixture()
def workload():
    return WorkloadSpec(
        batch_size=1,
        prefill_seq_len=128,
        decode_context_len=256,
        decode_tokens=64,
    )


# ──────────────────────────────────────────────────────────────────────────────
# KV allocator fixture
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def kv_allocator():
    return KVAllocator(
        num_layers=4,
        num_kv_heads=2,
        head_dim=16,
        page_size=8,
        dtype=torch.float32,
    )
