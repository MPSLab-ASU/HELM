from types import SimpleNamespace

import torch
import torch.nn as nn

from experiments import dev_pipeline
from experiments.dev_pipeline import _capture_leaf_module_trace


class TinyGemmaDecoderLayer(nn.Module):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        cache_position=None,
        use_cache=False,
    ):
        return hidden_states


class _TinyGemmaInner(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(4, 4)
        self.layers = nn.ModuleList([TinyGemmaDecoderLayer()])


class TinyGemmaLikeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _TinyGemmaInner()
        self.lm_head = nn.Linear(4, 4, bias=False)
        self.config = SimpleNamespace(
            hidden_size=4,
            model_type="gemma2",
            final_logit_softcapping=None,
        )

        with torch.no_grad():
            self.model.embed_tokens.weight.copy_(torch.eye(4))
            self.lm_head.weight.copy_(torch.eye(4))


def test_capture_leaf_module_trace_applies_gemma_embedding_scaling():
    model = TinyGemmaLikeModel().eval()
    input_ids = torch.tensor([[0, 1]], dtype=torch.long)
    attention_mask = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
    position_ids = torch.tensor([[0, 1]], dtype=torch.long)
    cache_position = torch.tensor([0, 1], dtype=torch.long)

    gm = _capture_leaf_module_trace(
        model,
        (input_ids, attention_mask, position_ids, cache_position),
    )

    logits = gm(input_ids)
    expected = model.lm_head(model.model.embed_tokens(input_ids) * (model.config.hidden_size ** 0.5))

    assert torch.allclose(logits, expected)


def test_build_compile_options_propagates_kv_offload_for_decode():
    args = SimpleNamespace(
        plan="auto",
        cpu_layers=None,
        gpu_layers=None,
        model="dummy-model",
        kv_offload=True,
        max_input_tokens=128,
        max_new_tokens=16,
    )

    opts = dev_pipeline._build_compile_options(args, lower_stages=True, graph_kind="decode")

    assert opts.graph_kind == "decode"
    assert opts.lower_stages is True
    assert opts.kv_offload is True
    assert opts.workload["decode_tokens"] == 16
