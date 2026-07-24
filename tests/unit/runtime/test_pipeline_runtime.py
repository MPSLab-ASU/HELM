import torch

from helm.runtime.pipeline_runtime import PipelineRuntime


class _RecordingRuntime(PipelineRuntime):
    def __init__(self):
        super().__init__(
            prefill_executor=object(),
            decode_executor=object(),
            dtype=torch.float16,
        )
        self.prefill_attention_mask = None

    def _reset_decode_cache(self):
        return None

    def prefill(self, input_ids, attention_mask=None):
        self.prefill_attention_mask = attention_mask.clone() if attention_mask is not None else None
        batch_size, seq_len = input_ids.shape
        logits = torch.zeros(batch_size, seq_len, 3, dtype=torch.float32)
        logits[0, 3, 1] = 10.0
        logits[1, 1, 2] = 10.0
        logits[1, 3, 0] = 9.0
        return logits

    def decode_step(self, input_ids, step_position):
        batch_size = input_ids.shape[0]
        return torch.zeros(batch_size, 1, 3, dtype=torch.float32)


def test_generate_uses_last_non_padding_prompt_logits():
    runtime = _RecordingRuntime()
    input_ids = torch.tensor(
        [
            [11, 12, 13, 14],
            [21, 22, 0, 0],
        ],
        dtype=torch.long,
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1],
            [1, 1, 0, 0],
        ],
        dtype=torch.long,
    )

    generated = runtime.generate(
        input_ids,
        max_new_tokens=1,
        attention_mask=attention_mask,
    )

    assert generated.tolist() == [[1], [2]]
    assert torch.equal(runtime.prefill_attention_mask, attention_mask)


def test_chunk_prefill_mask_is_causal_with_offset():
    rt = PipelineRuntime(object(), object(), dtype=torch.float16)
    mask = rt._build_chunk_prefill_mask(
        past=4, chunk_len=3, device=torch.device("cpu"),
        attention_mask=torch.ones(1, 7, dtype=torch.long),
    )
    assert mask.shape == (1, 1, 3, 7)
    min_val = torch.finfo(torch.float16).min
    # query 0 (abs pos 4) may attend keys 0..4
    assert mask[0, 0, 0, 4].item() == 0.0
    assert mask[0, 0, 0, 5].item() == min_val
    # query 2 (abs pos 6) may attend keys 0..6
    assert mask[0, 0, 2, 6].item() == 0.0


def test_prefill_chunk_size_env(monkeypatch):
    rt = PipelineRuntime(object(), object())
    monkeypatch.delenv("HELM_PREFILL_CHUNK", raising=False)
    assert rt._prefill_chunk_size(256) == 0
    assert rt._prefill_chunk_size(1024) == 512
    monkeypatch.setenv("HELM_PREFILL_CHUNK", "0")
    assert rt._prefill_chunk_size(4096) == 0
    monkeypatch.setenv("HELM_PREFILL_CHUNK", "128")
    assert rt._prefill_chunk_size(4096) == 128
