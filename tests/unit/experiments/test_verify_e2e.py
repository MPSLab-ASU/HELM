import os

import pytest
import torch

from experiments import verify_e2e


class _FakeRuntime:
    pass


class _ChatTokenizer:
    def __init__(self):
        self.chat_template = "{{ messages }}"
        self.formatted_text = "<chat-formatted>"
        self.messages = None
        self.called_text = None

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        self.messages = messages
        assert tokenize is False
        assert add_generation_prompt is True
        return self.formatted_text

    def __call__(self, text, *, return_tensors, truncation, max_length):
        self.called_text = text
        return {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        }


class _PlainTokenizer:
    def __init__(self):
        self.chat_template = None
        self.called_text = None

    def __call__(self, text, *, return_tensors, truncation, max_length):
        self.called_text = text
        return {
            "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
        }


def _run_main(monkeypatch, tmp_path, argv):
    calls = []
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    def fake_build_helm_runtime(*, model_name, dtype_str, cpu_threads, max_input_tokens, max_new_tokens, kv_offload):
        calls.append(
            {
                "model_name": model_name,
                "dtype_str": dtype_str,
                "cpu_threads": cpu_threads,
                "max_input_tokens": max_input_tokens,
                "max_new_tokens": max_new_tokens,
                "kv_offload": kv_offload,
            }
        )
        return _FakeRuntime(), object()

    monkeypatch.setattr(verify_e2e, "build_helm_runtime", fake_build_helm_runtime)
    monkeypatch.setattr(verify_e2e, "compute_hf_references", lambda **kwargs: [])
    monkeypatch.setattr(verify_e2e, "_load_sharegpt_prompts", lambda *args, **kwargs: [])
    monkeypatch.setattr(verify_e2e, "_make_output_dir", lambda base="artifacts": str(out_dir))
    monkeypatch.setattr(verify_e2e.random, "randint", lambda a, b: 123)
    monkeypatch.setattr(verify_e2e.sys, "argv", ["verify_e2e.py", *argv])

    with pytest.raises(SystemExit) as exc_info:
        verify_e2e.main()

    assert exc_info.value.code == 0
    assert len(calls) == 1
    return calls[0]


def test_main_disables_kv_offload_by_default(monkeypatch, tmp_path):
    call = _run_main(monkeypatch, tmp_path, [])
    assert call["kv_offload"] is False


def test_main_defaults_to_gpu_zero(monkeypatch, tmp_path):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    _run_main(monkeypatch, tmp_path, [])
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"


def test_main_uses_ungated_model_by_default(monkeypatch, tmp_path):
    call = _run_main(monkeypatch, tmp_path, [])
    assert call["model_name"] == "Qwen/Qwen3-0.6B"


def _reference(ids, logits_rows):
    return {"token_ids": ids, "logits": [torch.tensor(r) for r in logits_rows]}


def test_compare_to_reference_identical():
    ref = _reference([1, 2, 3], [[0.0, 5.0, 1.0, 0.0]] * 3)
    assert verify_e2e.compare_to_reference([1, 2, 3], ref) == {
        "match": "identical", "compared_tokens": 3}


def test_compare_to_reference_near_tie_is_not_a_failure():
    # Reference picks token 1 by 0.02 logits over token 2; HELM picks 2.
    ref = _reference([1, 1], [[0.0, 20.58, 20.56, 0.0], [0.0, 9.0, 1.0, 0.0]])
    out = verify_e2e.compare_to_reference([2, 1], ref, near_tie_tol=0.1)
    assert out["match"] == "near_tie"
    assert out["first_divergence"] == 0


def test_compare_to_reference_flags_real_divergence():
    ref = _reference([1, 1], [[0.0, 9.0, 1.0, 0.0], [0.0, 9.0, 1.0, 0.0]])
    out = verify_e2e.compare_to_reference([1, 3], ref, near_tie_tol=0.1)
    assert out["match"] == "diverged"
    assert out["first_divergence"] == 1
    assert out["logit_gap"] == 9.0


def test_main_enables_kv_offload_with_flag(monkeypatch, tmp_path):
    call = _run_main(monkeypatch, tmp_path, ["--kv-offload"])
    assert call["kv_offload"] is True


def test_compile_options_enable_explicit_static_analysis_fallback():
    opts = verify_e2e._build_compile_options_common(
        model_name="google/gemma-2-9b-it",
        workload={
            "batch_size": 1,
            "prefill_seq_len": 128,
            "decode_context_len": 128,
            "decode_tokens": 64,
            "dtype_size": 2,
        },
    )

    assert opts["allow_static_analysis_fallback"] is True


def test_build_prefill_inputs_formats_prompt_as_chat():
    tokenizer = _ChatTokenizer()

    verify_e2e._build_prefill_inputs(tokenizer, "hello there", max_input_tokens=8)

    assert tokenizer.messages == [{"role": "user", "content": "hello there"}]
    assert tokenizer.called_text == tokenizer.formatted_text


def test_build_prefill_inputs_falls_back_to_raw_prompt_without_chat_template():
    tokenizer = _PlainTokenizer()

    verify_e2e._build_prefill_inputs(tokenizer, "hello there", max_input_tokens=8)

    assert tokenizer.called_text == "hello there"


def test_load_sharegpt_prompts_raises_when_file_is_missing():
    import pytest

    with pytest.raises(FileNotFoundError, match="verify_e2e prompts file not found"):
        verify_e2e._load_sharegpt_prompts(
            "/definitely/missing/sharegpt.json",
            n=3,
            rng=verify_e2e.random.Random(0),
        )


def test_load_sharegpt_prompts_accepts_flat_list_of_strings(tmp_path):
    import json

    p = tmp_path / "prompts.json"
    p.write_text(json.dumps(["alpha", "beta", "gamma", "delta"]))

    prompts = verify_e2e._load_sharegpt_prompts(
        str(p), n=2, rng=verify_e2e.random.Random(0)
    )
    assert len(prompts) == 2
    assert set(prompts).issubset({"alpha", "beta", "gamma", "delta"})


def test_load_sharegpt_prompts_raises_on_invalid_json(tmp_path):
    import pytest

    p = tmp_path / "broken.json"
    p.write_text("{not valid json")

    with pytest.raises(RuntimeError, match="not valid JSON"):
        verify_e2e._load_sharegpt_prompts(
            str(p), n=3, rng=verify_e2e.random.Random(0)
        )


def test_default_prompts_file_exists_and_loads():
    # The committed prompts file must always be loadable so verify_e2e can
    # serve as an audit trail without depending on any external dataset.
    prompts = verify_e2e._load_sharegpt_prompts(
        verify_e2e._DEFAULT_PROMPTS_PATH,
        n=5,
        rng=verify_e2e.random.Random(0),
    )
    assert len(prompts) == 5
    assert all(isinstance(p, str) and p.strip() for p in prompts)
