"""Unit tests for the HelmInference router.

These tests cover only the routing decision (the load-bearing piece of
the facade). The heavy setup() path is exercised by verify_e2e.py and
the smoke test under experiments/, which need a real model and GPU.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from helm.compiler.partition.partition_plan import PartitionPlan, StageSpec
from helm.runtime.inference import (
    HelmInference,
    HelmInferenceConfig,
    _encode_prompt,
    _is_all_gpu,
)


def _plan(*device_ids: str) -> PartitionPlan:
    return PartitionPlan(
        stages=[
            StageSpec(stage_id=i, device_id=d, layer_start=i, layer_end=i + 1)
            for i, d in enumerate(device_ids)
        ]
    )


class TestIsAllGpu:
    """Routing predicate must match paper_bench's cc1bf42 detection."""

    def test_returns_false_for_none(self):
        assert _is_all_gpu(None) is False

    def test_returns_false_for_empty_plan(self):
        assert _is_all_gpu(PartitionPlan(stages=[])) is False

    def test_returns_true_for_single_cuda_stage(self):
        assert _is_all_gpu(_plan("cuda:0")) is True

    def test_returns_true_for_multiple_cuda_stages(self):
        assert _is_all_gpu(_plan("cuda:0", "cuda:0", "cuda:0")) is True

    def test_returns_false_when_any_stage_on_cpu(self):
        assert _is_all_gpu(_plan("cuda:0", "cpu", "cuda:0")) is False

    def test_returns_false_for_all_cpu(self):
        assert _is_all_gpu(_plan("cpu", "cpu")) is False

    def test_handles_missing_device_id(self):
        plan = PartitionPlan(stages=[StageSpec(stage_id=0, device_id="")])
        assert _is_all_gpu(plan) is False


class TestHelmInferenceConfig:
    def test_defaults_to_auto_routing(self):
        cfg = HelmInferenceConfig(model_id="dummy/model")
        assert cfg.route_to_vllm_when_all_gpu is None

    def test_rejects_non_boolean_routing(self):
        with pytest.raises(ValueError, match="route_to_vllm_when_all_gpu"):
            HelmInferenceConfig(model_id="dummy/model", route_to_vllm_when_all_gpu="yes")

    def test_is_frozen(self):
        cfg = HelmInferenceConfig(model_id="dummy/model")
        with pytest.raises((AttributeError, Exception)):
            cfg.model_id = "other"  # type: ignore[misc]


class TestRoutingDecision:
    """Drive HelmInference.setup() with a mocked compile to assert routing."""

    def _patch_compile(self, monkeypatch, plan: PartitionPlan):
        prefill_artifact = MagicMock()
        model = MagicMock()
        tokenizer = MagicMock()
        traced_seq_len = 7

        def fake_compile(_config):
            return plan, prefill_artifact, model, tokenizer, traced_seq_len

        monkeypatch.setattr(
            "helm.runtime.inference._compile_partition_plan",
            fake_compile,
        )
        return prefill_artifact, model, tokenizer

    def test_all_gpu_routes_to_vllm(self, monkeypatch):
        plan = _plan("cuda:0", "cuda:0")
        self._patch_compile(monkeypatch, plan)

        vllm_setup = MagicMock()
        helm_setup = MagicMock()
        monkeypatch.setattr(
            "helm.runtime.inference._VLLMHandle.setup", vllm_setup
        )
        monkeypatch.setattr(
            "helm.runtime.inference._HelmHandle.setup", helm_setup
        )

        inference = HelmInference(HelmInferenceConfig(model_id="dummy/model", route_to_vllm_when_all_gpu=True))
        inference.setup()

        assert inference.routed_to_vllm is True
        assert inference.partition_plan is plan
        vllm_setup.assert_called_once()
        helm_setup.assert_not_called()

    def test_hybrid_plan_routes_to_helm(self, monkeypatch):
        plan = _plan("cuda:0", "cpu", "cuda:0")
        self._patch_compile(monkeypatch, plan)

        vllm_setup = MagicMock()
        helm_setup = MagicMock()
        monkeypatch.setattr(
            "helm.runtime.inference._VLLMHandle.setup", vllm_setup
        )
        monkeypatch.setattr(
            "helm.runtime.inference._HelmHandle.setup", helm_setup
        )

        inference = HelmInference(HelmInferenceConfig(model_id="dummy/model"))
        inference.setup()

        assert inference.routed_to_vllm is False
        vllm_setup.assert_not_called()
        helm_setup.assert_called_once()

    def test_opt_out_keeps_helm_even_on_all_gpu(self, monkeypatch):
        plan = _plan("cuda:0", "cuda:0")
        self._patch_compile(monkeypatch, plan)

        vllm_setup = MagicMock()
        helm_setup = MagicMock()
        monkeypatch.setattr(
            "helm.runtime.inference._VLLMHandle.setup", vllm_setup
        )
        monkeypatch.setattr(
            "helm.runtime.inference._HelmHandle.setup", helm_setup
        )

        inference = HelmInference(
            HelmInferenceConfig(model_id="dummy/model", route_to_vllm_when_all_gpu=False)
        )
        inference.setup()

        assert inference.routed_to_vllm is False
        vllm_setup.assert_not_called()
        helm_setup.assert_called_once()

    @pytest.mark.parametrize("vllm_installed,expect_vllm", [(True, True), (False, False)])
    def test_auto_routes_all_gpu_plan_to_vllm_only_when_installed(
        self, monkeypatch, vllm_installed, expect_vllm
    ):
        plan = _plan("cuda:0", "cuda:0")
        self._patch_compile(monkeypatch, plan)
        vllm_setup = MagicMock()
        helm_setup = MagicMock()
        monkeypatch.setattr("helm.runtime.inference._VLLMHandle.setup", vllm_setup)
        monkeypatch.setattr("helm.runtime.inference._HelmHandle.setup", helm_setup)
        monkeypatch.setattr("helm.runtime.inference._vllm_available", lambda: vllm_installed)

        inference = HelmInference(HelmInferenceConfig(model_id="dummy/model"))
        inference.setup()

        assert inference.routed_to_vllm is expect_vllm
        assert vllm_setup.call_count == int(expect_vllm)
        assert helm_setup.call_count == int(not expect_vllm)

    def test_auto_keeps_hybrid_plan_on_helm_even_with_vllm(self, monkeypatch):
        plan = _plan("cpu", "cuda:0")
        self._patch_compile(monkeypatch, plan)
        vllm_setup = MagicMock()
        helm_setup = MagicMock()
        monkeypatch.setattr("helm.runtime.inference._VLLMHandle.setup", vllm_setup)
        monkeypatch.setattr("helm.runtime.inference._HelmHandle.setup", helm_setup)
        monkeypatch.setattr("helm.runtime.inference._vllm_available", lambda: True)

        inference = HelmInference(HelmInferenceConfig(model_id="dummy/model"))
        inference.setup()

        assert inference.routed_to_vllm is False
        vllm_setup.assert_not_called()
        helm_setup.assert_called_once()

    def test_generate_before_setup_raises(self):
        inference = HelmInference(HelmInferenceConfig(model_id="dummy/model"))
        with pytest.raises(RuntimeError, match="setup"):
            inference.generate(["hello"])


class TestVLLMUnavailable:
    """User explicitly chose raise-on-vLLM-unavailable: prove it."""

    def test_missing_vllm_raises_with_actionable_message(self, monkeypatch):
        import builtins
        from helm.runtime.inference import _VLLMHandle, HelmInferenceConfig

        real_import = builtins.__import__

        def hide_vllm(name, *a, **kw):
            if name == "vllm" or name.startswith("vllm."):
                raise ImportError("No module named 'vllm'")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", hide_vllm)

        handle = _VLLMHandle(HelmInferenceConfig(model_id="dummy/model"), tokenizer=None)
        with pytest.raises(RuntimeError, match="route_to_vllm_when_all_gpu=False"):
            handle.setup()


# ---------------------------------------------------------------------------
# Lifecycle + request-handling regression tests (Task C).
#
# These exercise the real HelmInference.generate() validation/coercion and the
# setup()/teardown()/context-manager lifecycle without any model download, GPU,
# or vLLM import. The compile step is faked (a hybrid plan routes to the native
# HELM path) and _HelmHandle.setup/generate are replaced with recording fakes so
# the object binds a working handle whose per-call behaviour we can observe.
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """Minimal tokenizer: token count == whitespace-split word count.

    Encodes to a 1xN int tensor so _encode_prompt's shape[1] length check and
    the empty-encoding guard both work against a real tensor.
    """

    def __init__(self, empty_for=None):
        self._empty_for = empty_for or set()

    def __call__(self, prompt, return_tensors=None, truncation=False, max_length=None):
        import torch as _torch

        if prompt in self._empty_for:
            n = 0
        else:
            n = len(prompt.split())
        return {"input_ids": _torch.zeros((1, n), dtype=_torch.long)}


def _install_helm_handle(monkeypatch, generate_impl):
    """Route setup() through the native HELM path with a recording fake handle.

    A hybrid (CPU+GPU) plan forces _HelmHandle. _HelmHandle.setup is stubbed to
    a no-op and _HelmHandle.generate is replaced by ``generate_impl``. Returns a
    dict recorder tracking compile-call and setup-call counts.
    """
    from helm.runtime.inference import _HelmHandle

    rec = {"compile_calls": 0, "setup_calls": 0}

    plan = _plan("cuda:0", "cpu", "cuda:0")

    def fake_compile(_config):
        rec["compile_calls"] += 1
        return plan, MagicMock(), MagicMock(), _FakeTokenizer(), 3

    def fake_setup(self):
        rec["setup_calls"] += 1
        self._runtime = object()  # non-None so generate() proceeds

    monkeypatch.setattr(
        "helm.runtime.inference._compile_partition_plan", fake_compile
    )
    monkeypatch.setattr(_HelmHandle, "setup", fake_setup)
    monkeypatch.setattr(_HelmHandle, "generate", generate_impl)
    return rec


class TestRepeatedRequests:
    """Several generate() calls against one setup() reuse the same handle."""

    def test_repeated_generate_reuses_single_setup(self, monkeypatch):
        calls = []

        def gen(self, prompts, max_new_tokens):
            calls.append(list(prompts))
            return [f"out:{p}" for p in prompts]

        rec = _install_helm_handle(monkeypatch, gen)

        inf = HelmInference(HelmInferenceConfig(model_id="dummy/model"))
        inf.setup()

        assert inf.generate(["a"]) == ["out:a"]
        assert inf.generate(["b b"]) == ["out:b b"]
        assert inf.generate(["c", "d d"]) == ["out:c", "out:d d"]

        # setup ran exactly once across all three requests.
        assert rec["compile_calls"] == 1
        assert rec["setup_calls"] == 1
        assert calls == [["a"], ["b b"], ["c", "d d"]]
        inf.teardown()


class TestVariableLengthPrompts:
    """Mixed and differing prompt lengths handled; oversize raises, no truncate."""

    def test_batch_mixes_short_and_long_prompts(self, monkeypatch):
        def gen(self, prompts, max_new_tokens):
            return [str(len(p.split())) for p in prompts]

        _install_helm_handle(monkeypatch, gen)
        inf = HelmInference(
            HelmInferenceConfig(model_id="dummy/model", max_input_tokens=10)
        )
        inf.setup()
        # one short (1 tok) and one long (5 tok) prompt in the same batch.
        assert inf.generate(["hi", "one two three four five"]) == ["1", "5"]
        inf.teardown()

    def test_successive_calls_with_differing_lengths(self, monkeypatch):
        seen = []

        def gen(self, prompts, max_new_tokens):
            seen.extend(len(p.split()) for p in prompts)
            return ["ok" for _ in prompts]

        _install_helm_handle(monkeypatch, gen)
        inf = HelmInference(
            HelmInferenceConfig(model_id="dummy/model", max_input_tokens=10)
        )
        inf.setup()
        inf.generate(["a"])
        inf.generate(["a b c"])
        inf.generate(["a b c d e f"])
        assert seen == [1, 3, 6]
        inf.teardown()

    def test_prompt_over_max_input_tokens_raises_actionable_error(self):
        cfg = HelmInferenceConfig(model_id="dummy/model", max_input_tokens=3)
        tok = _FakeTokenizer()
        with pytest.raises(ValueError, match="exceeding max_input_tokens=3") as ei:
            _encode_prompt(tok, "one two three four", cfg, max_new_tokens=1)
        # Actionable guidance, and it does NOT silently truncate.
        assert "shorten the prompt or increase the limit" in str(ei.value)

    def test_prompt_at_limit_is_accepted(self):
        cfg = HelmInferenceConfig(model_id="dummy/model", max_input_tokens=3)
        tok = _FakeTokenizer()
        encoded = _encode_prompt(tok, "one two three", cfg, max_new_tokens=1)
        assert encoded["input_ids"].shape[1] == 3

    def test_empty_encoding_rejected(self):
        cfg = HelmInferenceConfig(model_id="dummy/model")
        tok = _FakeTokenizer(empty_for={"???"})
        with pytest.raises(ValueError, match="at least one token"):
            _encode_prompt(tok, "???", cfg, max_new_tokens=1)

    def test_prompt_plus_new_tokens_over_context_limit_raises(self):
        cfg = HelmInferenceConfig(model_id="dummy/model", max_input_tokens=100)
        tok = _FakeTokenizer()
        # 3 prompt tokens + 5 new tokens > context_limit 6.
        with pytest.raises(ValueError, match="context limit 6"):
            _encode_prompt(tok, "a b c", cfg, max_new_tokens=5, context_limit=6)


class TestMaxNewTokensOverride:
    """Per-call max_new_tokens override, zero-token behaviour, and rejections."""

    def test_override_is_forwarded_to_handle(self, monkeypatch):
        seen = {}

        def gen(self, prompts, max_new_tokens):
            seen["n"] = max_new_tokens
            return ["x"]

        _install_helm_handle(monkeypatch, gen)
        inf = HelmInference(
            HelmInferenceConfig(model_id="dummy/model", max_new_tokens=64)
        )
        inf.setup()
        inf.generate(["hi"], max_new_tokens=7)
        assert seen["n"] == 7
        inf.teardown()

    def test_default_uses_config_max_new_tokens(self, monkeypatch):
        seen = {}

        def gen(self, prompts, max_new_tokens):
            seen["n"] = max_new_tokens
            return ["x"]

        _install_helm_handle(monkeypatch, gen)
        inf = HelmInference(
            HelmInferenceConfig(model_id="dummy/model", max_new_tokens=11)
        )
        inf.setup()
        inf.generate(["hi"])  # no override -> falls back to config value
        assert seen["n"] == 11
        inf.teardown()

    def test_zero_tokens_is_valid_and_forwarded(self, monkeypatch):
        seen = {}

        def gen(self, prompts, max_new_tokens):
            seen["n"] = max_new_tokens
            return ["" for _ in prompts]

        _install_helm_handle(monkeypatch, gen)
        inf = HelmInference(HelmInferenceConfig(model_id="dummy/model"))
        inf.setup()
        # max_new_tokens=0 is a documented, accepted value (empty generation).
        assert inf.generate(["hi"], max_new_tokens=0) == [""]
        assert seen["n"] == 0
        inf.teardown()

    def test_negative_max_new_tokens_rejected(self, monkeypatch):
        _install_helm_handle(monkeypatch, lambda self, p, n: ["x"])
        inf = HelmInference(HelmInferenceConfig(model_id="dummy/model"))
        inf.setup()
        with pytest.raises(ValueError, match="max_new_tokens must be an integer"):
            inf.generate(["hi"], max_new_tokens=-1)
        inf.teardown()

    def test_bool_max_new_tokens_rejected(self, monkeypatch):
        _install_helm_handle(monkeypatch, lambda self, p, n: ["x"])
        inf = HelmInference(HelmInferenceConfig(model_id="dummy/model"))
        inf.setup()
        with pytest.raises(ValueError, match="max_new_tokens must be an integer"):
            inf.generate(["hi"], max_new_tokens=True)
        inf.teardown()

    def test_non_int_max_new_tokens_rejected(self, monkeypatch):
        _install_helm_handle(monkeypatch, lambda self, p, n: ["x"])
        inf = HelmInference(HelmInferenceConfig(model_id="dummy/model"))
        inf.setup()
        with pytest.raises(ValueError, match="max_new_tokens must be an integer"):
            inf.generate(["hi"], max_new_tokens=1.5)
        inf.teardown()


class TestPromptCoercionAndValidation:
    """Bare str coercion; non-string sequence elements rejected."""

    def test_bare_str_prompt_coerced_to_single_element_list(self, monkeypatch):
        seen = {}

        def gen(self, prompts, max_new_tokens):
            seen["prompts"] = prompts
            return ["out" for _ in prompts]

        _install_helm_handle(monkeypatch, gen)
        inf = HelmInference(HelmInferenceConfig(model_id="dummy/model"))
        inf.setup()
        result = inf.generate("just a string")
        assert seen["prompts"] == ["just a string"]
        assert result == ["out"]
        inf.teardown()

    def test_non_string_sequence_element_rejected(self, monkeypatch):
        _install_helm_handle(monkeypatch, lambda self, p, n: ["x"])
        inf = HelmInference(HelmInferenceConfig(model_id="dummy/model"))
        inf.setup()
        with pytest.raises(ValueError, match="sequence of strings"):
            inf.generate(["ok", 123])
        inf.teardown()


class TestLifecycle:
    """setup idempotency, teardown reentrancy, failure cleanup, context manager."""

    def test_setup_twice_binds_handle_once(self, monkeypatch):
        rec = _install_helm_handle(monkeypatch, lambda self, p, n: ["x"])
        inf = HelmInference(HelmInferenceConfig(model_id="dummy/model"))
        inf.setup()
        handle_first = inf._handle
        inf.setup()  # second call is a no-op: handle already bound
        assert inf._handle is handle_first
        assert rec["compile_calls"] == 1
        assert rec["setup_calls"] == 1
        inf.teardown()

    def test_teardown_twice_is_noop_second_time(self, monkeypatch):
        _install_helm_handle(monkeypatch, lambda self, p, n: ["x"])
        inf = HelmInference(HelmInferenceConfig(model_id="dummy/model"))
        inf.setup()
        inf.teardown()
        assert inf._handle is None
        assert inf.partition_plan is None
        assert inf.routed_to_vllm is False
        # Second teardown must not raise even though there is nothing to release.
        inf.teardown()
        assert inf._handle is None

    def test_generate_after_teardown_raises_runtimeerror(self, monkeypatch):
        _install_helm_handle(monkeypatch, lambda self, p, n: ["x"])
        inf = HelmInference(HelmInferenceConfig(model_id="dummy/model"))
        inf.setup()
        inf.teardown()
        with pytest.raises(RuntimeError, match="setup"):
            inf.generate(["hi"])

    def test_failed_setup_leaves_object_torn_down_and_propagates(self, monkeypatch):
        boom = RuntimeError("compile blew up")

        def fake_compile(_config):
            raise boom

        monkeypatch.setattr(
            "helm.runtime.inference._compile_partition_plan", fake_compile
        )
        inf = HelmInference(HelmInferenceConfig(model_id="dummy/model"))
        with pytest.raises(RuntimeError, match="compile blew up") as ei:
            inf.setup()
        # The original exception propagates unchanged.
        assert ei.value is boom
        # Failed setup releases partially constructed state.
        assert inf._handle is None
        assert inf.partition_plan is None
        assert inf.routed_to_vllm is False

    def test_close_is_alias_of_teardown(self, monkeypatch):
        _install_helm_handle(monkeypatch, lambda self, p, n: ["x"])
        inf = HelmInference(HelmInferenceConfig(model_id="dummy/model"))
        inf.setup()
        assert inf._handle is not None
        inf.close()
        assert inf._handle is None

    def test_context_manager_sets_up_and_tears_down(self, monkeypatch):
        rec = _install_helm_handle(monkeypatch, lambda self, p, n: ["out"])
        inf = HelmInference(HelmInferenceConfig(model_id="dummy/model"))
        with inf as ctx:
            assert ctx is inf
            assert inf._handle is not None
            assert rec["setup_calls"] == 1
            assert inf.generate(["hi"]) == ["out"]
        # __exit__ tore the handle down.
        assert inf._handle is None

    def test_context_manager_tears_down_even_when_body_raises(self, monkeypatch):
        _install_helm_handle(monkeypatch, lambda self, p, n: ["out"])
        inf = HelmInference(HelmInferenceConfig(model_id="dummy/model"))
        with pytest.raises(ValueError, match="body failure"):
            with inf:
                assert inf._handle is not None
                raise ValueError("body failure")
        # teardown still ran despite the exception inside the with-block.
        assert inf._handle is None
        assert inf.partition_plan is None
