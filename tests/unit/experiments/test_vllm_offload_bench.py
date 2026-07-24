from types import SimpleNamespace


def test_build_targets_uses_custom_model_id_for_smoke_runs():
    from experiments import vllm_offload_bench as bench

    args = SimpleNamespace(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        name=None,
        offload_gb=0.0,
        only=None,
    )

    assert bench.build_targets(args) == [
        ("Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct", 0.0)
    ]


def test_build_targets_keeps_existing_only_filter_for_paper_models():
    from experiments import vllm_offload_bench as bench

    args = SimpleNamespace(
        model_id=None,
        name=None,
        offload_gb=0.0,
        only="LLaMA-2-13B,OLMo-2-13B",
    )

    assert bench.build_targets(args) == [
        ("LLaMA-2-13B", "NousResearch/Llama-2-13b-hf", 8),
        ("OLMo-2-13B", "allenai/OLMo-2-1124-13B-Instruct", 8),
    ]
