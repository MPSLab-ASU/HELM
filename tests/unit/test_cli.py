import pytest
import torch

from helm.cli import _build_parser, _to_benchmark_argv, _to_inference_config, main


def test_cli_dry_run_uses_existing_entrypoint(tmp_path):
    artifacts_dir = tmp_path / "artifacts"

    main([
        "--model",
        "dummy",
        "--mode",
        "dry_run",
        "--save-artifacts-dir",
        str(artifacts_dir),
    ])

    assert (artifacts_dir / "run_001" / "metrics.json").is_file()


def test_cli_rejects_unimplemented_execute_full_mode():
    parser = _build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--model", "dummy", "--mode", "execute_full"])


def test_public_inference_exports():
    from helm import HelmInference, HelmInferenceConfig, helm_backend
    from helm.runtime.inference import HelmInference as Implementation
    from helm.runtime.inference import HelmInferenceConfig as ConfigImplementation

    assert HelmInference is Implementation
    assert HelmInferenceConfig is ConfigImplementation
    assert callable(helm_backend)


def test_cli_preserves_diagnostic_default_and_native_generation_backend():
    args = _build_parser().parse_args(["--model", "dummy"])

    assert args.mode == "plan"
    assert args.backend == "auto"
    assert _to_inference_config(args).route_to_vllm_when_all_gpu is None
    assert _to_benchmark_argv(args)[:2] == ["--mode", "plan"]


def test_cli_generation_config_maps_runtime_options():
    args = _build_parser().parse_args([
        "--model", "local/model", "--mode", "generate",
        "--backend", "router", "--dtype", "float32",
        "--max-input-tokens", "23", "--max-new-tokens", "7",
        "--compiler-plan", "manual", "--compiler-cpu-layers", "0:2",
        "--compiler-gpu-layers", "3:4", "--cpu-threads", "2", "--kv-offload",
    ])

    config = _to_inference_config(args)
    assert config.model_id == "local/model"
    assert config.dtype == torch.float32
    assert config.max_input_tokens == 23
    assert config.max_new_tokens == 7
    assert config.plan_mode == "manual"
    assert config.cpu_layers == "0:2"
    assert config.gpu_layers == "3:4"
    assert config.cpu_threads == 2
    assert config.kv_offload is True
    assert config.route_to_vllm_when_all_gpu is True


@pytest.mark.parametrize("backend,expected", [("auto", None), ("router", True), ("helm", False)])
def test_cli_backend_maps_to_routing_mode(backend, expected):
    args = _build_parser().parse_args(["--model", "dummy", "--backend", backend])
    assert _to_inference_config(args).route_to_vllm_when_all_gpu is expected


@pytest.mark.parametrize("backend", ["vllm", "unknown"])
def test_cli_rejects_unknown_backend(backend):
    with pytest.raises(SystemExit):
        _build_parser().parse_args([
            "--model", "dummy", "--mode", "generate", "--backend", backend,
        ])


@pytest.mark.parametrize("flag,value", [
    ("--max-input-tokens", "0"),
    ("--max-new-tokens", "-1"),
    ("--cpu-threads", "0"),
])
def test_cli_rejects_invalid_generation_config_before_loading(flag, value, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--model", "unused-local-model", "--mode", "generate", flag, value])

    assert exc.value.code == 2
    assert "error:" in capsys.readouterr().err
