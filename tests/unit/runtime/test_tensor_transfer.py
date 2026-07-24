"""
Unit tests for move_tensor (helm/runtime/tensor_transfer.py).
"""
import pytest
import torch

from helm.runtime.tensor_transfer import move_tensor


# ──────────────────────────────────────────────────────────────────────────────
# Non-tensor passthrough
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("val", [42, "hello", None, 3.14, True])
def test_non_tensor_passthrough(val):
    assert move_tensor(val, "cpu") is val


# ──────────────────────────────────────────────────────────────────────────────
# Same-device no-copy
# ──────────────────────────────────────────────────────────────────────────────

def test_cpu_to_cpu_returns_same_object():
    t = torch.zeros(3)
    assert move_tensor(t, "cpu") is t


def test_cpu_to_cpu_values_unchanged():
    t = torch.tensor([1.0, 2.0, 3.0])
    result = move_tensor(t, "cpu")
    assert torch.equal(result, t)


# ──────────────────────────────────────────────────────────────────────────────
# Nested containers
# ──────────────────────────────────────────────────────────────────────────────

def test_dict_of_cpu_tensors():
    d = {"a": torch.zeros(2), "b": torch.ones(3)}
    result = move_tensor(d, "cpu")
    assert isinstance(result, dict)
    assert set(result.keys()) == {"a", "b"}
    assert all(isinstance(v, torch.Tensor) for v in result.values())


def test_list_of_cpu_tensors():
    lst = [torch.zeros(2), torch.ones(3)]
    result = move_tensor(lst, "cpu")
    assert isinstance(result, list)
    assert len(result) == 2


def test_tuple_of_cpu_tensors():
    t = (torch.zeros(2), torch.ones(3))
    result = move_tensor(t, "cpu")
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_nested_dict_list():
    nested = {"x": [torch.zeros(2), torch.ones(2)], "y": torch.zeros(1)}
    result = move_tensor(nested, "cpu")
    assert isinstance(result["x"], list)
    assert len(result["x"]) == 2


def test_mixed_dict_non_tensor_values():
    d = {"a": torch.zeros(2), "b": 42, "c": "hello"}
    result = move_tensor(d, "cpu")
    assert result["b"] == 42
    assert result["c"] == "hello"
    assert isinstance(result["a"], torch.Tensor)


def test_empty_dict():
    assert move_tensor({}, "cpu") == {}


def test_empty_list():
    assert move_tensor([], "cpu") == []


def test_empty_tuple():
    assert move_tensor((), "cpu") == ()


def test_nested_tuple_of_dicts():
    t = ({"x": torch.zeros(2)}, {"y": torch.ones(2)})
    result = move_tensor(t, "cpu")
    assert isinstance(result, tuple)
    assert all(isinstance(v, dict) for v in result)


# ──────────────────────────────────────────────────────────────────────────────
# CUDA tests (skipped if no GPU)
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cpu_to_cuda():
    t = torch.zeros(3)
    result = move_tensor(t, "cuda:0")
    assert result.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_same_device_no_copy():
    t = torch.zeros(3, device="cuda:0")
    assert move_tensor(t, "cuda:0") is t


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_same_device_no_index_no_copy():
    t = torch.zeros(3, device="cuda:0")
    # "cuda" without index should also be treated as same device
    result = move_tensor(t, "cuda")
    assert result.device.type == "cuda"


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_bare_cuda_targets_current_device_not_gpu_zero():
    source = torch.tensor([1.0, 2.0], device="cuda:0")
    with torch.cuda.device(1):
        result = move_tensor(source, "cuda")
    assert result.device == torch.device("cuda:1")
    assert torch.equal(result.cpu(), source.cpu())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_to_cpu():
    t = torch.zeros(3, device="cuda:0")
    result = move_tensor(t, "cpu")
    assert result.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_dict_cuda_tensors_moved():
    d = {"a": torch.zeros(2), "b": torch.ones(3)}
    result = move_tensor(d, "cuda:0")
    for v in result.values():
        assert v.device.type == "cuda"
