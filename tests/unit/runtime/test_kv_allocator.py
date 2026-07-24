"""
Unit tests for KVAllocator (helm/runtime/kv_allocator.py).
"""
import pytest
import torch

from helm.runtime.kv_allocator import KVAllocator


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _alloc(**kwargs):
    defaults = dict(num_layers=2, num_kv_heads=2, head_dim=8, page_size=4,
                    dtype=torch.float32)
    defaults.update(kwargs)
    return KVAllocator(**defaults)


CPU = torch.device("cpu")


# ──────────────────────────────────────────────────────────────────────────────
# allocate
# ──────────────────────────────────────────────────────────────────────────────

def test_allocate_cpu_state():
    a = _alloc()
    page = a.allocate(CPU)
    assert page.state == "CPU"
    assert page.device.type == "cpu"


def test_allocate_resets_metadata():
    a = _alloc()
    page = a.allocate(CPU)
    page.layer_id = 99
    page.start_token = 500
    page.used_tokens = 3
    a.free(page)
    page2 = a.allocate(CPU)
    assert page2.layer_id == -1
    assert page2.start_token == 0
    assert page2.used_tokens == 0


def test_allocate_triggers_batch_reserve():
    """Empty pool triggers reserve_batch, growing pool beyond 1."""
    a = _alloc()
    assert a.cpu_total_pages == 0
    a.allocate(CPU)
    assert a.cpu_total_pages > 1


def test_page_shape():
    a = KVAllocator(num_layers=2, num_kv_heads=3, head_dim=16, page_size=8)
    page = a.allocate(CPU)
    assert page.k_tensor.shape == (1, 3, 8, 16)
    assert page.v_tensor.shape == (1, 3, 8, 16)


def test_page_dtype():
    a = _alloc(dtype=torch.float16)
    page = a.allocate(CPU)
    assert page.k_tensor.dtype == torch.float16


def test_page_ids_are_unique():
    a = _alloc()
    p0 = a.allocate(CPU)
    p1 = a.allocate(CPU)
    assert p0.page_id != p1.page_id


# ──────────────────────────────────────────────────────────────────────────────
# free
# ──────────────────────────────────────────────────────────────────────────────

def test_free_state_becomes_free():
    a = _alloc()
    page = a.allocate(CPU)
    a.free(page)
    assert page.state == "FREE"


def test_free_returns_to_pool():
    a = _alloc()
    page = a.allocate(CPU)
    before = len(a.cpu_free_pool)
    a.free(page)
    assert len(a.cpu_free_pool) == before + 1


def test_freed_page_reused():
    a = _alloc()
    p1 = a.allocate(CPU)
    a.free(p1)
    p2 = a.allocate(CPU)
    assert p2 is p1  # pool pops the last freed page


# ──────────────────────────────────────────────────────────────────────────────
# reserve_batch
# ──────────────────────────────────────────────────────────────────────────────

def test_reserve_batch_cpu():
    a = _alloc()
    a.reserve_batch(5, CPU)
    assert a.cpu_total_pages == 5
    assert len(a.cpu_free_pool) == 5


def test_reserve_batch_accumulates():
    a = _alloc()
    a.reserve_batch(3, CPU)
    a.reserve_batch(4, CPU)
    assert a.cpu_total_pages == 7


# ──────────────────────────────────────────────────────────────────────────────
# move_page
# ──────────────────────────────────────────────────────────────────────────────

def test_move_page_same_device_returns_same():
    a = _alloc()
    page = a.allocate(CPU)
    result = a.move_page(page, CPU)
    assert result is page


def test_move_page_preserves_k_v_data():
    a = _alloc()
    page = a.allocate(CPU)
    page.used_tokens = 2
    page.k_tensor[:, :, :2, :] = 42.0
    page.v_tensor[:, :, :2, :] = 99.0
    # Same device → same page
    result = a.move_page(page, CPU)
    assert torch.all(result.k_tensor[:, :, :2, :] == 42.0)
    assert torch.all(result.v_tensor[:, :, :2, :] == 99.0)


# ──────────────────────────────────────────────────────────────────────────────
# report_usage
# ──────────────────────────────────────────────────────────────────────────────

def test_report_usage_keys():
    a = _alloc()
    r = a.report_usage()
    for key in ("gpu_pool_size", "cpu_pool_size", "active_gpu_pages", "active_cpu_pages",
                "gpu_free_pages", "cpu_free_pages", "gpu_reserved_bytes",
                "cpu_reserved_bytes", "gpu_used_bytes", "cpu_used_bytes"):
        assert key in r


def test_report_usage_active_cpu_pages():
    a = _alloc()
    a.allocate(CPU)
    r = a.report_usage()
    assert r["active_cpu_pages"] >= 1


def test_report_usage_bytes_consistent():
    a = _alloc()
    a.allocate(CPU)
    r = a.report_usage()
    assert r["cpu_used_bytes"] <= r["cpu_reserved_bytes"]


# ──────────────────────────────────────────────────────────────────────────────
# CUDA tests
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_allocate_gpu_state():
    a = _alloc()
    page = a.allocate(torch.device("cuda:0"))
    assert page.state == "GPU"
    assert page.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_move_page_cpu_to_gpu_copies_data():
    a = _alloc()
    page = a.allocate(CPU)
    page.layer_id = 2
    page.start_token = 8
    page.used_tokens = 3
    page.k_tensor[:, :, :3, :] = 7.0
    page.v_tensor[:, :, :3, :] = 8.0
    new_page = a.move_page(page, torch.device("cuda:0"))
    assert new_page.device.type == "cuda"
    assert new_page.layer_id == 2
    assert new_page.start_token == 8
    assert new_page.used_tokens == 3
    assert torch.allclose(new_page.k_tensor[:, :, :3, :].cpu(),
                          torch.full((1, 2, 3, 8), 7.0))
    assert torch.allclose(new_page.v_tensor[:, :, :3, :].cpu(),
                          torch.full((1, 2, 3, 8), 8.0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_move_page_cpu_to_gpu_frees_old():
    a = _alloc()
    page = a.allocate(CPU)
    cpu_free_before = len(a.cpu_free_pool)
    a.move_page(page, torch.device("cuda:0"))
    assert len(a.cpu_free_pool) == cpu_free_before + 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_move_page_empty_tokens_no_copy():
    """A page with used_tokens=0 should still move without error."""
    a = _alloc()
    page = a.allocate(CPU)
    page.used_tokens = 0
    new_page = a.move_page(page, torch.device("cuda:0"))
    assert new_page.device.type == "cuda"
    assert new_page.used_tokens == 0
