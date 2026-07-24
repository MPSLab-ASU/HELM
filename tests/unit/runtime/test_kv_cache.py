"""
Unit tests for KVCacheManager and LayerKVCache (helm/runtime/kv_cache.py).
"""
import pytest
import torch

from helm.runtime.kv_allocator import KVAllocator
from helm.runtime.kv_cache import KVCacheManager, LayerKVCache


# ──────────────────────────────────────────────────────────────────────────────
# Helpers / fixtures
# ──────────────────────────────────────────────────────────────────────────────

NUM_HEADS = 2
HEAD_DIM = 8
PAGE_SIZE = 4
NUM_LAYERS = 4


@pytest.fixture()
def allocator():
    return KVAllocator(
        num_layers=NUM_LAYERS,
        num_kv_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        page_size=PAGE_SIZE,
        dtype=torch.float32,
    )


@pytest.fixture()
def cache(allocator):
    # watermark=0 means eviction is triggered every time, but since pages are
    # on CPU (not cuda), evict_pages() won't find any GPU pages to move.
    mgr = KVCacheManager(allocator, gpu_high_watermark_bytes=0)
    mgr.initialize_layer_caches(num_layers=NUM_LAYERS)
    return mgr


def _kv(seq_len):
    k = torch.randn(1, NUM_HEADS, seq_len, HEAD_DIM)
    v = torch.randn(1, NUM_HEADS, seq_len, HEAD_DIM)
    return k, v


# ──────────────────────────────────────────────────────────────────────────────
# LayerKVCache
# ──────────────────────────────────────────────────────────────────────────────

def test_layer_cache_append_page(allocator):
    layer = LayerKVCache(layer_id=0)
    page = allocator.allocate(torch.device("cpu"))
    layer.append_page(page)
    assert len(layer.pages) == 1
    assert layer.tail_page is page


def test_layer_cache_tail_updates(allocator):
    layer = LayerKVCache(layer_id=0)
    p1 = allocator.allocate(torch.device("cpu"))
    p2 = allocator.allocate(torch.device("cpu"))
    layer.append_page(p1)
    layer.append_page(p2)
    assert layer.tail_page is p2


# ──────────────────────────────────────────────────────────────────────────────
# initialize_layer_caches
# ──────────────────────────────────────────────────────────────────────────────

def test_initialize_creates_n_layers(cache):
    assert cache.num_layers() == NUM_LAYERS
    assert sorted(cache.layers_keys()) == list(range(NUM_LAYERS))


def test_initial_seq_len_zero(cache):
    assert cache.seq_len() == 0


# ──────────────────────────────────────────────────────────────────────────────
# append_prefill
# ──────────────────────────────────────────────────────────────────────────────

def test_prefill_single_page_when_short(cache):
    k, v = _kv(3)  # 3 < PAGE_SIZE=4
    cache.append_prefill(0, k, v)
    assert len(cache.layers[0].pages) == 1
    assert cache.layers[0].total_tokens == 3


def test_prefill_multiple_pages_when_long(cache):
    k, v = _kv(9)  # ceil(9/4) = 3 pages
    cache.append_prefill(0, k, v)
    assert len(cache.layers[0].pages) == 3
    assert cache.layers[0].total_tokens == 9


def test_prefill_exact_one_page(cache):
    k, v = _kv(PAGE_SIZE)
    cache.append_prefill(0, k, v)
    assert len(cache.layers[0].pages) == 1
    assert cache.layers[0].tail_page.used_tokens == PAGE_SIZE


def test_prefill_data_preserved(cache):
    k, v = _kv(2)
    cache.append_prefill(0, k, v)
    page = cache.layers[0].pages[0]
    assert torch.allclose(page.k_tensor[:, :, :2, :], k)
    assert torch.allclose(page.v_tensor[:, :, :2, :], v)


def test_prefill_page_start_tokens(cache):
    k, v = _kv(8)  # 2 pages of 4
    cache.append_prefill(0, k, v)
    assert cache.layers[0].pages[0].start_token == 0
    assert cache.layers[0].pages[1].start_token == 4


def test_prefill_auto_creates_layer(cache):
    """append_prefill creates the layer if it doesn't exist."""
    del cache.layers[3]
    k, v = _kv(2)
    cache.append_prefill(3, k, v)
    assert 3 in cache.layers


# ──────────────────────────────────────────────────────────────────────────────
# append_decode
# ──────────────────────────────────────────────────────────────────────────────

def test_decode_increments_token_count(cache):
    k, v = _kv(1)
    cache.append_decode(0, k, v)
    assert cache.layers[0].total_tokens == 1


def test_decode_updates_seq_len(cache):
    k, v = _kv(1)
    cache.append_decode(0, k, v)
    assert cache.seq_len() == 1


def test_decode_within_existing_page(cache):
    k_pre, v_pre = _kv(2)
    cache.append_prefill(0, k_pre, v_pre)
    k_dec, v_dec = _kv(1)
    cache.append_decode(0, k_dec, v_dec)
    assert len(cache.layers[0].pages) == 1
    assert cache.layers[0].tail_page.used_tokens == 3


def test_decode_new_page_when_tail_full(cache):
    k_pre, v_pre = _kv(PAGE_SIZE)  # fills one page exactly
    cache.append_prefill(0, k_pre, v_pre)
    k_dec, v_dec = _kv(1)
    cache.append_decode(0, k_dec, v_dec)
    assert len(cache.layers[0].pages) == 2
    assert cache.layers[0].tail_page.used_tokens == 1


def test_decode_multiple_tokens_across_pages(cache):
    for _ in range(PAGE_SIZE + 2):  # fill one page + 2 in new
        k, v = _kv(1)
        cache.append_decode(0, k, v)
    assert len(cache.layers[0].pages) == 2
    assert cache.layers[0].total_tokens == PAGE_SIZE + 2


# ──────────────────────────────────────────────────────────────────────────────
# iterate_layer_pages / seq_len
# ──────────────────────────────────────────────────────────────────────────────

def test_iterate_layer_pages_returns_list(cache):
    k, v = _kv(3)
    cache.append_prefill(0, k, v)
    pages = cache.iterate_layer_pages(0)
    assert isinstance(pages, list)
    assert len(pages) == 1


def test_iterate_missing_layer_returns_empty(cache):
    assert cache.iterate_layer_pages(99) == []


def test_seq_len_from_layer_0(cache):
    k, v = _kv(5)
    cache.append_prefill(0, k, v)
    assert cache.seq_len() == 5


def test_seq_len_zero_with_no_layer_0(cache):
    del cache.layers[0]
    assert cache.seq_len() == 0


# ──────────────────────────────────────────────────────────────────────────────
# report_bytes_per_tier
# ──────────────────────────────────────────────────────────────────────────────

def test_report_bytes_keys(cache):
    r = cache.report_bytes_per_tier()
    assert "gpu_active_bytes" in r
    assert "cpu_active_bytes" in r


def test_report_bytes_cpu_after_cpu_append(cache):
    k, v = _kv(3)
    cache.append_prefill(0, k, v)
    r = cache.report_bytes_per_tier()
    # All pages are on CPU (state == 'CPU')
    assert r["cpu_active_bytes"] >= 0
    assert r["gpu_active_bytes"] == 0


# ──────────────────────────────────────────────────────────────────────────────
# clear
# ──────────────────────────────────────────────────────────────────────────────

def test_clear_resets_all_layers(cache):
    k, v = _kv(3)
    cache.append_prefill(0, k, v)
    cache.clear()
    assert cache.num_layers() == 0
    assert cache.seq_len() == 0


def test_clear_frees_pages_back_to_pool(cache):
    alloc = cache.allocator
    k, v = _kv(PAGE_SIZE)
    cache.append_prefill(0, k, v)
    free_before = len(alloc.cpu_free_pool)
    cache.clear()
    assert len(alloc.cpu_free_pool) > free_before


# ──────────────────────────────────────────────────────────────────────────────
# Eviction (CPU-only: evict_pages finds no GPU pages — just checks no crash)
# ──────────────────────────────────────────────────────────────────────────────

def test_evict_pages_no_gpu_pages_no_crash(cache):
    k, v = _kv(8)
    cache.append_prefill(0, k, v)
    cache.evict_pages(99999)  # no GPU pages → nothing to evict, no exception


def test_watermark_no_crash_cpu_only(cache):
    k, v = _kv(8)
    cache.append_prefill(0, k, v)
    # _enforce_residency_policy runs but finds no GPU pages
    r = cache.report_bytes_per_tier()
    assert r is not None


# ──────────────────────────────────────────────────────────────────────────────
# CUDA eviction tests
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_evict_gpu_pages_to_cpu():
    alloc = KVAllocator(num_layers=2, num_kv_heads=2, head_dim=8,
                        page_size=4, dtype=torch.float32)
    # Zero watermark → evict everything possible after each append
    mgr = KVCacheManager(alloc, gpu_high_watermark_bytes=0)
    mgr.initialize_layer_caches(2)
    gpu = torch.device("cuda:0")
    k = torch.randn(1, 2, 8, 8, device=gpu)
    v = torch.randn(1, 2, 8, 8, device=gpu)
    k_snapshot = k.detach().cpu().clone()
    v_snapshot = v.detach().cpu().clone()
    mgr.append_prefill(0, k, v)
    # Force eviction
    mgr.evict_pages(999999)
    # All fully-written non-tail pages should be on CPU — and the K/V data
    # must survive the GPU->CPU round-trip bit-exactly.
    token_cursor = 0
    for page in mgr.layers[0].pages[:-1]:
        assert page.state in ("CPU", "FREE"), f"Expected CPU, got {page.state}"
        if page.state == "CPU" and page.used_tokens:
            used = page.used_tokens
            assert torch.equal(
                page.k_tensor[:, :, :used, :].cpu(),
                k_snapshot[:, :, token_cursor:token_cursor + used, :],
            ), "evicted K page content differs from what was appended"
            assert torch.equal(
                page.v_tensor[:, :, :used, :].cpu(),
                v_snapshot[:, :, token_cursor:token_cursor + used, :],
            ), "evicted V page content differs from what was appended"
        token_cursor += page.used_tokens


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_tail_page_protected_from_eviction():
    alloc = KVAllocator(num_layers=2, num_kv_heads=2, head_dim=8,
                        page_size=4, dtype=torch.float32)
    mgr = KVCacheManager(alloc, gpu_high_watermark_bytes=0)
    mgr.initialize_layer_caches(2)
    gpu = torch.device("cuda:0")
    k = torch.randn(1, 2, 4, 8, device=gpu)
    v = torch.randn(1, 2, 4, 8, device=gpu)
    mgr.append_prefill(0, k, v)
    tail = mgr.layers[0].tail_page
    # Try to evict everything
    mgr.evict_pages(999999)
    # The active tail page must NOT be evicted: the evict_pages guard checks
    # `page != layer.tail_page`. Assert the ORIGINAL tail object's state, not
    # merely that some tail reference exists.
    assert tail is mgr.layers[0].tail_page, "tail_page reference must survive eviction"
    assert tail.state == "GPU", f"tail page was evicted (state={tail.state})"
    assert tail.device.type == "cuda"


# ──────────────────────────────────────────────────────────────────────────────
# Paging stats counters (instrumentation)
#
# Process-global counters for KV page eviction (GPU->CPU) and prefetch (CPU->GPU)
# so a benchmark can reset before a single request and read the totals after.
# ──────────────────────────────────────────────────────────────────────────────

_ZERO_STATS = {
    "evict_calls": 0, "pages_evicted": 0, "bytes_evicted": 0,
    "prefetch_calls": 0, "pages_prefetched": 0, "bytes_prefetched": 0,
}


def test_paging_stats_reset_zeroes_all():
    from helm.runtime.kv_cache import reset_paging_stats, get_paging_stats
    reset_paging_stats()
    assert get_paging_stats() == _ZERO_STATS


def test_paging_stats_record_eviction_accumulates():
    from helm.runtime.kv_cache import reset_paging_stats, get_paging_stats, _record_eviction
    reset_paging_stats()
    _record_eviction(n_pages=2, n_bytes=1000)
    _record_eviction(n_pages=3, n_bytes=1500)
    s = get_paging_stats()
    assert s["evict_calls"] == 2
    assert s["pages_evicted"] == 5
    assert s["bytes_evicted"] == 2500


def test_paging_stats_record_prefetch_accumulates():
    from helm.runtime.kv_cache import reset_paging_stats, get_paging_stats, _record_prefetch
    reset_paging_stats()
    _record_prefetch(n_pages=4, n_bytes=4096)
    s = get_paging_stats()
    assert s["prefetch_calls"] == 1
    assert s["pages_prefetched"] == 4
    assert s["bytes_prefetched"] == 4096


def test_paging_stats_zero_record_is_noop():
    from helm.runtime.kv_cache import (
        reset_paging_stats, get_paging_stats, _record_eviction, _record_prefetch,
    )
    reset_paging_stats()
    _record_eviction(0, 0)
    _record_prefetch(0, 0)
    assert get_paging_stats() == _ZERO_STATS


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_evict_pages_increments_counters():
    """Wiring guard (GPU host): real GPU->CPU eviction must bump the counters."""
    from helm.runtime.kv_cache import reset_paging_stats, get_paging_stats
    alloc = KVAllocator(num_layers=2, num_kv_heads=2, head_dim=8,
                        page_size=4, dtype=torch.float32)
    # High watermark so append_prefill does NOT evict during insertion; the
    # explicit evict_pages() call below is what we are counting.
    mgr = KVCacheManager(alloc, gpu_high_watermark_bytes=10**12)
    mgr.initialize_layer_caches(2)
    gpu = torch.device("cuda:0")
    k = torch.randn(1, 2, 8, 8, device=gpu)
    v = torch.randn(1, 2, 8, 8, device=gpu)
    mgr.append_prefill(0, k, v)
    reset_paging_stats()
    mgr.evict_pages(999999)
    s = get_paging_stats()
    assert s["pages_evicted"] >= 1
    assert s["bytes_evicted"] > 0
    assert s["evict_calls"] >= 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_streaming_attention_counts_prefetched_cpu_pages():
    """Wiring guard (GPU host): a GPU-side attention call over CPU-resident pages
    must record one prefetch of those pages."""
    from helm.runtime.kv_cache import (
        KVPage, perform_streaming_attention, reset_paging_stats, get_paging_stats,
    )

    def _mk(dev_str, state, start):
        dev = torch.device(dev_str)
        k = torch.randn(1, 2, 4, 8, device=dev)
        v = torch.randn(1, 2, 4, 8, device=dev)
        return KVPage(page_id=start, layer_id=0, start_token=start, used_tokens=4,
                      capacity_tokens=4, k_tensor=k, v_tensor=v, device=dev, state=state)

    pages = [_mk("cuda:0", "GPU", 0), _mk("cpu", "CPU", 4)]  # one GPU, one evicted CPU page
    query = torch.randn(1, 2, 1, 8, device=torch.device("cuda:0"))
    attn_mask = torch.zeros(1, 1, 1, 8, device=torch.device("cuda:0"))
    reset_paging_stats()
    out = perform_streaming_attention(query, pages, attention_mask=attn_mask)
    s = get_paging_stats()
    assert s["prefetch_calls"] == 1
    assert s["pages_prefetched"] == 1
    assert s["bytes_prefetched"] > 0
    assert out.shape == query.shape


# ──────────────────────────────────────────────────────────────────────────────
# Cross-stream lifetime: async CPU→GPU prefetch buffers must be protected with
# record_stream() so the caching allocator cannot recycle them (e.g. for the
# *next* page's H2D copy on the copy stream) while the compute stream's matmul
# is still reading them. Regression for a silent cross-stream data race.
# ──────────────────────────────────────────────────────────────────────────────

def _mk_page(dev_str, state, start, used=4, cap=4):
    dev = torch.device(dev_str)
    k = torch.randn(1, 2, used, 8, device=dev)
    v = torch.randn(1, 2, used, 8, device=dev)
    from helm.runtime.kv_cache import KVPage
    return KVPage(page_id=start, layer_id=0, start_token=start, used_tokens=used,
                  capacity_tokens=cap, k_tensor=k, v_tensor=v, device=dev, state=state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_streaming_attention_records_stream_on_cross_device_buffers_masked_path(monkeypatch):
    """Mixed-residency + attention_mask/softcap takes the online-softmax
    prefetch branch (kv_cache.py ~line 470). Each CPU page copied to GPU on
    the dedicated copy stream must have record_stream() called against the
    compute (current) stream before it is consumed there."""
    from helm.runtime.kv_cache import perform_streaming_attention

    calls = []
    orig_record_stream = torch.Tensor.record_stream

    def spy(self, stream):
        calls.append(stream)
        return orig_record_stream(self, stream)

    monkeypatch.setattr(torch.Tensor, "record_stream", spy)

    pages = [_mk_page("cuda:0", "GPU", 0), _mk_page("cpu", "CPU", 4)]
    query = torch.randn(1, 2, 1, 8, device=torch.device("cuda:0"))
    attn_mask = torch.zeros(1, 1, 1, 8, device=torch.device("cuda:0"))

    out = perform_streaming_attention(query, pages, attention_mask=attn_mask)
    torch.cuda.synchronize()

    # k_buf + v_buf for the one CPU-resident page → 2 record_stream calls.
    # The GPU-resident page never crosses streams, so it must not be recorded.
    assert len(calls) == 2
    current = torch.cuda.current_stream(torch.device("cuda:0"))
    assert all(s == current for s in calls)
    assert out.shape == query.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_streaming_attention_records_stream_on_cross_device_buffers_unmasked_path(monkeypatch):
    """Mixed-residency without a mask/softcap takes the dedicated 'Path 3' branch
    (kv_cache.py ~line 604), which must apply the same record_stream protection."""
    from helm.runtime.kv_cache import perform_streaming_attention

    calls = []
    orig_record_stream = torch.Tensor.record_stream

    def spy(self, stream):
        calls.append(stream)
        return orig_record_stream(self, stream)

    monkeypatch.setattr(torch.Tensor, "record_stream", spy)

    pages = [_mk_page("cuda:0", "GPU", 0), _mk_page("cpu", "CPU", 4)]
    query = torch.randn(1, 2, 1, 8, device=torch.device("cuda:0"))

    out = perform_streaming_attention(query, pages)  # no mask/softcap
    torch.cuda.synchronize()

    assert len(calls) == 2
    current = torch.cuda.current_stream(torch.device("cuda:0"))
    assert all(s == current for s in calls)
    assert out.shape == query.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_kv_copy_stream_memoized_per_device():
    """The copy stream must be a per-device singleton (dict-keyed), not a single
    shared global that silently gets reused for a different device."""
    from helm.runtime.kv_cache import _get_kv_copy_stream

    dev = torch.device("cuda:0")
    s1 = _get_kv_copy_stream(dev)
    s2 = _get_kv_copy_stream(dev)
    assert s1 is s2
    assert s1.device == dev


@pytest.mark.parametrize("capacity", [64, 640, 768, 2048])
def test_build_contiguous_honors_small_capacities(capacity):
    # Regression: capacities below 1024 (the default max_input_tokens + 512)
    # used to skip allocation entirely, silently disabling the all-GPU fast path.
    alloc = KVAllocator(num_layers=2, num_kv_heads=2, head_dim=8,
                        page_size=4, dtype=torch.float32)
    mgr = KVCacheManager(alloc, gpu_high_watermark_bytes=0)
    assert mgr.build_contiguous(num_layers=2, num_kv_heads=2, head_dim=8,
                                capacity=capacity, device=torch.device("cpu"),
                                dtype=torch.float32)
    assert mgr.use_contiguous
    assert mgr.cont_capacity == capacity
    assert mgr.cont_K[0].shape == (1, 2, capacity, 8)
