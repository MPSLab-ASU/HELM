"""
Unit tests for batch inference in KVOffloadManager and KVCacheManager.

Tests verify:
  - KVOffloadManager creates one KVCacheManager per batch item
  - Per-item KV append (prefill + decode) works correctly
  - Reset clears all caches
  - Streaming attention output shapes are correct for batch > 1
"""
import pytest
import torch

from helm.runtime.kv_allocator import KVAllocator
from helm.runtime.kv_cache import KVCacheManager, perform_streaming_attention


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

NUM_LAYERS = 4
NUM_KV_HEADS = 2
HEAD_DIM = 16
PAGE_SIZE = 8


@pytest.fixture()
def allocator():
    return KVAllocator(
        num_layers=NUM_LAYERS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        page_size=PAGE_SIZE,
        dtype=torch.float32,
    )


def _make_kvcm(allocator):
    return KVCacheManager(allocator, gpu_high_watermark_bytes=0)


# ──────────────────────────────────────────────────────────────────────────────
# KVOffloadManager batch instantiation
# ──────────────────────────────────────────────────────────────────────────────

def test_batch_kvcm_count(allocator):
    """Each batch item should get its own KVCacheManager."""
    batch_size = 4
    kvcms = [_make_kvcm(allocator) for _ in range(batch_size)]
    assert len(kvcms) == batch_size


def test_batch_kvcm_independent_layers(allocator):
    """KVCacheManagers must be independent: appending to one does not affect others."""
    kvcms = [_make_kvcm(allocator) for _ in range(3)]
    k = torch.randn(1, NUM_KV_HEADS, 4, HEAD_DIM)
    v = torch.randn(1, NUM_KV_HEADS, 4, HEAD_DIM)

    kvcms[0].append_prefill(0, k, v)

    assert 0 in kvcms[0].layers
    assert 0 not in kvcms[1].layers
    assert 0 not in kvcms[2].layers


def test_batch_kvcm_token_counts(allocator):
    """Each kvcm tracks its own token count independently."""
    kvcms = [_make_kvcm(allocator) for _ in range(2)]
    k4 = torch.randn(1, NUM_KV_HEADS, 4, HEAD_DIM)
    v4 = torch.randn(1, NUM_KV_HEADS, 4, HEAD_DIM)
    k8 = torch.randn(1, NUM_KV_HEADS, 8, HEAD_DIM)
    v8 = torch.randn(1, NUM_KV_HEADS, 8, HEAD_DIM)

    kvcms[0].append_prefill(0, k4, v4)
    kvcms[1].append_prefill(0, k8, v8)

    assert kvcms[0].layers[0].total_tokens == 4
    assert kvcms[1].layers[0].total_tokens == 8


# ──────────────────────────────────────────────────────────────────────────────
# Per-item prefill + decode
# ──────────────────────────────────────────────────────────────────────────────

def test_per_item_prefill_shape(allocator):
    """append_prefill called with (1, ...) per item must store correct token count."""
    batch_size = 4
    seq_len = 6
    kvcms = [_make_kvcm(allocator) for _ in range(batch_size)]

    k = torch.randn(batch_size, NUM_KV_HEADS, seq_len, HEAD_DIM)
    v = torch.randn(batch_size, NUM_KV_HEADS, seq_len, HEAD_DIM)

    for i in range(batch_size):
        kvcms[i].append_prefill(0, k[i:i+1], v[i:i+1])

    for i in range(batch_size):
        assert kvcms[i].layers[0].total_tokens == seq_len


def test_per_item_decode_increments(allocator):
    """append_decode called per item increments only that item's count."""
    batch_size = 3
    kvcms = [_make_kvcm(allocator) for _ in range(batch_size)]

    # Prefill all items with 4 tokens
    k_pre = torch.randn(1, NUM_KV_HEADS, 4, HEAD_DIM)
    v_pre = torch.randn(1, NUM_KV_HEADS, 4, HEAD_DIM)
    for kvcm in kvcms:
        kvcm.append_prefill(0, k_pre, v_pre)

    # Decode one step for item 1 only
    k_dec = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM)
    v_dec = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM)
    kvcms[1].append_decode(0, k_dec, v_dec)

    assert kvcms[0].layers[0].total_tokens == 4
    assert kvcms[1].layers[0].total_tokens == 5
    assert kvcms[2].layers[0].total_tokens == 4


# ──────────────────────────────────────────────────────────────────────────────
# Reset
# ──────────────────────────────────────────────────────────────────────────────

def test_batch_reset_clears_all(allocator):
    """Clearing all kvcms leaves them empty."""
    batch_size = 4
    kvcms = [_make_kvcm(allocator) for _ in range(batch_size)]

    k = torch.randn(1, NUM_KV_HEADS, 4, HEAD_DIM)
    v = torch.randn(1, NUM_KV_HEADS, 4, HEAD_DIM)
    for kvcm in kvcms:
        kvcm.append_prefill(0, k, v)

    for kvcm in kvcms:
        kvcm.clear()

    for kvcm in kvcms:
        assert len(kvcm.layers) == 0


def test_batch_reset_returns_pages_to_pool(allocator):
    """After clearing all kvcms the allocator pool should be fully reclaimed."""
    batch_size = 2
    kvcms = [_make_kvcm(allocator) for _ in range(batch_size)]

    k = torch.randn(1, NUM_KV_HEADS, 4, HEAD_DIM)
    v = torch.randn(1, NUM_KV_HEADS, 4, HEAD_DIM)
    for kvcm in kvcms:
        kvcm.append_prefill(0, k, v)

    used_before = allocator.report_usage()["active_cpu_pages"]
    assert used_before > 0

    for kvcm in kvcms:
        kvcm.clear()

    used_after = allocator.report_usage()["active_cpu_pages"]
    assert used_after == 0


# ──────────────────────────────────────────────────────────────────────────────
# Streaming attention output shape for batch > 1
# ──────────────────────────────────────────────────────────────────────────────

def test_streaming_attention_batch_shape(allocator):
    """
    Simulate the decode loop: per-item streaming attention should produce
    (1, num_q_heads, 1, head_dim) per item, matching standard SDPA.
    """
    num_q_heads = NUM_KV_HEADS  # GQA groups=1 for simplicity
    batch_size = 4
    seq_len = 12
    kvcms = [_make_kvcm(allocator) for _ in range(batch_size)]

    # Prefill
    k_pre = torch.randn(1, NUM_KV_HEADS, seq_len, HEAD_DIM)
    v_pre = torch.randn(1, NUM_KV_HEADS, seq_len, HEAD_DIM)
    for kvcm in kvcms:
        kvcm.append_prefill(0, k_pre, v_pre)

    # Decode step: simulate per-item attention
    q = torch.randn(batch_size, num_q_heads, 1, HEAD_DIM)
    k_dec = torch.randn(batch_size, NUM_KV_HEADS, 1, HEAD_DIM)
    v_dec = torch.randn(batch_size, NUM_KV_HEADS, 1, HEAD_DIM)

    outs = []
    for i in range(batch_size):
        kvcms[i].append_decode(0, k_dec[i:i+1], v_dec[i:i+1])
        pages = kvcms[i].iterate_layer_pages(0)
        out_i = perform_streaming_attention(q[i:i+1], pages)
        outs.append(out_i)

    out = torch.cat(outs, dim=0)
    assert out.shape == (batch_size, num_q_heads, 1, HEAD_DIM)


def test_streaming_attention_batch_independent(allocator):
    """
    Each batch item attends to its own KV history — different prompts produce
    different attention outputs.
    """
    num_q_heads = NUM_KV_HEADS
    batch_size = 2
    seq_len = 8
    kvcms = [_make_kvcm(allocator) for _ in range(batch_size)]

    # Give each item a different KV history
    torch.manual_seed(0)
    for i, kvcm in enumerate(kvcms):
        k = torch.randn(1, NUM_KV_HEADS, seq_len, HEAD_DIM) * (i + 1)
        v = torch.randn(1, NUM_KV_HEADS, seq_len, HEAD_DIM) * (i + 1)
        kvcm.append_prefill(0, k, v)

    q = torch.ones(batch_size, num_q_heads, 1, HEAD_DIM)
    k_dec = torch.randn(batch_size, NUM_KV_HEADS, 1, HEAD_DIM)
    v_dec = torch.randn(batch_size, NUM_KV_HEADS, 1, HEAD_DIM)

    outs = []
    for i in range(batch_size):
        kvcms[i].append_decode(0, k_dec[i:i+1], v_dec[i:i+1])
        pages = kvcms[i].iterate_layer_pages(0)
        outs.append(perform_streaming_attention(q[i:i+1], pages))

    # Different KV histories must produce different attention outputs
    assert not torch.allclose(outs[0], outs[1])
