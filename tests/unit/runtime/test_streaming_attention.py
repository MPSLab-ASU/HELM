"""
Unit tests for perform_streaming_attention (helm/runtime/kv_cache.py).

The online-softmax streaming attention must produce identical results to
standard scaled dot-product attention when given the same K/V data.
"""
import math
import pytest
import torch

from helm.runtime.kv_cache import KVPage, perform_streaming_attention


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _page(k: torch.Tensor, v: torch.Tensor, page_id: int = 0, start_token: int = 0) -> KVPage:
    """Wrap K/V tensors in a KVPage."""
    used = k.size(2)
    return KVPage(
        page_id=page_id,
        layer_id=0,
        start_token=start_token,
        used_tokens=used,
        capacity_tokens=used,
        k_tensor=k.clone(),
        v_tensor=v.clone(),
        device=k.device,
        state="CPU",
    )


def _eager(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale=None) -> torch.Tensor:
    """Reference: standard scaled dot-product attention."""
    if scale is None:
        scale = 1.0 / math.sqrt(q.size(-1))
    w = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    w = torch.softmax(w, dim=-1)
    return torch.matmul(w, v.float()).to(q.dtype)


# ──────────────────────────────────────────────────────────────────────────────
# Correctness vs. eager attention
# ──────────────────────────────────────────────────────────────────────────────

def test_single_page_matches_eager():
    B, H, S, D = 1, 2, 4, 8
    q = torch.randn(B, H, 1, D)
    k = torch.randn(B, H, S, D)
    v = torch.randn(B, H, S, D)
    result = perform_streaming_attention(q, [_page(k, v)])
    expected = _eager(q, k, v)
    assert result.shape == expected.shape
    assert torch.allclose(result, expected, atol=1e-4)


def test_two_pages_match_eager():
    B, H, D = 1, 2, 8
    k1 = torch.randn(B, H, 4, D)
    v1 = torch.randn(B, H, 4, D)
    k2 = torch.randn(B, H, 3, D)
    v2 = torch.randn(B, H, 3, D)
    q = torch.randn(B, H, 1, D)

    result = perform_streaming_attention(q, [_page(k1, v1, 0, 0), _page(k2, v2, 1, 4)])

    k_full = torch.cat([k1, k2], dim=2)
    v_full = torch.cat([v1, v2], dim=2)
    expected = _eager(q, k_full, v_full)
    assert torch.allclose(result, expected, atol=1e-4)


def test_three_pages_match_eager():
    B, H, D = 1, 1, 16
    sizes = [3, 5, 2]
    q = torch.randn(B, H, 1, D)
    pages, ks, vs, start = [], [], [], 0
    for i, sz in enumerate(sizes):
        k = torch.randn(B, H, sz, D)
        v = torch.randn(B, H, sz, D)
        ks.append(k)
        vs.append(v)
        pages.append(_page(k, v, i, start))
        start += sz
    result = perform_streaming_attention(q, pages)
    expected = _eager(q, torch.cat(ks, 2), torch.cat(vs, 2))
    assert torch.allclose(result, expected, atol=1e-4)


# ──────────────────────────────────────────────────────────────────────────────
# Edge cases
# ──────────────────────────────────────────────────────────────────────────────

def test_no_pages_returns_zeros():
    B, H, D = 1, 2, 8
    q = torch.randn(B, H, 1, D)
    result = perform_streaming_attention(q, [])
    assert result.shape == (B, H, 1, D)
    assert torch.all(result == 0)


def test_page_with_zero_used_tokens_skipped():
    B, H, D = 1, 2, 8
    q = torch.randn(B, H, 1, D)
    k = torch.randn(B, H, 4, D)
    v = torch.randn(B, H, 4, D)
    empty = _page(k, v)
    empty.used_tokens = 0
    result = perform_streaming_attention(q, [empty])
    assert result.shape == (B, H, 1, D)
    # No tokens → denominator is ~0 → output zeros
    assert torch.allclose(result, torch.zeros_like(result), atol=1e-6)


def test_mixed_empty_and_nonempty_page():
    """One empty page followed by one real page — result should equal single-page."""
    B, H, D = 1, 2, 8
    q = torch.randn(B, H, 1, D)
    k = torch.randn(B, H, 4, D)
    v = torch.randn(B, H, 4, D)
    empty = _page(torch.zeros(B, H, 4, D), torch.zeros(B, H, 4, D), page_id=0)
    empty.used_tokens = 0
    real = _page(k, v, page_id=1)
    result_two = perform_streaming_attention(q, [empty, real])
    result_one = perform_streaming_attention(q, [real])
    assert torch.allclose(result_two, result_one, atol=1e-5)


# ──────────────────────────────────────────────────────────────────────────────
# Custom scale
# ──────────────────────────────────────────────────────────────────────────────

def test_custom_scale():
    B, H, D = 1, 1, 4
    q = torch.randn(B, H, 1, D)
    k = torch.randn(B, H, 3, D)
    v = torch.randn(B, H, 3, D)
    scale = 0.25
    result = perform_streaming_attention(q, [_page(k, v)], scale=scale)
    expected = _eager(q, k, v, scale=scale)
    assert torch.allclose(result, expected, atol=1e-4)


def test_default_scale_is_inv_sqrt_head_dim():
    """Default scale = 1/sqrt(D), same as standard attention."""
    B, H, D = 1, 1, 16
    q = torch.randn(B, H, 1, D)
    k = torch.randn(B, H, 4, D)
    v = torch.randn(B, H, 4, D)
    result_default = perform_streaming_attention(q, [_page(k, v)])
    result_explicit = perform_streaming_attention(q, [_page(k, v)],
                                                  scale=1.0 / math.sqrt(D))
    assert torch.allclose(result_default, result_explicit, atol=1e-5)


# ──────────────────────────────────────────────────────────────────────────────
# Dtype preservation
# ──────────────────────────────────────────────────────────────────────────────

def test_output_dtype_matches_query_float32():
    B, H, D = 1, 2, 8
    q = torch.randn(B, H, 1, D, dtype=torch.float32)
    k = torch.randn(B, H, 4, D, dtype=torch.float32)
    v = torch.randn(B, H, 4, D, dtype=torch.float32)
    result = perform_streaming_attention(q, [_page(k, v)])
    assert result.dtype == torch.float32


def test_output_dtype_matches_query_float16():
    B, H, D = 1, 2, 8
    q = torch.randn(B, H, 1, D, dtype=torch.float16)
    k = torch.randn(B, H, 4, D, dtype=torch.float16)
    v = torch.randn(B, H, 4, D, dtype=torch.float16)
    result = perform_streaming_attention(q, [_page(k, v)])
    assert result.dtype == torch.float16


def test_output_shape_matches_query():
    B, H, Sq, D = 2, 4, 1, 16
    q = torch.randn(B, H, Sq, D)
    k = torch.randn(B, H, 6, D)
    v = torch.randn(B, H, 6, D)
    result = perform_streaming_attention(q, [_page(k, v)])
    assert result.shape == (B, H, Sq, D)


# ──────────────────────────────────────────────────────────────────────────────
# Grouped-Query Attention (GQA)
# ──────────────────────────────────────────────────────────────────────────────

def test_gqa_matches_eager_expanded():
    """GQA: num_kv_heads < num_q_heads — K/V are repeat_interleaved to match."""
    B, Hq, Hkv, S, D = 1, 4, 2, 6, 8
    q = torch.randn(B, Hq, 1, D)
    k = torch.randn(B, Hkv, S, D)
    v = torch.randn(B, Hkv, S, D)

    result = perform_streaming_attention(q, [_page(k, v)])

    groups = Hq // Hkv
    k_exp = k.repeat_interleave(groups, dim=1)
    v_exp = v.repeat_interleave(groups, dim=1)
    expected = _eager(q, k_exp, v_exp)

    assert result.shape == (B, Hq, 1, D)
    assert torch.allclose(result, expected, atol=1e-4)


def test_gqa_multi_page_matches_eager():
    B, Hq, Hkv, D = 1, 4, 2, 8
    sizes = [4, 3]
    q = torch.randn(B, Hq, 1, D)
    pages, ks, vs, start = [], [], [], 0
    for i, sz in enumerate(sizes):
        k = torch.randn(B, Hkv, sz, D)
        v = torch.randn(B, Hkv, sz, D)
        ks.append(k)
        vs.append(v)
        pages.append(_page(k, v, i, start))
        start += sz
    result = perform_streaming_attention(q, pages)
    groups = Hq // Hkv
    k_full = torch.cat(ks, 2).repeat_interleave(groups, dim=1)
    v_full = torch.cat(vs, 2).repeat_interleave(groups, dim=1)
    expected = _eager(q, k_full, v_full)
    assert torch.allclose(result, expected, atol=1e-4)


# ──────────────────────────────────────────────────────────────────────────────
# Numerical stability (large attention logits)
# ──────────────────────────────────────────────────────────────────────────────

def test_large_logits_no_nan():
    """Online softmax should stay numerically stable even with large logits."""
    B, H, D = 1, 1, 64
    q = torch.ones(B, H, 1, D) * 100.0
    k = torch.ones(B, H, 8, D) * 100.0
    v = torch.randn(B, H, 8, D)
    result = perform_streaming_attention(q, [_page(k, v)])
    assert not torch.isnan(result).any()
    assert not torch.isinf(result).any()


# ──────────────────────────────────────────────────────────────────────────────
# Online-softmax streaming path
#
# An attention_mask or softcap forces the running-max/sum accumulation loop
# (the paper's paged streaming-attention math) instead of the concat+SDPA
# shortcut.  These tests pin that path against an eager full-context reference.
# ──────────────────────────────────────────────────────────────────────────────

def _eager_masked(q, k, v, mask=None, softcap=None, scale=None):
    """Reference: full-context attention with additive mask and tanh softcap."""
    if scale is None:
        scale = 1.0 / math.sqrt(q.size(-1))
    if k.size(1) != q.size(1):  # GQA expansion
        groups = q.size(1) // k.size(1)
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
    w = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    if softcap is not None:
        w = torch.tanh(w / softcap) * softcap
    if mask is not None:
        w = w + mask.float()
    w = torch.softmax(w, dim=-1)
    return torch.matmul(w, v.float()).to(q.dtype)


def _paged(sizes, B=1, H=2, D=8, kv_heads=None):
    """Random K/V split into pages of the given sizes; returns (pages, K, V)."""
    kv_heads = kv_heads or H
    pages, ks, vs, start = [], [], [], 0
    for i, sz in enumerate(sizes):
        k = torch.randn(B, kv_heads, sz, D)
        v = torch.randn(B, kv_heads, sz, D)
        pages.append(_page(k, v, i, start))
        ks.append(k)
        vs.append(v)
        start += sz
    return pages, torch.cat(ks, dim=2), torch.cat(vs, dim=2)


def test_online_softmax_zero_mask_matches_unmasked_eager():
    """An all-zero additive mask must reproduce the unmasked result exactly,
    proving the streaming accumulation itself (not the mask) is correct."""
    torch.manual_seed(0)
    B, H, D = 1, 2, 8
    pages, K, V = _paged([3, 5, 2], B=B, H=H, D=D)
    q = torch.randn(B, H, 1, D)
    mask = torch.zeros(B, 1, 1, K.size(2))

    result = perform_streaming_attention(q, pages, attention_mask=mask)
    expected = _eager_masked(q, K, V)
    assert torch.allclose(result, expected, atol=1e-5)


def test_online_softmax_padding_mask_matches_masked_eager():
    """-inf-style padding entries spanning page boundaries must match eager."""
    torch.manual_seed(1)
    B, H, D = 1, 2, 8
    pages, K, V = _paged([4, 4, 3], B=B, H=H, D=D)
    q = torch.randn(B, H, 1, D)
    mask = torch.zeros(B, 1, 1, K.size(2))
    mask[..., [0, 3, 4, 10]] = torch.finfo(torch.float32).min  # crosses pages

    result = perform_streaming_attention(q, pages, attention_mask=mask)
    expected = _eager_masked(q, K, V, mask=mask)
    assert torch.allclose(result, expected, atol=1e-5)


def test_online_softmax_fully_masked_page_matches_eager():
    """A page whose every token is masked must contribute nothing (the
    nan_to_num running-max rescale guards exactly this case)."""
    torch.manual_seed(2)
    B, H, D = 1, 2, 8
    pages, K, V = _paged([4, 4], B=B, H=H, D=D)
    q = torch.randn(B, H, 1, D)
    mask = torch.zeros(B, 1, 1, 8)
    mask[..., :4] = torch.finfo(torch.float32).min  # first page fully masked

    result = perform_streaming_attention(q, pages, attention_mask=mask)
    expected = _eager_masked(q, K, V, mask=mask)
    assert torch.allclose(result, expected, atol=1e-5)


def test_online_softmax_softcap_matches_eager():
    """Gemma-2-style tanh softcap through the streaming path."""
    torch.manual_seed(3)
    B, H, D = 1, 2, 8
    pages, K, V = _paged([3, 5], B=B, H=H, D=D)
    q = torch.randn(B, H, 1, D)

    result = perform_streaming_attention(q, pages, softcap=50.0)
    expected = _eager_masked(q, K, V, softcap=50.0)
    assert torch.allclose(result, expected, atol=1e-5)


def test_online_softmax_gqa_expansion_with_mask():
    """Grouped-query attention (kv_heads < q_heads) through the masked path."""
    torch.manual_seed(4)
    B, H, D = 1, 4, 8
    pages, K, V = _paged([4, 3], B=B, H=H, D=D, kv_heads=2)
    q = torch.randn(B, H, 1, D)
    mask = torch.zeros(B, 1, 1, K.size(2))
    mask[..., 1] = torch.finfo(torch.float32).min

    result = perform_streaming_attention(q, pages, attention_mask=mask)
    expected = _eager_masked(q, K, V, mask=mask)
    assert torch.allclose(result, expected, atol=1e-5)


def test_online_softmax_fp16_query_matches_eager():
    """fp16 inputs accumulate in fp32 and cast back without over/underflow."""
    torch.manual_seed(5)
    B, H, D = 1, 2, 8
    pages, K, V = _paged([4, 4], B=B, H=H, D=D)
    pages = [_page(p.k_tensor.half(), p.v_tensor.half(), p.page_id, p.start_token) for p in pages]
    q = torch.randn(B, H, 1, D, dtype=torch.float16)
    mask = torch.zeros(B, 1, 1, 8)

    result = perform_streaming_attention(q, pages, attention_mask=mask)
    expected = _eager_masked(q, K.half(), V.half(), mask=mask)
    assert result.dtype == torch.float16
    assert torch.allclose(result.float(), expected.float(), atol=2e-3)
