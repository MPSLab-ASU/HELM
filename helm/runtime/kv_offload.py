"""
KV cache offloading integration for HELM.

Patches model attention modules to use a paged KV cache (KVCacheManager)
with CPU offloading. Streaming attention is used for decode steps so that
older KV pages evicted to CPU are fetched page-by-page without materialising
the full context on GPU.

Prefill  (q_len > 1): first chunk uses local SDPA; later chunks route
                    through paged KV streaming attention over all cached pages
Decode   (q_len == 1): append_decode → paged cache, perform_streaming_attention

The decoder layers remain FX leaf modules — no graph changes needed.
Patches are applied at the class level so every instance (including those
inside stage GraphModules) uses the offloaded path.
"""

import math
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from types import MethodType
from typing import Optional

from helm.runtime.kv_allocator import KVAllocator
from helm.runtime.kv_cache import KVCacheManager, perform_streaming_attention


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class KVOffloadConfig:
    num_layers: int
    num_kv_heads: int
    head_dim: int
    page_size: int = 64
    dtype: torch.dtype = torch.float16
    # Total GPU KV bytes to keep hot before evicting older pages to CPU.
    # Default: keep ~512 tokens per layer on GPU.
    gpu_watermark_bytes: Optional[int] = None
    # Pre-allocated contiguous KV buffer capacity (tokens). Set this explicitly
    # for all-GPU fast-path experiments; the offload default is paged KV so the
    # residency watermark can evict cold pages to CPU RAM.
    cont_capacity: int = 0

    def __post_init__(self):
        if self.gpu_watermark_bytes is None:
            elem = torch.tensor([], dtype=self.dtype).element_size()
            # 2 tensors (K+V) × kv_heads × head_dim × elem_size × 512 tokens
            per_token_bytes = 2 * self.num_kv_heads * self.head_dim * elem
            self.gpu_watermark_bytes = 512 * per_token_bytes * self.num_layers

    @staticmethod
    def from_model(model, page_size: int = 64,
                   gpu_watermark_bytes: Optional[int] = None,
                   cont_capacity: Optional[int] = None) -> "KVOffloadConfig":
        cfg = model.config
        num_layers   = cfg.num_hidden_layers
        num_kv_heads = getattr(cfg, "num_key_value_heads",
                               cfg.num_attention_heads)
        # Some configs (e.g. Mistral) declare `head_dim` but leave it None and
        # fall back to hidden_size // num_attention_heads internally. Mirror that
        # so a None never propagates into the watermark byte math.
        head_dim     = getattr(cfg, "head_dim", None) or (
            cfg.hidden_size // cfg.num_attention_heads)
        dtype = next(model.parameters()).dtype
        if cont_capacity is None:
            cont_capacity = 0
        return KVOffloadConfig(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_size=page_size,
            dtype=dtype,
            gpu_watermark_bytes=gpu_watermark_bytes,
            cont_capacity=cont_capacity,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Shared decode step helper
# ─────────────────────────────────────────────────────────────────────────────

def _decode_batched(kvcms, q, k, v, layer_idx: int, scale: float,
                    attention_mask: Optional[torch.Tensor] = None):
    """
    Decode step shared by all architecture forward patches.

    Three execution paths, selected in order:

    1. Contiguous path (all-GPU, sequence within pre-allocated capacity):
       Writes K/V in-place into a pre-allocated [bsz, heads, capacity, head_dim]
       buffer and passes a zero-copy view to SDPA.  No append_decode, no
       repeat_interleave — equivalent to Accelerate's decode path.

    2. Paged GPU path (all-GPU, capacity exceeded or contiguous disabled):
       Assembles batched K/V with torch.cat across pages and issues a single
       batched SDPA call with enable_gqa=True (no KV head expansion).

    3. Paged streaming path (mixed/CPU residency):
       Per-item perform_streaming_attention with async H2D prefetch and online
       softmax.  This is HELM's core offloading path.

    `attention_mask`, when given, is the caller's per-item key mask (additive,
    broadcastable to [bsz, *, 1, kv_len]) — prefill caches every position it
    is handed, including pad positions from batch padding, so without this
    mask a shorter prompt in the batch would attend to its own pad-token K/V.

    Returns out: [bsz, num_q_heads, 1, head_dim]
    """
    bsz = q.shape[0]

    # ── Path 1: contiguous in-place fast path ────────────────────────────────
    if kvcms[0].use_contiguous:
        if kvcms[0].cont_seq_len < kvcms[0].cont_capacity:
            # Write new token in-place — no append_decode, no paged overhead.
            for i in range(bsz):
                kvcms[i].write_decode_contiguous(layer_idx, k[i:i+1], v[i:i+1])

            # K/V views: zero-copy slice into pre-allocated buffer.
            if bsz == 1:
                K, V = kvcms[0].get_kv_contiguous(layer_idx)
            else:
                Ks, Vs = zip(*(kvcms[i].get_kv_contiguous(layer_idx) for i in range(bsz)))
                K, V = torch.cat(Ks, dim=0), torch.cat(Vs, dim=0)

            # Advance committed token count after the last transformer layer.
            if layer_idx == kvcms[0].cont_num_layers - 1:
                for kvcm in kvcms:
                    kvcm.advance_contiguous()

            # enable_gqa lets the flash-attention kernel handle GQA natively —
            # no repeat_interleave, no extra tensor allocation.
            return F.scaled_dot_product_attention(
                q, K, V, attn_mask=attention_mask, scale=scale, enable_gqa=True
            )
        else:
            # Capacity exceeded — bulk-migrate decode tokens to pages, then fall through.
            for kvcm in kvcms:
                kvcm.migrate_contiguous_to_pages()

    # ── Path 2 & 3: paged path ───────────────────────────────────────────────
    all_pages = []
    for i in range(bsz):
        kvcms[i].append_decode(layer_idx, k[i:i+1], v[i:i+1], skip_residency=False)
        all_pages.append(kvcms[i].iterate_layer_pages(layer_idx))

    all_gpu = all(kvcms[i].all_gpu_resident(layer_idx) for i in range(bsz))

    # ── Path 2: all GPU-resident — single batched SDPA ───────────────────────
    if all_gpu:
        K_list, V_list, seq_len_ref, fast_ok = [], [], None, True
        for pages in all_pages:
            active = [p for p in pages if p.used_tokens > 0]
            if not active:
                fast_ok = False
                break
            K_i = torch.cat([p.k_tensor[:, :, :p.used_tokens, :] for p in active], dim=2)
            V_i = torch.cat([p.v_tensor[:, :, :p.used_tokens, :] for p in active], dim=2)
            if seq_len_ref is None:
                seq_len_ref = K_i.shape[2]
            elif K_i.shape[2] != seq_len_ref:
                fast_ok = False
                break
            K_list.append(K_i)
            V_list.append(V_i)
        if fast_ok and K_list:
            K = torch.cat(K_list, dim=0)
            V = torch.cat(V_list, dim=0)
            return F.scaled_dot_product_attention(
                q, K, V, attn_mask=attention_mask, scale=scale, enable_gqa=True
            )

    # ── Path 3: mixed/CPU residency — per-item streaming attention ────────────
    outs = [
        perform_streaming_attention(
            q[i:i+1],
            all_pages[i],
            scale=scale,
            attention_mask=attention_mask[i:i+1] if attention_mask is not None else None,
        )
        for i in range(bsz)
    ]
    return torch.cat(outs, dim=0)


def _cached_tokens(kvcm, layer_idx: int) -> int:
    return sum(
        page.used_tokens
        for page in kvcm.iterate_layer_pages(layer_idx)
        if page.used_tokens > 0 and getattr(page, "state", None) != "DEAD"
    )


def _append_prefill_batched(kvcms, k, v, layer_idx: int) -> list[int]:
    past_lengths = []
    for i in range(k.shape[0]):
        past_lengths.append(_cached_tokens(kvcms[i], layer_idx))
        kvcms[i].append_prefill(layer_idx, k[i:i + 1], v[i:i + 1])
        if getattr(kvcms[i], "use_contiguous", False):
            kvcms[i].prefill_contiguous(layer_idx, k[i:i + 1], v[i:i + 1])
    return past_lengths


def _stream_prefill_batched(
    kvcms,
    q,
    layer_idx: int,
    scale: float,
    attention_mask: Optional[torch.Tensor] = None,
    softcap: Optional[float] = None,
):
    outs = []
    for i in range(q.shape[0]):
        item_mask = attention_mask[i:i + 1] if attention_mask is not None else None
        outs.append(
            perform_streaming_attention(
                q[i:i + 1],
                kvcms[i].iterate_layer_pages(layer_idx),
                scale=scale,
                attention_mask=item_mask,
                softcap=softcap,
            )
        )
    return torch.cat(outs, dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# Per-architecture patched forward factories
# ─────────────────────────────────────────────────────────────────────────────

def _apply_softcap(attn_weights: torch.Tensor, softcap: Optional[float]) -> torch.Tensor:
    if softcap is None:
        return attn_weights
    return torch.tanh(attn_weights / softcap) * softcap


def _gemma2_sliding_window_mask(
    attn_self,
    attention_mask: Optional[torch.Tensor],
    cache_position: Optional[torch.Tensor],
    key_length: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """Overlay Gemma-2's per-layer sliding window on top of the caller mask."""
    if attention_mask is not None:
        query_length = attention_mask.shape[-2]
        attention_mask = attention_mask[:, :, :query_length, :key_length]
    elif cache_position is not None:
        query_length = int(cache_position.numel())
    else:
        query_length = key_length

    sliding_window = getattr(attn_self, "sliding_window", None)
    if sliding_window is None:
        return attention_mask

    if cache_position is None:
        query_positions = torch.arange(query_length, device=device, dtype=torch.long)
    else:
        query_positions = cache_position.to(device=device, dtype=torch.long).reshape(-1)
        if query_positions.numel() != query_length:
            raise ValueError(
                "Gemma2 sliding-window mask requires cache_position to match "
                f"query length ({query_length}), got {query_positions.numel()}."
            )

    key_positions = torch.arange(key_length, device=device, dtype=torch.long)
    min_val = torch.finfo(dtype).min
    allowed = (key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)) & (
        key_positions.unsqueeze(0) > (query_positions.unsqueeze(1) - sliding_window)
    )
    sliding_mask = torch.full(
        (query_length, key_length),
        min_val,
        device=device,
        dtype=dtype,
    )
    sliding_mask.masked_fill_(allowed, 0)
    sliding_mask = sliding_mask.unsqueeze(0).unsqueeze(0)

    if attention_mask is None:
        return sliding_mask
    return attention_mask + sliding_mask


def _make_qwen2_forward(kvcms):
    """kvcms: list of KVCacheManager, one per batch item."""
    def forward(
        attn_self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        past_key_values=None,      # ignored — kvcm handles storage
        cache_position=None,
        **kwargs,
    ):
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

        bsz, q_len, _ = hidden_states.shape
        hidden_shape = (bsz, q_len, -1, attn_self.head_dim)

        q = attn_self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        k = attn_self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        v = attn_self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        scale = 1.0 / math.sqrt(attn_self.head_dim)
        num_kv = k.shape[1]
        groups  = q.shape[1] // num_kv

        if q_len > 1:
            # First prefill chunk can attend over local K/V. Later chunks must
            # attend over every paged KV token already cached for this layer.
            past_lengths = _append_prefill_batched(kvcms, k, v, attn_self.layer_idx)
            if any(past > 0 for past in past_lengths):
                out = _stream_prefill_batched(
                    kvcms,
                    q,
                    attn_self.layer_idx,
                    scale,
                    attention_mask,
                )
            else:
                k_exp = k.repeat_interleave(groups, dim=1)
                v_exp = v.repeat_interleave(groups, dim=1)
                out = F.scaled_dot_product_attention(
                    q, k_exp, v_exp,
                    attn_mask=attention_mask,
                    dropout_p=attn_self.attention_dropout if attn_self.training else 0.0,
                    scale=scale,
                )
        else:
            out = _decode_batched(kvcms, q, k, v, attn_self.layer_idx, scale, attention_mask)

        out = out.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        out = attn_self.o_proj(out)
        return out, None

    return forward


def _make_qwen3_forward(kvcms):
    """kvcms: list of KVCacheManager, one per batch item."""
    def forward(
        attn_self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ):
        from transformers.models.qwen3.modeling_qwen3 import (
            apply_rotary_pos_emb,
            repeat_kv,
        )

        bsz, q_len, _ = hidden_states.shape
        hidden_shape = (bsz, q_len, -1, attn_self.head_dim)

        q = attn_self.q_norm(attn_self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        k = attn_self.k_norm(attn_self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        v = attn_self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        scale = 1.0 / math.sqrt(attn_self.head_dim)

        if q_len > 1:
            past_lengths = _append_prefill_batched(kvcms, k, v, attn_self.layer_idx)
            if any(past > 0 for past in past_lengths):
                out = _stream_prefill_batched(
                    kvcms,
                    q,
                    attn_self.layer_idx,
                    scale,
                    attention_mask,
                )
            else:
                k_exp = repeat_kv(k, attn_self.num_key_value_groups)
                v_exp = repeat_kv(v, attn_self.num_key_value_groups)
                out = F.scaled_dot_product_attention(
                    q, k_exp, v_exp,
                    attn_mask=attention_mask,
                    dropout_p=attn_self.attention_dropout if attn_self.training else 0.0,
                    scale=scale,
                )
        else:
            out = _decode_batched(kvcms, q, k, v, attn_self.layer_idx, scale, attention_mask)

        out = out.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        out = attn_self.o_proj(out)
        return out, None

    return forward


def _make_olmo2_forward(kvcms):
    """OLMo-2 attention patch (kvcms: list of KVCacheManager, one per batch item).

    Mirrors Qwen3 except q_norm/k_norm apply to the un-viewed projection
    (post-q_proj, pre-reshape) per the upstream Olmo2Attention.forward.
    """
    def forward(
        attn_self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ):
        from transformers.models.olmo2.modeling_olmo2 import (
            apply_rotary_pos_emb,
            repeat_kv,
        )

        bsz, q_len, _ = hidden_states.shape
        hidden_shape = (bsz, q_len, -1, attn_self.head_dim)

        q = attn_self.q_norm(attn_self.q_proj(hidden_states)).view(hidden_shape).transpose(1, 2)
        k = attn_self.k_norm(attn_self.k_proj(hidden_states)).view(hidden_shape).transpose(1, 2)
        v = attn_self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        scale = 1.0 / math.sqrt(attn_self.head_dim)

        if q_len > 1:
            past_lengths = _append_prefill_batched(kvcms, k, v, attn_self.layer_idx)
            if any(past > 0 for past in past_lengths):
                out = _stream_prefill_batched(
                    kvcms,
                    q,
                    attn_self.layer_idx,
                    scale,
                    attention_mask,
                )
            else:
                k_exp = repeat_kv(k, attn_self.num_key_value_groups)
                v_exp = repeat_kv(v, attn_self.num_key_value_groups)
                out = F.scaled_dot_product_attention(
                    q, k_exp, v_exp,
                    attn_mask=attention_mask,
                    dropout_p=attn_self.attention_dropout if attn_self.training else 0.0,
                    scale=scale,
                )
        else:
            out = _decode_batched(kvcms, q, k, v, attn_self.layer_idx, scale, attention_mask)

        out = out.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        out = attn_self.o_proj(out)
        return out, None

    return forward


def _make_llama_forward(kvcms):
    """kvcms: list of KVCacheManager, one per batch item."""
    def forward(
        attn_self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ):
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

        bsz, q_len, _ = hidden_states.shape
        hidden_shape = (bsz, q_len, -1, attn_self.head_dim)

        q = attn_self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        k = attn_self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        v = attn_self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        scale = 1.0 / math.sqrt(attn_self.head_dim)
        groups = q.shape[1] // k.shape[1]

        if q_len > 1:
            past_lengths = _append_prefill_batched(kvcms, k, v, attn_self.layer_idx)
            if any(past > 0 for past in past_lengths):
                out = _stream_prefill_batched(
                    kvcms,
                    q,
                    attn_self.layer_idx,
                    scale,
                    attention_mask,
                )
            else:
                k_exp = k.repeat_interleave(groups, dim=1)
                v_exp = v.repeat_interleave(groups, dim=1)
                out = F.scaled_dot_product_attention(
                    q, k_exp, v_exp,
                    attn_mask=attention_mask,
                    dropout_p=attn_self.attention_dropout if attn_self.training else 0.0,
                    scale=scale,
                )
        else:
            out = _decode_batched(kvcms, q, k, v, attn_self.layer_idx, scale, attention_mask)

        out = out.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        out = attn_self.o_proj(out)
        return out, None

    return forward


def _gemma2_eager_attention(
    attn_self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scale: float,
    softcap: Optional[float],
) -> torch.Tensor:
    from transformers.models.gemma2.modeling_gemma2 import repeat_kv

    key_states = repeat_kv(key, attn_self.num_key_value_groups)
    value_states = repeat_kv(value, attn_self.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scale
    attn_weights = _apply_softcap(attn_weights, softcap)
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask.to(device=attn_weights.device, dtype=attn_weights.dtype)

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(
        attn_weights,
        p=attn_self.attention_dropout if attn_self.training else 0.0,
        training=attn_self.training,
    )
    return torch.matmul(attn_weights, value_states)


def _make_gemma2_forward(kvcms):
    """kvcms: list of KVCacheManager, one per batch item."""
    def forward(
        attn_self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ):
        from transformers.models.gemma2.modeling_gemma2 import apply_rotary_pos_emb

        bsz, q_len, _ = hidden_states.shape
        hidden_shape = (bsz, q_len, -1, attn_self.head_dim)

        q = attn_self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        k = attn_self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        v = attn_self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        scale = getattr(attn_self, "scaling", 1.0 / math.sqrt(attn_self.head_dim))
        softcap = getattr(attn_self, "attn_logit_softcapping", None)
        if q_len > 1:
            past_lengths = _append_prefill_batched(kvcms, k, v, attn_self.layer_idx)
            if any(past > 0 for past in past_lengths):
                key_length = max(_cached_tokens(kvcm, attn_self.layer_idx) for kvcm in kvcms)
                masked_attention = _gemma2_sliding_window_mask(
                    attn_self,
                    attention_mask,
                    cache_position,
                    key_length,
                    device=q.device,
                    dtype=q.dtype,
                )
                out = _stream_prefill_batched(
                    kvcms,
                    q,
                    attn_self.layer_idx,
                    scale,
                    masked_attention,
                    softcap,
                )
            else:
                masked_attention = _gemma2_sliding_window_mask(
                    attn_self,
                    attention_mask,
                    cache_position,
                    k.shape[-2],
                    device=q.device,
                    dtype=q.dtype,
                )
                out = _gemma2_eager_attention(
                    attn_self,
                    q,
                    k,
                    v,
                    masked_attention,
                    scale,
                    softcap,
                )
        else:
            outs = []
            for i in range(bsz):
                kvcms[i].append_decode(attn_self.layer_idx, k[i:i+1], v[i:i+1])
                pages = kvcms[i].iterate_layer_pages(attn_self.layer_idx)
                key_length = sum(
                    page.used_tokens
                    for page in pages
                    if page.used_tokens > 0 and page.state != "DEAD"
                )
                item_mask = (
                    _gemma2_sliding_window_mask(
                        attn_self,
                        attention_mask[i:i+1] if attention_mask is not None else None,
                        cache_position,
                        key_length,
                        device=q.device,
                        dtype=q.dtype,
                    )
                    if key_length > 0
                    else None
                )
                outs.append(
                    perform_streaming_attention(
                        q[i:i+1],
                        pages,
                        scale=scale,
                        attention_mask=item_mask,
                        softcap=softcap,
                    )
                )
            out = torch.cat(outs, dim=0)

        out = out.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        out = attn_self.o_proj(out)
        return out, None

    return forward


# ─────────────────────────────────────────────────────────────────────────────
# Manager
# ─────────────────────────────────────────────────────────────────────────────

class KVOffloadManager:
    """
    Installs paged-KV attention patches on the model and manages the cache
    lifecycle across generate() calls.

    Supports batch_size > 1: one KVCacheManager is maintained per batch item,
    all sharing a single KVAllocator pool.  Attention patches loop over the
    batch dimension for KV operations and concatenate results.

    Usage:
        mgr = KVOffloadManager(model, KVOffloadConfig.from_model(model), batch_size=4)
        runtime = PipelineRuntime(..., kv_offload_mgr=mgr)
        runtime.generate(input_ids, max_new_tokens=32)  # input_ids: (4, S)
    """

    def __init__(self, model, config: KVOffloadConfig, batch_size: int = 1):
        self.config = config
        self.batch_size = batch_size
        self.allocator = KVAllocator(
            num_layers=config.num_layers,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            page_size=config.page_size,
            dtype=config.dtype,
        )
        # Split the GPU watermark budget across batch items so total GPU KV
        # usage stays within the original budget.
        watermark_per_item = max(config.gpu_watermark_bytes // batch_size, 1)
        self.kvcms = [
            KVCacheManager(self.allocator, gpu_high_watermark_bytes=watermark_per_item)
            for _ in range(batch_size)
        ]
        self._patched: dict[str, tuple] = {}  # arch → (cls, replacement_forward)
        self._patched_instance_forwards: dict[int, tuple[object, bool, object]] = {}
        self._apply_patches(model)
        self.patch_module_roots(model)

        # Optional all-GPU fast path for experiments.  Disabled by default so
        # paged KV residency can enforce the CPU-offload watermark.
        if config.cont_capacity > 0:
            device = next(model.parameters()).device
            if device.type == "cuda":
                built = [
                    kvcm.build_contiguous(
                        num_layers=config.num_layers,
                        num_kv_heads=config.num_kv_heads,
                        head_dim=config.head_dim,
                        capacity=config.cont_capacity,
                        device=device,
                        dtype=config.dtype,
                    )
                    for kvcm in self.kvcms
                ]
                if not all(built):  # partial OOM — drop all to keep things consistent
                    for kvcm in self.kvcms:
                        kvcm.drop_contiguous()

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self):
        """Clear all paged caches and re-arm contiguous path for next generate()."""
        for kvcm in self.kvcms:
            kvcm.clear()
            kvcm.reset_contiguous()

    def restore(self):
        """Restore original attention forwards (e.g. after generation)."""
        for module, had_instance_forward, original_forward in self._patched_instance_forwards.values():
            if had_instance_forward:
                module.forward = original_forward
            else:
                module.__dict__.pop("forward", None)
        self._patched_instance_forwards.clear()
        self._patched.clear()

    def report(self) -> dict:
        return self.allocator.report_usage()

    def patch_module_roots(self, *roots):
        """
        Rebind attention forwards on copied runtime modules.

        FX/GraphModule copies can carry their own instance-level `forward`
        attributes. Rebinding only the concrete instances keeps unrelated
        models isolated and staged runtime copies aligned with the
        manager's captured KV cache list.
        """
        for root in roots:
            if root is None:
                continue
            for cls, patched_forward in self._patched.values():
                for module in root.modules():
                    if not isinstance(module, cls):
                        continue
                    module_key = id(module)
                    if module_key in self._patched_instance_forwards:
                        continue
                    had_instance_forward = "forward" in module.__dict__
                    original_forward = module.__dict__.get("forward")
                    module.forward = MethodType(patched_forward, module)
                    self._patched_instance_forwards[module_key] = (
                        module,
                        had_instance_forward,
                        original_forward,
                    )

    # ── Patch dispatch ────────────────────────────────────────────────────────

    def _apply_patches(self, model):
        name = model.__class__.__name__.lower()
        if "qwen3" in name:
            self._patch("qwen3")
        elif "qwen2" in name:
            self._patch("qwen2")
        elif "gemma2" in name:
            self._patch("gemma2")
        elif "ministral" in name:
            # Note: "mistral" is not a substring of "ministral" — keep this branch
            # before the "mistral" check below. Ministral's attention class mirrors
            # Llama's exactly except for an unused `sliding_window` kwarg, so we
            # reuse the llama forward and only switch the patched class.
            self._patch("ministral")
        elif "olmo2" in name:
            self._patch("olmo2")
        elif "mistral" in name:
            # Mistral (incl. Mistral-Nemo) defines its own MistralAttention
            # (nn.Module, NOT a LlamaAttention subclass), so patching
            # LlamaAttention would silently miss every attention module and KV
            # offload would never engage. Patch MistralAttention directly; its
            # forward signature/projections match Llama's, and Nemo has no
            # sliding window, so the llama forward is correct here.
            self._patch("mistral")
        elif "llama" in name:
            self._patch("llama")
        else:
            raise NotImplementedError(
                f"KVOffloadManager: no attention patch for {model.__class__.__name__}"
            )

    def _patch(self, arch: str):
        if arch == "qwen2":
            import transformers.models.qwen2.modeling_qwen2 as m
            cls = m.Qwen2Attention
            new_fwd = _make_qwen2_forward(self.kvcms)
        elif arch == "qwen3":
            import transformers.models.qwen3.modeling_qwen3 as m
            cls = m.Qwen3Attention
            new_fwd = _make_qwen3_forward(self.kvcms)
        elif arch == "llama":
            import transformers.models.llama.modeling_llama as m
            cls = m.LlamaAttention
            new_fwd = _make_llama_forward(self.kvcms)
        elif arch == "gemma2":
            import transformers.models.gemma2.modeling_gemma2 as m
            cls = m.Gemma2Attention
            new_fwd = _make_gemma2_forward(self.kvcms)
        elif arch == "mistral":
            import transformers.models.mistral.modeling_mistral as m
            cls = m.MistralAttention
            new_fwd = _make_llama_forward(self.kvcms)
        elif arch == "ministral":
            import transformers.models.ministral.modeling_ministral as m
            cls = m.MinistralAttention
            new_fwd = _make_llama_forward(self.kvcms)
        elif arch == "olmo2":
            import transformers.models.olmo2.modeling_olmo2 as m
            cls = m.Olmo2Attention
            new_fwd = _make_olmo2_forward(self.kvcms)
        else:
            raise ValueError(arch)

        # Bind only this model and its runtime copies. Mutating cls.forward
        # would redirect unrelated models to this manager's request cache.
        self._patched[arch] = (cls, new_fwd)
