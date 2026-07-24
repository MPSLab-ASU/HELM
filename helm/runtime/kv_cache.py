import math
import os
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F


# ── KV paging instrumentation ──────────────────────────────────────────────────
# Process-global counters for KV page eviction (GPU->CPU) and prefetch (CPU->GPU).
# A benchmark resets before a single serialized request and reads totals after,
# so it can report how eviction/prefetch frequency scales with context length.
# These counters never alter cache behavior; they only accumulate integers.
@dataclass
class KVPagingStats:
    evict_calls: int = 0
    pages_evicted: int = 0
    bytes_evicted: int = 0
    prefetch_calls: int = 0
    pages_prefetched: int = 0
    bytes_prefetched: int = 0


_PAGING_STATS = KVPagingStats()


def reset_paging_stats() -> None:
    """Zero the global paging counters (call before a measured request)."""
    global _PAGING_STATS
    _PAGING_STATS = KVPagingStats()


def get_paging_stats() -> dict:
    """Return the global paging counters as a plain dict."""
    s = _PAGING_STATS
    return {
        "evict_calls": s.evict_calls,
        "pages_evicted": s.pages_evicted,
        "bytes_evicted": s.bytes_evicted,
        "prefetch_calls": s.prefetch_calls,
        "pages_prefetched": s.pages_prefetched,
        "bytes_prefetched": s.bytes_prefetched,
    }


def _record_eviction(n_pages: int, n_bytes: int) -> None:
    """Record one eviction call that moved n_pages (n_bytes) GPU->CPU."""
    if n_pages <= 0:
        return
    _PAGING_STATS.evict_calls += 1
    _PAGING_STATS.pages_evicted += n_pages
    _PAGING_STATS.bytes_evicted += n_bytes


def _record_prefetch(n_pages: int, n_bytes: int) -> None:
    """Record one streaming-attention call that prefetched n_pages (n_bytes) CPU->GPU."""
    if n_pages <= 0:
        return
    _PAGING_STATS.prefetch_calls += 1
    _PAGING_STATS.pages_prefetched += n_pages
    _PAGING_STATS.bytes_prefetched += n_bytes


@dataclass
class KVPage:
    page_id: int
    layer_id: int
    start_token: int
    used_tokens: int
    capacity_tokens: int
    k_tensor: torch.Tensor
    v_tensor: torch.Tensor
    device: torch.device
    state: str  # 'FREE', 'GPU', 'CPU'

class LayerKVCache:
    def __init__(self, layer_id: int):
        self.layer_id = layer_id
        self.pages: List[KVPage] = []
        self.tail_page: Optional[KVPage] = None
        self.total_tokens: int = 0

    def append_page(self, page: KVPage):
        self.pages.append(page)
        self.tail_page = page

class KVCacheManager:
    """
    KVCacheManager owns all layer KV state.
    Implements append semantics, residency tracking, and high-level 
    prefill/decode integration.
    """
    def __init__(self, allocator, gpu_high_watermark_bytes: int = 512 * 1024 * 1024):
        self.allocator = allocator
        self.layers = {}
        self.gpu_high_watermark_bytes = gpu_high_watermark_bytes
        # Contiguous (Accelerate-style) KV buffers for the all-GPU fast path.
        # Pre-allocated once; tokens are written in-place — no torch.cat per step.
        # Keyed by layer_id. Empty until build_contiguous() succeeds.
        self.cont_K: dict = {}   # layer_id → Tensor[1, kv_heads, capacity, head_dim]
        self.cont_V: dict = {}
        self.cont_capacity: int = 0
        self.cont_num_layers: int = 0
        self.cont_seq_len: int = 0     # tokens committed (prefill + completed decode steps)
        self.cont_prefill_len: int = 0  # tokens from prefill phase
        self.use_contiguous: bool = False

    def initialize_layer_caches(self, num_layers: int):
        for i in range(num_layers):
            self.layers[i] = LayerKVCache(layer_id=i)

    def append_prefill(self, layer_id: int, K_prefill: torch.Tensor, V_prefill: torch.Tensor):
        """Append prefill KV tensor chunk to the layer cache in page-sized blocks."""
        if layer_id not in self.layers:
            self.layers[layer_id] = LayerKVCache(layer_id)
        
        layer = self.layers[layer_id]
        seq_len = K_prefill.size(2)
        page_size = self.allocator.page_size
        device = K_prefill.device
        
        for start_idx in range(0, seq_len, page_size):
            chunk_len = min(page_size, seq_len - start_idx)
            end_idx = start_idx + chunk_len

            k_chunk = K_prefill[:, :, start_idx:end_idx, :]
            v_chunk = V_prefill[:, :, start_idx:end_idx, :]

            page = self.allocator.allocate(device)
            page.layer_id = layer_id
            page.start_token = layer.total_tokens
            page.used_tokens = chunk_len

            page.k_tensor[:, :, :chunk_len, :].copy_(k_chunk)
            page.v_tensor[:, :, :chunk_len, :].copy_(v_chunk)

            layer.append_page(page)
            layer.total_tokens += chunk_len

            # Enforce after every page so GPU KV never spikes beyond watermark
            # even during long prefills (previously only ran once at end of layer).
            self._enforce_residency_policy()

    def append_decode(self, layer_id: int, K_new: torch.Tensor, V_new: torch.Tensor,
                      skip_residency: bool = False):
        """Append a single token KV pair for the given layer."""
        if layer_id not in self.layers:
            self.layers[layer_id] = LayerKVCache(layer_id)

        layer = self.layers[layer_id]
        device = K_new.device

        if layer.tail_page is None or layer.tail_page.used_tokens == layer.tail_page.capacity_tokens:
            page = self.allocator.allocate(device)
            page.layer_id = layer_id
            page.start_token = layer.total_tokens
            page.used_tokens = 0
            layer.append_page(page)

        page = layer.tail_page
        idx = page.used_tokens

        page.k_tensor[:, :, idx:idx+1, :].copy_(K_new)
        page.v_tensor[:, :, idx:idx+1, :].copy_(V_new)

        page.used_tokens += 1
        layer.total_tokens += 1

        if not skip_residency:
            self._enforce_residency_policy()

    def iterate_layer_pages(self, layer_id: int):
        if layer_id not in self.layers:
            return []
        # Return a shallow copy of the list so it isn't mutated during iteration
        return list(self.layers[layer_id].pages)

    def evict_pages(self, bytes_to_evict: int):
        """Global oldest page first eviction."""
        if bytes_to_evict <= 0:
            return

        gpu_pages = []
        for layer in self.layers.values():
            for page in layer.pages:
                if page.state == 'GPU' and page.device.type == 'cuda':
                    # Protect tail page
                    if page != layer.tail_page:
                        # Only evict fully written pages
                        if page.used_tokens == page.capacity_tokens:
                            gpu_pages.append((layer, page))
                            
        # Sort by oldest allocation globally (using start_token instead of page_id for chronological age)
        gpu_pages.sort(key=lambda t: t[1].start_token)
        
        evicted_bytes = 0
        n_moved = 0
        for layer, page in gpu_pages:
            if evicted_bytes >= bytes_to_evict:
                break

            # move_page returns a newly allocated CPU page with copied data
            original_start_token = page.start_token
            new_page = self.allocator.move_page(page, torch.device('cpu'))

            # Ensure ordering invariants are protected
            assert new_page.start_token == original_start_token, "Page invariant broken during migration"

            # Replace old page with new page in the layer's list
            idx = layer.pages.index(page)
            layer.pages[idx] = new_page

            # Keep tail page reference correct if it happened to be the one (though protected above)
            if layer.tail_page is page:
                layer.tail_page = new_page

            evicted_bytes += new_page.k_tensor.numel() * new_page.k_tensor.element_size() * 2
            n_moved += 1

        # Instrumentation only: records GPU->CPU page spills.
        _record_eviction(n_moved, evicted_bytes)

    def _enforce_residency_policy(self):
        # Note: We only record bytes for pages that are fully written and not the active tail.
        # Thus, the watermark strictly applies to evictable pages, not total GPU KV footprint.
        gpu_bytes = 0
        for layer in self.layers.values():
            for page in layer.pages:
                if page.state == 'GPU' and page.device.type == 'cuda':
                    if page != layer.tail_page and page.used_tokens == page.capacity_tokens:
                        gpu_bytes += page.k_tensor.numel() * page.k_tensor.element_size() * 2
        
        if gpu_bytes > self.gpu_high_watermark_bytes:
            self.evict_pages(gpu_bytes - self.gpu_high_watermark_bytes)

    def seq_len(self):
        """Return the current cached sequence length (from layer 0)."""
        if 0 in self.layers:
            return self.layers[0].total_tokens
        return 0

    def num_layers(self):
        return len(self.layers)

    def clear(self):
        for layer in self.layers.values():
            for page in layer.pages:
                # DEAD pages had their tensors replaced with placeholders and
                # were never added to any pool — skip them here.
                if page.state != 'DEAD':
                    self.allocator.free(page)
            layer.pages.clear()
            layer.tail_page = None
            layer.total_tokens = 0
        self.layers.clear()

    def all_gpu_resident(self, layer_id: int) -> bool:
        """Return True if every used page for this layer is GPU-resident (no CPU pages)."""
        if layer_id not in self.layers:
            return True  # empty cache — trivially all-GPU
        for page in self.layers[layer_id].pages:
            if page.used_tokens > 0 and page.state != 'GPU':
                return False
        return True

    # ── Contiguous (Accelerate-style) fast path ───────────────────────────────

    def build_contiguous(self, num_layers: int, num_kv_heads: int, head_dim: int,
                         capacity: int, device: torch.device, dtype: torch.dtype) -> bool:
        """
        Pre-allocate contiguous K/V buffers for all layers.
        Tries the requested capacity first; on OOM, halves it and retries down
        to a floor of min(capacity, 1024) tokens.
        Returns True on success with the actual capacity used, False if all attempts fail.
        """
        cap = capacity
        floor = max(1, min(capacity, 1024))
        while cap >= floor:
            try:
                self.cont_K.clear()
                self.cont_V.clear()
                for lid in range(num_layers):
                    self.cont_K[lid] = torch.zeros(
                        1, num_kv_heads, cap, head_dim, device=device, dtype=dtype)
                    self.cont_V[lid] = torch.zeros(
                        1, num_kv_heads, cap, head_dim, device=device, dtype=dtype)
                self.cont_capacity = cap
                self.cont_num_layers = num_layers
                self.use_contiguous = True
                return True
            except (torch.cuda.OutOfMemoryError, RuntimeError):
                self.cont_K.clear()
                self.cont_V.clear()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                cap //= 2
        self.use_contiguous = False
        return False

    def prefill_contiguous(self, layer_id: int, K: torch.Tensor, V: torch.Tensor):
        """Copy prefill KV into the contiguous buffer (called after append_prefill)."""
        n = K.shape[2]
        self.cont_K[layer_id][:, :, :n, :].copy_(K)
        self.cont_V[layer_id][:, :, :n, :].copy_(V)
        if layer_id == 0:
            self.cont_prefill_len = n
            self.cont_seq_len = n

    def write_decode_contiguous(self, layer_id: int, K_new: torch.Tensor, V_new: torch.Tensor):
        """Write one decode token in-place at cont_seq_len (the next free position)."""
        idx = self.cont_seq_len
        self.cont_K[layer_id][:, :, idx:idx + 1, :].copy_(K_new)
        self.cont_V[layer_id][:, :, idx:idx + 1, :].copy_(V_new)

    def get_kv_contiguous(self, layer_id: int):
        """Return (K, V) views including the token just written — zero allocation."""
        n = self.cont_seq_len + 1  # +1 to include the in-progress token
        return self.cont_K[layer_id][:, :, :n, :], self.cont_V[layer_id][:, :, :n, :]

    def advance_contiguous(self):
        """Commit the current decode token. Call once after all layers are written."""
        self.cont_seq_len += 1

    def migrate_contiguous_to_pages(self):
        """
        Bulk-copy decode tokens from the contiguous buffer into the paged cache,
        then drop the contiguous buffers.  Called when cont_capacity is exceeded.
        Prefill pages are already populated (written during the prefill phase);
        only the decode portion needs appending.
        """
        if not self.cont_K:
            return
        decode_start = self.cont_prefill_len
        decode_end   = self.cont_seq_len
        if decode_end > decode_start:
            for layer_id in range(self.cont_num_layers):
                decode_K = self.cont_K[layer_id][:, :, decode_start:decode_end, :]
                decode_V = self.cont_V[layer_id][:, :, decode_start:decode_end, :]
                self.append_prefill(layer_id, decode_K, decode_V)
        self.drop_contiguous()

    def drop_contiguous(self):
        """Free contiguous buffers and revert to paged streaming."""
        self.cont_K.clear()
        self.cont_V.clear()
        self.use_contiguous = False

    def reset_contiguous(self):
        """Re-enable contiguous path for the next generate() (keeps buffers allocated)."""
        if self.cont_K:
            self.use_contiguous = True
            self.cont_seq_len = 0
            self.cont_prefill_len = 0

    def layers_keys(self):
        return sorted(self.layers.keys())

    def report_bytes_per_tier(self) -> dict:
        gpu_bytes = 0
        cpu_bytes = 0
        for layer in self.layers.values():
            for page in layer.pages:
                if page.state == 'DEAD':
                    continue
                page_bytes = page.k_tensor.numel() * page.k_tensor.element_size() * 2
                if page.state == 'GPU':
                    gpu_bytes += page_bytes
                elif page.state == 'CPU':
                    cpu_bytes += page_bytes
                    
        return {
            "gpu_active_bytes": gpu_bytes,
            "cpu_active_bytes": cpu_bytes
        }
        
        
# Persistent copy streams for async CPU→GPU KV page prefetch, keyed by device.
# A torch.cuda.Stream is bound to the device it was created on; keying by
# device (rather than one shared global) prevents a stream created for one
# CUDA device from being reused — incorrectly — against a different device.
# Lazily initialised on first call to perform_streaming_attention that needs it.
_kv_copy_streams: dict = {}


def _get_kv_copy_stream(device: torch.device) -> torch.cuda.Stream:
    stream = _kv_copy_streams.get(device)
    if stream is None:
        stream = torch.cuda.Stream(device=device)
        _kv_copy_streams[device] = stream
    return stream


def _async_kv_prefetch_enabled() -> bool:
    """Gate async CPU->GPU page prefetch for ablations and debugging."""
    return not bool(os.environ.get("HELM_DISABLE_ASYNC_KV"))


def _apply_softcap(attn_weights: torch.Tensor, softcap: Optional[float]) -> torch.Tensor:
    if softcap is None:
        return attn_weights
    return torch.tanh(attn_weights / softcap) * softcap


def perform_streaming_attention(
    query: torch.Tensor,
    pages: List[KVPage],
    scale: float = None,
    attention_mask: Optional[torch.Tensor] = None,
    softcap: Optional[float] = None,
):
    """
    Runtime attention helper that streams over KV pages using online softmax.

    Four execution paths depending on page residency:

    1. All GPU  — single F.scaled_dot_product_attention call (flash-attention
                  backend on CUDA; no Python loop, TC-enabled fp16 kernels).
    2. All CPU  — CPU-stage layers: query and all pages on CPU.  Concatenate
                  all pages and issue a single SDPA call; avoids the per-page
                  Python loop and repeated small-tensor allocations.
    3. Mixed    — GPU-stage layers with some pages evicted to CPU.  GPU pages
                  are batched into one matmul.  CPU pages are prefetched
                  asynchronously: all H2D copies are submitted upfront on a
                  dedicated copy stream with per-page CUDA events; the compute
                  stream waits for each event individually, overlapping the
                  H2D transfer of page N+1 with the matmul for page N.
    4. Mixed fallback — same as 3 but without CUDA (should never fire in
                  normal operation since CPU pages with a CUDA query only arise
                  from GPU-stage attention).
    """
    device = query.device
    num_q_heads = query.size(1)

    if scale is None:
        scale = 1.0 / math.sqrt(query.size(-1))

    dtype = query.dtype
    active_pages = [p for p in pages if p.used_tokens > 0 and p.state != 'DEAD']

    if not active_pages:
        return torch.zeros_like(query)

    # Partition active pages by device.  Exclude DEAD pages (evicted GPU pages
    # whose tensors have been replaced with empty placeholders).
    gpu_pages = [p for p in active_pages if p.state == 'GPU' and p.device.type == "cuda"]
    cpu_pages = [p for p in active_pages if p.state == 'CPU' and p.device.type != "cuda"]

    # Instrumentation only: when the query is on GPU, every
    # CPU-resident page must be streamed CPU->GPU for this attention call. This
    # is the single, path-independent definition of a prefetch this step.
    if device.type == "cuda" and cpu_pages:
        _cpu_bytes = sum(
            p.k_tensor[:, :, :p.used_tokens, :].numel() * p.k_tensor.element_size() * 2
            for p in cpu_pages
        )
        _record_prefetch(len(cpu_pages), _cpu_bytes)

    def _expand_kv(k: torch.Tensor, v: torch.Tensor):
        """Expand KV heads to match Q heads for grouped-query attention."""
        num_kv = k.size(1)
        if num_kv != num_q_heads:
            groups = num_q_heads // num_kv
            k = k.repeat_interleave(groups, dim=1)
            v = v.repeat_interleave(groups, dim=1)
        return k, v

    if attention_mask is not None or softcap is not None:
        q32 = query.float()
        bsz, _, q_len, _ = query.shape
        out = torch.zeros(bsz, num_q_heads, q_len, query.size(-1), device=device, dtype=torch.float32)
        running_max = torch.full((bsz, num_q_heads, q_len, 1), -float("inf"), device=device, dtype=torch.float32)
        running_sum = torch.zeros((bsz, num_q_heads, q_len, 1), device=device, dtype=torch.float32)

        use_prefetch = (
            device.type == "cuda"
            and torch.cuda.is_available()
            and _async_kv_prefetch_enabled()
            and any(p.device.type != device.type for p in active_pages)
        )

        if use_prefetch:
            _kv_copy_stream = _get_kv_copy_stream(device)

            page_chunks = []
            token_cursor = 0
            for page in active_pages:
                used = page.used_tokens
                if page.device.type == device.type:
                    k_buf = page.k_tensor[:, :, :used, :].float()
                    v_buf = page.v_tensor[:, :, :used, :].float()
                    ev = None
                else:
                    with torch.cuda.stream(_kv_copy_stream):
                        k_buf = page.k_tensor[:, :, :used, :].to(device, non_blocking=True).float()
                        v_buf = page.v_tensor[:, :, :used, :].to(device, non_blocking=True).float()
                        ev = torch.cuda.Event()
                        ev.record()
                page_mask = None
                if attention_mask is not None:
                    page_mask = attention_mask[:, :, :, token_cursor : token_cursor + used]
                page_chunks.append((k_buf, v_buf, ev, page_mask))
                token_cursor += used

            current_stream = torch.cuda.current_stream(device)
            for k_buf, v_buf, ev, page_mask in page_chunks:
                if ev is not None:
                    current_stream.wait_event(ev)
                    # k_buf/v_buf were allocated on the copy stream but are
                    # consumed here on current_stream. Without record_stream,
                    # the caching allocator only tracks the allocating stream
                    # and may recycle this memory for a new copy-stream
                    # allocation (e.g. the next page, or the next call) before
                    # current_stream's matmul below has actually read it —
                    # a silent cross-stream data race.
                    k_buf.record_stream(current_stream)
                    v_buf.record_stream(current_stream)
                k, v = _expand_kv(k_buf, v_buf)
                scores = torch.matmul(q32, k.transpose(-2, -1)) * scale
                scores = _apply_softcap(scores, softcap)
                if page_mask is not None:
                    scores = scores + page_mask.to(device=device, dtype=torch.float32)
                block_max = scores.amax(dim=-1, keepdim=True)
                new_max = torch.maximum(running_max, block_max)
                old_scale = torch.nan_to_num(torch.exp(running_max - new_max), nan=0.0)
                block_exp = torch.exp(scores - new_max)

                out = out * old_scale + torch.matmul(block_exp, v)
                running_sum = running_sum * old_scale + block_exp.sum(dim=-1, keepdim=True)
                running_max = new_max
        else:
            token_cursor = 0
            for page in active_pages:
                used = page.used_tokens
                k_buf = page.k_tensor[:, :, :used, :].to(device).float()
                v_buf = page.v_tensor[:, :, :used, :].to(device).float()
                k, v = _expand_kv(k_buf, v_buf)

                scores = torch.matmul(q32, k.transpose(-2, -1)) * scale
                scores = _apply_softcap(scores, softcap)
                if attention_mask is not None:
                    page_mask = attention_mask[:, :, :, token_cursor : token_cursor + used]
                    scores = scores + page_mask.to(device=device, dtype=torch.float32)
                block_max = scores.amax(dim=-1, keepdim=True)
                new_max = torch.maximum(running_max, block_max)
                old_scale = torch.nan_to_num(torch.exp(running_max - new_max), nan=0.0)
                block_exp = torch.exp(scores - new_max)

                out = out * old_scale + torch.matmul(block_exp, v)
                running_sum = running_sum * old_scale + block_exp.sum(dim=-1, keepdim=True)
                running_max = new_max
                token_cursor += used

        out = out / torch.clamp(running_sum, min=1e-12)
        return out.to(dtype)

    # ── Path 1: all pages GPU-resident ───────────────────────────────────────
    # Single SDPA call — flash-attention backend, TC-enabled fp16, no loop.
    # This path is decode-only (q_len == 1): no future tokens exist so causal
    # masking is a no-op.  If q_len > 1 (prefill), we must pass is_causal=True
    # to avoid attending to future positions.
    if not cpu_pages:
        K = torch.cat([p.k_tensor[:, :, :p.used_tokens, :] for p in gpu_pages], dim=2)
        V = torch.cat([p.v_tensor[:, :, :p.used_tokens, :] for p in gpu_pages], dim=2)
        K, V = _expand_kv(K, V)
        q_len = query.size(2)
        is_causal = q_len > 1
        return F.scaled_dot_product_attention(query, K, V, scale=scale, is_causal=is_causal)

    # ── Path 2: all pages CPU-resident, query on CPU ─────────────────────────
    # CPU-stage layers: no PCIe involved.  Concatenate all pages and call SDPA
    # once — avoids the N-iteration Python loop and per-page tensor allocations.
    if not gpu_pages and device.type != "cuda":
        K = torch.cat([p.k_tensor[:, :, :p.used_tokens, :] for p in cpu_pages], dim=2).float()
        V = torch.cat([p.v_tensor[:, :, :p.used_tokens, :] for p in cpu_pages], dim=2).float()
        K, V = _expand_kv(K, V)
        return F.scaled_dot_product_attention(query.float(), K, V, scale=scale).to(dtype)

    # ── Path 3: mixed residency, CUDA query (GPU-stage with evicted pages) ────
    # Submit all CPU→GPU copies upfront on a dedicated copy stream.  Record a
    # per-page CUDA event after each copy+cast.  The compute stream then waits
    # on each event in order, so it only stalls until *that page* is ready —
    # the H2D transfer for page N+1 runs concurrently with the matmul for page N.
    q32 = query.float()
    bsz, _, q_len, _ = query.shape
    out         = torch.zeros(bsz, num_q_heads, q_len, query.size(-1),
                              device=device, dtype=torch.float32)
    running_max = torch.full((bsz, num_q_heads, q_len, 1), -float("inf"),
                             device=device, dtype=torch.float32)
    running_sum = torch.zeros((bsz, num_q_heads, q_len, 1),
                              device=device, dtype=torch.float32)

    # GPU pages: one fused matmul
    if gpu_pages:
        K_gpu = torch.cat([p.k_tensor[:, :, :p.used_tokens, :] for p in gpu_pages], dim=2).float()
        V_gpu = torch.cat([p.v_tensor[:, :, :p.used_tokens, :] for p in gpu_pages], dim=2).float()
        K_gpu, V_gpu = _expand_kv(K_gpu, V_gpu)

        scores    = torch.matmul(q32, K_gpu.transpose(-2, -1)) * scale
        block_max = scores.amax(dim=-1, keepdim=True)
        new_max   = torch.maximum(running_max, block_max)
        old_scale = torch.nan_to_num(torch.exp(running_max - new_max), nan=0.0)
        block_exp = torch.exp(scores - new_max)

        out         = out * old_scale + torch.matmul(block_exp, V_gpu)
        running_sum = running_sum * old_scale + block_exp.sum(dim=-1, keepdim=True)
        running_max = new_max

    if not _async_kv_prefetch_enabled():
        for page in cpu_pages:
            k_buf = page.k_tensor[:, :, :page.used_tokens, :].to(device).float()
            v_buf = page.v_tensor[:, :, :page.used_tokens, :].to(device).float()
            k, v = _expand_kv(k_buf, v_buf)

            scores    = torch.matmul(q32, k.transpose(-2, -1)) * scale
            block_max = scores.amax(dim=-1, keepdim=True)
            new_max   = torch.maximum(running_max, block_max)
            old_scale = torch.nan_to_num(torch.exp(running_max - new_max), nan=0.0)
            block_exp = torch.exp(scores - new_max)

            out         = out * old_scale + torch.matmul(block_exp, v)
            running_sum = running_sum * old_scale + block_exp.sum(dim=-1, keepdim=True)
            running_max = new_max
    else:
        # Async prefetch all CPU pages to GPU, recording a per-page event.
        _kv_copy_stream = _get_kv_copy_stream(device)

        kv_bufs: List[tuple] = []
        for page in cpu_pages:
            with torch.cuda.stream(_kv_copy_stream):
                k_buf = page.k_tensor[:, :, :page.used_tokens, :].to(device, non_blocking=True).float()
                v_buf = page.v_tensor[:, :, :page.used_tokens, :].to(device, non_blocking=True).float()
                ev = torch.cuda.Event()
                ev.record()
            kv_bufs.append((k_buf, v_buf, ev))

        # Process each page: wait only for that page's event, so H2D for page N+1
        # overlaps with the matmul for page N. Use the query's device explicitly
        # (not the implicit default device) so this matches the device the copy
        # stream and events were created on.
        current_stream = torch.cuda.current_stream(device)
        for k_buf, v_buf, ev in kv_bufs:
            current_stream.wait_event(ev)
            # See the identical record_stream note above: k_buf/v_buf were
            # produced on _kv_copy_stream and must not be recycled by it while
            # current_stream is still reading them below.
            k_buf.record_stream(current_stream)
            v_buf.record_stream(current_stream)
            k, v = _expand_kv(k_buf, v_buf)

            scores    = torch.matmul(q32, k.transpose(-2, -1)) * scale
            block_max = scores.amax(dim=-1, keepdim=True)
            new_max   = torch.maximum(running_max, block_max)
            old_scale = torch.nan_to_num(torch.exp(running_max - new_max), nan=0.0)
            block_exp = torch.exp(scores - new_max)

            out         = out * old_scale + torch.matmul(block_exp, v)
            running_sum = running_sum * old_scale + block_exp.sum(dim=-1, keepdim=True)
            running_max = new_max

    out = out / torch.clamp(running_sum, min=1e-12)
    return out.to(dtype)
