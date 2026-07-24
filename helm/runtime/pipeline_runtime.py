import logging
import os
from typing import Callable, Optional

import torch

logger = logging.getLogger(__name__)


class PipelineRuntime:
    """
    Sequential CPU-GPU pipeline runtime.

    Architecture:
        - Prefill:  run the decode executor on the prompt.  Long prompts are
                    processed in chunks (``HELM_PREFILL_CHUNK``, default 512) so
                    activation memory stays O(chunk) instead of O(seq_len).
                    KV accumulates across chunks; only the final chunk's logits
                    seed the first decode token.
        - Decode:   run the decode executor one token at a time.
                    Each call extends the same DynamicCache, giving the
                    model full context over all previous tokens.

    Both GPU stages and CPU stages share the same DynamicCache object
    (it lives as a constant inside the traced wrapper module).  GPU layers
    write their KV to GPU tensors; CPU layers write theirs to CPU tensors.
    No cross-device copies are needed because each layer only accesses its
    own slice of the cache.
    """

    def __init__(
        self,
        prefill_executor,
        decode_executor,
        tokenizer=None,
        dtype=torch.bfloat16,
        kv_offload_mgr=None,
        decode_wrapper=None,
    ):
        self.prefill_executor = prefill_executor   # kept for future use
        self.decode_executor = decode_executor
        self.tokenizer = tokenizer
        self.dtype = dtype
        self.kv_offload_mgr = kv_offload_mgr      # KVOffloadManager or None
        self.decode_wrapper = decode_wrapper       # direct ref to _Wrapper for reliable reset

    # ------------------------------------------------------------------ #
    #  Mask helpers
    # ------------------------------------------------------------------ #

    def _normalize_prompt_attention_mask(self, input_ids, attention_mask=None):
        if attention_mask is None:
            return torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)
        if attention_mask.dim() != 2:
            raise ValueError("attention_mask must have shape [batch, seq_len]")
        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                f"attention_mask shape {tuple(attention_mask.shape)} does not match "
                f"input_ids shape {tuple(input_ids.shape)}"
            )
        return attention_mask.to(device=input_ids.device, dtype=torch.long)

    def _build_position_ids(self, attention_mask):
        positions = attention_mask.cumsum(dim=1) - 1
        positions = positions.clamp_min(0)
        return positions * attention_mask

    def _build_causal_mask(self, seq_len, device, attention_mask=None):
        """Full causal mask for prefill: (B, 1, S, S) when padding is present."""
        min_val = torch.finfo(self.dtype).min
        mask = torch.full((seq_len, seq_len), min_val, device=device, dtype=self.dtype)
        mask = torch.triu(mask, diagonal=1)
        mask = mask[None, None, :, :]
        if attention_mask is None:
            return mask
        key_padding = (1.0 - attention_mask.to(device=device, dtype=self.dtype)) * min_val
        return mask + key_padding[:, None, None, :]

    def _build_decode_mask(self, total_len, device, attention_mask=None):
        """Decode mask: (1, 1, 1, total_len).  All prior positions visible."""
        if attention_mask is None:
            return torch.zeros((1, 1, 1, total_len), device=device, dtype=self.dtype)
        if attention_mask.dim() != 2:
            raise ValueError("decode attention_mask must have shape [batch, total_len]")
        if attention_mask.shape[1] != total_len:
            raise ValueError(
                f"decode attention_mask width {attention_mask.shape[1]} does not match "
                f"total_len={total_len}"
            )
        min_val = torch.finfo(self.dtype).min
        key_padding = (1.0 - attention_mask.to(device=device, dtype=self.dtype)) * min_val
        return key_padding[:, None, None, :]

    def _build_chunk_prefill_mask(self, past, chunk_len, device, attention_mask=None):
        """Causal mask for one prefill chunk: queries at [past, past+chunk_len)
        attend to keys [0, past+chunk_len).  Shape (1, 1, chunk_len, past+chunk_len).
        """
        total = past + chunk_len
        min_val = torch.finfo(self.dtype).min
        q_idx = torch.arange(chunk_len, device=device).view(chunk_len, 1)
        k_idx = torch.arange(total, device=device).view(1, total)
        allowed = k_idx <= (past + q_idx)
        mask = torch.where(
            allowed,
            torch.zeros((), device=device, dtype=self.dtype),
            torch.full((), min_val, device=device, dtype=self.dtype),
        )
        mask = mask[None, None, :, :]
        if attention_mask is not None:
            key_padding = (1.0 - attention_mask[:, :total].to(
                device=device, dtype=self.dtype)) * min_val
            mask = mask + key_padding[:, None, None, :]
        return mask

    def _prefill_chunk_size(self, seq_len: int) -> int:
        """Return chunk size for long prompts; 0 means single-pass prefill."""
        raw = os.environ.get("HELM_PREFILL_CHUNK", "512")
        try:
            chunk = int(raw)
        except ValueError:
            chunk = 512
        if chunk <= 0 or seq_len <= chunk:
            return 0
        return chunk

    def _select_next_token_logits(self, logits, prompt_lengths):
        # The decode graph slices hidden states to the last position before
        # lm_head (see decode_tracer), so prefill logits arrive pre-reduced as
        # [batch, 1, vocab]. In that case the single row IS the last position;
        # just drop the seq dim. The gather path is kept for the legacy/full
        # [batch, seq_len, vocab] shape (per-sequence last-valid position).
        if logits.shape[1] == 1:
            return logits[:, -1, :]
        last_positions = prompt_lengths.to(device=logits.device, dtype=torch.long).clamp_min(1) - 1
        gather_index = last_positions.view(-1, 1, 1).expand(-1, 1, logits.shape[-1])
        return logits.gather(1, gather_index).squeeze(1)

    # ------------------------------------------------------------------ #
    #  Prefill
    # ------------------------------------------------------------------ #

    def prefill(self, input_ids, attention_mask=None):
        """
        Run prefill using the decode executor.

        Short prompts use one forward pass.  Longer prompts are split into
        chunks of ``HELM_PREFILL_CHUNK`` tokens (default 512; set 0 to force a
        single pass).  KV is accumulated across chunks; logits from the final
        chunk seed the first decode token (the graph only materialises logits
        for the last position in each chunk).
        """
        device = input_ids.device
        seq_len = input_ids.shape[1]
        prompt_attention_mask = self._normalize_prompt_attention_mask(input_ids, attention_mask)

        chunk_size = self._prefill_chunk_size(seq_len)
        if chunk_size == 0:
            causal_mask = self._build_causal_mask(seq_len, device, prompt_attention_mask)
            position_ids = self._build_position_ids(prompt_attention_mask)
            cache_position = torch.arange(seq_len, dtype=torch.long, device=device)
            outputs = self.decode_executor.run({
                "input_ids": input_ids,
                "attention_mask": causal_mask,
                "position_ids": position_ids,
                "cache_position": cache_position,
            })
            if isinstance(outputs, dict):
                return outputs["logits"]
            return outputs

        full_position_ids = self._build_position_ids(prompt_attention_mask)
        last_logits = None
        past = 0
        while past < seq_len:
            chunk_len = min(chunk_size, seq_len - past)
            chunk_mask = self._build_chunk_prefill_mask(
                past, chunk_len, device, prompt_attention_mask)
            outputs = self.decode_executor.run({
                "input_ids": input_ids[:, past:past + chunk_len],
                "attention_mask": chunk_mask,
                "position_ids": full_position_ids[:, past:past + chunk_len],
                "cache_position": torch.arange(
                    past, past + chunk_len, dtype=torch.long, device=device),
            })
            if isinstance(outputs, dict):
                last_logits = outputs["logits"]
            else:
                last_logits = outputs
            past += chunk_len
        return last_logits

    # ------------------------------------------------------------------ #
    #  Decode step
    # ------------------------------------------------------------------ #

    def decode_step(self, input_ids, step_position, attention_mask=None, position_ids=None):
        """
        Single autoregressive decode step.

        The DynamicCache is updated in-place by the stage modules, so
        the model attends to all prompt tokens plus all previously
        generated tokens without any extra bookkeeping here.
        """
        device = input_ids.device
        batch_size = input_ids.shape[0]

        if position_ids is None:
            position_ids = torch.full(
                (batch_size, 1),
                step_position,
                dtype=torch.long,
                device=device,
            )
        else:
            position_ids = position_ids.to(device=device, dtype=torch.long)
        cache_position = torch.tensor([step_position], dtype=torch.long, device=device)
        decode_mask = self._build_decode_mask(step_position + 1, device, attention_mask)

        outputs = self.decode_executor.run({
            "input_ids": input_ids,
            "attention_mask": decode_mask,
            "position_ids": position_ids,
            "cache_position": cache_position,
        })

        if isinstance(outputs, dict):
            return outputs["logits"]
        return outputs

    # ------------------------------------------------------------------ #
    #  Generation loop
    # ------------------------------------------------------------------ #

    def _reset_decode_cache(self):
        """
        Reset the DynamicCache in the decode wrapper before each generation.

        Fast path: if self.decode_wrapper is set (direct reference to the
        _Wrapper from DecodeTracer), call reset_cache() on it immediately.

        Slow path: search stage submodules for a _Wrapper by checking for the
        reset_cache() method.  This handles the rare case where decode_wrapper
        was not passed at construction time.  We look for the callable directly
        on the module rather than introspecting hook closure internals, which
        is an implementation detail that breaks if the hook uses weakrefs.
        """
        if self.decode_wrapper is not None:
            self.decode_wrapper.reset_cache()
            return
        for stage in self.decode_executor.stages:
            for _, module in stage.module.named_modules():
                if callable(getattr(module, "reset_cache", None)) and hasattr(module, "_kv_hooks"):
                    module.reset_cache()
                    return

    def generate(
        self,
        input_ids,
        max_new_tokens=8,
        eos_token_id=None,
        pad_token_id=None,
        attention_mask=None,
        token_callback: Optional[Callable[[int, torch.Tensor, torch.Tensor], None]] = None,
    ):
        """
        Autoregressive generation.

        1. Prefill  → seeds DynamicCache with all prompt KV, returns logits.
        2. Decode loop → each step extends DynamicCache by one token.
        """
        if self.kv_offload_mgr is not None:
            self.kv_offload_mgr.reset()
        # Always reset DynamicCache regardless of kv_offload_mgr.
        # kv_offload_mgr.reset() only clears the paged KV cache; the
        # DynamicCache embedded in the executor stages must also be cleared so
        # that _update_causal_mask sees past_seen_tokens=0 on the next run.
        self._reset_decode_cache()
        seq_len = input_ids.shape[1]
        prompt_attention_mask = self._normalize_prompt_attention_mask(input_ids, attention_mask)
        prompt_lengths = prompt_attention_mask.sum(dim=1)
        logits = self.prefill(input_ids, attention_mask=prompt_attention_mask)
        next_token_logits = self._select_next_token_logits(logits, prompt_lengths)

        generated = []
        batch_size = input_ids.shape[0]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=next_token_logits.device)
        terminal_token_id = pad_token_id if pad_token_id is not None else eos_token_id
        decode_attention_mask = prompt_attention_mask.to(next_token_logits.device)
        decode_position_ids = prompt_lengths.to(next_token_logits.device).clone()
        for step in range(max_new_tokens):
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            if finished.any() and terminal_token_id is not None:
                next_token = next_token.clone()
                next_token[finished] = terminal_token_id
            generated.append(next_token)
            if eos_token_id is not None:
                finished = finished | (next_token.squeeze(-1) == eos_token_id)
            if token_callback is not None:
                token_callback(step, next_token.detach().clone(), finished.detach().clone())
            if bool(finished.all()):
                break
            if step + 1 >= max_new_tokens:
                break
            active_rows = ~finished
            decode_attention_mask = torch.cat(
                [
                    decode_attention_mask,
                    active_rows.unsqueeze(1).to(
                        dtype=decode_attention_mask.dtype,
                        device=decode_attention_mask.device,
                    ),
                ],
                dim=1,
            )
            step_position = seq_len + step
            # When KV offload is active, the paged KV cache owns all KV state.
            # Reset DynamicCache each step so it never accumulates stale GPU KV
            # tensors that bypass the watermark-eviction logic.
            if self.kv_offload_mgr is not None:
                self._reset_decode_cache()
            logits = self.decode_step(
                next_token,
                step_position,
                attention_mask=decode_attention_mask,
                position_ids=decode_position_ids.unsqueeze(1),
            )
            decode_position_ids = decode_position_ids + active_rows.to(decode_position_ids.dtype)
            next_token_logits = logits[:, -1, :]

        generations = torch.cat(generated, dim=1) if generated else torch.empty(
            (batch_size, 0), dtype=torch.long, device=input_ids.device
        )

        if self.tokenizer:
            decoded = self.tokenizer.batch_decode(generations, skip_special_tokens=True)
            logger.debug("[Generated Token Sequence]: %s", decoded[0])

        return generations
