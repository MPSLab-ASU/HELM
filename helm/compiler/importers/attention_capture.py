import contextlib
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Any


@dataclass
class CapturedQKV:
    q: Optional[Any] = None
    k: Optional[Any] = None
    v: Optional[Any] = None


class BaseAttentionCaptureAdapter:
    def matches(self, model) -> bool:
        raise NotImplementedError

    @contextlib.contextmanager
    def patch(self, model, capture: CapturedQKV):
        raise NotImplementedError


class Qwen2AttentionCaptureAdapter(BaseAttentionCaptureAdapter):
    def matches(self, model) -> bool:
        name = model.__class__.__name__.lower()
        return "qwen2" in name

    @contextlib.contextmanager
    def patch(self, model, capture: CapturedQKV):
        import transformers.models.qwen2.modeling_qwen2 as modeling_qwen2

        old_forward = modeling_qwen2.Qwen2Attention.forward

        def patched_forward(
            attn_self,
            hidden_states,
            position_embeddings=None,
            attention_mask=None,
            past_key_value=None,
            cache_position=None,
            **kwargs,
        ):
            bsz, q_len, _ = hidden_states.shape

            q = attn_self.q_proj(hidden_states)
            k = attn_self.k_proj(hidden_states)
            v = attn_self.v_proj(hidden_states)

            head_dim = attn_self.head_dim
            num_heads = q.shape[-1] // head_dim
            num_kv_heads = k.shape[-1] // head_dim

            q = q.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
            k = k.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
            v = v.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

            capture.q = q
            capture.k = k
            capture.v = v

            # Re-implement FX-friendly forward to avoid Proxy iteration on *input_shape
            # (which happens in the original modeling_qwen2.py)
            
            # 1. Apply RoPE
            if position_embeddings is None:
                raise RuntimeError("Qwen2AttentionCaptureAdapter: position_embeddings is missing")
            
            cos, sin = position_embeddings
            q, k = modeling_qwen2.apply_rotary_pos_emb(q, k, cos, sin)

            # 2. Update Cache
            if past_key_value is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                k, v = past_key_value.update(k, v, attn_self.layer_idx, cache_kwargs)

            repeat_factor = num_heads // num_kv_heads
            k_attn = k.repeat_interleave(repeat_factor, dim=1)
            v_attn = v.repeat_interleave(repeat_factor, dim=1)

            # 3. Attention
            attn_output = F.scaled_dot_product_attention(
                q, k_attn, v_attn,
                attn_mask=attention_mask,
                dropout_p=attn_self.attention_dropout if attn_self.training else 0.0,
                is_causal=False,
            )

            # 4. Output projection
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(bsz, q_len, -1)
            attn_output = attn_self.o_proj(attn_output)

            return attn_output, None

        modeling_qwen2.Qwen2Attention.forward = patched_forward
        try:
            yield
        finally:
            modeling_qwen2.Qwen2Attention.forward = old_forward


class LlamaAttentionCaptureAdapter(BaseAttentionCaptureAdapter):
    def matches(self, model) -> bool:
        name = model.__class__.__name__.lower()
        return "llama" in name or "mistral" in name

    @contextlib.contextmanager
    def patch(self, model, capture: CapturedQKV):
        import transformers.models.llama.modeling_llama as modeling_llama

        old_forward = modeling_llama.LlamaAttention.forward

        def patched_forward(
            attn_self,
            hidden_states,
            position_embeddings=None,
            attention_mask=None,
            past_key_value=None,
            cache_position=None,
            **kwargs,
        ):
            bsz, q_len, _ = hidden_states.shape

            q = attn_self.q_proj(hidden_states)
            k = attn_self.k_proj(hidden_states)
            v = attn_self.v_proj(hidden_states)

            head_dim = attn_self.head_dim
            num_heads = q.shape[-1] // head_dim
            num_kv_heads = k.shape[-1] // head_dim

            q = q.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
            k = k.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
            v = v.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

            capture.q = q
            capture.k = k
            capture.v = v

            # 1. Apply RoPE
            if position_embeddings is None:
                 raise RuntimeError("LlamaAttentionCaptureAdapter: position_embeddings is missing")
            
            cos, sin = position_embeddings
            q, k = modeling_llama.apply_rotary_pos_emb(q, k, cos, sin)

            # 2. Update Cache
            if past_key_value is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                k, v = past_key_value.update(k, v, attn_self.layer_idx, cache_kwargs)

            repeat_factor = num_heads // num_kv_heads
            k_attn = k.repeat_interleave(repeat_factor, dim=1)
            v_attn = v.repeat_interleave(repeat_factor, dim=1)

            # 3. Attention
            attn_output = F.scaled_dot_product_attention(
                q, k_attn, v_attn,
                attn_mask=attention_mask,
                dropout_p=attn_self.attention_dropout if attn_self.training else 0.0,
                is_causal=False,
            )

            # 4. Output projection
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(bsz, q_len, -1)
            attn_output = attn_self.o_proj(attn_output)

            return attn_output, None

        modeling_llama.LlamaAttention.forward = patched_forward
        try:
            yield
        finally:
            modeling_llama.LlamaAttention.forward = old_forward


class Qwen3AttentionCaptureAdapter(BaseAttentionCaptureAdapter):
    def matches(self, model) -> bool:
        name = model.__class__.__name__.lower()
        return "qwen3" in name

    @contextlib.contextmanager
    def patch(self, model, capture: CapturedQKV):
        import transformers.models.qwen3.modeling_qwen3 as modeling_qwen3

        old_forward = modeling_qwen3.Qwen3Attention.forward

        def patched_forward(
            attn_self,
            hidden_states,
            position_embeddings,
            attention_mask=None,
            past_key_values=None,
            cache_position=None,
            **kwargs,
        ):
            bsz, q_len, _ = hidden_states.shape

            head_dim = attn_self.head_dim
            hidden_shape = (bsz, q_len, -1, head_dim)

            q = attn_self.q_norm(attn_self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            k = attn_self.k_norm(attn_self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            v = attn_self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            cos, sin = position_embeddings
            q, k = modeling_qwen3.apply_rotary_pos_emb(q, k, cos, sin)

            capture.q = q
            capture.k = k
            capture.v = v

            if past_key_values is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                k, v = past_key_values.update(k, v, attn_self.layer_idx, cache_kwargs)

            k_attn = modeling_qwen3.repeat_kv(k, attn_self.num_key_value_groups)
            v_attn = modeling_qwen3.repeat_kv(v, attn_self.num_key_value_groups)

            attn_output = F.scaled_dot_product_attention(
                q,
                k_attn,
                v_attn,
                attn_mask=attention_mask,
                dropout_p=attn_self.attention_dropout if attn_self.training else 0.0,
                is_causal=False,
            )

            attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
            attn_output = attn_self.o_proj(attn_output)

            return attn_output, None

        modeling_qwen3.Qwen3Attention.forward = patched_forward
        try:
            yield
        finally:
            modeling_qwen3.Qwen3Attention.forward = old_forward


def resolve_capture_adapter(model):
    adapters = [
        Qwen3AttentionCaptureAdapter(),
        Qwen2AttentionCaptureAdapter(),
        LlamaAttentionCaptureAdapter(),
    ]
    for adapter in adapters:
        if adapter.matches(model):
            return adapter
    raise NotImplementedError(
        f"No decode attention adapter registered for {model.__class__.__name__}"
    )
