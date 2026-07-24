import contextlib
import torch.nn.functional as F

class BaseAttentionAdapter:
    def matches(self, model) -> bool:
        raise NotImplementedError

    @contextlib.contextmanager
    def patched(self, model):
        raise NotImplementedError

class Qwen2AttentionAdapter(BaseAttentionAdapter):
    def matches(self, model) -> bool:
        return model.__class__.__name__.lower().startswith("qwen2")

    @contextlib.contextmanager
    def patched(self, model):
        import transformers.models.qwen2.modeling_qwen2 as modeling_qwen2
        old_forward = modeling_qwen2.Qwen2Attention.forward

        def patched_forward(attn_self, hidden_states, *args, **kwargs):
            bsz = hidden_states.size(0)
            q_len = hidden_states.size(1)

            query_states = attn_self.q_proj(hidden_states)
            key_states = attn_self.k_proj(hidden_states)
            value_states = attn_self.v_proj(hidden_states)

            head_dim = attn_self.head_dim
            num_heads = query_states.shape[-1] // head_dim
            num_kv_heads = key_states.shape[-1] // head_dim

            query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

            kv_seq_len = key_states.shape[-2]
            past = kwargs.get("past_key_value")
            if past is not None:
                kv_seq_len += past.get_usable_length(kv_seq_len, attn_self.layer_idx)

            if "position_embeddings" in kwargs:
                cos, sin = kwargs["position_embeddings"]
            else:
                # Some versions might pass them differently or we might need to compute them
                # if we're tracing from a point where they aren't pre-computed.
                # For Qwen2.5-1.5B with current transformers, they should be in kwargs.
                raise RuntimeError("Qwen2AttentionAdapter: 'position_embeddings' not found in kwargs")

            query_states, key_states = modeling_qwen2.apply_rotary_pos_emb(
                query_states, key_states, cos, sin, kwargs.get("position_ids")
            )

            if past is not None:
                cache_kwargs = {"sin": sin, "cos": cos}
                cache_position = kwargs.get("cache_position")
                if cache_position is not None:
                    cache_kwargs["cache_position"] = cache_position
                key_states, value_states = past.update(
                    key_states, value_states, attn_self.layer_idx, cache_kwargs
                )

            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=kwargs.get("attention_mask"),
                dropout_p=attn_self.attention_dropout if attn_self.training else 0.0,
                is_causal=False,
            )

            attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
            attn_output = attn_self.o_proj(attn_output)
            return attn_output, None

        modeling_qwen2.Qwen2Attention.forward = patched_forward
        try:
            yield
        finally:
            modeling_qwen2.Qwen2Attention.forward = old_forward

class LlamaAttentionAdapter(BaseAttentionAdapter):
    def matches(self, model) -> bool:
        name = model.__class__.__name__.lower()
        return name.startswith("llama") or name.startswith("mistral")

    @contextlib.contextmanager
    def patched(self, model):
        import transformers.models.llama.modeling_llama as modeling_llama
        # Mistral uses very similar structure, often we can share the patch if attributes match
        # For simplicity, focus on Llama first.
        old_forward = modeling_llama.LlamaAttention.forward

        def patched_forward(attn_self, hidden_states, *args, **kwargs):
            bsz = hidden_states.size(0)
            q_len = hidden_states.size(1)

            query_states = attn_self.q_proj(hidden_states)
            key_states = attn_self.k_proj(hidden_states)
            value_states = attn_self.v_proj(hidden_states)

            head_dim = attn_self.head_dim
            num_heads = query_states.shape[-1] // head_dim
            num_kv_heads = key_states.shape[-1] // head_dim

            query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

            kv_seq_len = key_states.shape[-2]
            past = kwargs.get("past_key_value")
            if past is not None:
                kv_seq_len += past.get_usable_length(kv_seq_len, attn_self.layer_idx)

            cos, sin = attn_self.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = modeling_llama.apply_rotary_pos_emb(
                query_states, key_states, cos, sin, kwargs.get("position_ids")
            )

            if past is not None:
                cache_kwargs = {"sin": sin, "cos": cos}
                cache_position = kwargs.get("cache_position")
                if cache_position is not None:
                    cache_kwargs["cache_position"] = cache_position
                key_states, value_states = past.update(
                    key_states, value_states, attn_self.layer_idx, cache_kwargs
                )

            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=kwargs.get("attention_mask"),
                dropout_p=attn_self.attention_dropout if attn_self.training else 0.0,
                is_causal=False,
            )

            attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
            attn_output = attn_self.o_proj(attn_output)
            return attn_output, None

        modeling_llama.LlamaAttention.forward = patched_forward
        try:
            yield
        finally:
            modeling_llama.LlamaAttention.forward = old_forward

def resolve_attention_adapter(model):
    adapters = [
        Qwen2AttentionAdapter(),
        LlamaAttentionAdapter(),
    ]
    for adapter in adapters:
        if adapter.matches(model):
            return adapter
    raise NotImplementedError(
        f"No decode attention adapter registered for {model.__class__.__name__}"
    )
