from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig
from .model_norm_lora import LayerNorm, RMSNorm, make_norm, LoRALinear, apply_lora_adapters, freeze_non_lora_parameters, lora_state_dict, load_lora_state_dict, merge_lora_adapters, lora_parameter_count

class RotaryEmbedding(nn.Module):
    """Rotary positional embedding cache for attention heads."""

    def __init__(self, head_size: int, context_length: int, theta: float) -> None:
        """Create RoPE caches.

        Args:
            head_size: Attention head dimension.
            context_length: Maximum context length.
            theta: Frequency base.
        """

        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_size, 2).float() / head_size))
        positions = torch.arange(context_length, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, query: torch.Tensor, key: torch.Tensor, start_pos: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to query and key tensors.

        Args:
            query: Query tensor with shape ``[batch, heads, tokens, head_size]``.
            key: Key tensor with shape ``[batch, heads, tokens, head_size]``.
            start_pos: Absolute starting token position.

        Returns:
            Rotated query and key tensors.
        """

        token_count = query.size(-2)
        cos = self.cos[:, :, start_pos : start_pos + token_count, :]
        sin = self.sin[:, :, start_pos : start_pos + token_count, :]
        return (query * cos) + (_rotate_half(query) * sin), (key * cos) + (_rotate_half(key) * sin)


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension in RoPE pairs.

    Args:
        value: Tensor to rotate.

    Returns:
        Rotated tensor.
    """

    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class CausalSelfAttention(nn.Module):
    """Causal multi-head self-attention block."""

    def __init__(self, config: ModelConfig) -> None:
        """Create attention module.

        Args:
            config: Model architecture configuration.
        """

        super().__init__()
        self.head_count = config.head_count
        self.kv_head_count = config.resolved_kv_head_count()
        self.embedding_size = config.embedding_size
        self.position_encoding = config.position_encoding
        self.attention_backend = config.attention_backend
        self.attention_window = config.attention_window
        self.head_size = config.embedding_size // config.head_count
        self.kv_embedding_size = self.kv_head_count * self.head_size
        self.c_attn = nn.Linear(
            config.embedding_size,
            config.embedding_size + (2 * self.kv_embedding_size),
            bias=config.bias,
        )
        self.c_proj = nn.Linear(config.embedding_size, config.embedding_size, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.rotary = (
            RotaryEmbedding(self.head_size, config.context_length, config.rope_theta)
            if config.position_encoding == "rope"
            else None
        )
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.context_length, config.context_length, dtype=torch.bool)).view(
                1, 1, config.context_length, config.context_length
            ),
        )

    def forward(
        self,
        value: torch.Tensor,
        past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        start_pos: int = 0,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Apply causal self-attention.

        Args:
            value: Input hidden states.
            past_kv: Optional cached key/value tensors.
            start_pos: Absolute starting token position.
            use_cache: Whether to return updated key/value cache.

        Returns:
            Attention output tensor, plus cache when requested.
        """

        batch_size, token_count, channel_count = value.size()
        qkv = self.c_attn(value)
        query, key, val = qkv.split((self.embedding_size, self.kv_embedding_size, self.kv_embedding_size), dim=2)

        key = key.view(batch_size, token_count, self.kv_head_count, self.head_size).transpose(1, 2)
        query = query.view(batch_size, token_count, self.head_count, self.head_size).transpose(1, 2)
        val = val.view(batch_size, token_count, self.kv_head_count, self.head_size).transpose(1, 2)
        if self.rotary is not None:
            query, key = self.rotary(query, key, start_pos=start_pos)

        if past_kv is not None:
            past_key, past_val = past_kv
            key = torch.cat((past_key, key), dim=-2)
            val = torch.cat((past_val, val), dim=-2)
            if key.size(-2) > self.mask.size(-1):
                key = key[:, :, -self.mask.size(-1) :, :]
                val = val[:, :, -self.mask.size(-1) :, :]
        present = (key, val)
        expanded_key = self._expand_kv(key)
        expanded_val = self._expand_kv(val)

        key_count = expanded_key.size(-2)
        if past_kv is None:
            mask = self.mask[:, :, :token_count, :key_count]
        else:
            start = max(0, key_count - token_count)
            mask = self.mask[:, :, start : start + token_count, :key_count]
        if self.attention_window > 0:
            positions = torch.arange(key_count, device=value.device)
            query_positions = torch.arange(key_count - token_count, key_count, device=value.device)
            window_mask = positions[None, :] >= (query_positions[:, None] - self.attention_window + 1)
            mask = mask & window_mask.view(1, 1, token_count, key_count)

        if self.attention_backend == "sdpa" and hasattr(F, "scaled_dot_product_attention"):
            if self.attention_window <= 0 and past_kv is None:
                # Plain full-sequence causal attention (the common training
                # case): the mask built above is mathematically identical to
                # is_causal=True. Passing an explicit attn_mask tensor here
                # instead can prevent PyTorch from dispatching to the fused
                # FlashAttention kernel on supported hardware, falling back
                # to the slower/more memory-hungry "efficient" or "math"
                # backends even when the SDPA/Flash backend is selected.
                y = F.scaled_dot_product_attention(
                    query,
                    expanded_key,
                    expanded_val,
                    is_causal=True,
                    dropout_p=self.attn_dropout.p if self.training else 0.0,
                )
            else:
                attn_mask = mask[:, :, :, :].bool()
                y = F.scaled_dot_product_attention(
                    query,
                    expanded_key,
                    expanded_val,
                    attn_mask=attn_mask,
                    dropout_p=self.attn_dropout.p if self.training else 0.0,
                )
        else:
            attention = (query @ expanded_key.transpose(-2, -1)) * (1.0 / math.sqrt(expanded_key.size(-1)))
            attention = attention.masked_fill(mask == 0, float("-inf"))
            attention = F.softmax(attention, dim=-1)
            attention = self.attn_dropout(attention)
            y = attention @ expanded_val
        y = y.transpose(1, 2).contiguous().view(batch_size, token_count, channel_count)
        output = self.resid_dropout(self.c_proj(y))
        if use_cache:
            return output, present
        return output

    def _expand_kv(self, value: torch.Tensor) -> torch.Tensor:
        """Expand grouped key/value heads to query head count.

        Args:
            value: Key or value tensor with key/value head count.

        Returns:
            Tensor with one key/value head per query head.
        """

        if self.kv_head_count == self.head_count:
            return value
        repeat_count = self.head_count // self.kv_head_count
        return value.repeat_interleave(repeat_count, dim=1)


class MLP(nn.Module):
    """Feed-forward network inside a transformer block."""

    def __init__(self, config: ModelConfig) -> None:
        """Create feed-forward network.

        Args:
            config: Model architecture configuration.
        """

        super().__init__()
        self.mlp_type = config.mlp_type
        hidden_size = 4 * config.embedding_size
        if self.mlp_type == "swiglu":
            self.w1 = nn.Linear(config.embedding_size, hidden_size, bias=config.bias)
            self.w2 = nn.Linear(hidden_size, config.embedding_size, bias=config.bias)
            self.w3 = nn.Linear(config.embedding_size, hidden_size, bias=config.bias)
            self.dropout = nn.Dropout(config.dropout)
        else:
            self.net = nn.Sequential(
                nn.Linear(config.embedding_size, hidden_size, bias=config.bias),
                nn.GELU(),
                nn.Linear(hidden_size, config.embedding_size, bias=config.bias),
                nn.Dropout(config.dropout),
            )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Apply feed-forward transformation.

        Args:
            value: Input hidden states.

        Returns:
            Transformed hidden states.
        """

        if self.mlp_type == "swiglu":
            return self.dropout(self.w2(F.silu(self.w1(value)) * self.w3(value)))
        return self.net(value)


class Block(nn.Module):
    """Transformer block with attention and MLP."""

    def __init__(self, config: ModelConfig) -> None:
        """Create a transformer block.

        Args:
            config: Model architecture configuration.
        """

        super().__init__()
        self.ln_1 = make_norm(config)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = make_norm(config)
        self.mlp = MLP(config)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Apply transformer block.

        Args:
            value: Input hidden states.

        Returns:
            Updated hidden states.
        """

        value = value + self.attn(self.ln_1(value))
        value = value + self.mlp(self.ln_2(value))
        return value

    def forward_with_cache(
        self,
        value: torch.Tensor,
        past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Apply transformer block and return updated KV cache.

        Args:
            value: Input hidden states.
            past_kv: Optional cached key/value tensors.
            start_pos: Absolute starting token position.

        Returns:
            Updated hidden states and key/value cache.
        """

        attention_output, present = self.attn(self.ln_1(value), past_kv=past_kv, start_pos=start_pos, use_cache=True)
        value = value + attention_output
        value = value + self.mlp(self.ln_2(value))
        return value, present



