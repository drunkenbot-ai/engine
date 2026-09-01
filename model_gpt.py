from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional
from .config import ModelConfig
from .model_norm_lora import make_norm
from .model_transformer import Block


class MicroGPT(nn.Module):
    """Small GPT-style causal language model."""

    def __init__(self, config: ModelConfig) -> None:
        """Create the model.

        Args:
            config: Model architecture configuration.
        """

        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_size)
        self.position_embedding = (
            nn.Embedding(config.context_length, config.embedding_size)
            if config.position_encoding == "learned"
            else None
        )
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.layer_count)])
        self.ln_f = make_norm(config)
        self.lm_head = nn.Linear(config.embedding_size, config.vocab_size, bias=False)
        self.gradient_checkpointing = False
        self.token_embedding.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def enable_gradient_checkpointing(self, enabled: bool = True) -> None:
        """Trade extra compute for substantially lower activation memory."""

        self.gradient_checkpointing = bool(enabled)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize module weights.

        Args:
            module: Module to initialize.
        """

        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            idx: Token IDs with shape ``[batch, tokens]``.

        Returns:
            Logits with shape ``[batch, tokens, vocab]``.

        Raises:
            ValueError: If the sequence is longer than context length.
        """

        _, token_count = idx.size()
        if token_count > self.config.context_length:
            raise ValueError("Input sequence is longer than context_length")
        value = self.token_embedding(idx)
        if self.position_embedding is not None:
            positions = torch.arange(0, token_count, dtype=torch.long, device=idx.device)
            value = value + self.position_embedding(positions)
        value = self.drop(value)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                value = checkpoint(block, value, use_reentrant=False)
            else:
                value = block(value)
        value = self.ln_f(value)
        return self.lm_head(value)

    def forward_with_cache(
        self,
        idx: torch.Tensor,
        past_kv: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Run a forward pass and return updated KV cache.

        Args:
            idx: Token IDs with shape ``[batch, tokens]``.
            past_kv: Optional per-layer key/value cache.
            start_pos: Absolute starting token position.

        Returns:
            Logits and updated per-layer KV cache.
        """

        _, token_count = idx.size()
        if token_count > self.config.context_length:
            raise ValueError("Input sequence is longer than context_length")
        value = self.token_embedding(idx)
        if self.position_embedding is not None:
            positions = torch.arange(start_pos, start_pos + token_count, dtype=torch.long, device=idx.device)
            positions = positions.clamp(max=self.config.context_length - 1)
            value = value + self.position_embedding(positions)
        value = self.drop(value)
        next_cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        for index, block in enumerate(self.blocks):
            layer_cache = past_kv[index] if past_kv is not None and index < len(past_kv) else None
            value, present = block.forward_with_cache(value, past_kv=layer_cache, start_pos=start_pos)
            next_cache.append(present)
        value = self.ln_f(value)
        return self.lm_head(value), next_cache

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
        use_kv_cache: bool = True,
    ) -> torch.Tensor:
        """Autoregressively sample new tokens.

        Args:
            idx: Starting token IDs.
            max_new_tokens: Number of tokens to generate.
            temperature: Sampling temperature.
            top_k: Optional top-k cutoff.
            use_kv_cache: Whether to reuse key/value tensors during generation.

        Returns:
            Token IDs including the original context and generated tokens.
        """

        if max_new_tokens <= 0:
            return idx

        past_kv: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None
        cached_logits: Optional[torch.Tensor] = None
        if use_kv_cache:
            idx_cond = idx[:, -self.config.context_length :]
            cached_logits, past_kv = self.forward_with_cache(idx_cond, start_pos=0)

        for step in range(max_new_tokens):
            if use_kv_cache and cached_logits is not None:
                logits = cached_logits[:, -1, :] / max(temperature, 1e-5)
            else:
                idx_cond = idx[:, -self.config.context_length :]
                logits = self(idx_cond)[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            if use_kv_cache and step < max_new_tokens - 1:
                if past_kv:
                    cached_length = int(past_kv[0][0].size(-2))
                    rolling_start = min(cached_length, self.config.context_length - 1)
                    cached_logits, past_kv = self.forward_with_cache(
                        idx[:, -1:],
                        past_kv=past_kv,
                        start_pos=rolling_start,
                    )
                else:
                    idx_cond = idx[:, -self.config.context_length :]
                    cached_logits, past_kv = self.forward_with_cache(idx_cond, start_pos=0)
        return idx

