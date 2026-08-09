from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class LayerNorm(nn.Module):
    """Layer normalization with optional bias."""

    def __init__(self, size: int, bias: bool) -> None:
        """Create layer normalization.

        Args:
            size: Feature dimension.
            bias: Whether to include a bias vector.
        """

        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.bias = nn.Parameter(torch.zeros(size)) if bias else None

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Normalize an input tensor.

        Args:
            value: Tensor to normalize.

        Returns:
            Normalized tensor.
        """

        return F.layer_norm(value, self.weight.shape, self.weight, self.bias, 1e-5)


class RMSNorm(nn.Module):
    """Root mean square normalization used by Llama-style models."""

    def __init__(self, size: int, eps: float = 1e-6) -> None:
        """Create RMSNorm.

        Args:
            size: Feature dimension.
            eps: Numerical stability value.
        """

        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Normalize by root mean square.

        Args:
            value: Input tensor.

        Returns:
            Normalized tensor.
        """

        return self.weight * value * torch.rsqrt(value.pow(2).mean(dim=-1, keepdim=True) + self.eps)


def make_norm(config: ModelConfig) -> nn.Module:
    """Create the configured normalization layer.

    Args:
        config: Model configuration.

    Returns:
        Normalization module.
    """

    if config.norm_type == "rmsnorm":
        return RMSNorm(config.embedding_size)
    return LayerNorm(config.embedding_size, bias=config.bias)


class LoRALinear(nn.Module):
    """Linear layer with trainable low-rank LoRA adapters."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float) -> None:
        """Create a LoRA wrapper around an existing linear layer.

        Args:
            base: Frozen base linear layer.
            rank: Adapter rank.
            alpha: LoRA scaling alpha.
            dropout: Dropout probability before the adapter.
        """

        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Parameter(torch.zeros(rank, base.in_features, device=base.weight.device, dtype=base.weight.dtype))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Apply the base projection plus LoRA update.

        Args:
            value: Input tensor.

        Returns:
            Projected tensor.
        """

        update = F.linear(F.linear(self.dropout(value), self.lora_a), self.lora_b) * self.scaling
        return self.base(value) + update

    def merged_linear(self) -> nn.Linear:
        """Return a plain linear layer with LoRA weights merged.

        Returns:
            Linear layer equivalent to base plus LoRA update.
        """

        merged = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None)
        merged.weight.data.copy_(self.base.weight.data + (self.lora_b @ self.lora_a) * self.scaling)
        if self.base.bias is not None and merged.bias is not None:
            merged.bias.data.copy_(self.base.bias.data)
        return merged


def _set_nested_module(root: nn.Module, module_name: str, module: nn.Module) -> None:
    """Replace a nested module by dotted name.

    Args:
        root: Root module.
        module_name: Dotted module name.
        module: Replacement module.
    """

    parent_name, child_name = module_name.rsplit(".", 1) if "." in module_name else ("", module_name)
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, child_name, module)


def _lora_target_names(model: nn.Module, target_modules: str) -> set[str]:
    """Resolve LoRA target module names.

    Args:
        model: Model to inspect.
        target_modules: Comma-separated target groups.

    Returns:
        Set of module names to wrap.
    """

    groups = {part.strip().lower() for part in target_modules.split(",") if part.strip()}
    if "all" in groups:
        groups.update({"attention", "mlp"})
    names: set[str] = set()
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name.endswith("lm_head"):
            continue
        is_attention = ".attn." in name
        is_mlp = ".mlp." in name
        if ("attention" in groups and is_attention) or ("mlp" in groups and is_mlp):
            names.add(name)
    return names


def apply_lora_adapters(model: nn.Module, rank: int, alpha: float, dropout: float, target_modules: str) -> int:
    """Attach LoRA adapters to selected linear layers.

    Args:
        model: Model to modify in place.
        rank: LoRA rank.
        alpha: LoRA alpha.
        dropout: LoRA dropout.
        target_modules: Comma-separated target groups.

    Returns:
        Number of wrapped modules.
    """

    names = _lora_target_names(model, target_modules)
    for name in sorted(names):
        module = model.get_submodule(name)
        if isinstance(module, nn.Linear):
            _set_nested_module(model, name, LoRALinear(module, rank, alpha, dropout))
    return len(names)


def freeze_non_lora_parameters(model: nn.Module) -> None:
    """Freeze all parameters except LoRA adapter parameters.

    Args:
        model: Model to update.
    """

    for name, parameter in model.named_parameters():
        parameter.requires_grad_(("lora_a" in name) or ("lora_b" in name))


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return trainable LoRA adapter tensors.

    Args:
        model: Model containing LoRA adapters.

    Returns:
        LoRA-only state dictionary.
    """

    return {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if ".lora_a" in name or ".lora_b" in name
    }


def load_lora_state_dict(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Load LoRA adapter tensors into a model.

    Args:
        model: Model containing LoRA adapters.
        state: LoRA-only state dictionary.
    """

    model.load_state_dict(state, strict=False)


def merge_lora_adapters(model: nn.Module) -> int:
    """Merge LoRA adapters into plain linear layers.

    Args:
        model: Model to modify in place.

    Returns:
        Number of merged LoRA modules.
    """

    merged = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            # merged_linear() builds a fresh nn.Linear, which defaults to
            # CPU regardless of what device the original layer was on.
            # Moving it explicitly avoids ending up with a model whose
            # merged layers are silently on a different device than
            # everything else in it.
            merged_linear = module.merged_linear().to(module.base.weight.device)
            _set_nested_module(model, name, merged_linear)
            merged += 1
    return merged


def lora_parameter_count(model: nn.Module) -> int:
    """Count trainable LoRA parameters.

    Args:
        model: Model containing LoRA adapters.

    Returns:
        Number of trainable adapter parameters.
    """

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)



