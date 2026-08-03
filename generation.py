from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from .config import ModelConfig
from .model import MicroGPT
from .tokenizer import EOS_TOKEN, load_tokenizer, token_id


def load_model_from_checkpoint(checkpoint_path: Path, device: Optional[str] = None) -> MicroGPT:
    """Load a trained MicroGPT checkpoint.

    Args:
        checkpoint_path: Path to a saved model checkpoint.
        device: Optional device override.

    Returns:
        Loaded model in evaluation mode.
    """

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = ModelConfig(**checkpoint["model_config"])
    model = MicroGPT(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def generate_text(
    checkpoint_path: Path,
    tokenizer_path: Path,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: Optional[int] = 50,
    device: Optional[str] = None,
    use_kv_cache: bool = True,
) -> str:
    """Generate text from a trained checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint.
        tokenizer_path: Path to tokenizer JSON.
        prompt: Prompt text.
        max_new_tokens: Maximum tokens to sample.
        temperature: Sampling temperature.
        top_k: Optional top-k sampling cutoff.
        device: Optional device override.
        use_kv_cache: Whether to use key/value cache during generation.

    Returns:
        Decoded generated text.
    """

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(tokenizer_path)
    model = load_model_from_checkpoint(checkpoint_path, device=device)
    input_ids = tokenizer.encode(prompt).ids
    context = torch.tensor([input_ids], dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens, temperature=temperature, top_k=top_k, use_kv_cache=use_kv_cache)
    eos_id = token_id(tokenizer, EOS_TOKEN)
    output_ids = generated[0].tolist()
    if eos_id in output_ids[len(input_ids) :]:
        eos_index = output_ids.index(eos_id, len(input_ids))
        output_ids = output_ids[:eos_index]
    return tokenizer.decode(output_ids)

