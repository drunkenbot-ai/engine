from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Optional

import torch

from .generation import load_model_from_checkpoint
from .lineage import read_json, utc_timestamp, write_json
from .tokenizer import EOS_TOKEN, load_tokenizer, token_id


DEFAULT_BENCHMARK_PROMPTS = [
    "Explain what a Python function is and give a tiny example.",
    "Write a Python function that adds two numbers.",
    "Review this code and explain any issue:\n```python\ndef add(a, b):\nprint(a + b)\n```",
]


@dataclass
class BenchmarkResult:
    """Result returned after running benchmark prompts.

    Attributes:
        output_path: JSON file containing benchmark outputs.
        prompt_count: Number of prompts evaluated.
        total_seconds: Total elapsed time.
        total_generated_tokens: Total generated tokens across prompts.
        tokens_per_second: Average generated-token throughput.
    """

    output_path: Path
    prompt_count: int
    total_seconds: float
    total_generated_tokens: int = 0
    tokens_per_second: float = 0.0


def normalize_prompts(raw_prompts: str) -> list[str]:
    """Split raw benchmark prompt text into prompts.

    Args:
        raw_prompts: Text containing prompts separated by blank lines.

    Returns:
        Non-empty prompt list.
    """

    prompts = [part.strip() for part in raw_prompts.replace("\r\n", "\n").split("\n\n") if part.strip()]
    return prompts or DEFAULT_BENCHMARK_PROMPTS


def evaluate_checkpoint(
    model_dir: Path,
    prompts: list[str],
    output_dir: Optional[Path] = None,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_k: int = 50,
    device: Optional[str] = None,
    use_kv_cache: bool = True,
    progress: Optional[Callable[[Any], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> BenchmarkResult:
    """Evaluate a trained checkpoint with benchmark prompts.

    Args:
        model_dir: Folder containing ``final_model.pt`` and ``tokenizer.json``.
        prompts: Benchmark prompts.
        output_dir: Optional output folder for benchmark JSON.
        max_new_tokens: Maximum new tokens per prompt.
        temperature: Sampling temperature.
        top_k: Top-k sampling cutoff.
        device: Optional device override.
        use_kv_cache: Whether to reuse key/value cache during generation.
        progress: Optional progress callback.
        should_stop: Optional cancellation callback.

    Returns:
        Benchmark result summary.

    Raises:
        FileNotFoundError: If checkpoint or tokenizer is missing.
    """

    model_dir = Path(model_dir)
    checkpoint_path = model_dir / "final_model.pt"
    tokenizer_path = model_dir / "tokenizer.json"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Final checkpoint not found: {checkpoint_path}")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

    output_dir = output_dir or model_dir / "benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    lineage = read_json(model_dir / "model_lineage.json", default={}) or {}
    tokenizer = load_tokenizer(tokenizer_path)
    model = load_model_from_checkpoint(checkpoint_path, device=device)
    eos_id = token_id(tokenizer, EOS_TOKEN)

    started = perf_counter()
    outputs: list[dict[str, Any]] = []
    for index, prompt in enumerate(prompts, start=1):
        if should_stop and should_stop():
            raise RuntimeError("Benchmark stopped by user.")
        if progress:
            progress({"message": f"Benchmark prompt {index}/{len(prompts)}...", "percent": int(90 * (index - 1) / max(len(prompts), 1))})
        prompt_started = perf_counter()
        input_ids = tokenizer.encode(prompt).ids
        context = torch.tensor([input_ids], dtype=torch.long, device=device)
        generated = model.generate(
            context,
            max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            use_kv_cache=use_kv_cache,
        )
        output_ids = generated[0].tolist()
        generated_token_count = max(len(output_ids) - len(input_ids), 0)
        if eos_id in output_ids[len(input_ids) :]:
            eos_index = output_ids.index(eos_id, len(input_ids))
            output_ids = output_ids[:eos_index]
            generated_token_count = max(len(output_ids) - len(input_ids), 0)
        text = tokenizer.decode(output_ids)
        elapsed = perf_counter() - prompt_started
        outputs.append(
            {
                "index": index,
                "prompt": prompt,
                "output": text,
                "elapsed_seconds": elapsed,
                "generated_tokens": generated_token_count,
                "tokens_per_second": generated_token_count / max(elapsed, 1e-9),
                "characters": len(text),
            }
        )

    total = perf_counter() - started
    total_generated_tokens = sum(int(item.get("generated_tokens", 0)) for item in outputs)
    tokens_per_second = total_generated_tokens / max(total, 1e-9)
    payload = {
        "schema": "micro_llm_benchmark",
        "version": 1,
        "created_at": utc_timestamp(),
        "model_dir": str(model_dir),
        "checkpoint_path": str(checkpoint_path),
        "tokenizer_path": str(tokenizer_path),
        "device": device,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "use_kv_cache": use_kv_cache,
        "model_lineage": lineage,
        "prompt_count": len(prompts),
        "total_seconds": total,
        "total_generated_tokens": total_generated_tokens,
        "tokens_per_second": tokens_per_second,
        "results": outputs,
    }
    output_path = output_dir / f"benchmark_{utc_timestamp()}.json"
    write_json(output_path, payload)
    if progress:
        progress({"message": f"Benchmark saved: {output_path}", "percent": 100})
    return BenchmarkResult(output_path, len(prompts), total, total_generated_tokens, tokens_per_second)

