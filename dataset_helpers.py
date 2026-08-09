from __future__ import annotations
import logging
import json
from pathlib import Path
from typing import Any, Callable, Optional
from .config import DatasetConfig

LOGGER = logging.getLogger(__name__)


def _local_structured_dataset_paths(config: DatasetConfig) -> list[
    tuple[Path, str, str]]:
    """Return configured local structured dataset paths.

    Args:
        config: Dataset configuration.

    Returns:
        Tuples of path, document kind, and progress label.
    """

    items: list[tuple[Path, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for path in [config.conversation_dataset_path,
                 *config.conversation_dataset_paths]:
        if path is None or not str(path).strip():
            continue
        key = ("conversation", str(Path(path)))
        if key not in seen:
            seen.add(key)
            items.append((Path(path), "conversation", "local conversation"))
    for path in [config.instruction_dataset_path,
                 *config.instruction_dataset_paths]:
        if path is None or not str(path).strip():
            continue
        key = ("instruction", str(Path(path)))
        if key not in seen:
            seen.add(key)
            items.append((Path(path), "instruction", "local instruction"))
    return items
def _emit(progress: Optional[Callable[[Any], None]], message: str,
          percent: Optional[int] = None) -> None:
    """Emit a progress event if a callback is available.

    Args:
        progress: Optional callback for progress dictionaries.
        message: Human-readable progress message.
        percent: Optional progress percentage.
    """

    LOGGER.info(message)
    if progress:
        progress({"message": message, "percent": percent})
def _cache_key(config: DatasetConfig) -> str:
    """Return a cache key for extraction-affecting options.

    Args:
        config: Dataset configuration.

    Returns:
        Cache key string.
    """

    return json.dumps(
        {
            "lowercase": config.lowercase,
            "code_training_mode": config.code_training_mode,
            "include_prose": config.include_prose,
            "include_source_code": config.include_source_code,
            "extract_code_blocks": config.extract_code_blocks,
            "preserve_indentation": config.preserve_indentation,
            "generate_instruction_samples": config.generate_instruction_samples,
            "reasoning_sample_mode": config.reasoning_sample_mode,
            "dataset_stage": config.dataset_stage,
            "conversation_datasets": config.conversation_datasets,
            "conversation_sample_limit": config.conversation_sample_limit,
            "conversation_dataset_path": str(
                config.conversation_dataset_path or ""),
            "instruction_dataset_path": str(
                config.instruction_dataset_path or ""),
            "conversation_dataset_paths": [str(path) for path in
                                           config.conversation_dataset_paths],
            "instruction_dataset_paths": [str(path) for path in
                                          config.instruction_dataset_paths],
            "default_data_paths": [str(path) for path in
                                   config.default_data_paths],
        },
        sort_keys=True,
    )

