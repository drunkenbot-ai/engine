from __future__ import annotations
import shutil
from pathlib import Path
from typing import Any, Callable, Optional
from .config import DatasetConfig
from .dataset_helpers import _emit
from .tokenizer import load_tokenizer, train_tokenizer


def _resolve_tokenizer_strategy(config: DatasetConfig, tokenizer_path: Path) -> \
tuple[str, bool]:
    """Resolve tokenizer strategy into an executable mode.

    Args:
        config: Dataset configuration.
        tokenizer_path: Dataset tokenizer output path.

    Returns:
        Strategy name and whether the dataset tokenizer should be reused.
    """

    strategy = config.tokenizer_strategy or "auto"
    if strategy == "auto":
        return strategy, config.prepare_mode == "incremental" and tokenizer_path.exists()
    if strategy == "reuse_dataset":
        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"Cannot reuse dataset tokenizer because tokenizer.json was not found in {config.output_dir}."
            )
        return strategy, True
    if strategy in {"train_new", "import_tokenizer"}:
        return strategy, False
    raise ValueError(f"Unsupported tokenizer strategy: {strategy}")


def _load_or_create_tokenizer(
        config: DatasetConfig,
        corpus_path: Path,
        tokenizer_path: Path,
        selected_vocab_size: int,
        progress: Optional[Callable[[Any], None]],
        should_stop: Optional[Callable[[], bool]],
) -> tuple[Any, bool, bool, Optional[str]]:
    """Load, import, or train a tokenizer for the prepared corpus.

    Args:
        config: Dataset configuration.
        corpus_path: Normalized training corpus path.
        tokenizer_path: Dataset tokenizer output path.
        selected_vocab_size: Vocabulary size used when training a new tokenizer.
        progress: Optional progress callback.
        should_stop: Optional cancellation callback.

    Returns:
        Tokenizer, reused flag, imported flag, and optional source path.
    """

    strategy, reuse_tokenizer = _resolve_tokenizer_strategy(config,
                                                            tokenizer_path)
    imported = False
    source_path: Optional[str] = None

    if reuse_tokenizer:
        _emit(progress, "Reusing existing dataset tokenizer.json...", 62)
        return load_tokenizer(tokenizer_path), True, imported, source_path

    if strategy == "import_tokenizer":
        if config.tokenizer_path is None:
            raise ValueError(
                "Choose a tokenizer.json file when tokenizer strategy is Import tokenizer.json.")
        import_path = Path(config.tokenizer_path)
        if not import_path.exists():
            raise FileNotFoundError(
                f"Tokenizer import file not found: {import_path}")
        _emit(progress, f"Importing tokenizer from {import_path}...", 62)
        tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        if import_path.resolve() != tokenizer_path.resolve():
            shutil.copy2(import_path, tokenizer_path)
        return load_tokenizer(tokenizer_path), False, True, str(import_path)

    corpus_size_bytes = corpus_path.stat().st_size
    training_mb = corpus_size_bytes / (1024 * 1024)
    max_training_bytes = (
        int(config.tokenizer_training_max_gb * 1024**3)
        if config.tokenizer_training_max_gb > 0
        else None
    )
    if max_training_bytes is not None and corpus_size_bytes > max_training_bytes:
        _emit(
            progress,
            (
                f"Training tokenizer on a {config.tokenizer_training_max_gb:.1f} GiB sample of the "
                f"{training_mb:.1f} MB corpus (tokenizer_training_max_gb)..."
            ),
            62,
        )
    else:
        _emit(
            progress,
            f"Training tokenizer on the full {training_mb:.1f} MB corpus...",
            62,
        )
    tokenizer = train_tokenizer(
        corpus_path,
        tokenizer_path,
        vocab_size=selected_vocab_size,
        min_frequency=config.min_frequency,
        should_stop=should_stop,
        max_training_bytes=max_training_bytes,
    )
    return tokenizer, False, imported, source_path

