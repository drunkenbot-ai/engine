from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional
import numpy as np
from .config import DatasetConfig, dataclass_to_jsonable
from .data import file_sha256
from .dataset_corpus import _StreamingCorpusBuilder
from .dataset_loader import _load_documents_with_cache
from .dataset_mixture import MAX_REPETITIVE_UNIT_RATIO
from .dataset_quality import _dataset_quality_report
from .dataset_tokenizer import _load_or_create_tokenizer
from .lineage import record_dataset_version, write_json
from .tokenizer import encode_file_to_npy, save_tokenizer_package, \
    token_dtype_for_vocab, validate_training_tokenizer

LOGGER = logging.getLogger(__name__)


@dataclass
class DatasetBuildResult:
    """Result returned after dataset preparation.

    Attributes:
        output_dir: Prepared dataset folder.
        tokenizer_path: Path to tokenizer JSON.
        document_count: Number of loaded samples.
        token_count: Total encoded tokens.
        train_window_count: Number of sliding training windows.
        val_window_count: Number of sliding validation windows.
        sequence_token_stats: Approximate min/avg/median/max source token lengths.
        vocab_size: Final tokenizer vocabulary size.
        character_count: Total corpus characters.
        suggested_vocab_size: Automatically estimated vocabulary size.
        warning: Optional dataset quality warning.
        code_sample_count: Number of code samples.
        prose_sample_count: Number of prose samples.
        conversation_sample_count: Number of conversation/instruction samples.
        cached_file_count: Number of unchanged source files reused from cache.
        processed_file_count: Number of source files extracted this run.
        skipped_file_count: Number of files with no readable text.
        failed_file_count: Number of files that failed extraction.
        dataset_version_id: Unique dataset version identifier.
        dataset_version_number: One-based dataset version number.
        mixture_report: Per-source family sampling report.
        quality_score: Dataset quality score from 0 to 100.
        quality_stars: Dataset quality rating from 0 to 5.
        quality_label: Human-readable dataset quality label.
        quality_reasons: Short reasons behind the quality score.
        duplicate_block_count: Number of repeated blocks in the written corpus.
        unique_block_count: Number of unique blocks in the written corpus.
        corpus_block_count: Number of non-empty blocks inspected in the written corpus.
        duplicate_block_ratio: Fraction of repeated text blocks in the written corpus.
        unique_block_ratio: Fraction of unique text blocks in the written corpus.
    """

    output_dir: Path
    tokenizer_path: Path
    document_count: int
    token_count: int
    vocab_size: int
    character_count: int
    suggested_vocab_size: int
    train_window_count: int = 0
    val_window_count: int = 0
    sequence_token_stats: dict[str, float] = field(default_factory=dict)
    warning: Optional[str] = None
    code_sample_count: int = 0
    prose_sample_count: int = 0
    conversation_sample_count: int = 0
    cached_file_count: int = 0
    processed_file_count: int = 0
    skipped_file_count: int = 0
    failed_file_count: int = 0
    dataset_version_id: str = ""
    dataset_version_number: int = 0
    mixture_report: dict[str, Any] = field(default_factory=dict)
    quality_score: float = 0.0
    quality_stars: float = 0.0
    quality_label: str = "Not rated"
    quality_reasons: list[str] = field(default_factory=list)
    duplicate_block_count: int = 0
    unique_block_count: int = 0
    corpus_block_count: int = 0
    duplicate_block_ratio: float = 0.0
    unique_block_ratio: float = 1.0


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


def estimate_vocab_size(character_count: int, unique_word_count: int) -> int:
    """Estimate a reasonable tokenizer vocabulary size.

    Args:
        character_count: Number of corpus characters.
        unique_word_count: Approximate number of unique whitespace words.

    Returns:
        Suggested vocabulary size.
    """

    if character_count < 20_000:
        ceiling = 1_000
    elif character_count < 100_000:
        ceiling = 4_000
    elif character_count < 500_000:
        ceiling = 8_000
    elif character_count < 2_000_000:
        ceiling = 16_000
    else:
        ceiling = 32_000

    desired = max(512, int(unique_word_count * 1.7), int(character_count / 45))
    return max(256, min(ceiling, desired))


def content_warning(character_count: int) -> Optional[str]:
    """Return a corpus-size warning when the dataset is small.

    Args:
        character_count: Number of corpus characters.

    Returns:
        Warning text, or ``None`` when the corpus is large enough.
    """

    if character_count < 10_000:
        return "The corpus is very small. Training can run, but the model will only be useful for smoke tests."
    if character_count < 100_000:
        return "The corpus is modest. Use more text for better generations and reasoning behavior."
    return None


def build_dataset(
        config: DatasetConfig,
        progress: Optional[Callable[[Any], None]] = None,
        should_stop: Optional[Callable[[], bool]] = None,
) -> DatasetBuildResult:
    """Build a tokenizer-ready dataset project.

    Args:
        config: Dataset preparation settings.
        progress: Optional callback receiving progress event dictionaries.
        should_stop: Optional callback returning true when the user requested stop.

    Returns:
        Dataset build summary.

    Raises:
        ValueError: If no supported documents are found.
    """

    from .training import split_tokens_to_files

    config.output_dir.mkdir(parents=True, exist_ok=True)
    _emit(progress, "Scanning source folder...", 3)
    corpus_path = config.output_dir / "corpus.txt"
    corpus_builder = _StreamingCorpusBuilder(
        corpus_path,
        code_training_mode=config.code_training_mode,
        generate_instruction_samples=config.generate_instruction_samples,
        reasoning_sample_mode=config.reasoning_sample_mode,
    )
    # Loading, exact-duplicate removal, low-diversity filtering, and corpus
    # writing all happen inside this single streaming pass -- each document
    # is evaluated and written (or dropped) as it arrives, so at no point is
    # the full document set held in memory at once. See
    # _StreamingCorpusBuilder / _load_documents_with_cache.
    (
        manifest,
        cached_file_count,
        processed_file_count,
        skipped_file_count,
        failed_file_count,
    ) = _load_documents_with_cache(config, corpus_builder, progress,
                                   should_stop)
    duplicate_report = corpus_builder.close()
    stats = corpus_builder.stats
    if should_stop and should_stop():
        raise RuntimeError("Dataset preparation stopped by user.")
    if stats.accepted_document_count == 0:
        corpus_path.unlink(missing_ok=True)
        raise ValueError(
            "No supported text, PDF, JSONL, or structured JSON documents were found.")
    if stats.exact_duplicates_removed:
        _emit(
            progress,
            f"Removed {stats.exact_duplicates_removed:,} exact duplicate extracted document(s).",
            44,
        )
    if stats.low_diversity_removed:
        _emit(
            progress,
            (
                "Excluded "
                f"{stats.low_diversity_removed:,} low-diversity document(s) "
                f"({stats.low_diversity_removed_characters:,} characters) "
                "instead of padding the corpus with repeated templates."
            ),
            45,
        )
        _emit(progress, "Low-diversity files excluded:", 45)
        for excluded in stats.low_diversity_examples:
            _emit(progress, f"  - {excluded['path']}", 45)
    mixture_report = {
        "applied": False,
        "reason": "Dataset mixture disabled",
    }

    character_count = stats.character_count
    unique_words = len(stats.unique_words)
    suggested_vocab_size = estimate_vocab_size(character_count, unique_words)
    selected_vocab_size = config.vocab_size or suggested_vocab_size
    warning = content_warning(character_count)
    if character_count < 1_000_000:
        low_corpus_message = (
            "Prepared corpus is below 1M characters after quality filtering. "
            "Add licensed, provenance-tracked sources or select an approved online dataset; "
            "the app will not pad training data with synthetic repetition."
        )
        warning = f"{warning} {low_corpus_message}" if warning else low_corpus_message
    code_sample_count = stats.code_sample_count
    conversation_sample_count = stats.conversation_sample_count
    prose_sample_count = stats.prose_sample_count
    document_count = stats.accepted_document_count
    _emit(progress,
          f"Content size: {character_count:,} characters across {document_count} files.",
          45)
    if config.code_training_mode:
        _emit(progress,
              f"Code mode: {code_sample_count:,} code samples, {prose_sample_count:,} prose samples.",
              46)
    if conversation_sample_count:
        _emit(progress,
              f"Conversation data: {conversation_sample_count:,} dialogue/instruction samples.",
              46)
    if cached_file_count or processed_file_count:
        _emit(progress,
              f"Cache: reused {cached_file_count:,} file(s), processed {processed_file_count:,} file(s).",
              47)
    if skipped_file_count or failed_file_count:
        _emit(progress,
              f"Quality: skipped {skipped_file_count:,} empty file(s), failed {failed_file_count:,} file(s).",
              48)
    _emit(progress, f"Unique word estimate: {unique_words:,}.", 48)
    _emit(progress, f"Auto vocabulary size: {selected_vocab_size:,}.", 50)
    if warning:
        _emit(progress, f"Warning: {warning}")

    _emit(progress, "Corpus written.", 56)
    if should_stop and should_stop():
        raise RuntimeError("Dataset preparation stopped by user.")
    _emit(
        progress,
        (
            "Corpus diversity: "
            f"{duplicate_report['unique_block_count']:,}/{duplicate_report['block_count']:,} unique blocks, "
            f"{duplicate_report['duplicate_block_ratio'] * 100:.1f}% repeated."
        ),
        74,
    )
    tokenizer_path = config.output_dir / "tokenizer.json"
    if should_stop and should_stop():
        raise RuntimeError("Dataset preparation stopped by user.")
    tokenizer, reuse_tokenizer, tokenizer_imported, tokenizer_source_path = _load_or_create_tokenizer(
        config,
        corpus_path,
        tokenizer_path,
        selected_vocab_size,
        progress,
        should_stop,
    )
    validate_training_tokenizer(tokenizer)
    save_tokenizer_package(tokenizer, tokenizer_path,
                           model_max_length=config.context_length)

    _emit(progress, "Encoding corpus into token IDs...", 78)
    if should_stop and should_stop():
        raise RuntimeError("Dataset preparation stopped by user.")
    # Encode straight to a memmap-friendly .npy file. Encoding is streamed in
    # bounded batches (never a full in-memory token list), and the resulting
    # file is then opened read-only as a memmap so the split step below also
    # never holds the full token stream in RAM.
    token_dtype = token_dtype_for_vocab(tokenizer.get_vocab_size())
    all_tokens_path = config.output_dir / "all_tokens.npy"
    token_count = encode_file_to_npy(
        tokenizer, corpus_path, all_tokens_path, token_dtype,
        should_stop=should_stop
    )
    _emit(progress, f"Encoded {token_count:,} tokens.", 86)

    token_density = (token_count / max(character_count,
                                       1)) if character_count else 0.0
    document_token_lengths = [max(1, int(round(char_len * token_density)))
                              for char_len in stats.document_char_lengths if
                              char_len]
    if document_token_lengths:
        sequence_stats = {
            "min": min(document_token_lengths),
            "average": sum(document_token_lengths) / len(
                document_token_lengths),
            "median": statistics.median(document_token_lengths),
            "max": max(document_token_lengths),
        }
    else:
        sequence_stats = {"min": 0, "average": 0.0, "median": 0.0, "max": 0}
    _emit(
        progress,
        (
            "Token distribution: "
            f"min {int(sequence_stats['min']):,}, "
            f"avg {float(sequence_stats['average']):,.0f}, "
            f"median {float(sequence_stats['median']):,.0f}, "
            f"max {int(sequence_stats['max']):,}."
        ),
        88,
    )
    if should_stop and should_stop():
        raise RuntimeError("Dataset preparation stopped by user.")
    all_tokens = np.load(all_tokens_path, mmap_mode="r")
    train_token_count, val_token_count = split_tokens_to_files(
        all_tokens,
        config.output_dir / "train_tokens.npy",
        config.output_dir / "val_tokens.npy",
        config.validation_split,
        dtype=token_dtype,
        should_stop=should_stop,
    )
    del all_tokens  # release the memmap handle before deleting the backing file
    all_tokens_path.unlink(missing_ok=True)
    train_window_count = max(0, train_token_count - config.context_length)
    val_window_count = max(0, val_token_count - config.context_length)
    _emit(progress,
          f"Training tokens: {train_token_count:,}; validation tokens: {val_token_count:,}.",
          92)
    _emit(progress,
          f"Training windows: {train_window_count:,}; validation windows: {val_window_count:,}.",
          92)
    quality_report = _dataset_quality_report(
        document_count=document_count,
        token_count=token_count,
        vocab_size=tokenizer.get_vocab_size(),
        unique_words=unique_words,
        train_window_count=train_window_count,
        val_window_count=val_window_count,
        code_sample_count=code_sample_count,
        prose_sample_count=prose_sample_count,
        conversation_sample_count=conversation_sample_count,
        skipped_file_count=skipped_file_count,
        failed_file_count=failed_file_count,
        warning=warning,
        sequence_stats=sequence_stats,
        duplicate_report=duplicate_report,
    )
    _emit(
        progress,
        f"Dataset rating: {quality_report['stars']:.1f}/5 stars ({quality_report['label']}, score {quality_report['score']:.1f}/100).",
        94,
    )

    summary = {
        "dataset_config": dataclass_to_jsonable(config),
        "document_count": document_count,
        "character_count": character_count,
        "token_count": token_count,
        "train_token_count": train_token_count,
        "val_token_count": val_token_count,
        "train_tokens_path": "train_tokens.npy",
        "val_tokens_path": "val_tokens.npy",
        "token_storage_format": "npy",
        "train_window_count": train_window_count,
        "val_window_count": val_window_count,
        "context_length": config.context_length,
        "sequence_token_stats": sequence_stats,
        "code_sample_count": code_sample_count,
        "prose_sample_count": prose_sample_count,
        "conversation_sample_count": conversation_sample_count,
        "dataset_stage": config.dataset_stage,
        "conversation_datasets": config.conversation_datasets,
        "conversation_sample_limit": config.conversation_sample_limit,
        "conversation_dataset_path": str(
            config.conversation_dataset_path or ""),
        "instruction_dataset_path": str(config.instruction_dataset_path or ""),
        "conversation_dataset_paths": [str(path) for path in
                                       config.conversation_dataset_paths],
        "instruction_dataset_paths": [str(path) for path in
                                      config.instruction_dataset_paths],
        "tool_call_dataset_paths": [str(path) for path in
                                    config.tool_call_dataset_paths],
        "default_data_paths": [str(path) for path in
                               config.default_data_paths],
        "mixture_weights": config.mixture_weights,
        "mixture_report": mixture_report,
        "exact_duplicate_documents_removed": stats.exact_duplicates_removed,
        "exact_duplicate_document_examples": stats.exact_duplicate_examples,
        "low_diversity_documents_removed": stats.low_diversity_removed,
        "low_diversity_characters_removed": stats.low_diversity_removed_characters,
        "low_diversity_duplicate_unit_threshold": MAX_REPETITIVE_UNIT_RATIO,
        "low_diversity_document_examples": stats.low_diversity_examples,
        "suggested_vocab_size": suggested_vocab_size,
        "tokenizer_vocab_size": tokenizer.get_vocab_size(),
        "tokenizer_sha256": file_sha256(tokenizer_path),
        "warning": warning,
        "source_files": stats.source_files,
        "source_files_truncated": stats.source_files_truncated,
        "cached_file_count": cached_file_count,
        "processed_file_count": processed_file_count,
        "skipped_file_count": skipped_file_count,
        "failed_file_count": failed_file_count,
        "source_file_count": manifest.count(),
        "prepare_mode": config.prepare_mode,
        "tokenizer_strategy": config.tokenizer_strategy,
        "reasoning_sample_mode": config.reasoning_sample_mode,
        "tokenizer_reused": reuse_tokenizer,
        "tokenizer_imported": tokenizer_imported,
        "tokenizer_source_path": tokenizer_source_path,
        "quality_score": quality_report["score"],
        "quality_stars": quality_report["stars"],
        "quality_label": quality_report["label"],
        "quality_reasons": quality_report["reasons"],
        "quality_components": quality_report["components"],
        "duplicate_block_count": duplicate_report["duplicate_block_count"],
        "unique_block_count": duplicate_report["unique_block_count"],
        "corpus_block_count": duplicate_report["block_count"],
        "duplicate_block_ratio": duplicate_report["duplicate_block_ratio"],
        "unique_block_ratio": duplicate_report["unique_block_ratio"],
        "most_repeated_block_count": duplicate_report[
            "most_repeated_block_count"],
        "top_repeated_blocks": duplicate_report["top_repeated_blocks"],
    }
    dataset_version = record_dataset_version(config.output_dir, summary,
                                             manifest)
    write_json(config.output_dir / "dataset_summary.json", summary)
    manifest.close()
    _emit(progress,
          f"Dataset version recorded: {dataset_version['version_id']}.", 98)
    _emit(progress, f"Dataset ready: {config.output_dir}", 100)
    return DatasetBuildResult(
        config.output_dir,
        tokenizer_path,
        document_count,
        token_count,
        tokenizer.get_vocab_size(),
        character_count,
        suggested_vocab_size,
        train_window_count,
        val_window_count,
        sequence_stats,
        warning,
        code_sample_count,
        prose_sample_count,
        conversation_sample_count,
        cached_file_count,
        processed_file_count,
        skipped_file_count,
        failed_file_count,
        str(dataset_version["version_id"]),
        int(dataset_version["version_number"]),
        mixture_report,
        float(quality_report["score"]),
        float(quality_report["stars"]),
        str(quality_report["label"]),
        list(quality_report["reasons"]),
        int(duplicate_report["duplicate_block_count"]),
        int(duplicate_report["unique_block_count"]),
        int(duplicate_report["block_count"]),
        float(duplicate_report["duplicate_block_ratio"]),
        float(duplicate_report["unique_block_ratio"]),
    )

__all__ = ["DatasetBuildResult", "build_dataset", "estimate_vocab_size", "content_warning"]
