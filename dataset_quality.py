from __future__ import annotations
import re
from typing import Any, Optional

def _bounded_ratio(value: float, target: float) -> float:
    """Return value/target clamped between 0 and 1.

    Args:
        value: Actual metric value.
        target: Metric value that should receive full credit.

    Returns:
        Clamped ratio.
    """

    if target <= 0:
        return 0.0
    return max(0.0, min(1.0, float(value) / float(target)))


def _canonical_corpus_block(text: str) -> str:
    """Normalize a corpus block for repeated-content checks.

    Args:
        text: Raw block text.

    Returns:
        Whitespace-normalized lowercase text.
    """

    return re.sub(r"\s+", " ", text).strip().lower()


def _dataset_quality_report(
        *,
        document_count: int,
        token_count: int,
        vocab_size: int,
        unique_words: int,
        train_window_count: int,
        val_window_count: int,
        code_sample_count: int,
        prose_sample_count: int,
        conversation_sample_count: int,
        skipped_file_count: int,
        failed_file_count: int,
        warning: Optional[str],
        sequence_stats: dict[str, float],
        duplicate_report: dict[str, Any],
) -> dict[str, Any]:
    """Rate a prepared dataset for small-LLM training readiness.

    Args:
        document_count: Prepared document/sample count.
        token_count: Total token count.
        vocab_size: Final tokenizer vocabulary size.
        unique_words: Estimated unique words in the corpus.
        train_window_count: Number of trainable context windows.
        val_window_count: Number of validation context windows.
        code_sample_count: Prepared code sample count.
        prose_sample_count: Prepared prose sample count.
        conversation_sample_count: Prepared conversation/instruction sample count.
        skipped_file_count: Empty or unreadable source files skipped.
        failed_file_count: Source files that failed extraction.
        warning: Size/content warning string.
        sequence_stats: Approximate per-document token distribution.
        duplicate_report: Repeated text-block report for the written corpus.

    Returns:
        Dataset quality dictionary with score, stars, label, and reasons.
    """

    reasons: list[str] = []
    token_score = 30.0 * _bounded_ratio(token_count, 1_000_000)
    window_score = 20.0 * _bounded_ratio(train_window_count, 50_000)
    vocab_target = max(4_000.0, min(32_000.0, unique_words * 0.8))
    vocab_score = 18.0 * _bounded_ratio(vocab_size, vocab_target)
    document_score = 12.0 * _bounded_ratio(document_count, 1_000)
    validation_score = 8.0 * _bounded_ratio(val_window_count, 2_000)
    families = sum(1 for count in (
    code_sample_count, prose_sample_count, conversation_sample_count) if
                   count > 0)
    diversity_score = 7.0 * _bounded_ratio(families, 3)
    average_sequence = float(sequence_stats.get("average", 0.0) or 0.0)
    sequence_score = 5.0 * _bounded_ratio(average_sequence, 256)
    penalty = min(20.0, failed_file_count * 3.0 + skipped_file_count * 0.5)
    duplicate_ratio = float(
        duplicate_report.get("duplicate_block_ratio", 0.0) or 0.0)
    duplicate_penalty = min(35.0, duplicate_ratio * 70.0)
    penalty += duplicate_penalty
    if warning and warning != "none":
        penalty += 5.0
    score = max(
        0.0,
        min(
            100.0,
            token_score
            + window_score
            + vocab_score
            + document_score
            + validation_score
            + diversity_score
            + sequence_score
            - penalty,
        ),
    )
    stars = round(score / 20.0 * 2.0) / 2.0
    if score >= 85:
        label = "Excellent"
    elif score >= 70:
        label = "Good"
    elif score >= 50:
        label = "Usable"
    elif score >= 30:
        label = "Weak"
    else:
        label = "Very weak"
    if token_count < 250_000:
        reasons.append("Token count is low for robust training.")
    else:
        reasons.append("Token count is sufficient for a small experiment.")
    if train_window_count < 5_000:
        reasons.append("Few training windows; model may memorize quickly.")
    if vocab_size < 4_000:
        reasons.append(
            "Vocabulary is small; language coverage may be limited.")
    elif vocab_size > 50_000:
        reasons.append(
            "Vocabulary is large; tiny models may spend capacity on tokens.")
    else:
        reasons.append("Vocabulary size is in a reasonable small-model range.")
    if families >= 2:
        reasons.append("Dataset includes multiple content families.")
    if skipped_file_count or failed_file_count:
        reasons.append(
            f"Extraction skipped {skipped_file_count} file(s) and failed {failed_file_count} file(s).")
    if duplicate_ratio >= 0.5:
        reasons.append(
            "Prepared corpus is heavily repeated; training may memorize instead of generalize.")
    elif duplicate_ratio >= 0.2:
        reasons.append(
            "Prepared corpus has many repeated blocks; add more varied data or deduplicate.")
    elif duplicate_ratio >= 0.05:
        reasons.append("Prepared corpus has some repeated blocks.")
    else:
        reasons.append("Prepared corpus block diversity looks healthy.")
    if warning and warning != "none":
        reasons.append(str(warning))
    return {
        "score": round(score, 1),
        "stars": stars,
        "label": label,
        "reasons": reasons,
        "components": {
            "tokens": round(token_score, 1),
            "windows": round(window_score, 1),
            "vocabulary": round(vocab_score, 1),
            "documents": round(document_score, 1),
            "validation": round(validation_score, 1),
            "diversity": round(diversity_score, 1),
            "sequence": round(sequence_score, 1),
            "duplicate_penalty": round(duplicate_penalty, 1),
            "penalty": round(penalty, 1),
        },
    }

