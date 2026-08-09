from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any, Callable, Optional

from .conversation_datasets import CONVERSATION_DATASET_PRESETS
from .data import Document

LOGGER = logging.getLogger(__name__)

AGGREGATE_MIXTURE_FAMILIES: set[str] = set()
MIXTURE_CHUNK_CHARS = 25_000
# A default corpus must not gain apparent scale by repeating a tiny template.
# This threshold is deliberately conservative: it only applies once a document
# has enough independently meaningful units to make the measurement useful.
MAX_REPETITIVE_UNIT_RATIO = 0.35
MIN_REPETITION_CHECK_UNITS = 20
MIN_REPETITION_CHECK_CHARS = 2_000


def _emit(progress: Optional[Callable[[Any], None]], message: str, percent: Optional[int] = None) -> None:
    LOGGER.info(message)
    if progress:
        progress({"message": message, "percent": percent})


def _canonical_corpus_block(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _slugify_category(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "general_prose"


def _mixture_label(category: str) -> str:
    return category.replace("_", " ").title()


def _default_data_category(path: Path) -> Optional[str]:
    parts_lower = [part.lower() for part in path.parts]
    data_roots = ("default_data", "training_data")
    root_indices = [parts_lower.index(root) for root in data_roots if root in parts_lower]
    if root_indices:
        default_index = max(root_indices)
        relative_parts = path.parts[default_index + 1 :]
    else:
        # Prepared documents can come from any user-selected root.  In that
        # case the containing folder is the only reliable folder metadata.
        relative_parts = path.parts
    # Categories are configured by directory layout.  The first directory
    # under the configured training-data root owns all nested files.
    if len(relative_parts) > 1:
        return _slugify_category(relative_parts[0])
    return None


def _deduplicate_documents(documents: list[Document]) -> tuple[list[Document], dict[str, Any]]:
    unique_documents: list[Document] = []
    seen: dict[str, Document] = {}
    duplicates: list[dict[str, str]] = []
    for document in documents:
        canonical_text = _canonical_corpus_block(document.text)
        if not canonical_text:
            unique_documents.append(document)
            continue
        digest = hashlib.sha256(
            f"{document.kind}\n{document.language or ''}\n{canonical_text}".encode("utf-8")
        ).hexdigest()
        original = seen.get(digest)
        if original is not None:
            duplicates.append(
                {
                    "path": str(document.path),
                    "duplicate_of": str(original.path),
                    "kind": document.kind,
                }
            )
            continue
        seen[digest] = document
        unique_documents.append(document)
    return unique_documents, {
        "removed_documents": len(duplicates),
        "duplicates": duplicates[:50],
    }


def _content_units_for_diversity(document: Document) -> list[str]:
    """Return comparable content units for a repetition-quality check.

    Prose sources are often whitespace-normalised during ingestion, so using
    source lines would miss repeated sentences.  Code remains line-oriented;
    prose and chat are instead split at sentence and turn boundaries.
    """

    text = document.text.replace("\r\n", "\n").replace("\r", "\n")
    if document.kind == "code":
        raw_units = text.split("\n")
    else:
        raw_units = re.split(r"(?<=[.!?])\s+|\n+(?=(?:User|Assistant|System|Instruction|Response):)", text)
    return [
        _canonical_corpus_block(unit)
        for unit in raw_units
        if len(_canonical_corpus_block(unit)) >= 24
    ]


def _filter_repetitive_documents(documents: list[Document]) -> tuple[list[Document], dict[str, Any]]:
    """Remove documents dominated by exact repeated content units.

    This is a quality gate, not a substitute for semantic deduplication.  It
    catches generated padding such as the old bundled curriculum files before
    it can dominate token counts and make a small corpus look large.
    """

    accepted: list[Document] = []
    rejected: list[dict[str, Any]] = []
    for document in documents:
        units = _content_units_for_diversity(document)
        if len(document.text) < MIN_REPETITION_CHECK_CHARS or len(units) < MIN_REPETITION_CHECK_UNITS:
            accepted.append(document)
            continue
        duplicate_ratio = 1.0 - (len(set(units)) / len(units))
        if duplicate_ratio > MAX_REPETITIVE_UNIT_RATIO:
            rejected.append(
                {
                    "path": str(document.path),
                    "kind": document.kind,
                    "unit_count": len(units),
                    "duplicate_unit_ratio": round(duplicate_ratio, 4),
                }
            )
            continue
        accepted.append(document)
    rejected_paths = {item["path"] for item in rejected}
    return accepted, {
        "removed_documents": len(rejected),
        "removed_characters": sum(
            len(document.text)
            for document in documents
            if str(document.path) in rejected_paths
        ),
        "threshold": MAX_REPETITIVE_UNIT_RATIO,
        "examples": rejected[:50],
    }


def _document_mixture_family(document: Document) -> str:
    default_category = _default_data_category(document.path)
    if default_category:
        return default_category
    if document.kind == "code":
        return "source_code"
    if document.kind == "instruction":
        return "instruction"
    if document.kind == "conversation":
        return "conversation"
    dataset_id = str(document.language or "")
    preset = CONVERSATION_DATASET_PRESETS.get(dataset_id)
    if preset and preset.stage == "base":
        return "online_base"
    if "__hf_datasets__" in document.path.parts:
        for part in document.path.parts:
            preset = CONVERSATION_DATASET_PRESETS.get(part)
            if preset and preset.stage == "base":
                return "online_base"
    return "local_prose"


def _stable_document_sort_key(document: Document) -> str:
    text_digest = hashlib.sha256(document.text[:4096].encode("utf-8", errors="ignore")).hexdigest()
    key = f"{document.path}|{document.kind}|{document.language or ''}|{len(document.text)}|{text_digest}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _chunk_document_for_mixture(document: Document, chunk_chars: int = MIXTURE_CHUNK_CHARS) -> list[Document]:
    text = document.text
    if len(text) <= chunk_chars:
        return [document]
    chunks: list[Document] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_chars)
        if end < len(text):
            boundary = text.rfind("\n\n", start, end)
            if boundary <= start + int(chunk_chars * 0.5):
                boundary = text.rfind("\n", start, end)
            if boundary > start + int(chunk_chars * 0.5):
                end = boundary
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(
                Document(
                    path=document.path,
                    text=chunk_text,
                    kind=document.kind,
                    language=document.language,
                )
            )
        start = max(end, start + 1)
    return chunks or [document]


def _chunk_documents_for_mixture(documents: list[Document]) -> list[Document]:
    chunks: list[Document] = []
    for document in documents:
        chunks.extend(_chunk_document_for_mixture(document))
    return chunks


def _empty_mixture_report(weights: dict[str, float], documents: list[Document], applied: bool, reason: str = "") -> dict[str, Any]:
    document_families = {_document_mixture_family(document) for document in documents}
    families_to_report = sorted({*MIXTURE_LABELS, *weights, *document_families})
    by_family: dict[str, list[Document]] = {key: [] for key in families_to_report}
    for document in documents:
        by_family.setdefault(_document_mixture_family(document), []).append(document)
    total_chars = sum(len(document.text) for document in documents)
    families = {}
    for family in families_to_report:
        available = by_family.get(family, [])
        available_chars = sum(len(document.text) for document in available)
        families[family] = {
            "label": _mixture_label(family),
            "requested_weight": float(weights.get(family, 0.0) or 0.0),
            "available_documents": len(available),
            "available_characters": available_chars,
            "selected_documents": len(available) if not applied else 0,
            "selected_characters": available_chars if not applied else 0,
            "actual_percent": (available_chars * 100.0 / total_chars) if total_chars else 0.0,
            "dropped_documents": 0,
            "dropped_characters": 0,
        }
    return {
        "applied": applied,
        "reason": reason,
        "total_available_documents": len(documents),
        "total_selected_documents": len(documents) if not applied else 0,
        "total_available_characters": total_chars,
        "total_selected_characters": total_chars if not applied else 0,
        "families": families,
    }


def _apply_dataset_mixture(
    documents: list[Document],
    weights: dict[str, float],
    progress: Optional[Callable[[Any], None]],
) -> tuple[list[Document], dict[str, Any]]:
    """
    Apply Dataset Blueprint percentages independently.

    Unlike the previous implementation, percentages are NOT normalized.

    100% means:
        Include every document in that category.

    50% means:
        Include approximately half of the documents from that category.

    Categories never reduce one another.
    """

    original_document_count = len(documents)

    documents = _chunk_documents_for_mixture(documents)

    if len(documents) != original_document_count:
        _emit(
            progress,
            f"Dataset mixture: split {original_document_count:,} source file(s) into "
            f"{len(documents):,} sampling chunk(s).",
            49,
        )

    # ---------------------------------------------------------
    # Group documents by category
    # ---------------------------------------------------------

    families: dict[str, list[Document]] = {}

    for doc in documents:
        family = _document_mixture_family(doc)
        families.setdefault(family, []).append(doc)

    selected_documents: list[Document] = []

    report = {
        "applied": True,
        "reason": "",
        "total_available_documents": len(documents),
        "total_selected_documents": 0,
        "total_available_characters": sum(len(d.text) for d in documents),
        "total_selected_characters": 0,
        "families": {},
    }

    # ---------------------------------------------------------
    # Process each category independently
    # ---------------------------------------------------------

    selected_documents = []

    for family in sorted(families.keys()):

        docs = families[family]

        docs.sort(key=_stable_document_sort_key)

        percentage = float(weights.get(family, 100.0))

        percentage = max(0.0, min(100.0, percentage))

        available_documents = len(docs)
        available_characters = sum(len(d.text) for d in docs)

        if percentage >= 100.0:

            chosen = docs

        elif percentage <= 0.0:

            chosen = []

        else:

            keep = round(available_documents * percentage / 100.0)

            chosen = docs[:keep]

        selected_documents.extend(chosen)

        selected_characters = sum(len(d.text) for d in chosen)

        report["families"][family] = {
            "label": _mixture_label(family),
            "requested_weight": percentage,
            "available_documents": available_documents,
            "available_characters": available_characters,
            "selected_documents": len(chosen),
            "selected_characters": selected_characters,
            "effective_requested_percent": percentage,
            "actual_percent": (
                len(chosen) * 100.0 / available_documents
                if available_documents
                else 0.0
            ),
            "dropped_documents": available_documents - len(chosen),
            "dropped_characters": available_characters - selected_characters,
        }

    report["total_selected_documents"] = len(selected_documents)
    report["total_selected_characters"] = sum(
        len(d.text) for d in selected_documents
    )

    _emit(
        progress,
        f"Dataset mixture selected "
        f"{len(selected_documents):,} of {len(documents):,} document chunks.",
        50,
    )

    return selected_documents, report

__all__ = [
    "MIXTURE_LABELS",
    "AGGREGATE_MIXTURE_FAMILIES",
    "MIXTURE_CHUNK_CHARS",
    "_apply_dataset_mixture",
    "_filter_repetitive_documents",
    "MAX_REPETITIVE_UNIT_RATIO",
]

