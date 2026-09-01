"""Standalone document extraction helpers, safe for worker processes.

This module intentionally imports only from :mod:`.data`. ``dataset_build.py``
(via ``.config``, ``.tokenizer``, ``.training``) pulls in ``torch`` and other
heavy dependencies; when a function defined in ``dataset_build.py`` is used as
a ``ProcessPoolExecutor`` target under the ``spawn`` start method, every
worker process has to re-import that entire chain just to resolve the
function, which is slow and memory-heavy for something that only needs to
read and clean one text file. Keeping the actual worker function here instead
means worker processes only pay for what they use.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

from .data import (
    Document,
    document_to_dict,
    expand_code_documents,
    load_jsonl_documents,
    read_supported_document,
)


def bad_extraction_reasons(
    path: Path,
    preview: Optional[dict[str, str]],
    size: int,
) -> list[str]:
    """Return quality reasons when extracted preview text looks suspicious.

    Args:
        path: Source file path.
        preview: Preview dictionary with at least ``preview`` and ``kind``.
        size: Source file size in bytes.

    Returns:
        Human-readable list of extraction quality concerns, empty if none.
    """

    suffix = path.suffix.lower()
    if preview is None:
        return ["no readable preview text"]
    text = str(preview.get("preview", ""))
    if not text.strip():
        return ["empty preview text"]
    reasons: list[str] = []
    visible = [char for char in text if not char.isspace()]
    if len(text) < 80 and suffix == ".pdf":
        reasons.append("very little text extracted from PDF preview")
    if size > 250_000 and suffix == ".pdf" and len(text) < 200:
        reasons.append("large PDF produced very little readable text")
    if visible:
        alpha_ratio = sum(char.isalpha() for char in visible) / len(visible)
        symbol_ratio = sum(not char.isalnum() for char in visible) / len(visible)
        if alpha_ratio < 0.25 and str(preview.get("kind")) != "code":
            reasons.append("low alphabetic text ratio")
        if symbol_ratio > 0.45:
            reasons.append("high symbol/noise ratio")
    if re.search(r"(.)\1{18,}", text):
        reasons.append("long repeated character run")
    if text.count("\ufffd") >= 3 or "\u00c3\u0192" in text[:500]:
        reasons.append("encoding artifacts detected")
    words = re.findall(r"[A-Za-z]{2,}", text)
    if suffix in {".pdf", ".txt", ".md", ".text"} and len(set(words)) < 8 and len(text) > 200:
        reasons.append("very low word variety")
    return reasons


def extract_documents_worker(
    path: Path,
    lowercase: bool,
    code_training_mode: bool,
    preserve_indentation: bool,
    include_prose: bool,
    extract_code_blocks: bool,
) -> dict[str, Any]:
    """Extract and expand one source file into documents.

    Intended as a ``ProcessPoolExecutor`` target: runs inside a worker
    process so CPU-bound extraction work (PDF parsing, regex text cleaning,
    code-block detection) for multiple large files genuinely runs in
    parallel across CPU cores, rather than serialized behind the GIL as with
    threads. Only one source file's text is ever resident in a given worker
    process at a time.

    Args:
        path: Source file path.
        lowercase: Whether to lowercase loaded content.
        code_training_mode: Enables code-aware loading and expansion.
        preserve_indentation: Keeps code formatting where possible.
        include_prose: Keeps prose documents in code-aware mode.
        extract_code_blocks: Extracts code-like blocks from prose documents.

    Returns:
        Dict with the source path, extracted documents (as plain dicts, for
        safe pickling back to the main process), an optional error message,
        and any bad-extraction-quality reasons.
    """

    try:
        if path.suffix.lower() == ".jsonl":
            record_errors: list[str] = []
            source_documents = load_jsonl_documents(
                path,
                lowercase=lowercase,
                on_invalid=record_errors.append,
            )
            return {
                "path": str(path),
                "documents": [document_to_dict(doc) for doc in source_documents],
                "error": None,
                "record_errors": record_errors,
                "bad_extraction_reasons": [],
            }
        source_doc = read_supported_document(
            path,
            lowercase=lowercase,
            code_training_mode=code_training_mode,
            preserve_indentation=preserve_indentation,
        )
    except Exception as exc:  # noqa: BLE001 - reported back to the main process
        return {
            "path": str(path),
            "documents": [],
            "error": str(exc),
            "record_errors": [],
            "bad_extraction_reasons": [],
        }

    if source_doc is None:
        return {
            "path": str(path),
            "documents": [],
            "error": None,
            "record_errors": [],
            "bad_extraction_reasons": [],
        }

    reasons = bad_extraction_reasons(
        path,
        {
            "path": str(path),
            "kind": source_doc.kind,
            "language": source_doc.language or "",
            "characters": str(len(source_doc.text)),
            "preview": source_doc.text[:1200],
        },
        path.stat().st_size,
    )
    if path.suffix.lower() == ".pdf" and reasons:
        return {
            "path": str(path),
            "documents": [],
            "error": None,
            "record_errors": [],
            "bad_extraction_reasons": reasons,
        }

    source_documents: list[Document] = [source_doc]
    if code_training_mode:
        source_documents = expand_code_documents(
            source_documents,
            include_prose=include_prose,
            extract_code_blocks=extract_code_blocks,
            preserve_indentation=preserve_indentation,
            should_stop=None,
        )
    return {
        "path": str(path),
        "documents": [document_to_dict(doc) for doc in source_documents],
        "error": None,
        "record_errors": [],
        "bad_extraction_reasons": [],
    }


__all__ = ["bad_extraction_reasons", "extract_documents_worker"]
