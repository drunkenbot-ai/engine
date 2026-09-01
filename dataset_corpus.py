from __future__ import annotations
import hashlib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from .data import Document, format_document_for_training
from .dataset_mixture import (MAX_REPETITIVE_UNIT_RATIO,
                              MIN_REPETITION_CHECK_CHARS,
                                MIN_REPETITION_CHECK_UNITS,
                                _canonical_corpus_block,
                                _content_units_for_diversity
                              )


@dataclass
class _CorpusBuildStats:
    """Streaming accumulator for corpus-wide statistics.

    Every field here is either a small counter, a hash, or a length-capped
    example list -- never full document text. This is what keeps
    :class:`_StreamingCorpusBuilder` bounded in memory regardless of how
    large the source corpus is.
    """

    character_count: int = 0
    unique_words: set[str] = field(default_factory=set)
    code_sample_count: int = 0
    prose_sample_count: int = 0
    conversation_sample_count: int = 0
    accepted_document_count: int = 0
    document_char_lengths: list[int] = field(default_factory=list)
    source_files: list[str] = field(default_factory=list)
    source_files_truncated: bool = False
    exact_duplicates_removed: int = 0
    exact_duplicate_examples: list[dict[str, str]] = field(default_factory=list)
    low_diversity_removed: int = 0
    low_diversity_removed_characters: int = 0
    low_diversity_examples: list[dict[str, Any]] = field(default_factory=list)
    block_counts: Counter = field(default_factory=Counter)
    block_examples: dict[str, str] = field(default_factory=dict)
    block_total: int = 0
    block_ignored: int = 0


class _StreamingCorpusBuilder:
    """Filters and writes the training corpus one document at a time.

    Replaces the previous pipeline of "load every document into one list,
    then run exact-dedup over the whole list, then run a repetition filter
    over the whole list, then write the whole list to disk" -- each of which
    held the entire prepared corpus in memory at once. Here, each document
    is deduplicated, quality-checked, written to ``corpus.txt``, and then
    immediately eligible for garbage collection, so at most one document's
    text is resident at a time (aside from small bookkeeping state).
    """

    _EXAMPLE_CAP = 50
    _SOURCE_FILE_CAP = 1000
    _BLOCK_EXAMPLE_CAP = 8

    def __init__(
        self,
        corpus_path: Path,
        code_training_mode: bool,
        generate_instruction_samples: bool,
        reasoning_sample_mode: str,
    ) -> None:
        """Open the corpus file for streaming writes.

        Args:
            corpus_path: Destination corpus text file.
            code_training_mode: Whether to use code/prose tags.
            generate_instruction_samples: Whether code samples should include
                a simple instruction wrapper.
            reasoning_sample_mode: Instruction/reasoning style for code
                samples.
        """

        self._code_training_mode = code_training_mode
        self._generate_instruction_samples = generate_instruction_samples
        self._reasoning_sample_mode = reasoning_sample_mode
        self._seen_digests: dict[str, str] = {}
        self.stats = _CorpusBuildStats()
        corpus_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = corpus_path.open("w", encoding="utf-8")

    def submit(self, document: Document) -> None:
        """Filter, count, and write one document, then let it be freed.

        Args:
            document: Candidate document to evaluate and possibly write.
        """

        canonical = _canonical_corpus_block(document.text)
        if not canonical:
            return

        digest = hashlib.sha256(
            f"{document.kind}\n{document.language or ''}\n{canonical}".encode("utf-8")
        ).hexdigest()
        original_path = self._seen_digests.get(digest)
        if original_path is not None:
            self.stats.exact_duplicates_removed += 1
            if len(self.stats.exact_duplicate_examples) < self._EXAMPLE_CAP:
                self.stats.exact_duplicate_examples.append(
                    {"path": str(document.path), "duplicate_of": original_path, "kind": document.kind}
                )
            return
        self._seen_digests[digest] = str(document.path)

        if self._is_low_diversity(document):
            self.stats.low_diversity_removed += 1
            self.stats.low_diversity_removed_characters += len(document.text)
            if len(self.stats.low_diversity_examples) < self._EXAMPLE_CAP:
                self.stats.low_diversity_examples.append(
                    {"path": str(document.path), "kind": document.kind}
                )
            return

        self._accept(document, canonical)

    @staticmethod
    def _is_low_diversity(document: Document) -> bool:
        """Return whether a document is dominated by repeated content units.

        Args:
            document: Candidate document.

        Returns:
            True when the document should be excluded as low-diversity.
        """

        units = _content_units_for_diversity(document)
        if len(document.text) < MIN_REPETITION_CHECK_CHARS or len(units) < MIN_REPETITION_CHECK_UNITS:
            return False
        duplicate_ratio = 1.0 - (len(set(units)) / len(units))
        return duplicate_ratio > MAX_REPETITIVE_UNIT_RATIO

    def _accept(self, document: Document, canonical: str) -> None:
        """Record stats for and write one accepted document.

        Args:
            document: Accepted document.
            canonical: Canonicalized text used for block-duplicate hashing.
        """

        stats = self.stats
        stats.accepted_document_count += 1
        stats.character_count += len(document.text) + 1
        stats.unique_words.update(word.lower() for word in document.text.split())
        stats.document_char_lengths.append(len(document.text))
        if document.kind == "code":
            stats.code_sample_count += 1
        elif document.kind in {"conversation", "instruction"}:
            stats.conversation_sample_count += 1
        else:
            stats.prose_sample_count += 1
        if len(stats.source_files) < self._SOURCE_FILE_CAP:
            stats.source_files.append(str(document.path))
        else:
            stats.source_files_truncated = True

        if len(canonical) >= 12:
            block_digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            stats.block_counts[block_digest] += 1
            stats.block_examples.setdefault(block_digest, canonical[:240])
            stats.block_total += 1
        else:
            stats.block_ignored += 1

        if self._code_training_mode:
            self._file.write(
                format_document_for_training(
                    document,
                    generate_instruction_samples=self._generate_instruction_samples,
                    reasoning_sample_mode=self._reasoning_sample_mode,
                )
            )
        else:
            self._file.write(document.text)
        self._file.write("\n\n")

    def close(self) -> dict[str, Any]:
        """Flush the corpus file and compute the final duplicate-block report.

        Returns:
            Duplicate-block report dictionary, matching the shape previously
            produced by scanning the fully written corpus file.
        """

        self._file.close()
        stats = self.stats
        unique_blocks = len(stats.block_counts)
        duplicate_blocks = sum(count - 1 for count in stats.block_counts.values() if count > 1)
        duplicate_ratio = duplicate_blocks / max(stats.block_total, 1)
        unique_ratio = unique_blocks / max(stats.block_total, 1)
        repeated = [
            {"count": count, "sample": stats.block_examples[digest]}
            for digest, count in stats.block_counts.most_common(self._BLOCK_EXAMPLE_CAP)
            if count > 1
        ]
        return {
            "block_count": stats.block_total,
            "unique_block_count": unique_blocks,
            "duplicate_block_count": duplicate_blocks,
            "duplicate_block_ratio": duplicate_ratio,
            "unique_block_ratio": unique_ratio,
            "ignored_block_count": stats.block_ignored,
            "truncated": False,
            "most_repeated_block_count": repeated[0]["count"] if repeated else 1,
            "top_repeated_blocks": repeated,
        }

