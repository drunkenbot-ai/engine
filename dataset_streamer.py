"""Streaming ingestion engine for high-density frontier pretraining datasets.

Streams pre-filtered, deduplicated open datasets (FineWeb-Edu, StarCoder2, OpenWebMath)
directly into cluster-safe 28.0 MB JSONL shards without requiring massive local disk downloads.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

LOGGER = logging.getLogger(__name__)

PARTITION_MAX_BYTES = 28 * 1024 * 1024  # 28.0 MB ceiling for cluster workers


class PartitionedJsonlWriter:
    """Streams JSON records into strictly bounded partition files (<= 28.0 MB)."""

    def __init__(self, output_dir: Path, prefix: str = "part", max_bytes: int = PARTITION_MAX_BYTES) -> None:
        self.output_dir = Path(output_dir)
        self.prefix = prefix
        self.max_bytes = max_bytes
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.current_part_idx = 1
        self.current_part_file: Optional[Any] = None
        self.current_part_bytes = 0
        self.total_records = 0
        self.total_tokens_estimated = 0
        self.total_bytes = 0
        self.partition_files: list[str] = []
        self._open_new_partition()

    def _open_new_partition(self) -> None:
        if self.current_part_file is not None:
            self.current_part_file.close()
        filename = f"{self.prefix}_{self.current_part_idx:04d}.jsonl"
        filepath = self.output_dir / filename
        self.partition_files.append(filename)
        self.current_part_file = filepath.open("w", encoding="utf-8")
        self.current_part_bytes = 0

    def write_record(self, record: dict[str, Any]) -> None:
        """Serialize and append record, rotating partition if 28MB limit is reached."""
        line = json.dumps(record, ensure_ascii=False) + "\n"
        encoded = line.encode("utf-8")
        line_len = len(encoded)

        if self.current_part_bytes + line_len > self.max_bytes and self.current_part_bytes > 0:
            self.current_part_idx += 1
            self._open_new_partition()

        self.current_part_file.write(line)
        self.current_part_bytes += line_len
        self.total_bytes += line_len
        self.total_records += 1

        # Token estimate: ~3.85 bytes per token
        self.total_tokens_estimated += max(1, int(line_len / 3.85))

    def close(self) -> dict[str, Any]:
        """Flush final shard and return manifest summary."""
        if self.current_part_file is not None:
            self.current_part_file.close()
            self.current_part_file = None

        manifest = {
            "total_records": self.total_records,
            "total_bytes": self.total_bytes,
            "total_mb": round(self.total_bytes / (1024 * 1024), 2),
            "estimated_tokens": self.total_tokens_estimated,
            "partitions_count": len(self.partition_files),
            "partition_files": self.partition_files,
        }
        manifest_path = self.output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        LOGGER.info("Closed writer for %s: %d records across %d shards.", self.output_dir.name, self.total_records, len(self.partition_files))
        return manifest


def stream_dataset_to_partitions(
    records_iterable: Iterator[dict[str, Any]],
    output_dir: Path,
    prefix: str,
    target_tokens: int = 50_000_000,
    text_extractor: Optional[Callable[[dict[str, Any]], str]] = None,
    filter_fn: Optional[Callable[[str], bool]] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> dict[str, Any]:
    """Stream generic dataset records into cluster-ready 28MB JSONL shards with deduplication.

    Args:
        records_iterable: Iterator yielding raw record dicts.
        output_dir: Output folder for partitions.
        prefix: Partition file prefix.
        target_tokens: Maximum target tokens to ingest before stopping.
        text_extractor: Function to extract plain text string from record.
        filter_fn: Optional predicate to accept/reject text.
        progress_callback: Optional (current_tokens, target_tokens) reporter.

    Returns:
        Manifest summary dictionary.
    """
    writer = PartitionedJsonlWriter(output_dir, prefix=prefix)
    seen_prefixes: set[str] = set()

    for raw in records_iterable:
        if writer.total_tokens_estimated >= target_tokens:
            break

        text = text_extractor(raw) if text_extractor else raw.get("text", "")
        if not text or not isinstance(text, str):
            continue

        clean = text.strip()
        if len(clean) < 60:
            continue

        if filter_fn and not filter_fn(clean):
            continue

        # Prefix deduplication: catch repetitive synthetic boilerplate
        prefix_sig = clean[:48].lower()
        if prefix_sig in seen_prefixes:
            continue
        if len(seen_prefixes) < 100_000:
            seen_prefixes.add(prefix_sig)

        record = {"text": clean, "meta": {"length": len(clean)}}
        writer.write_record(record)

        if progress_callback and writer.total_records % 1000 == 0:
            progress_callback(writer.total_tokens_estimated, target_tokens)

    return writer.close()
