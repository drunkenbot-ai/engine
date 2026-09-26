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


FRONTIER_SOURCE_PRESETS: dict[str, dict[str, Any]] = {
    "fineweb_edu": {
        "dataset_name": "HuggingFaceFW/fineweb-edu",
        "subset": "sample-10BT",
        "split": "train",
        "text_key": "text",
        "prefix": "fineweb_edu",
        "filter_fn": lambda r: r.get("score", 5.0) is None or float(r.get("score", 5.0)) >= 3.0,
    },
    "open_web_math": {
        "dataset_name": "open-web-math/open-web-math",
        "subset": None,
        "split": "train",
        "text_key": "text",
        "prefix": "open_web_math",
        "filter_fn": None,
    },
    "starcoder2": {
        "dataset_name": "bigcode/the-stack-smol",
        "subset": "data",
        "split": "train",
        "text_key": "content",
        "prefix": "starcoder2",
        "filter_fn": None,
    },
    "wikipedia": {
        "dataset_name": "wikimedia/wikipedia",
        "subset": "20231101.en",
        "split": "train",
        "text_key": "text",
        "prefix": "wikipedia_en",
        "filter_fn": None,
    },
}


def stream_hf_dataset(
    preset_or_name: str,
    output_dir: Path,
    target_tokens: int = 50_000_000,
    subset: Optional[str] = None,
    split: str = "train",
    text_key: str = "text",
    prefix: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> dict[str, Any]:
    """Stream records from a HuggingFace dataset into cluster-ready 28MB shards.

    Args:
        preset_or_name: Either a preset slug ('fineweb_edu', 'open_web_math', etc.) or HF dataset name.
        output_dir: Target directory for partition JSONL files.
        target_tokens: Ingestion token ceiling.
        subset: HF dataset configuration name.
        split: Dataset split to stream.
        text_key: Column holding text content.
        prefix: Shard filename prefix.
        progress_callback: Progress reporting callable.

    Returns:
        Manifest dictionary.
    """
    import datasets

    filter_fn = None
    if preset_or_name in FRONTIER_SOURCE_PRESETS:
        cfg = FRONTIER_SOURCE_PRESETS[preset_or_name]
        ds_name = cfg["dataset_name"]
        subset = subset or cfg["subset"]
        split = split or cfg["split"]
        text_key = cfg["text_key"]
        prefix = prefix or cfg["prefix"]
        filter_fn = cfg["filter_fn"]
    else:
        ds_name = preset_or_name
        prefix = prefix or "part"

    LOGGER.info("Connecting stream to %s (subset=%s, split=%s)...", ds_name, subset, split)
    ds = datasets.load_dataset(ds_name, name=subset, split=split, streaming=True)

    def extract_text(raw: dict[str, Any]) -> str:
        if filter_fn and not filter_fn(raw):
            return ""
        val = raw.get(text_key, "")
        return str(val) if val else ""

    return stream_dataset_to_partitions(
        records_iterable=ds,
        output_dir=Path(output_dir),
        prefix=prefix,
        target_tokens=target_tokens,
        text_extractor=extract_text,
        progress_callback=progress_callback,
    )


def main() -> None:
    """CLI entrypoint for streaming frontier tokens directly into partitioned shards."""
    import argparse

    parser = argparse.ArgumentParser(description="Stream frontier datasets into cluster-safe 28MB shards.")
    parser.add_argument(
        "--preset",
        type=str,
        default="fineweb_edu",
        choices=list(FRONTIER_SOURCE_PRESETS.keys()),
        help="Dataset preset to stream.",
    )
    parser.add_argument(
        "--target-tokens",
        type=str,
        default="50M",
        help="Target tokens to stream (e.g. 10M, 50M, 1B, 12B).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="E:/AI_Projects/dataset/fineweb_edu",
        help="Destination directory for 28MB shards.",
    )

    args = parser.parse_args()

    # Parse target tokens shorthand
    val_str = args.target_tokens.strip().upper()
    if val_str.endswith("B"):
        tokens = int(float(val_str[:-1]) * 1_000_000_000)
    elif val_str.endswith("M"):
        tokens = int(float(val_str[:-1]) * 1_000_000)
    elif val_str.endswith("K"):
        tokens = int(float(val_str[:-1]) * 1_000)
    else:
        tokens = int(val_str)

    print(f"[*] Starting streaming ingestion for preset '{args.preset}' -> {args.output_dir}")
    print(f"[*] Target token ceiling: {tokens:,} tokens")

    def report_progress(current: int, target: int) -> None:
        pct = (current / max(target, 1)) * 100
        print(f"  -> Progress: {current:,} / {target:,} tokens ({pct:.1f}%)")

    manifest = stream_hf_dataset(
        args.preset,
        output_dir=Path(args.output_dir),
        target_tokens=tokens,
        progress_callback=report_progress,
    )
    print("\n[+] Streaming Ingestion Complete!")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

