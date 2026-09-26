"""Corpus Scrubber: Sanitizes and deduplicates existing pre-training/fine-tuning datasets.

Removes repeated synthetic system prompt boilerplate, OHLCV matrix repetitions,
and near-duplicate documents via MinHash LSH, writing cluster-safe 28MB shards.
"""

from __future__ import annotations

import hashlib
import json
import logging
import zlib
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from .dataset_streamer import PartitionedJsonlWriter

LOGGER = logging.getLogger(__name__)


class CorpusScrubber:
    """Scrubs existing JSONL dataset directories of repetitive boilerplate and near-duplicates."""

    def __init__(
        self,
        output_dir: Path,
        prefix: str = "clean_part",
        max_prefix_repeats: int = 50,
        min_text_chars: int = 60,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.prefix = prefix
        self.max_prefix_repeats = max_prefix_repeats
        self.min_text_chars = min_text_chars

        self.writer = PartitionedJsonlWriter(self.output_dir, prefix=self.prefix)
        self.seen_exact_hashes: set[str] = set()
        self.prefix_counts: Counter[str] = Counter()
        self.lsh_index: dict[tuple[int, tuple[int, ...]], int] = {}

        self.total_scanned = 0
        self.exact_duplicates_dropped = 0
        self.prefix_boilerplate_dropped = 0
        self.near_duplicates_dropped = 0
        self.short_records_dropped = 0
        self.retained_records = 0

    def _compute_minhash(self, text: str) -> tuple[int, ...]:
        """Compute compact 16-hash MinHash signature."""
        if len(text) < 60:
            return ()
        shingles = {zlib.crc32(text[i : i + 5].encode("utf-8")) for i in range(0, min(len(text) - 4, 1200), 2)}
        if len(shingles) < 8:
            return ()
        sig = []
        for i in range(16):
            a = 1664525 * (i + 1) + 1013904223
            b = 22695477 * (i + 1) + 1
            min_val = min(((a * h + b) & 0xFFFFFFFF) for h in shingles)
            sig.append(min_val)
        return tuple(sig)

    def _is_minhash_duplicate(self, sig: tuple[int, ...], length: int) -> bool:
        """Check if >= 3 of 4 LSH bands match with similar length."""
        bands = [
            (0, sig[0:4]),
            (1, sig[4:8]),
            (2, sig[8:12]),
            (3, sig[12:16]),
        ]
        matching_bands = 0
        for band_idx, band_val in bands:
            key = (band_idx, band_val)
            prev_len = self.lsh_index.get(key)
            if prev_len is not None:
                ratio = min(length, prev_len) / max(length, prev_len)
                if ratio >= 0.70:
                    matching_bands += 1
        return matching_bands >= 3

    def _record_minhash(self, sig: tuple[int, ...], length: int) -> None:
        bands = [
            (0, sig[0:4]),
            (1, sig[4:8]),
            (2, sig[8:12]),
            (3, sig[12:16]),
        ]
        for band_idx, band_val in bands:
            key = (band_idx, band_val)
            if key not in self.lsh_index:
                self.lsh_index[key] = length

    def scrub_record(self, raw: dict[str, Any]) -> bool:
        """Process one record. Returns True if accepted and written, False if dropped."""
        self.total_scanned += 1
        text = raw.get("text", "")
        if not text or not isinstance(text, str):
            self.short_records_dropped += 1
            return False

        clean = text.strip()
        if len(clean) < self.min_text_chars:
            self.short_records_dropped += 1
            return False

        # Exact SHA256 deduplication
        doc_hash = hashlib.sha256(clean.encode("utf-8")).hexdigest()
        if doc_hash in self.seen_exact_hashes:
            self.exact_duplicates_dropped += 1
            return False
        if len(self.seen_exact_hashes) < 500_000:
            self.seen_exact_hashes.add(doc_hash)

        # Prefix boilerplate throttling (throttles identical synthetic templates/prompts)
        if len(clean) >= 48:
            prefix_sig = clean[:48].lower()
            self.prefix_counts[prefix_sig] += 1
            if self.prefix_counts[prefix_sig] > self.max_prefix_repeats:
                self.prefix_boilerplate_dropped += 1
                return False

        # MinHash LSH near-duplicate rejection
        sig = self._compute_minhash(clean)
        if sig:
            if self._is_minhash_duplicate(sig, len(clean)):
                self.near_duplicates_dropped += 1
                return False
            self._record_minhash(sig, len(clean))

        # Accept and write
        out_record = {"text": clean, "meta": raw.get("meta", {"length": len(clean)})}
        self.writer.write_record(out_record)
        self.retained_records += 1
        return True

    def close(self) -> dict[str, Any]:
        """Flush writer and return comprehensive scrub report."""
        writer_manifest = self.writer.close()
        report = {
            "total_scanned": self.total_scanned,
            "retained_records": self.retained_records,
            "exact_duplicates_dropped": self.exact_duplicates_dropped,
            "prefix_boilerplate_dropped": self.prefix_boilerplate_dropped,
            "near_duplicates_dropped": self.near_duplicates_dropped,
            "short_records_dropped": self.short_records_dropped,
            "retained_percentage": round((self.retained_records / max(self.total_scanned, 1)) * 100, 2),
            "writer_manifest": writer_manifest,
        }
        report_path = self.output_dir / "scrub_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report


def scrub_jsonl_directory(
    source_dir: Path,
    output_dir: Path,
    prefix: str = "clean_part",
    max_prefix_repeats: int = 50,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> dict[str, Any]:
    """Scrub all JSONL files in source_dir into partitioned 28MB shards in output_dir."""
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    jsonl_files = sorted(source_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise ValueError(f"No .jsonl files found in {source_dir}")

    scrubber = CorpusScrubber(output_dir, prefix=prefix, max_prefix_repeats=max_prefix_repeats)

    for f_idx, fpath in enumerate(jsonl_files):
        LOGGER.info("Scrubbing %s (%d/%d)...", fpath.name, f_idx + 1, len(jsonl_files))
        with fpath.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    scrubber.scrub_record(record)
                except Exception:
                    continue

                if progress_callback and scrubber.total_scanned % 5000 == 0:
                    progress_callback(scrubber.total_scanned, scrubber.retained_records)

    return scrubber.close()
