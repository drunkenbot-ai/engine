from __future__ import annotations

from typing import Any, Callable, Optional
from pathlib import Path

from .config import DatasetConfig
from .conversation_datasets import CONVERSATION_DATASET_PRESETS
from .data import file_fingerprint, load_structured_json_documents
from .dataset_helpers import _local_structured_dataset_paths
from .lineage import read_json, write_json
from .dataset_preview_health import (
    DatasetPreviewResult,
    _bad_extraction_reasons,
    _balance_label,
    _emit,
    _file_sha256_cancellable,
    _has_prepared_token_artifacts,
    _preview_fingerprint,
    _preview_supported_document,
    _readiness_report,
    _supported_source_paths_cancellable,
)

def scan_dataset_preview(
    config: DatasetConfig,
    sample_limit: int = 8,
    progress: Optional[Callable[[Any], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> DatasetPreviewResult:
    _emit(progress, "Scanning supported source files...", 5)
    local_structured_paths = _local_structured_dataset_paths(config)
    if config.input_dir.exists():
        paths = _supported_source_paths_cancellable(
            config.input_dir,
            config.code_training_mode,
            config.include_source_code,
            should_stop,
        )
    elif config.conversation_datasets or local_structured_paths:
        paths = []
    else:
        paths = _supported_source_paths_cancellable(
            config.input_dir,
            config.code_training_mode,
            config.include_source_code,
            should_stop,
        )
    suffix_counts: dict[str, int] = {}
    total_bytes = 0
    for path in paths:
        if should_stop and should_stop():
            raise RuntimeError("Dataset preview stopped by user.")
        suffix_counts[path.suffix.lower() or "<none>"] = suffix_counts.get(path.suffix.lower() or "<none>", 0) + 1
        try:
            total_bytes += path.stat().st_size
        except OSError:
            pass
    for path, _, _ in local_structured_paths:
        if should_stop and should_stop():
            raise RuntimeError("Dataset preview stopped by user.")
        path = Path(path)
        if path.exists() and path.is_file():
            suffix_counts[path.suffix.lower() or "<none>"] = suffix_counts.get(path.suffix.lower() or "<none>", 0) + 1
            try:
                total_bytes += path.stat().st_size
            except OSError:
                pass

    summary = read_json(config.output_dir / "dataset_summary.json", default={}) or {}
    prepared = _has_prepared_token_artifacts(config.output_dir)
    issues: list[str] = []
    if not paths and not config.conversation_datasets and not local_structured_paths:
        issues.append("No supported source files found.")
    if config.conversation_dataset_path:
        issues.append(f"Local conversation JSON selected: {config.conversation_dataset_path}.")
    if config.instruction_dataset_path:
        issues.append(f"Local instruction JSON selected: {config.instruction_dataset_path}.")
    if config.conversation_datasets:
        labels = [
            CONVERSATION_DATASET_PRESETS[item].label
            for item in config.conversation_datasets
            if item in CONVERSATION_DATASET_PRESETS
        ]
        issues.append(f"Conversation datasets selected: {', '.join(labels)}.")
    if total_bytes < 100_000:
        issues.append("Source content appears small for meaningful LLM training.")
    if prepared and summary:
        token_count = int(summary.get("token_count", 0) or 0)
        if token_count < 50_000:
            issues.append("Prepared token count is low; expect smoke-test quality only.")
        if summary.get("warning"):
            issues.append(str(summary["warning"]))
    elif config.output_dir.exists():
        issues.append("Dataset folder exists but does not contain a complete prepared dataset.")

    _emit(progress, "Reading a few preview samples...", 30)
    sample_previews: list[dict[str, str]] = []
    all_previews: list[dict[str, str]] = []
    bad_extraction_files: list[dict[str, str]] = []
    content_fingerprints: dict[str, list[str]] = {}
    preview_cache_path = config.output_dir / "preview_scan_cache.json"
    preview_cache = read_json(preview_cache_path, default={}) or {}
    cached_files = preview_cache.get("files") if isinstance(preview_cache.get("files"), dict) else {}
    updated_cache: dict[str, Any] = {}
    readable = 0
    scan_limit = min(len(paths), max(sample_limit * 8, 80))
    for path in paths[:scan_limit]:
        if should_stop and should_stop():
            raise RuntimeError("Dataset preview stopped by user.")
        try:
            stat = path.stat()
            size = stat.st_size
            mtime_ns = stat.st_mtime_ns
        except OSError:
            size = 0
            mtime_ns = 0
        cache_key = str(path.resolve())
        cached_entry = cached_files.get(cache_key, {}) if isinstance(cached_files, dict) else {}
        cached_preview = cached_entry.get("preview")
        cached_reasons = cached_entry.get("bad_extraction_reasons")
        if (
            cached_entry.get("size") == size
            and cached_entry.get("mtime_ns") == mtime_ns
            and isinstance(cached_preview, dict)
            and isinstance(cached_reasons, list)
        ):
            preview = cached_preview
            reasons = [str(item) for item in cached_reasons]
        else:
            try:
                preview = _preview_supported_document(path, config, should_stop)
            except RuntimeError:
                raise
            except Exception as exc:
                issues.append(f"Could not preview {path.name}: {exc}")
                continue
            reasons = _bad_extraction_reasons(path, preview, size)
        updated_cache[cache_key] = {
            "size": size,
            "mtime_ns": mtime_ns,
            "preview": preview,
            "bad_extraction_reasons": reasons,
            "duplicate_digest": cached_entry.get("duplicate_digest"),
            "duplicate_digest_mode": cached_entry.get("duplicate_digest_mode"),
            "strict_duplicate_digest": cached_entry.get("strict_duplicate_digest"),
        }
        if reasons:
            bad_extraction_files.append(
                {
                    "path": str(path),
                    "reasons": "; ".join(reasons),
                    "size": str(size),
                }
            )
        if preview is None:
            issues.append(f"{path.name} has no readable text.")
            continue
        readable += 1
        all_previews.append(preview)
        preview_text = preview.get("preview", "")
        if len(preview_text) >= 120:
            content_fingerprints.setdefault(_preview_fingerprint(preview_text), []).append(str(path))
        if len(sample_previews) < sample_limit:
            sample_previews.append(preview)
            percent = 30 + int(45 * len(sample_previews) / max(sample_limit, 1))
            _emit(progress, f"Previewed {path.name}.", percent)

    for local_path, kind, _ in local_structured_paths:
        if len(sample_previews) >= sample_limit:
            continue
        if should_stop and should_stop():
            raise RuntimeError("Dataset preview stopped by user.")
        try:
            local_documents = load_structured_json_documents(Path(local_path), kind=kind, lowercase=config.lowercase)
        except Exception as exc:
            issues.append(f"Could not preview {kind} JSON dataset: {exc}")
            continue
        for document in local_documents[: max(1, sample_limit - len(sample_previews))]:
            preview = {
                "path": str(document.path),
                "kind": document.kind,
                "language": document.language or "",
                "characters": str(len(document.text)),
                "preview": document.text[:1200],
            }
            all_previews.append(preview)
            sample_previews.append(preview)
            _emit(progress, f"Previewed {kind} JSON sample.", 75)
            if len(sample_previews) >= sample_limit:
                break

    if readable == 0 and paths:
        issues.append("Supported files were found, but none produced readable preview text.")
    _emit(progress, "Checking duplicate files and repeated extracted text...", 85)
    size_groups: dict[int, list[Path]] = {}
    for path in paths:
        if should_stop and should_stop():
            raise RuntimeError("Dataset preview stopped by user.")
        try:
            size_groups.setdefault(path.stat().st_size, []).append(path)
        except OSError:
            continue
    exact_hashes: dict[str, list[str]] = {}
    for same_size_paths in size_groups.values():
        if len(same_size_paths) < 2:
            continue
        for path in same_size_paths:
            cache_key = str(path.resolve())
            entry = updated_cache.get(cache_key)
            try:
                stat = path.stat()
                size = stat.st_size
                mtime_ns = stat.st_mtime_ns
            except OSError:
                continue
            if entry is None:
                cached_entry = cached_files.get(cache_key, {}) if isinstance(cached_files, dict) else {}
                entry = {
                    "size": size,
                    "mtime_ns": mtime_ns,
                    "preview": cached_entry.get("preview"),
                    "bad_extraction_reasons": cached_entry.get("bad_extraction_reasons"),
                    "duplicate_digest": cached_entry.get("duplicate_digest"),
                    "duplicate_digest_mode": cached_entry.get("duplicate_digest_mode"),
                    "strict_duplicate_digest": cached_entry.get("strict_duplicate_digest"),
                }
            digest_mode = "fast" if config.fast_scan_mode else "full"
            if (
                entry.get("size") == size
                and entry.get("mtime_ns") == mtime_ns
                and entry.get("duplicate_digest_mode") == digest_mode
                and isinstance(entry.get("duplicate_digest"), str)
            ):
                digest = str(entry["duplicate_digest"])
            else:
                if config.fast_scan_mode:
                    digest = file_fingerprint(path, fast=True, sample_bytes=config.fast_scan_sample_bytes)
                else:
                    digest = _file_sha256_cancellable(path, should_stop)
                entry["duplicate_digest"] = digest
                entry["duplicate_digest_mode"] = digest_mode
            entry["size"] = size
            entry["mtime_ns"] = mtime_ns
            updated_cache[cache_key] = entry
            exact_hashes.setdefault(digest, []).append(str(path))
    if config.fast_scan_mode and config.strict_duplicate_verification:
        verified_hashes: dict[str, list[str]] = {}
        for fast_group in exact_hashes.values():
            if len(fast_group) < 2:
                continue
            for path_str in fast_group:
                path = Path(path_str)
                cache_key = str(path.resolve())
                entry = updated_cache.get(cache_key, {})
                try:
                    stat = path.stat()
                    size = stat.st_size
                    mtime_ns = stat.st_mtime_ns
                except OSError:
                    continue
                if (
                    entry.get("size") == size
                    and entry.get("mtime_ns") == mtime_ns
                    and isinstance(entry.get("strict_duplicate_digest"), str)
                ):
                    strict_digest = str(entry["strict_duplicate_digest"])
                else:
                    strict_digest = _file_sha256_cancellable(path, should_stop)
                    entry["strict_duplicate_digest"] = strict_digest
                entry["size"] = size
                entry["mtime_ns"] = mtime_ns
                updated_cache[cache_key] = entry
                verified_hashes.setdefault(strict_digest, []).append(path_str)
        exact_hashes = verified_hashes
    duplicate_groups: list[dict[str, Any]] = []
    for group in exact_hashes.values():
        if len(group) > 1:
            duplicate_groups.append({"type": "exact file", "count": len(group), "files": group[:8]})
    for group in content_fingerprints.values():
        unique_group = sorted(set(group))
        if len(unique_group) > 1:
            duplicate_groups.append({"type": "similar extracted text", "count": len(unique_group), "files": unique_group[:8]})
    duplicate_files = {
        file_path
        for group in duplicate_groups
        for file_path in group.get("files", [])
    }
    duplicate_count = len(duplicate_files)
    if duplicate_groups:
        issues.append(f"Found {len(duplicate_groups)} likely duplicate group(s) involving {duplicate_count} file entries.")
    if bad_extraction_files:
        issues.append(f"Found {len(bad_extraction_files)} file(s) with suspicious extraction quality.")
    summary_code_count = int(summary.get("code_sample_count", 0) or 0)
    summary_prose_count = int(summary.get("prose_sample_count", 0) or 0)
    code_preview_count = summary_code_count or sum(1 for preview in all_previews if preview.get("kind") == "code")
    prose_preview_count = summary_prose_count or sum(1 for preview in all_previews if preview.get("kind") != "code")
    balance = _balance_label(code_preview_count, prose_preview_count, config.code_training_mode)
    effective_source_count = len(paths) + len(config.conversation_datasets) + len(local_structured_paths)
    readiness_score, readiness_label, readiness_reasons = _readiness_report(
        source_file_count=effective_source_count,
        total_bytes=total_bytes,
        prepared=prepared,
        summary=summary,
        duplicate_count=duplicate_count,
        bad_extraction_count=len(bad_extraction_files),
        code_count=code_preview_count,
        prose_count=prose_preview_count,
        code_training_mode=config.code_training_mode,
    )
    _emit(progress, "Dataset preview complete.", 100)
    if config.output_dir.exists():
        write_json(preview_cache_path, {"version": 1, "files": updated_cache})
    return DatasetPreviewResult(
        source_file_count=effective_source_count,
        prepared=prepared,
        total_bytes=total_bytes,
        suffix_counts=dict(sorted(suffix_counts.items())),
        sample_previews=sample_previews,
        issues=issues,
        summary=summary,
        duplicate_count=duplicate_count,
        duplicate_groups=duplicate_groups,
        bad_extraction_count=len(bad_extraction_files),
        bad_extraction_files=bad_extraction_files[:30],
        code_preview_count=code_preview_count,
        prose_preview_count=prose_preview_count,
        balance_label=balance,
        readiness_score=readiness_score,
        readiness_label=readiness_label,
        readiness_reasons=readiness_reasons,
    )


__all__ = [
    "DatasetPreviewResult",
    "scan_dataset_preview",
]
