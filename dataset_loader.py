from __future__ import annotations
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Optional
from .config import DatasetConfig, dataclass_to_jsonable
from .conversation_datasets import CONVERSATION_DATASET_PRESETS, dataset_ids_for_stage, load_conversation_documents
from .data import (
    Document,
    SUPPORTED_CODE_SUFFIXES,
    SUPPORTED_TEXT_SUFFIXES,
    JsonRecordDiagnostics,
    document_from_dict,
    file_fingerprint,
    load_structured_json_documents_with_diagnostics,
    structured_json_source_files,
    supported_source_paths,
)
from .document_extraction import bad_extraction_reasons as _bad_extraction_reasons, extract_documents_worker
from .manifest_store import ManifestStore
from .dataset_corpus import _StreamingCorpusBuilder
from .dataset_helpers import _emit, _cache_key, _local_structured_dataset_paths


def _load_documents_with_cache(
        config: DatasetConfig,
        corpus_builder: "_StreamingCorpusBuilder",
        progress: Optional[Callable[[Any], None]],
        should_stop: Optional[Callable[[], bool]],
) -> tuple[Any, int, int, int, int, int, int]:
    """Load documents using an extraction cache and stream them into the corpus.

    New (non-cached) files are extracted in parallel worker processes (see
    ``extract_documents_worker``), bounding peak memory to roughly
    ``config.max_workers`` files' worth of text at a time rather than the
    whole corpus. Every extracted or cached document is handed to
    ``corpus_builder.submit`` and then immediately released -- this function
    never accumulates a list of documents itself.

    Args:
        config: Dataset configuration.
        corpus_builder: Streaming builder that filters, counts, and writes
            each document as it arrives.
        progress: Optional progress callback.
        should_stop: Optional cancellation callback.

    Returns:
        Manifest plus cached, processed, partial, skipped, failed, and invalid
        record counts.
    """

    manifest_db_path = config.output_dir / "dataset_manifest.sqlite3"
    legacy_manifest_path = config.output_dir / "dataset_manifest.json"
    cache_dir = config.output_dir / "cache" / "documents"
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = ManifestStore.open(manifest_db_path,
                                  legacy_json_path=legacy_manifest_path)
    manifest.start_file_tracking()
    key = _cache_key(config)
    force_reprocess = config.prepare_mode == "force_reprocess"

    local_structured_paths = _local_structured_dataset_paths(config)
    configured_structured_files = {
        source.resolve()
        for local_path, _kind, _label in local_structured_paths
        for source in structured_json_source_files(Path(local_path))
    }
    selected_default_files = [
        Path(path)
        for path in config.default_data_paths
        if Path(path).exists() and Path(path).is_file()
    ]
    input_dir_resolved = config.input_dir.resolve() if config.input_dir.exists() else None
    default_files_under_input = bool(
        selected_default_files) and input_dir_resolved is not None and all(
        input_dir_resolved in candidate.resolve().parents or candidate.resolve() == input_dir_resolved
        for candidate in selected_default_files
    )
    if config.input_dir.exists() and not default_files_under_input:
        source_paths = supported_source_paths(
            config.input_dir,
            code_training_mode=config.code_training_mode,
            include_source_code=config.include_source_code,
        )
    elif config.conversation_datasets or local_structured_paths or config.default_data_paths:
        source_paths = []
    else:
        source_paths = supported_source_paths(
            config.input_dir,
            code_training_mode=config.code_training_mode,
            include_source_code=config.include_source_code,
        )
    default_paths = []
    seen_source_paths = {path.resolve() for path in source_paths if
                         path.exists()}
    for candidate in selected_default_files:
        if not candidate.exists() or not candidate.is_file():
            _emit(progress, f"Skipped bundled data file: {candidate}")
            continue
        suffix = candidate.suffix.lower()
        if suffix not in SUPPORTED_TEXT_SUFFIXES and suffix not in SUPPORTED_CODE_SUFFIXES and suffix not in {
            ".pdf", ".json", ".jsonl"}:
            _emit(progress,
                  f"Skipped unsupported bundled data file: {candidate.name}")
            continue
        resolved = candidate.resolve()
        if resolved in seen_source_paths:
            continue
        seen_source_paths.add(resolved)
        default_paths.append(candidate)
    if default_paths:
        source_paths.extend(default_paths)
        source_paths = sorted(source_paths)
        _emit(progress,
              f"Bundled starter data enabled: {len(default_paths)} file(s).",
              8)
    source_paths = [
        path
        for path in source_paths
        if path.resolve() not in configured_structured_files
    ]
    _emit(progress,
          f"Found {len(source_paths)} supported files in {config.input_dir}.",
          8)
    cached_count = 0
    processed_count = 0
    partial_count = 0
    skipped_count = 0
    failed_count = 0
    invalid_record_count = 0

    def _read_diagnostics(
        path: Path,
        payload: dict[str, Any],
    ) -> tuple[list[JsonRecordDiagnostics], bool, bool]:
        values = payload.get("record_diagnostics")
        if values:
            if isinstance(values, dict):
                values = [values]
            if not isinstance(values, list) or not all(
                isinstance(value, dict) for value in values
            ):
                return [], False, False
            try:
                diagnostics = [
                    JsonRecordDiagnostics.from_jsonable(value, path)
                    for value in values
                ]
            except (TypeError, ValueError, KeyError):
                return [], False, False
            if any(
                diagnostic.invalid_record_count <= 0
                for diagnostic in diagnostics
            ):
                return [], False, False
            return diagnostics, False, True
        legacy_value = payload.get("record_errors")
        if legacy_value is not None and not isinstance(legacy_value, list):
            return [], False, False
        legacy_messages = list(legacy_value or [])
        if not legacy_messages:
            return [], False, True
        return [
            JsonRecordDiagnostics.from_legacy_messages(path, legacy_messages)
        ], True, True

    def _diagnostics_payload(
        diagnostics: list[JsonRecordDiagnostics],
    ) -> list[dict[str, Any]]:
        return [diagnostic.to_jsonable() for diagnostic in diagnostics]

    def _upsert_manifest(
        manifest_key: str,
        entry: dict[str, Any],
    ) -> None:
        manifest.track_file(manifest_key)
        manifest.upsert(manifest_key, entry, commit=False)

    def _load_cached_payload(
        cache_path: Path,
        source_path: Path,
    ) -> Optional[
        tuple[
            list[dict[str, Any]],
            list[Document],
            list[JsonRecordDiagnostics],
            bool,
        ]
    ]:
        try:
            cached_payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None
        if isinstance(cached_payload, dict):
            cached_items_value = cached_payload.get("documents")
            if not isinstance(cached_items_value, list):
                return None
            cached_items = cached_items_value
            (
                cached_diagnostics,
                migrated_legacy_diagnostics,
                diagnostics_valid,
            ) = _read_diagnostics(source_path, cached_payload)
            if not diagnostics_valid:
                return None
        elif isinstance(cached_payload, list):
            if source_path.suffix.lower() == ".jsonl":
                return None
            cached_items = cached_payload
            cached_diagnostics = []
            migrated_legacy_diagnostics = False
        else:
            return None
        if not cached_items or not all(
            isinstance(item, dict) for item in cached_items
        ):
            return None
        try:
            cached_documents = [
                document_from_dict(item)
                for item in cached_items
            ]
        except (KeyError, TypeError, ValueError, AttributeError):
            return None
        return (
            cached_items,
            cached_documents,
            cached_diagnostics,
            migrated_legacy_diagnostics,
        )

    def _submit_cached(cached_documents: list[Document]) -> None:
        for document in cached_documents:
            corpus_builder.submit(document)

    # First pass: separate files that can be served from the extraction
    # cache (cheap disk read, done inline) from files that need real
    # extraction (CPU-heavy, farmed out to worker processes below).
    pending_extraction: list[Path] = []
    file_digests: dict[str, str] = {}
    file_stats: dict[str, Any] = {}
    for path in source_paths:
        if should_stop and should_stop():
            raise RuntimeError("Dataset preparation stopped by user.")
        stat = path.stat()
        digest = file_fingerprint(path, fast=config.fast_scan_mode,
                                  sample_bytes=config.fast_scan_sample_bytes)
        file_digests[str(path)] = digest
        file_stats[str(path)] = stat
        cache_path = cache_dir / f"{digest}.json"
        manifest_key = str(path.resolve())
        previous = manifest.get(manifest_key) or {}
        can_use_cache = (
                not force_reprocess
                and previous.get("sha256") == digest
                and previous.get("cache_key") == key
                and cache_path.exists()
        )
        if not can_use_cache:
            pending_extraction.append(path)
            continue

        cache_result = _load_cached_payload(cache_path, path)
        if cache_result is None:
            pending_extraction.append(path)
            _emit(
                progress,
                f"Reprocessing {path.name}: extraction cache is invalid.",
            )
            continue
        (
            cached_items,
            cached_documents,
            cached_diagnostics,
            migrated_legacy_diagnostics,
        ) = cache_result
        cached_extraction_reasons = []
        if path.suffix.lower() == ".pdf":
            cached_text = "\n".join(document.text for document in cached_documents)
            cached_extraction_reasons = _bad_extraction_reasons(
                path,
                {
                    "path": str(path),
                    "kind": cached_documents[0].kind if cached_documents else "prose",
                    "language": cached_documents[0].language if cached_documents else "",
                    "characters": str(len(cached_text)),
                    "preview": cached_text[:1200],
                },
                stat.st_size,
            )
        if cached_extraction_reasons:
            skipped_count += 1
            reason_text = "; ".join(cached_extraction_reasons)
            _emit(progress, f"Skipped cached {path.name}: suspicious PDF extraction ({reason_text}).")
            _upsert_manifest(
                manifest_key,
                {
                    "path": str(path), "sha256": digest, "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns, "cache_key": key,
                    "status": "skipped_bad_extraction", "reasons": cached_extraction_reasons,
                },
            )
            continue
        _submit_cached(cached_documents)
        cached_document_count = len(cached_documents)
        del cached_documents
        diagnostic_payload = _diagnostics_payload(cached_diagnostics)
        diagnostic_count = sum(
            item.invalid_record_count for item in cached_diagnostics
        )
        invalid_record_count += diagnostic_count
        if cached_diagnostics:
            partial_count += 1
            _emit(
                progress,
                f"{cached_diagnostics[0].summary()} Preserved "
                f"{cached_document_count:,} valid sample(s) from cache.",
                event_type="dataset_diagnostic",
                level="warning",
                outcome="partial",
                source_path=str(path),
                diagnostic=diagnostic_payload[0],
            )
        else:
            cached_count += 1
            _emit(progress, f"Reused {path.name} from cache.")
        if migrated_legacy_diagnostics:
            cache_path.write_text(
                json.dumps(
                    {
                        "documents": cached_items,
                        "record_diagnostics": diagnostic_payload,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        _upsert_manifest(
            manifest_key,
            {
                "path": str(path), "sha256": digest, "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns, "cache_key": key,
                "cache_file": str(cache_path.relative_to(config.output_dir)),
                "status": "cached_partial" if cached_diagnostics else "cached",
                "invalid_record_count": diagnostic_count,
                "record_diagnostics": diagnostic_payload,
            },
        )

    # Second pass: extract new/changed files in parallel worker processes.
    # ``max_workers`` bounds how many files' full text can be resident (one
    # per in-flight worker) at any moment, regardless of total corpus size.
    # Also capped by CPU count: this is CPU-bound work, and each worker
    # process is a full Python interpreter, so requesting more workers than
    # cores adds contention and (thanks to spawn re-importing this app's
    # dependency chain per worker) startup/memory overhead without a
    # throughput benefit.
    cpu_cap = max(1, os.cpu_count() or 1)
    worker_count = (
        max(1, min(config.max_workers, cpu_cap, len(pending_extraction)))
        if pending_extraction
        else 0
    )
    if pending_extraction:
        _emit(
            progress,
            f"Extracting {len(pending_extraction):,} file(s) with {worker_count} worker process(es)...",
            10,
        )
    if worker_count:
        # build_dataset() itself typically already runs inside a spawned
        # child process (see ui/workers.py's ProcessTaskWorker, used with
        # isolate_process=True). This app loads torch/CUDA and Qt, and
        # forking a process that may already have CUDA initialized is a
        # known source of crashes and hangs -- the app's own worker
        # deliberately uses "spawn" for exactly that reason. This pool must
        # match that choice explicitly rather than rely on the platform
        # default (which is "fork" on Linux).
        mp_context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=mp_context) as executor:
            future_map = {
                executor.submit(
                    extract_documents_worker,
                    path,
                    config.lowercase,
                    config.code_training_mode,
                    config.preserve_indentation,
                    config.include_prose,
                    config.extract_code_blocks,
                ): path
                for path in pending_extraction
            }
            completed = 0
            for future in as_completed(future_map):
                if should_stop and should_stop():
                    # cancel_futures drops any not-yet-started work
                    # immediately; already-running extractions in worker
                    # processes still have to finish their current file
                    # (there is no safe way to interrupt mid-extraction),
                    # matching the previous ThreadPoolExecutor's behavior.
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise RuntimeError("Dataset preparation stopped by user.")
                path = future_map[future]
                completed += 1
                percent = 10 + int(32 * completed / max(len(pending_extraction), 1))
                digest = file_digests[str(path)]
                stat = file_stats[str(path)]
                manifest_key = str(path.resolve())
                cache_path = cache_dir / f"{digest}.json"
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001 - reported to the user
                    failed_count += 1
                    _emit(progress, f"Failed {path.name}: {exc}", percent)
                    _upsert_manifest(
                        manifest_key,
                        {
                            "path": str(path), "sha256": digest, "size": stat.st_size,
                            "mtime_ns": stat.st_mtime_ns, "cache_key": key,
                            "status": "failed", "error": str(exc),
                        },
                    )
                    continue

                if result["error"] is not None:
                    failed_count += 1
                    _emit(progress, f"Failed {path.name}: {result['error']}", percent)
                    _upsert_manifest(
                        manifest_key,
                        {
                            "path": str(path), "sha256": digest, "size": stat.st_size,
                            "mtime_ns": stat.st_mtime_ns, "cache_key": key,
                            "status": "failed", "error": result["error"],
                        },
                    )
                    continue

                if result["bad_extraction_reasons"]:
                    skipped_count += 1
                    reason_text = "; ".join(result["bad_extraction_reasons"])
                    _emit(progress, f"Skipped {path.name}: suspicious PDF extraction ({reason_text}).", percent)
                    _upsert_manifest(
                        manifest_key,
                        {
                            "path": str(path), "sha256": digest, "size": stat.st_size,
                            "mtime_ns": stat.st_mtime_ns, "cache_key": key,
                            "status": "skipped_bad_extraction", "reasons": result["bad_extraction_reasons"],
                        },
                    )
                    continue

                record_diagnostics = [
                    JsonRecordDiagnostics.from_jsonable(value, path)
                    for value in result.get("record_diagnostics") or []
                    if isinstance(value, dict)
                ]
                diagnostic_payload = _diagnostics_payload(record_diagnostics)
                diagnostic_count = sum(
                    item.invalid_record_count for item in record_diagnostics
                )
                invalid_record_count += diagnostic_count

                if not result["documents"]:
                    if record_diagnostics:
                        failed_count += 1
                        _emit(
                            progress,
                            f"Failed {record_diagnostics[0].summary()} "
                            "No valid records remain.",
                            percent,
                            event_type="dataset_diagnostic",
                            level="error",
                            outcome="failed",
                            source_path=str(path),
                            diagnostic=diagnostic_payload[0],
                        )
                        _upsert_manifest(
                            manifest_key,
                            {
                                "path": str(path), "sha256": digest,
                                "size": stat.st_size,
                                "mtime_ns": stat.st_mtime_ns, "cache_key": key,
                                "status": "failed_invalid_records",
                                "invalid_record_count": diagnostic_count,
                                "record_diagnostics": diagnostic_payload,
                            },
                        )
                        continue
                    skipped_count += 1
                    _emit(progress, f"Skipped {path.name}: no readable text found.", percent)
                    _upsert_manifest(
                        manifest_key,
                        {
                            "path": str(path), "sha256": digest, "size": stat.st_size,
                            "mtime_ns": stat.st_mtime_ns, "cache_key": key,
                            "status": "skipped_empty",
                        },
                    )
                    continue

                cache_path.write_text(
                    json.dumps(
                        {
                            "documents": result["documents"],
                            "record_diagnostics": diagnostic_payload,
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                for item in result["documents"]:
                    corpus_builder.submit(document_from_dict(item))
                if record_diagnostics:
                    partial_count += 1
                    _emit(
                        progress,
                        f"{record_diagnostics[0].summary()} Preserved "
                        f"{len(result['documents']):,} valid sample(s).",
                        percent,
                        event_type="dataset_diagnostic",
                        level="warning",
                        outcome="partial",
                        source_path=str(path),
                        diagnostic=diagnostic_payload[0],
                    )
                else:
                    processed_count += 1
                    _emit(progress, f"Processed {path.name}: {len(result['documents'])} sample(s).", percent)
                _upsert_manifest(
                    manifest_key,
                    {
                        "path": str(path), "sha256": digest, "size": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns, "cache_key": key,
                        "cache_file": str(cache_path.relative_to(config.output_dir)),
                        "status": "partial" if record_diagnostics else "processed",
                        "invalid_record_count": diagnostic_count,
                        "record_diagnostics": diagnostic_payload,
                    },
                )

    for local_path, kind, label in local_structured_paths:
        if should_stop and should_stop():
            raise RuntimeError("Dataset preparation stopped by user.")
        local_path = Path(local_path)
        local_root = str(local_path.resolve())
        _emit(progress, f"Loading {label} JSON/JSONL dataset: {local_path}",
              42)
        local_result = load_structured_json_documents_with_diagnostics(
            local_path,
            kind=kind,
            lowercase=config.lowercase,
            on_invalid=lambda _message: None,
        )
        for document in local_result.documents:
            corpus_builder.submit(document)
        local_document_count = len(local_result.documents)
        invalid_files = {
            diagnostic.path.resolve()
            for diagnostic in local_result.diagnostics
        }
        document_files = {
            Path(str(document.path).rsplit("#", 1)[0]).resolve()
            for document in local_result.documents
        }
        partial_files = invalid_files & document_files
        failed_files = invalid_files - document_files
        clean_files = {
            path.resolve() for path in local_result.source_files
        } - invalid_files
        processed_count += len(clean_files)
        partial_count += len(partial_files)
        failed_count += len(failed_files)
        local_invalid_count = sum(
            diagnostic.invalid_record_count
            for diagnostic in local_result.diagnostics
        )
        invalid_record_count += local_invalid_count
        diagnostics_by_file = {
            diagnostic.path.resolve(): diagnostic
            for diagnostic in local_result.diagnostics
        }
        documents_by_file: dict[Path, int] = {}
        for document in local_result.documents:
            document_path = Path(str(document.path).rsplit("#", 1)[0]).resolve()
            documents_by_file[document_path] = (
                documents_by_file.get(document_path, 0) + 1
            )
        for diagnostic in local_result.diagnostics:
            valid_count = documents_by_file.get(diagnostic.path.resolve(), 0)
            suffix = (
                f" Preserved {valid_count:,} valid sample(s)."
                if valid_count
                else " No valid records remain."
            )
            diagnostic_outcome = "partial" if valid_count else "failed"
            _emit(
                progress,
                diagnostic.summary() + suffix,
                42,
                event_type="dataset_diagnostic",
                level="warning" if valid_count else "error",
                outcome=diagnostic_outcome,
                source_path=str(diagnostic.path),
                diagnostic=diagnostic.to_jsonable(),
            )
        for source_file in local_result.source_files:
            resolved_source = source_file.resolve()
            diagnostic = diagnostics_by_file.get(resolved_source)
            valid_count = documents_by_file.get(resolved_source, 0)
            status = (
                "partial"
                if diagnostic is not None and valid_count
                else "failed_invalid_records"
                if diagnostic is not None
                else "processed"
            )
            _upsert_manifest(
                f"local-{kind}://{resolved_source}",
                {
                    "path": str(source_file),
                    "local_dataset_root": local_root,
                    "kind": kind,
                    "sample_count": valid_count,
                    "cache_key": key,
                    "status": status,
                    "invalid_record_count": (
                        diagnostic.invalid_record_count
                        if diagnostic is not None
                        else 0
                    ),
                    "record_diagnostics": (
                        [diagnostic.to_jsonable()]
                        if diagnostic is not None
                        else []
                    ),
                },
            )
        _emit(
            progress,
            f"Loaded {local_document_count:,} valid {kind} sample(s) from "
            f"{local_path.name}; {local_invalid_count:,} invalid record(s).",
            43,
        )

    if config.conversation_datasets:
        allowed_dataset_ids = set(dataset_ids_for_stage(config.dataset_stage))
        skipped_stage_ids = [dataset_id for dataset_id in
                             config.conversation_datasets if
                             dataset_id not in allowed_dataset_ids]
        selected_dataset_ids = [dataset_id for dataset_id in
                                config.conversation_datasets if
                                dataset_id in allowed_dataset_ids]
        if skipped_stage_ids:
            skipped_labels = [
                CONVERSATION_DATASET_PRESETS[item].label
                for item in skipped_stage_ids
                if item in CONVERSATION_DATASET_PRESETS
            ]
            _emit(progress,
                  f"Skipping dataset(s) not recommended for {config.dataset_stage}: {', '.join(skipped_labels)}.")
        if not selected_dataset_ids:
            _emit(progress,
                  f"No online datasets selected for {config.dataset_stage}; continuing with local sources only.")
            config.conversation_datasets = []
            manifest.set_meta("dataset_config", dataclass_to_jsonable(config),
                              commit=False)
            manifest.set_meta("cache_key", key, commit=False)
            manifest.prune_untracked_files()
            manifest.commit()
            return (
                manifest,
                cached_count,
                processed_count,
                partial_count,
                skipped_count,
                failed_count,
                invalid_record_count,
            )
        hf_cache_dir = config.output_dir / "cache" / "huggingface"
        labels = [
            CONVERSATION_DATASET_PRESETS[item].label
            for item in selected_dataset_ids
            if item in CONVERSATION_DATASET_PRESETS
        ]
        _emit(progress,
              f"Online training datasets enabled: {', '.join(labels)}.", 8)
        _emit(progress,
              f"Online training datasets will be cached in: {hf_cache_dir}", 8)
        hf_documents = load_conversation_documents(
            selected_dataset_ids,
            config.conversation_sample_limit,
            hf_cache_dir,
            lowercase=config.lowercase,
            progress=progress,
            should_stop=should_stop,
        )
        for document in hf_documents:
            corpus_builder.submit(document)
        del hf_documents
        config.conversation_datasets = selected_dataset_ids
        for dataset_id in selected_dataset_ids:
            preset = CONVERSATION_DATASET_PRESETS.get(dataset_id)
            _upsert_manifest(
                f"hf://{dataset_id}",
                {
                    "path": f"hf://{dataset_id}",
                    "dataset": preset.hf_path if preset else dataset_id,
                    "config_name": preset.config_name if preset else "",
                    "split": preset.split if preset else "",
                    "sample_limit": config.conversation_sample_limit,
                    "cache_key": key,
                    "status": "processed",
                },
            )
        processed_count += len(selected_dataset_ids)

    manifest.set_meta("dataset_config", dataclass_to_jsonable(config),
                      commit=False)
    manifest.set_meta("cache_key", key, commit=False)
    manifest.prune_untracked_files()
    manifest.commit()
    return (
        manifest,
        cached_count,
        processed_count,
        partial_count,
        skipped_count,
        failed_count,
        invalid_record_count,
    )
