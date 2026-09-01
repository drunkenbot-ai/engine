from __future__ import annotations

import json
from concurrent.futures import Future
from pathlib import Path

from engine.config import DatasetConfig
from engine.data import JsonRecordDiagnostics
from engine.dataset_build import build_dataset
from engine.dataset_loader import _load_documents_with_cache
from engine.manifest_store import ManifestStore


class _Corpus:
    def __init__(self) -> None:
        self.documents = []

    def submit(self, document) -> None:
        self.documents.append(document)


class _InlineExecutor:
    def __init__(self, **_kwargs) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        pass

    def submit(self, function, *args) -> Future:
        future = Future()
        future.set_result(function(*args))
        return future


def _load(tmp_path: Path, source_text: str, monkeypatch):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir(exist_ok=True)
    source = input_dir / "records.jsonl"
    source.write_text(source_text, encoding="utf-8")
    monkeypatch.setattr(
        "engine.dataset_loader.ProcessPoolExecutor",
        _InlineExecutor,
    )
    corpus = _Corpus()
    events = []
    (
        manifest,
        cached,
        processed,
        partial,
        skipped,
        failed,
        invalid_records,
    ) = _load_documents_with_cache(
        DatasetConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            max_workers=1,
        ),
        corpus,
        progress=events.append,
        should_stop=None,
    )
    entry = manifest.get(str(source.resolve()))
    manifest.close()
    return (
        source,
        output_dir,
        corpus,
        entry,
        (cached, processed, partial, skipped, failed, invalid_records),
        events,
    )


def test_mixed_jsonl_preserves_good_records_and_marks_file_partial(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source, _output, corpus, entry, counts, events = _load(
        tmp_path,
        '{"instruction":"Keep me","output":"Valid"}\n'
        '{"instruction":\n'
        '{}\n',
        monkeypatch,
    )

    assert len(corpus.documents) == 1
    assert counts == (0, 0, 1, 0, 0, 2)
    assert entry["status"] == "partial"
    assert entry["invalid_record_count"] == 2
    diagnostic = entry["record_diagnostics"][0]
    assert diagnostic["filename"] == source.name
    assert diagnostic["location_ranges"] == [{"start": 2, "end": 3}]
    assert diagnostic["omitted_location_count"] == 0
    assert sum("invalid records" in event["message"] for event in events) == 1
    diagnostic_event = next(
        event
        for event in events
        if event.get("event_type") == "dataset_diagnostic"
    )
    assert diagnostic_event["outcome"] == "partial"
    assert diagnostic_event["level"] == "warning"
    assert diagnostic_event["diagnostic"]["invalid_record_count"] == 2

    _, _, cached_corpus, cached_entry, cached_counts, cached_events = _load(
        tmp_path,
        source.read_text(encoding="utf-8"),
        monkeypatch,
    )
    assert len(cached_corpus.documents) == 1
    assert cached_counts == (0, 0, 1, 0, 0, 2)
    assert cached_entry["status"] == "cached_partial"
    assert sum(
        "invalid records" in event["message"] for event in cached_events
    ) == 1


def test_wholly_invalid_jsonl_is_failed_not_skipped_or_processed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source, _output, corpus, entry, counts, events = _load(
        tmp_path,
        '{"instruction":\n{}\n',
        monkeypatch,
    )

    assert corpus.documents == []
    assert counts == (0, 0, 0, 0, 1, 2)
    assert entry["status"] == "failed_invalid_records"
    assert entry["invalid_record_count"] == 2
    assert entry["record_diagnostics"][0]["filename"] == source.name
    diagnostic_events = [
        event for event in events if "invalid records" in event["message"]
    ]
    assert len(diagnostic_events) == 1
    assert "No valid records remain" in diagnostic_events[0]["message"]
    assert diagnostic_events[0]["outcome"] == "failed"
    assert diagnostic_events[0]["level"] == "error"


def test_large_invalid_run_is_compacted_and_manifest_stays_bounded(
    tmp_path: Path,
    monkeypatch,
) -> None:
    invalid_count = 30_000
    source_text = '{"text":"Keep this independent record."}\n' + (
        "{}\n" * invalid_count
    )
    source, output, corpus, entry, counts, events = _load(
        tmp_path,
        source_text,
        monkeypatch,
    )

    assert len(corpus.documents) == 1
    assert counts == (0, 0, 1, 0, 0, invalid_count)
    assert entry["invalid_record_count"] == invalid_count
    diagnostic = entry["record_diagnostics"][0]
    assert diagnostic["location_ranges"] == [
        {"start": 2, "end": invalid_count + 1}
    ]
    assert f"lines 2-{invalid_count + 1:,}".replace(",", "") in (
        diagnostic["summary"].replace(",", "")
    )
    assert source.name in diagnostic["summary"]
    assert len(json.dumps(entry["record_diagnostics"])) < 2_000
    cache_file = output / entry["cache_file"]
    assert cache_file.stat().st_size < 5_000
    assert sum("invalid records" in event["message"] for event in events) == 1


def test_scattered_location_preview_is_bounded_with_omitted_count() -> None:
    diagnostics = JsonRecordDiagnostics(Path("educational_instructions.jsonl"))
    for line_number in range(1, 60_000, 2):
        diagnostics.add(
            line_number,
            "missing instruction/input/output or text fields",
        )

    payload = diagnostics.to_jsonable()
    assert payload["invalid_record_count"] == 30_000
    assert len(payload["location_ranges"]) == 12
    assert payload["omitted_location_count"] == 29_988
    assert "29,988 more" in payload["summary"]
    assert "educational_instructions.jsonl" in payload["summary"]
    assert len(payload["summary"]) < 500


def test_malformed_cached_diagnostics_force_safe_reprocessing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_text = '{"text":"Keep me."}\n{}\n'
    _source, output, _corpus, entry, _counts, _events = _load(
        tmp_path,
        source_text,
        monkeypatch,
    )
    cache_file = output / entry["cache_file"]
    payload = json.loads(cache_file.read_text(encoding="utf-8"))
    payload["record_diagnostics"] = {
        "invalid_record_count": "not-an-integer"
    }
    cache_file.write_text(json.dumps(payload), encoding="utf-8")

    _source, _output, corpus, entry, counts, events = _load(
        tmp_path,
        source_text,
        monkeypatch,
    )

    assert len(corpus.documents) == 1
    assert counts == (0, 0, 1, 0, 0, 1)
    assert entry["status"] == "partial"
    assert any(
        "extraction cache is invalid" in event["message"]
        for event in events
    )

    payload = json.loads(cache_file.read_text(encoding="utf-8"))
    cache_file.write_text(
        json.dumps(payload["documents"]),
        encoding="utf-8",
    )
    _source, _output, corpus, entry, counts, events = _load(
        tmp_path,
        source_text,
        monkeypatch,
    )
    assert len(corpus.documents) == 1
    assert counts == (0, 0, 1, 0, 0, 1)
    assert entry["status"] == "partial"
    assert any(
        "extraction cache is invalid" in event["message"]
        for event in events
    )

    cache_file.write_text("{", encoding="utf-8")
    _source, _output, corpus, entry, counts, events = _load(
        tmp_path,
        source_text,
        monkeypatch,
    )
    assert len(corpus.documents) == 1
    assert counts == (0, 0, 1, 0, 0, 1)
    assert entry["status"] == "partial"
    assert any(
        "extraction cache is invalid" in event["message"]
        for event in events
    )


def test_local_structured_directory_records_one_manifest_row_per_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    local_dir = tmp_path / "instructions"
    output_dir = tmp_path / "output"
    local_dir.mkdir()
    partial_source = local_dir / "partial.jsonl"
    failed_source = local_dir / "failed.jsonl"
    partial_source.write_text(
        '{"instruction":"Keep","output":"this."}\n{}\n',
        encoding="utf-8",
    )
    failed_source.write_text("{}\n", encoding="utf-8")
    corpus = _Corpus()
    events = []

    (
        manifest,
        cached,
        processed,
        partial,
        skipped,
        failed,
        invalid_records,
    ) = _load_documents_with_cache(
        DatasetConfig(
            input_dir=local_dir,
            output_dir=output_dir,
            instruction_dataset_path=local_dir,
            max_workers=1,
        ),
        corpus,
        progress=events.append,
        should_stop=None,
    )
    entries = dict(manifest.iter_files())
    manifest.close()

    assert len(corpus.documents) == 1
    assert (cached, processed, partial, skipped, failed, invalid_records) == (
        0,
        0,
        1,
        0,
        1,
        2,
    )
    partial_entry = entries[f"local-instruction://{partial_source.resolve()}"]
    failed_entry = entries[f"local-instruction://{failed_source.resolve()}"]
    assert partial_entry["status"] == "partial"
    assert failed_entry["status"] == "failed_invalid_records"
    assert len(partial_entry["record_diagnostics"]) == 1
    assert len(failed_entry["record_diagnostics"]) == 1
    assert sum(
        event.get("event_type") == "dataset_diagnostic"
        for event in events
    ) == 2


def test_manifest_prunes_sources_missing_from_current_build(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    obsolete = input_dir / "obsolete.txt"
    obsolete.write_text("This source will be removed.", encoding="utf-8")
    source_text = '{"text":"Current source."}\n{}\n'
    _source, output, _corpus, _entry, _counts, _events = _load(
        tmp_path,
        source_text,
        monkeypatch,
    )
    first_manifest = ManifestStore.open(output / "dataset_manifest.sqlite3")
    assert first_manifest.count() == 2
    first_manifest.close()

    obsolete.unlink()
    _load(tmp_path, source_text, monkeypatch)

    current_manifest = ManifestStore.open(output / "dataset_manifest.sqlite3")
    assert current_manifest.count() == 1
    current_manifest.close()


def test_partial_build_emits_one_terminal_completed_with_warnings_event(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    diverse_text = " ".join(f"unique-token-{index}" for index in range(500))
    (input_dir / "records.jsonl").write_text(
        json.dumps({"text": diverse_text}) + "\n" + "{}\n" * 100,
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "engine.dataset_loader.ProcessPoolExecutor",
        _InlineExecutor,
    )
    events = []

    result = build_dataset(
        DatasetConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            vocab_size=256,
            min_frequency=1,
            context_length=8,
            validation_split=0.2,
            max_workers=1,
        ),
        progress=events.append,
    )

    completion_events = [
        event for event in events if event.get("event_type") == "completion"
    ]
    assert len(completion_events) == 1
    assert completion_events[0] == {
        "message": f"Dataset ready: {output_dir}",
        "percent": 100,
        "event_type": "completion",
        "outcome": "completed_with_warnings",
        "partial_file_count": 1,
        "failed_file_count": 0,
        "invalid_record_count": 100,
    }
    assert result.partial_file_count == 1
    assert result.failed_file_count == 0
    assert result.invalid_record_count == 100
    assert result.preparation_outcome == "completed_with_warnings"
    summary = json.loads(
        (output_dir / "dataset_summary.json").read_text(encoding="utf-8")
    )
    assert summary["preparation_outcome"] == "completed_with_warnings"
    assert summary["partial_file_count"] == 1
    assert summary["invalid_record_count"] == 100
