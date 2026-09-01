from __future__ import annotations

from concurrent.futures import Future
from pathlib import Path

from engine.config import DatasetConfig
from engine.dataset_loader import _load_documents_with_cache


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
    manifest, cached, processed, skipped, failed = _load_documents_with_cache(
        DatasetConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            max_workers=1,
        ),
        corpus,
        progress=None,
        should_stop=None,
    )
    entry = manifest.get(str(source.resolve()))
    manifest.close()
    return source, corpus, entry, (cached, processed, skipped, failed)


def test_mixed_jsonl_preserves_good_records_and_marks_file_partial(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source, corpus, entry, counts = _load(
        tmp_path,
        '{"instruction":"Keep me","output":"Valid"}\n'
        '{"instruction":\n'
        '{}\n',
        monkeypatch,
    )

    assert len(corpus.documents) == 1
    assert counts == (0, 1, 0, 1)
    assert entry["status"] == "partial"
    assert entry["record_error_count"] == 2
    assert any(f"{source.name}, line 2" in error for error in entry["record_errors"])
    assert any(f"{source.name}, line 3" in error for error in entry["record_errors"])

    _, cached_corpus, cached_entry, cached_counts = _load(
        tmp_path,
        source.read_text(encoding="utf-8"),
        monkeypatch,
    )
    assert len(cached_corpus.documents) == 1
    assert cached_counts == (1, 0, 0, 1)
    assert cached_entry["status"] == "cached_partial"


def test_wholly_invalid_jsonl_is_failed_not_skipped_or_processed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source, corpus, entry, counts = _load(
        tmp_path,
        '{"instruction":\n{}\n',
        monkeypatch,
    )

    assert corpus.documents == []
    assert counts == (0, 0, 0, 1)
    assert entry["status"] == "failed_invalid_records"
    assert entry["record_error_count"] == 2
    assert all(source.name in error for error in entry["record_errors"])
