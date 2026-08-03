from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterator, Optional


# How many upserts to batch into one SQLite transaction/commit. Committing
# after every single file would be safest against a crash mid-scan, but
# each commit costs a real fsync; batching keeps that cost bounded while
# still checkpointing progress regularly during a very large scan (e.g.
# millions of source files) instead of risking the whole scan's progress
# on one final commit.
COMMIT_BATCH_SIZE = 1000


class ManifestStore:
    """SQLite-backed replacement for the old single-JSON-file manifest.

    The previous manifest was one JSON file holding a dict with one entry
    per source file (or per online dataset). For a project with a very
    large number of source files (hundreds of thousands to millions), that
    dict -- and the JSON string built from it -- had to be held entirely in
    memory, which could raise ``MemoryError`` during dataset preparation.

    This store keeps one row per file in a small SQLite database instead,
    with each row's entry stored as its own small JSON blob. Entries for
    local files and for online (``hf://...``) datasets have different
    shapes (different fields), so a generic per-row blob is simpler and
    more flexible than a rigid fixed-column schema, while still keeping
    the one property that actually matters here: every lookup or update
    touches exactly one row, so memory use stays flat no matter how many
    files are tracked, and there is never a large in-memory dict or JSON
    string covering the whole manifest.
    """

    def __init__(self, connection: sqlite3.Connection, db_path: Optional[Path] = None) -> None:
        """Wrap an open SQLite connection.

        Args:
            connection: Open SQLite connection with the manifest schema
                already created.
            db_path: Path to the underlying database file, if known. Used
                by callers that need to copy the raw file (for example, to
                snapshot it into a dataset version folder).
        """

        self._connection = connection
        self._pending_since_commit = 0
        self.db_path = db_path

    @classmethod
    def open(cls, db_path: Path, legacy_json_path: Optional[Path] = None) -> "ManifestStore":
        """Open (creating if needed) a manifest database.

        If ``db_path`` does not exist yet but ``legacy_json_path`` does,
        the old JSON manifest is imported once, so existing projects
        prepared before this change keep their file cache instead of
        starting from scratch.

        Args:
            db_path: SQLite database file path.
            legacy_json_path: Old ``dataset_manifest.json`` path to migrate
                from, if present.

        Returns:
            Opened manifest store.
        """

        db_path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not db_path.exists()
        connection = sqlite3.connect(str(db_path))
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS files (
                manifest_key TEXT PRIMARY KEY,
                entry_json TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT
            )
            """
        )
        connection.commit()
        store = cls(connection, db_path=db_path)
        if is_new and legacy_json_path is not None and legacy_json_path.exists():
            store._migrate_from_json(legacy_json_path)
        return store

    def _migrate_from_json(self, legacy_json_path: Path) -> None:
        """One-time import of an old JSON manifest into this store.

        Reads the legacy file once (the same memory cost the old code
        always had for that one file), writes every entry into SQLite, and
        does not touch the legacy file itself -- it is left in place as a
        harmless leftover the user can delete later.

        Args:
            legacy_json_path: Old ``dataset_manifest.json`` path.
        """

        try:
            legacy = json.loads(legacy_json_path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(legacy, dict):
            return
        files = legacy.get("files", {})
        if isinstance(files, dict):
            for manifest_key, entry in files.items():
                if isinstance(entry, dict):
                    self.upsert(manifest_key, entry, commit=False)
        for meta_key in ("dataset_config", "cache_key", "dataset_id", "dataset_version"):
            if meta_key in legacy:
                self.set_meta(meta_key, legacy[meta_key], commit=False)
        self.commit()

    def get(self, manifest_key: str) -> Optional[dict[str, Any]]:
        """Return one file's manifest entry, or ``None`` if not tracked.

        Args:
            manifest_key: Resolved source path (or ``hf://dataset_id``)
                used as the row's primary key.

        Returns:
            Entry dictionary, or ``None``.
        """

        row = self._connection.execute(
            "SELECT entry_json FROM files WHERE manifest_key = ?",
            (manifest_key,),
        ).fetchone()
        if row is None:
            return None
        try:
            return json.loads(row[0])
        except Exception:
            return None

    def upsert(self, manifest_key: str, entry: dict[str, Any], commit: bool = True) -> None:
        """Insert or update one file's manifest entry.

        Args:
            manifest_key: Resolved source path (or ``hf://dataset_id``).
            entry: Entry fields. Any JSON-serializable shape is accepted --
                local files and online datasets store different fields.
            commit: Whether to commit immediately. Callers doing many
                upserts in a row (e.g. scanning thousands of files) can
                pass ``False`` and let the internal batch-size safety net
                (or an explicit call to ``commit()``) checkpoint instead,
                to amortize the fsync cost of committing.
        """

        self._connection.execute(
            """
            INSERT INTO files (manifest_key, entry_json) VALUES (?, ?)
            ON CONFLICT(manifest_key) DO UPDATE SET entry_json=excluded.entry_json
            """,
            (manifest_key, json.dumps(entry, ensure_ascii=False, default=str)),
        )
        self._pending_since_commit += 1
        if commit:
            self.commit()
        elif self._pending_since_commit >= COMMIT_BATCH_SIZE:
            # Safety net for callers doing many commit=False upserts in a
            # row (a large file scan): checkpoint periodically so a crash
            # partway through doesn't lose the entire scan's progress, even
            # though the caller hasn't explicitly called commit() yet.
            self.commit()

    def iter_files(self) -> Iterator[tuple[str, dict[str, Any]]]:
        """Yield every tracked file's ``(manifest_key, entry)`` pair.

        Rows are streamed directly from SQLite in primary-key order, one at
        a time, rather than being loaded into one big list or dict first --
        this is what makes fingerprinting and export safe for very large
        file counts.

        Yields:
            Pairs of manifest key and entry dictionary.
        """

        cursor = self._connection.execute("SELECT manifest_key, entry_json FROM files ORDER BY manifest_key")
        for manifest_key, entry_json in cursor:
            try:
                entry = json.loads(entry_json)
            except Exception:
                entry = {}
            yield manifest_key, entry

    def count(self) -> int:
        """Return how many files are tracked.

        Returns:
            Row count.
        """

        row = self._connection.execute("SELECT COUNT(*) FROM files").fetchone()
        return int(row[0]) if row else 0

    def get_meta(self, key: str, default: Any = None) -> Any:
        """Return a small top-level metadata value (not per-file).

        Args:
            key: Metadata key, such as ``"dataset_config"`` or
                ``"cache_key"``.
            default: Value returned if the key is not set.

        Returns:
            Parsed JSON value, or ``default``.
        """

        row = self._connection.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        if row is None:
            return default
        try:
            return json.loads(row[0])
        except Exception:
            return default

    def set_meta(self, key: str, value: Any, commit: bool = True) -> None:
        """Set a small top-level metadata value (not per-file).

        Args:
            key: Metadata key.
            value: JSON-serializable value.
            commit: Whether to commit immediately.
        """

        self._connection.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value, ensure_ascii=False, default=str)),
        )
        if commit:
            self.commit()

    def commit(self) -> None:
        """Commit any pending writes."""

        self._connection.commit()
        self._pending_since_commit = 0

    def close(self) -> None:
        """Commit and close the underlying connection."""

        self.commit()
        self._connection.close()

    def __enter__(self) -> "ManifestStore":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()
