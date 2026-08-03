from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from .manifest_store import ManifestStore


def utc_timestamp() -> str:
    """Return a compact UTC timestamp for artifact identifiers.

    Returns:
        Timestamp string.
    """

    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def stable_json_hash(value: Any) -> str:
    """Return a deterministic short hash for JSON-serializable data.

    Uses streaming hashing so extremely large manifests do not need to be
    converted into one huge JSON string first.

    Args:
        value: Any JSON-serializable object.

    Returns:
        Twelve-character SHA-256 prefix.
    """

    digest = hashlib.sha256()

    def _update(obj: Any) -> None:
        if obj is None:
            digest.update(b"null")

        elif isinstance(obj, bool):
            digest.update(b"true" if obj else b"false")

        elif isinstance(obj, (int, float)):
            digest.update(str(obj).encode("utf-8"))

        elif isinstance(obj, str):
            digest.update(obj.encode("utf-8"))

        elif isinstance(obj, dict):
            digest.update(b"{")
            for key in sorted(obj.keys(), key=str):
                digest.update(str(key).encode("utf-8"))
                digest.update(b":")
                _update(obj[key])
                digest.update(b",")
            digest.update(b"}")

        elif isinstance(obj, (list, tuple)):
            digest.update(b"[")
            for item in obj:
                _update(item)
                digest.update(b",")
            digest.update(b"]")

        elif isinstance(obj, set):
            digest.update(b"<set>")
            for item in sorted(obj, key=str):
                _update(item)
                digest.update(b",")

        else:
            digest.update(str(obj).encode("utf-8"))

    _update(value)

    return digest.hexdigest()[:12]


def read_json(path: Path, default: Optional[Any] = None) -> Any:
    """Read JSON from disk.

    Args:
        path: JSON file path.
        default: Value returned when the file does not exist or cannot be read.

    Returns:
        Parsed JSON or default.
    """

    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )


def next_version_number(lineage: dict[str, Any]) -> int:
    """Return the next dataset version number.

    Args:
        lineage: Existing lineage dictionary.

    Returns:
        Next one-based version number.
    """

    versions = lineage.get("versions", [])
    return len(versions) + 1


def ensure_dataset_lineage(output_dir: Path) -> dict[str, Any]:
    """Load or create dataset lineage metadata.

    Args:
        output_dir: Dataset output folder.

    Returns:
        Dataset lineage dictionary.
    """

    lineage_path = output_dir / "dataset_lineage.json"
    lineage = read_json(lineage_path, default=None)
    if isinstance(lineage, dict) and lineage.get("dataset_id"):
        lineage.setdefault("versions", [])
        return lineage
    return {
        "schema": "micro_llm_dataset_lineage",
        "version": 1,
        "dataset_id": f"ds_{uuid4().hex[:12]}",
        "created_at": utc_timestamp(),
        "versions": [],
    }


def record_dataset_version(output_dir: Path, summary: dict[str, Any], manifest_store: ManifestStore) -> dict[str, Any]:
    """Record a new dataset version and snapshot its metadata.

    Args:
        output_dir: Dataset output folder.
        summary: Dataset summary dictionary.
        manifest_store: Open manifest store tracking every source file.

    Returns:
        Version metadata appended to lineage.
    """

    lineage = ensure_dataset_lineage(output_dir)
    version_number = next_version_number(lineage)

    # ------------------------------------------------------------------
    # Build a memory-efficient fingerprint by streaming rows straight from
    # SQLite (already returned in manifest_key order) instead of loading
    # every file's metadata into one big dict or JSON string first. This
    # keeps fingerprinting cheap in RAM no matter how many source files a
    # project has.
    # ------------------------------------------------------------------

    digest = hashlib.sha256()

    for manifest_key, info in manifest_store.iter_files():
        digest.update(manifest_key.encode("utf-8"))
        digest.update(str(info.get("sha256", "")).encode("utf-8"))
        digest.update(str(info.get("size", "")).encode("utf-8"))
        digest.update(str(info.get("mtime_ns", "")).encode("utf-8"))

    digest.update(
        json.dumps(
            summary.get("dataset_config", {}),
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        ).encode("utf-8")
    )

    digest.update(str(summary.get("tokenizer_vocab_size")).encode("utf-8"))
    digest.update(str(summary.get("tokenizer_sha256")).encode("utf-8"))
    digest.update(str(summary.get("tokenizer_strategy")).encode("utf-8"))

    source_fingerprint = digest.hexdigest()[:12]

    # ------------------------------------------------------------------

    version_id = f"v{version_number:03d}_{utc_timestamp()}_{source_fingerprint}"

    version = {
        "version_number": version_number,
        "version_id": version_id,
        "created_at": utc_timestamp(),
        "source_fingerprint": source_fingerprint,
        "document_count": summary.get("document_count"),
        "character_count": summary.get("character_count"),
        "token_count": summary.get("token_count"),
        "tokenizer_vocab_size": summary.get("tokenizer_vocab_size"),
        "tokenizer_sha256": summary.get("tokenizer_sha256"),
        "code_sample_count": summary.get("code_sample_count"),
        "prose_sample_count": summary.get("prose_sample_count"),
        "prepare_mode": summary.get("prepare_mode"),
        "tokenizer_strategy": summary.get("tokenizer_strategy"),
        "summary_path": "dataset_summary.json",
        "manifest_path": "dataset_manifest.sqlite3",
        "snapshot_dir": f"versions/{version_id}",
        "file_count": manifest_store.count(),
    }

    lineage["updated_at"] = utc_timestamp()
    lineage.setdefault("versions", []).append(version)

    summary["dataset_id"] = lineage["dataset_id"]
    summary["dataset_version"] = version

    manifest_store.set_meta("dataset_id", lineage["dataset_id"])
    manifest_store.set_meta("dataset_version", version)

    snapshot_dir = output_dir / "versions" / version_id
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    write_json(snapshot_dir / "dataset_summary.json", summary)
    # The manifest snapshot is a plain file copy of the SQLite database,
    # not a JSON export -- for a very large file count, serializing every
    # row to JSON here would hit exactly the same MemoryError this was
    # built to avoid. A copy is fast regardless of size and needs no
    # in-memory reconstruction of the data at all.
    manifest_store.commit()
    if manifest_store.db_path is not None and manifest_store.db_path.exists():
        shutil.copy2(manifest_store.db_path, snapshot_dir / "dataset_manifest.sqlite3")
    write_json(output_dir / "dataset_lineage.json", lineage)

    return version
