"""Download and manage datasets published as GitHub release assets."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import tempfile
import time
import urllib.request
import urllib.error
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional


LOGGER = logging.getLogger(__name__)
DEFAULT_MANIFEST_URL = (
    "https://github.com/drunkenbot-ai/dataset/releases/latest/download/manifest.json"
)


@dataclass(frozen=True)
class DatasetCategory:
    """Metadata for one downloadable dataset category."""

    name: str
    archive: str
    size_bytes: int
    file_count: int
    sha256: str


@dataclass(frozen=True)
class DatasetManifest:
    """Validated metadata published for an external dataset."""

    dataset_id: str
    version: str
    categories: tuple[DatasetCategory, ...]

    @classmethod
    def from_json(cls, payload: object) -> "DatasetManifest":
        """Build a manifest from decoded JSON.

        Args:
            payload: Decoded manifest object.

        Returns:
            Validated dataset manifest.

        Raises:
            ValueError: If required fields are missing or invalid.
        """
        if not isinstance(payload, dict):
            raise ValueError("Dataset manifest must be a JSON object")
        dataset_id = payload.get("dataset_id")
        version = payload.get("version")
        raw_categories = payload.get("categories")
        if not isinstance(dataset_id, str) or not dataset_id.strip():
            raise ValueError("Dataset manifest has no dataset_id")
        if not isinstance(version, str) or not version.strip():
            raise ValueError("Dataset manifest has no version")
        if not isinstance(raw_categories, list):
            raise ValueError("Dataset manifest categories must be a list")

        categories = []
        seen_names: set[str] = set()
        for raw in raw_categories:
            if not isinstance(raw, dict):
                raise ValueError("Dataset category must be an object")
            values = {key: raw.get(key) for key in ("name", "archive", "sha256")}
            if not all(isinstance(value, str) and value.strip() for value in values.values()):
                raise ValueError("Dataset category has invalid name, archive, or sha256")
            name = values["name"]
            if name in seen_names:
                raise ValueError(f"Duplicate dataset category: {name}")
            if Path(values["archive"]).name != values["archive"]:
                raise ValueError(f"Unsafe archive name: {values['archive']}")
            if not isinstance(raw.get("size_bytes"), int) or raw["size_bytes"] < 0:
                raise ValueError(f"Invalid size for dataset category: {name}")
            if not isinstance(raw.get("file_count"), int) or raw["file_count"] < 0:
                raise ValueError(f"Invalid file count for dataset category: {name}")
            if len(values["sha256"]) != 64:
                raise ValueError(f"Invalid SHA-256 for dataset category: {name}")
            categories.append(DatasetCategory(
                name=name,
                archive=values["archive"],
                size_bytes=raw["size_bytes"],
                file_count=raw["file_count"],
                sha256=values["sha256"].lower(),
            ))
            seen_names.add(name)
        return cls(dataset_id=dataset_id, version=version.strip(), categories=tuple(categories))


def parse_version(version: str) -> tuple[int, ...]:
    """Parse a numeric dotted dataset version.

    Args:
        version: Version such as ``"2.0.0"``.

    Returns:
        Numeric version components.

    Raises:
        ValueError: If the version is not numeric and dotted.
    """
    parts = version.strip().split(".")
    if not parts or any(not part.isdigit() for part in parts):
        raise ValueError(f"Invalid dataset version: {version!r}")
    return tuple(int(part) for part in parts)


def is_newer_version(remote: str, local: Optional[str]) -> bool:
    """Return whether a remote version is newer than a local version.

    Args:
        remote: Remote dataset version.
        local: Installed dataset version, or ``None``.

    Returns:
        True when the remote version is greater than the local version.
    """
    return local is None or parse_version(remote) > parse_version(local)


def _download(url: str, destination: Path, progress: Optional[Callable[[int, int], None]]) -> None:
    """Download a URL to a file while reporting byte progress."""
    request = urllib.request.Request(url, headers={"User-Agent": "DrunkenBot-LLM-IDE"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=45) as response, destination.open("wb") as output:
                total = int(response.headers.get("Content-Length") or 0)
                downloaded = 0
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    output.write(block)
                    downloaded += len(block)
                    if progress:
                        progress(downloaded, total)
            return
        except (TimeoutError, OSError, urllib.error.URLError):
            if attempt == 2:
                raise
            LOGGER.warning("Download attempt %d failed for %s; retrying", attempt + 1, url)
            time.sleep(2 ** attempt)


def _sha256_file(path: Path) -> str:
    """Calculate a file checksum without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _extract_archive(archive: Path, destination: Path) -> None:
    """Extract an archive while rejecting paths outside the staging folder."""
    with zipfile.ZipFile(archive) as package:
        root = destination.resolve()
        for member in package.infolist():
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise OSError(f"Unsafe path in dataset archive: {member.filename}")
        package.extractall(destination)


def load_manifest(url: str = DEFAULT_MANIFEST_URL) -> DatasetManifest:
    """Download and validate the current remote dataset manifest.

    Args:
        url: Manifest URL.

    Returns:
        Validated remote manifest.
    """
    with urllib.request.urlopen(
        urllib.request.Request(url, headers={"User-Agent": "DrunkenBot-LLM-IDE"}),
        timeout=60,
    ) as response:
        payload = json.loads(response.read().decode("utf-8"))
    manifest = DatasetManifest.from_json(payload)
    LOGGER.info("Loaded dataset manifest %s version %s", manifest.dataset_id, manifest.version)
    return manifest


def install_categories(
    manifest: DatasetManifest,
    destination: Path,
    categories: Optional[list[str]] = None,
    progress: Optional[Callable[[int, int], None]] = None,
    manifest_url: str = DEFAULT_MANIFEST_URL,
) -> Path:
    """Download, verify, and atomically install selected dataset categories.

    Args:
        manifest: Validated dataset manifest to install.
        destination: Managed dataset root.
        categories: Category names to install, or all categories when omitted.
        progress: Optional callback receiving downloaded and total bytes.
        manifest_url: URL used to resolve release asset URLs.

    Returns:
        Destination dataset root.

    Raises:
        ValueError: If a requested category is not in the manifest.
        OSError: If download, extraction, or replacement fails.
        zipfile.BadZipFile: If an archive is invalid.
    """
    selected_names = set(categories) if categories is not None else {
        category.name for category in manifest.categories if category.file_count > 0
    }
    by_name = {category.name: category for category in manifest.categories}
    unknown = selected_names - by_name.keys()
    if unknown:
        raise ValueError(f"Unknown dataset categories: {', '.join(sorted(unknown))}")
    release_root = manifest_url.rsplit("/", 1)[0]
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="dataset-install-", dir=destination.parent) as temp_name:
        staging = Path(temp_name) / "dataset"
        staging.mkdir()
        if destination.is_dir():
            shutil.copytree(destination, staging, dirs_exist_ok=True)
        ordered_names = sorted(selected_names)
        for category_index, name in enumerate(ordered_names, start=1):
            category = by_name[name]
            if _category_is_installed(destination, category.name, manifest.version):
                LOGGER.info("Skipping already-installed dataset category %s", name)
                continue
            archive = staging / category.archive
            LOGGER.info("Downloading dataset category %s", name)
            archive_progress = (
                (lambda downloaded, total: progress({
                    "message": f"Downloaded {category.name}: {downloaded} bytes",
                    "button_text": f"Downloading {category.archive} dataset ({category_index} of {len(ordered_names)})",
                    "percent": int(downloaded / total * 100) if total else 0,
                }))
                if progress
                else None
            )
            _download(f"{release_root}/{category.archive}", archive, archive_progress)
            if archive.stat().st_size != category.size_bytes:
                raise OSError(f"Size mismatch for dataset category {name}")
            digest = _sha256_file(archive)
            if digest != category.sha256:
                raise OSError(f"SHA-256 mismatch for dataset category {name}")
            _extract_archive(archive, staging)
            archive.unlink()

        (staging / "version.txt").write_text(manifest.version + "\n", encoding="utf-8")
        (staging / "manifest.json").write_text(
            json.dumps({
                "dataset_id": manifest.dataset_id,
                "version": manifest.version,
                "categories": [category.__dict__ for category in manifest.categories],
            }, indent=2) + "\n",
            encoding="utf-8",
        )
        backup = destination.with_name(destination.name + ".previous")
        if backup.exists():
            shutil.rmtree(backup)
        if destination.exists():
            destination.replace(backup)
        staging.replace(destination)
        if backup.exists():
            shutil.rmtree(backup)
    LOGGER.info("Installed dataset version %s at %s", manifest.version, destination)
    return destination


def _category_is_installed(destination: Path, category_name: str, version: str) -> bool:
    """Return whether a category is already installed at the requested version.

    Args:
        destination: Managed dataset root.
        category_name: Category directory name.
        version: Manifest version required by the caller.

    Returns:
        True when the category directory exists and the installed version matches.
    """
    category_path = destination / category_name
    version_path = destination / "version.txt"
    if not category_path.is_dir() or not version_path.is_file():
        return False
    return version_path.read_text(encoding="utf-8").strip() == version


def download_latest_dataset(
    destination: Path,
    categories: Optional[list[str]] = None,
    progress: Optional[Callable[[int, int], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    manifest_url: str = DEFAULT_MANIFEST_URL,
) -> DatasetManifest:
    """Download and install all categories from the latest release.

    Args:
        destination: Managed dataset root.
        progress: Optional callback receiving downloaded and total bytes.
        should_stop: Optional cooperative cancellation callback.

    Returns:
        The installed release manifest.
    """
    if should_stop and should_stop():
        raise RuntimeError("Dataset download stopped by user.")
    manifest = load_manifest(manifest_url)
    install_categories(
        manifest,
        destination,
        categories=categories,
        progress=progress,
        manifest_url=manifest_url,
    )
    return manifest

