from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Optional

from engine.contracts import TrainingJobSpec


def default_artifact_root() -> Path:
    """Return the default artifact serving root.

    Returns:
        Artifact root path.
    """

    return Path.home() / ".drunkenbot_ide" / "artifacts"


def create_job_artifact_bundle(
    job: TrainingJobSpec,
    artifact_root: Optional[Path] = None,
    base_url: str = "/artifacts",
) -> Path:
    """Create a portable artifact bundle for a training job.

    The bundle contains the prepared dataset under ``dataset/`` and a
    ``job.json`` manifest. Workers extract it into their local workspace and
    rewrite dataset/output paths before training.

    Args:
        job: Training job specification.
        artifact_root: Root folder where bundles are written.
        base_url: URL prefix used by the coordinator artifact route.

    Returns:
        Bundle path.
    """

    root = Path(artifact_root) if artifact_root else default_artifact_root()
    root.mkdir(parents=True, exist_ok=True)
    bundle_path = root / f"{job.job_id}.zip"
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        _write_directory(archive, job.dataset.dataset_dir, "dataset")
        resume_path = job.training.resume_from_checkpoint
        if resume_path and Path(resume_path).is_file():
            archive.write(Path(resume_path), "checkpoints/resume_checkpoint.pt")
            job.metadata["resume_checkpoint_artifact"] = "checkpoints/resume_checkpoint.pt"
        base_path = job.training.fine_tune_from_checkpoint or job.model.base_checkpoint
        if base_path and Path(base_path).is_file():
            archive.write(Path(base_path), "checkpoints/base_checkpoint.pt")
            job.metadata["base_checkpoint_artifact"] = "checkpoints/base_checkpoint.pt"
        archive.writestr("job.json", json.dumps(job.to_jsonable(), indent=2))
    job.metadata["artifact_bundle_url"] = f"{base_url.rstrip('/')}/{bundle_path.name}"
    job.metadata["artifact_bundle_name"] = bundle_path.name
    return bundle_path


def create_result_artifact_bundle(job_id: str, output_dir: Path, bundle_path: Path) -> Path:
    """Create a portable output artifact bundle for a completed job.

    Args:
        job_id: Training job identifier.
        output_dir: Worker-local output directory to bundle.
        bundle_path: Destination zip path.

    Returns:
        Bundle path.
    """

    bundle_path = Path(bundle_path)
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        _write_directory(archive, output_dir, "model")
        archive.writestr("result.json", json.dumps({"job_id": job_id}, indent=2))
    return bundle_path


def _write_directory(archive: zipfile.ZipFile, source: Path, archive_root: str) -> None:
    """Write a directory into a zip archive.

    Args:
        archive: Zip archive.
        source: Source directory.
        archive_root: Archive root folder.

    Raises:
        FileNotFoundError: If the source directory does not exist.
    """

    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"Artifact source not found: {source}")
    for path in source.rglob("*"):
        if path.is_file():
            archive.write(path, Path(archive_root) / path.relative_to(source))

