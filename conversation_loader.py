from __future__ import annotations

import logging
import os
from pathlib import Path
import subprocess
import sys
from threading import Thread
from typing import Any, Callable, Optional

from .data import Document, clean_text
from .conversation_presets import *

LOGGER = logging.getLogger(__name__)

def load_conversation_documents(
    dataset_ids: list[str],
    sample_limit: int,
    cache_dir: Path,
    lowercase: bool = False,
    progress: Optional[Callable[[Any], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> list[Document]:
    """Load selected Hugging Face conversation datasets as training documents.

    Args:
        dataset_ids: Preset IDs to load.
        sample_limit: Maximum rows per dataset. Zero means no limit.
        cache_dir: Hugging Face dataset cache directory.
        lowercase: Whether to lowercase extracted text.
        progress: Optional progress callback.
        should_stop: Optional cancellation callback.

    Returns:
        Conversation/instruction documents.
    """

    if not dataset_ids:
        return []
    documents: list[Document] = []
    cache_dir.mkdir(parents=True, exist_ok=True)
    _emit(progress, f"Hugging Face dataset cache: {cache_dir}")
    LOGGER.info("Hugging Face dataset cache: %s", cache_dir)
    def load_one(preset_index: int, dataset_id: str) -> tuple[str, list[Document]]:
        """Load one preset and return its documents."""

        if should_stop and should_stop():
            raise RuntimeError("Dataset preparation stopped by user.")
        preset = CONVERSATION_DATASET_PRESETS.get(dataset_id)
        if preset is None and dataset_id.startswith("hf_custom:"):
            custom_path = dataset_id.removeprefix("hf_custom:")
            preset = ConversationDatasetPreset(
                dataset_id=dataset_id,
                label=f"Custom Hugging Face dataset ({custom_path})",
                hf_path=custom_path,
                config_name=None,
                split="train",
                stage="base",
                description="User-provided Hugging Face dataset.",
            )
        if preset is None:
            _emit(progress, f"Skipping unknown conversation dataset: {dataset_id}")
            LOGGER.warning("Skipping unknown conversation dataset: %s", dataset_id)
            return dataset_id, []
        _emit(
            progress,
            f"Downloading/loading {preset.label} into {cache_dir}...",
            8 + min(25, preset_index * 3),
        )
        LOGGER.info("Downloading/loading %s into %s", preset.label, cache_dir)
        rows = _load_preset_rows_in_subprocess(preset, sample_limit, cache_dir, lowercase, progress, should_stop)
        total = len(rows)
        loaded = 0
        preset_documents: list[Document] = []
        for row_index, row in enumerate(rows):
            if should_stop and should_stop():
                raise RuntimeError("Dataset preparation stopped by user.")
            kind = str(row.get("kind") or "conversation")
            messages = row.get("messages")
            if isinstance(messages, list):
                text = _render_message_list(messages)
            else:
                text = str(row.get("text") or "")
            if not text:
                continue
            preset_documents.append(
                Document(
                    path=Path("__hf_datasets__") / preset.dataset_id / f"{row_index}.txt",
                    text=text,
                    kind=kind,
                    language=preset.dataset_id,
                )
            )
            loaded += 1
            if loaded % 1000 == 0:
                _emit(progress, f"{preset.label}: loaded {loaded:,}/{total:,} sample(s).")
        _emit(progress, f"{preset.label}: added {loaded:,} sample(s).")
        LOGGER.info("%s added %s sample(s)", preset.label, f"{loaded:,}")
        return dataset_id, preset_documents

    max_workers = min(4, max(1, len(dataset_ids)))
    if len(dataset_ids) > 1:
        _emit(progress, f"Loading {len(dataset_ids)} online dataset(s) in parallel with {max_workers} worker(s).")
        LOGGER.info("Loading %s online dataset(s) in parallel with %s worker(s)", len(dataset_ids), max_workers)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        pending = {
            executor.submit(load_one, preset_index, dataset_id)
            for preset_index, dataset_id in enumerate(dataset_ids, start=1)
        }
        while pending:
            if should_stop and should_stop():
                for future in pending:
                    future.cancel()
                raise RuntimeError("Dataset preparation stopped by user.")
            done, pending = wait(pending, timeout=0.25, return_when=FIRST_COMPLETED)
            for future in done:
                if should_stop and should_stop():
                    for pending_future in pending:
                        pending_future.cancel()
                    raise RuntimeError("Dataset preparation stopped by user.")
                _, preset_documents = future.result()
                documents.extend(preset_documents)
        for future in pending:
            _, preset_documents = future.result()
            documents.extend(preset_documents)
    return documents


def _load_preset_rows_in_subprocess(
    preset: ConversationDatasetPreset,
    sample_limit: int,
    cache_dir: Path,
    lowercase: bool,
    progress: Optional[Callable[[Any], None]],
    should_stop: Optional[Callable[[], bool]],
) -> list[dict[str, str]]:
    """Extract a Hugging Face preset in a child process.

    Args:
        preset: Dataset preset to extract.
        sample_limit: Maximum rows to extract.
        cache_dir: Hugging Face cache directory.
        lowercase: Whether to lowercase text.
        progress: Optional progress callback.
        should_stop: Optional cancellation callback.

    Returns:
        Extracted text rows.
    """

    extract_dir = cache_dir / "_micro_llm_extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)
    output_path = extract_dir / f"{preset.dataset_id}_{max(sample_limit, 0)}_{int(lowercase)}.jsonl"
    if output_path.exists():
        output_path.unlink()
    command = [
        sys.executable,
        "-m",
        "engine.conversation_datasets",
        "extract",
        "--dataset-id",
        preset.dataset_id,
        "--sample-limit",
        str(sample_limit),
        "--cache-dir",
        str(cache_dir),
        "--output-jsonl",
        str(output_path),
    ]
    if preset.config_name:
        command.extend(["--config-name", preset.config_name])
    if lowercase:
        command.append("--lowercase")
    LOGGER.info("Starting Hugging Face extraction subprocess: %s", " ".join(command))
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=_hf_subprocess_environment(cache_dir),
    )
    assert process.stdout is not None

    output_queue: Queue[str] = Queue()

    def read_output() -> None:
        """Read child output without blocking cancellation polling."""

        assert process.stdout is not None
        for line in process.stdout:
            output_queue.put(line)

    reader = Thread(target=read_output, daemon=True)
    reader.start()
    while process.poll() is None:
        while True:
            try:
                line = output_queue.get_nowait()
            except Empty:
                break
            text = line.strip()
            if text:
                LOGGER.info("[hf:%s] %s", preset.dataset_id, text)
                _emit(progress, text)
        if should_stop and should_stop():
            _terminate_process(process, preset.dataset_id)
            raise RuntimeError("Dataset preparation stopped by user.")
        try:
            line = output_queue.get(timeout=0.2)
        except Empty:
            continue
        text = line.strip()
        if text:
            LOGGER.info("[hf:%s] %s", preset.dataset_id, text)
            _emit(progress, text)
    return_code = process.wait()
    reader.join(timeout=1)
    while True:
        try:
            line = output_queue.get_nowait()
        except Empty:
            break
        text = line.strip()
        if text:
            LOGGER.info("[hf:%s] %s", preset.dataset_id, text)
            _emit(progress, text)
    LOGGER.info("Hugging Face extraction subprocess finished for %s with code %s", preset.dataset_id, return_code)
    if return_code != 0:
        raise RuntimeError(
            f"Hugging Face dataset loader exited with code {return_code} while loading {preset.label}. "
            "Check drunkenbot_ide.log and drunkenbot_ide_faults.log."
        )
    if not output_path.exists():
        raise RuntimeError(f"Hugging Face extraction did not create output: {output_path}")
    rows: list[dict[str, str]] = []
    with output_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _terminate_process(process: subprocess.Popen[Any], dataset_id: str) -> None:
    """Terminate a child process and escalate to kill if it stays alive.

    Args:
        process: Running child process.
        dataset_id: Dataset ID used for logging.
    """

    LOGGER.info("Terminating Hugging Face extraction subprocess for %s", dataset_id)
    process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        LOGGER.warning("Killing unresponsive Hugging Face extraction subprocess for %s", dataset_id)
        process.kill()
        process.wait(timeout=3)


def _hf_subprocess_environment(cache_dir: Path) -> dict[str, str]:
    """Build a Hugging Face environment that stays inside the project cache.

    Args:
        cache_dir: Project-local Hugging Face cache directory.

    Returns:
        Environment variables for the extraction subprocess.
    """

    env = os.environ.copy()
    env["HF_HOME"] = str(cache_dir / "hf_home")
    env["HF_HUB_CACHE"] = str(cache_dir / "hub")
    env["HF_DATASETS_CACHE"] = str(cache_dir / "datasets")
    env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    return env


def _emit(progress: Optional[Callable[[Any], None]], message: str, percent: Optional[int] = None) -> None:
    """Emit a progress event if a callback is available."""

    if progress:
        progress({"message": message, "percent": percent})



