from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import engine.notifier as notifier_module
from engine.notifier import NotificationManager


def _enabled_config(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "progress_interval_seconds": 60,
                "telegram": {
                    "enabled": True,
                    "bot_token": "test-token",
                    "chat_id": "test-chat",
                },
                "email": {"enabled": False},
            }
        ),
        encoding="utf-8",
    )


def test_config_is_cached_between_explicit_or_mtime_refreshes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "notifier.json"
    _enabled_config(config_path)
    original_load = notifier_module.load_notifier_config
    loads = []

    def counted_load(path: Path):
        loads.append(path)
        return original_load(path)

    monkeypatch.setattr(notifier_module, "load_notifier_config", counted_load)
    manager = NotificationManager(
        config_path,
        config_check_interval_seconds=3600.0,
        minimum_progress_interval_seconds=0.0,
    )
    delivered = threading.Event()
    manager._send_progress = lambda *_args: delivered.set()
    for index in range(20):
        manager.notify_progress("training", "Training", [str(index)], index)
    assert delivered.wait(1.0)
    manager.close()

    assert len(loads) == 1


def test_progress_is_coalesced_and_terminal_delivery_is_not_dropped(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "notifier.json"
    _enabled_config(config_path)
    manager = NotificationManager(
        config_path,
        max_pending=3,
        config_check_interval_seconds=3600.0,
        minimum_progress_interval_seconds=0.0,
    )
    manager.config.progress_interval_seconds = 0.05
    first_started = threading.Event()
    release_first = threading.Event()
    delivered_progress: list[str] = []
    delivered_terminal: list[str] = []

    def send_progress(_stage: str, _subject: str, text: str) -> None:
        delivered_progress.append(text)
        if len(delivered_progress) == 1:
            first_started.set()
            release_first.wait(1.0)

    manager._send_progress = send_progress
    manager._send_completion = (
        lambda _stage, _subject, text: delivered_terminal.append(text)
    )
    manager.notify_progress("training", "Training", ["first"])
    assert first_started.wait(1.0)
    for index in range(20):
        manager.notify_progress("training", "Training", [f"latest-{index}"])
        manager.notify_progress(f"stage-{index}", "Other", [str(index)])
        assert manager.pending_count <= 3
    manager.notify_complete("training", "Training complete", ["done"])
    assert manager.pending_count <= 3
    release_first.set()
    manager.close(timeout=2.0)

    assert delivered_terminal
    assert "done" in delivered_terminal[-1]
    assert all("latest-0" not in text for text in delivered_progress[1:])


def test_config_refreshes_when_mtime_changes(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "notifier.json"
    _enabled_config(config_path)
    original_load = notifier_module.load_notifier_config
    loads = []

    def counted_load(path: Path):
        loads.append(path)
        return original_load(path)

    monkeypatch.setattr(notifier_module, "load_notifier_config", counted_load)
    manager = NotificationManager(
        config_path,
        config_check_interval_seconds=0.0,
        minimum_progress_interval_seconds=0.0,
    )
    manager._send_progress = lambda *_args: None
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["progress_interval_seconds"] = 120
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    stat = config_path.stat()
    os.utime(config_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))

    manager.notify_progress("training", "Training", ["updated"])
    manager.close()

    assert len(loads) == 2
    assert manager.config.progress_interval_seconds == 120
