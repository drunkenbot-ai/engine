from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import engine


def test_cli_help_and_train_parser_start_without_app_package(
    tmp_path: Path,
) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(engine.__file__).parent.parent)
    for arguments in (("--help",), ("train", "--help")):
        result = subprocess.run(
            [sys.executable, "-m", "engine.cli", *arguments],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "ModuleNotFoundError" not in result.stderr


def test_standalone_worker_protocol_declares_ui_integration_api() -> None:
    from engine import training_worker_protocol

    expected = {
        "create_worker_request",
        "launch_worker_process",
        "load_run_manifest",
        "manifest_is_stale",
        "write_stop_request",
        "write_worker_request",
    }
    assert expected <= set(training_worker_protocol.__all__)
