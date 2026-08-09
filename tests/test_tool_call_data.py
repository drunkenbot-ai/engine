"""Tests for OpenAI-style tool-call dataset formatting."""

from __future__ import annotations

from pathlib import Path

from engine.config import DatasetConfig
from engine.data_core import load_structured_json_documents
from engine.dataset_helpers import _local_structured_dataset_paths
from engine.tool_call_data import format_tool_call_record


def test_formats_openai_tool_calls_and_results() -> None:
    """Preserve tool schema, call arguments, IDs, and results."""

    record = {
        "tools": [{"type": "function", "function": {"name": "weather"}}],
        "messages": [
            {"role": "user", "content": "Will it rain?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "weather", "arguments": "{\"city\":\"Pune\"}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "{\"rain\":false}"},
        ],
    }

    text = format_tool_call_record(record)

    assert "<tools>" in text
    assert '"name":"weather"' in text
    assert "<tool_calls>" in text
    assert '"id":"call_1"' in text
    assert '<tool_result id="call_1">' in text


def test_structured_loader_marks_tool_call_documents(tmp_path: Path) -> None:
    """Load tool-call JSONL as a dedicated structured dataset kind."""

    source = tmp_path / "calls.jsonl"
    source.write_text(
        '{"messages":[{"role":"user","content":"Find flights"},'
        '{"role":"assistant","function_call":{"name":"search","arguments":"{}"}}]}\n',
        encoding="utf-8",
    )

    documents = load_structured_json_documents(source, "tool_call")

    assert len(documents) == 1
    assert documents[0].kind == "tool_call"
    assert "<tool_calls>" in documents[0].text


def test_tool_call_paths_are_discovered() -> None:
    """Expose configured tool-call paths to the dataset loader."""

    source = Path("tool_examples.jsonl")
    config = DatasetConfig(input_dir=Path("."), output_dir=Path("out"), tool_call_dataset_paths=[source])

    assert _local_structured_dataset_paths(config) == [(source, "tool_call", "local tool-call")]
