from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional

from .conversation_presets import *
from .conversation_loader import *

def _conversation_text_from_row(row: dict[str, Any]) -> tuple[str, str]:
    """Extract tagged conversation/instruction text from a dataset row.

    Args:
        row: Hugging Face row.

    Returns:
        Text and document kind.
    """

    messages = row.get("messages") or row.get("conversation") or row.get("conversations")
    if isinstance(messages, list):
        rendered = _render_message_list(messages)
        if rendered:
            return rendered, "conversation"
    dialogue = row.get("dialogue")
    if isinstance(dialogue, list):
        turns = [f"{'User' if index % 2 == 0 else 'Assistant'}: {value}" for index, value in enumerate(dialogue)]
        return "\n".join(turns), "conversation"
    instruction = str(row.get("instruction") or row.get("prompt") or row.get("question") or "").strip()
    input_text = str(row.get("input") or row.get("context") or row.get("problem") or "").strip()
    output = str(
        row.get("output")
        or row.get("response")
        or row.get("answer")
        or row.get("completion")
        or row.get("solution")
        or row.get("code")
        or ""
    ).strip()
    if instruction and output:
        user = instruction if not input_text else f"{instruction}\n\n{input_text}"
        return f"User: {user}\nAssistant: {output}", "instruction"
    for key in ("text", "story", "content"):
        value = row.get(key)
        if value:
            return str(value), "prose"
    return "", "prose"


def _chat_messages_from_text(text: str, kind: str) -> list[dict[str, str]]:
    """Convert tagged extracted text into the canonical chat message shape."""
    messages: list[dict[str, str]] = []
    for line in text.splitlines():
        if ": " not in line:
            continue
        role, content = line.split(": ", 1)
        role_key = role.strip().lower()
        if role_key in {"system", "user", "assistant"} and content.strip():
            messages.append({"role": role_key, "content": content.strip()})
    if kind == "instruction" and not any(item["role"] == "system" for item in messages):
        messages.insert(0, {"role": "system", "content": "You are a helpful assistant."})
    return messages


def _render_message_list(messages: list[Any]) -> str:
    """Render common message-list schemas into role-prefixed turns."""

    turns: list[str] = []
    for index, message in enumerate(messages):
        if isinstance(message, dict):
            role = str(message.get("role") or message.get("from") or message.get("speaker") or "").strip()
            content = str(message.get("content") or message.get("value") or message.get("text") or "").strip()
        else:
            role = "user" if index % 2 == 0 else "assistant"
            content = str(message).strip()
        if not content:
            continue
        label = "Assistant" if role.lower() in {"assistant", "gpt", "bot"} else "User"
        if role.lower() in {"system"}:
            label = "System"
        turns.append(f"{label}: {content}")
    return "\n".join(turns)


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


def _extract_preset_to_jsonl(
    dataset_id: str,
    sample_limit: int,
    cache_dir: Path,
    output_jsonl: Path,
    lowercase: bool,
) -> None:
    """Extract one Hugging Face preset to JSONL for the parent app.

    Args:
        dataset_id: Preset ID.
        sample_limit: Maximum rows to extract.
        cache_dir: Hugging Face cache directory.
        output_jsonl: JSONL output path.
        lowercase: Whether to lowercase extracted text.
    """

    preset = CONVERSATION_DATASET_PRESETS[dataset_id]
    os.environ.update(_hf_subprocess_environment(cache_dir))
    print(f"Importing datasets package for {preset.label}.", flush=True)
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the datasets package to use Hugging Face conversation datasets.") from exc

    cache_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading/loading {preset.label} into {cache_dir}.", flush=True)
    if preset.config_name:
        dataset = load_dataset(preset.hf_path, preset.config_name, split=preset.split, cache_dir=str(cache_dir))
    else:
        dataset = load_dataset(preset.hf_path, split=preset.split, cache_dir=str(cache_dir))
    row_count = len(dataset) if hasattr(dataset, "__len__") else 0
    print(f"Loaded {preset.hf_path} split {preset.split} with {row_count or 'unknown'} row(s).", flush=True)
    limit = row_count if sample_limit <= 0 or row_count <= 0 else min(sample_limit, row_count)
    if limit and hasattr(dataset, "select"):
        dataset = dataset.select(range(limit))
    loaded = 0
    with output_jsonl.open("w", encoding="utf-8") as file:
        if dataset_id == "dailydialog":
            loaded = _write_daily_dialog_rows(dataset, file, lowercase)
        else:
            for row in dataset:
                text, kind = _conversation_text_from_row(dict(row))
                text = clean_text(text, lowercase=lowercase)
                if not text:
                    continue
                if preset.stage == "code":
                    kind = "code"
                messages = _chat_messages_from_text(text, kind)
                record: dict[str, Any] = {"kind": kind}
                if messages:
                    record["messages"] = messages
                else:
                    record["text"] = text
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
                loaded += 1
                if loaded % 1000 == 0:
                    print(f"{preset.label}: extracted {loaded:,}/{limit:,} sample(s).", flush=True)
    print(f"{preset.label}: wrote {loaded:,} sample(s) to {output_jsonl}.", flush=True)


def _write_daily_dialog_rows(dataset: Any, file: Any, lowercase: bool) -> int:
    """Group DailyDialog utterance rows into dialogue samples.

    Args:
        dataset: Hugging Face dataset rows.
        file: Open JSONL file handle.
        lowercase: Whether to lowercase extracted text.

    Returns:
        Number of written dialogue samples.
    """

    dialogues: dict[str, list[str]] = {}
    order: list[str] = []
    for row in dataset:
        value = dict(row)
        dialog_id = str(value.get("dialog_id", len(order)))
        utterance = str(value.get("utterance") or "").strip()
        if not utterance:
            continue
        if dialog_id not in dialogues:
            dialogues[dialog_id] = []
            order.append(dialog_id)
        dialogues[dialog_id].append(utterance)
    written = 0
    for dialog_id in order:
        turns = dialogues[dialog_id]
        if not turns:
            continue
        text = "\n".join(
            f"{'User' if index % 2 == 0 else 'Assistant'}: {utterance}"
            for index, utterance in enumerate(turns)
        )
        text = clean_text(text, lowercase=lowercase)
        if not text:
            continue
        file.write(json.dumps({"text": text, "kind": "conversation"}, ensure_ascii=False) + "\n")
        written += 1
        if written % 1000 == 0:
            print(f"DailyDialog: extracted {written:,} dialogue sample(s).", flush=True)
    return written


def main() -> None:
    """Run conversation dataset helper commands."""

    parser = argparse.ArgumentParser(description="Micro LLM conversation dataset helper")
    subparsers = parser.add_subparsers(dest="command", required=True)
    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument("--dataset-id", required=True, choices=sorted(CONVERSATION_DATASET_PRESETS))
    extract_parser.add_argument("--sample-limit", type=int, default=20000)
    extract_parser.add_argument("--cache-dir", required=True)
    extract_parser.add_argument("--output-jsonl", required=True)
    extract_parser.add_argument("--config-name", default=None)
    extract_parser.add_argument("--lowercase", action="store_true")
    args = parser.parse_args()
    if args.command == "extract":
        _extract_preset_to_jsonl(
            dataset_id=args.dataset_id,
            sample_limit=args.sample_limit,
            cache_dir=Path(args.cache_dir),
            output_jsonl=Path(args.output_jsonl),
            lowercase=args.lowercase,
        )


if __name__ == "__main__":
    main()

