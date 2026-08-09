"""Format OpenAI-style tool-call records for language-model training."""

from __future__ import annotations

import json
from typing import Any


def format_tool_call_record(record: Any) -> str:
    """Render a structured tool-call example as a deterministic transcript.

    The input supports the OpenAI Chat Completions shape: top-level ``tools``
    plus ``messages`` whose assistant messages contain ``tool_calls`` or the
    legacy ``function_call`` field. Tool result messages use the ``tool`` or
    ``function`` role. Records without tool-call content return an empty
    string so existing instruction and conversation formatting is unchanged.

    Args:
        record: Decoded JSON training example.

    Returns:
        Tagged transcript suitable for tokenizer training, or an empty string.
    """

    if not isinstance(record, dict):
        return ""
    messages = record.get("messages", record.get("conversations"))
    has_tools = bool(record.get("tools") or record.get("functions"))
    if not isinstance(messages, list):
        return _format_prompt_completion_call(record) if has_tools else ""
    rendered_messages = [_format_message(message) for message in messages]
    rendered_messages = [message for message in rendered_messages if message]
    has_calls = any(_is_tool_message(message) for message in messages if isinstance(message, dict))
    if not has_tools and not has_calls:
        return ""
    lines = _format_tools(record.get("tools", record.get("functions", [])))
    lines.extend(rendered_messages)
    return "\n".join(lines).strip()


def _format_prompt_completion_call(record: dict[str, Any]) -> str:
    """Render a compact prompt/function-call record when messages are absent.

    Args:
        record: Decoded JSON training example.

    Returns:
        Tagged training transcript, or an empty string.
    """

    prompt = _text(record.get("prompt", record.get("input", record.get("instruction", ""))))
    call = record.get("tool_call", record.get("function_call"))
    if not prompt or not isinstance(call, dict):
        return ""
    lines = _format_tools(record.get("tools", record.get("functions", [])))
    lines.append(f"User: {prompt}")
    lines.append(_tool_calls_block([call]))
    result = record.get("tool_result", record.get("function_result"))
    if result is not None:
        lines.append(_tool_result_block(result, record.get("tool_call_id", "")))
    return "\n".join(lines)


def _format_tools(tools: Any) -> list[str]:
    """Render tool definitions when the record provides them.

    Args:
        tools: Tool or legacy function definition list.

    Returns:
        Lines containing a tagged, compact JSON definition block.
    """

    if not isinstance(tools, list) or not tools:
        return []
    return ["Tools:", "<tools>", _json(tools), "</tools>"]


def _format_message(message: Any) -> str:
    """Render one chat message, including calls and tool results.

    Args:
        message: One decoded structured message.

    Returns:
        Tagged message text, or an empty string.
    """

    if not isinstance(message, dict):
        return _text(message)
    role = str(message.get("role", message.get("from", "message"))).lower()
    if role in {"tool", "function"}:
        return _tool_result_block(message.get("content", message.get("result", "")), message.get("tool_call_id", message.get("name", "")))
    content = _text(message.get("content", message.get("value", message.get("text", ""))))
    label = {"user": "User", "assistant": "Assistant", "system": "System", "developer": "Developer"}.get(role, role.title())
    lines = [f"{label}: {content}".rstrip()] if content else []
    calls = message.get("tool_calls")
    if not isinstance(calls, list) and isinstance(message.get("function_call"), dict):
        calls = [message["function_call"]]
    if isinstance(calls, list) and calls:
        lines.append(_tool_calls_block(calls))
    return "\n".join(lines)


def _tool_calls_block(calls: list[Any]) -> str:
    """Render a tool-call payload block.

    Args:
        calls: Tool-call dictionaries.

    Returns:
        Tagged JSON call block.
    """

    normalized = [_normalize_call(call) for call in calls if isinstance(call, dict)]
    return "\n".join(["<tool_calls>", _json(normalized), "</tool_calls>"])


def _normalize_call(call: dict[str, Any]) -> dict[str, Any]:
    """Normalize legacy and current call shapes without losing arguments.

    Args:
        call: Tool-call dictionary.

    Returns:
        JSON-safe normalized call dictionary.
    """

    function = call.get("function") if isinstance(call.get("function"), dict) else call
    return {
        "id": _text(call.get("id", call.get("tool_call_id", ""))),
        "type": _text(call.get("type", "function")) or "function",
        "function": {
            "name": _text(function.get("name", "")),
            "arguments": _json_text(function.get("arguments", function.get("parameters", {}))),
        },
    }


def _tool_result_block(result: Any, call_id: Any) -> str:
    """Render one tool execution result.

    Args:
        result: Tool output value.
        call_id: Associated tool-call ID or tool name.

    Returns:
        Tagged tool result block.
    """

    identifier = _text(call_id)
    heading = f"<tool_result id=\"{identifier}\">" if identifier else "<tool_result>"
    return "\n".join([heading, _json_text(result), "</tool_result>"])


def _is_tool_message(message: dict[str, Any]) -> bool:
    """Return whether a message contains calls or a tool result.

    Args:
        message: Decoded chat message.

    Returns:
        Whether the message is tool-call related.
    """

    return bool(
        message.get("tool_calls")
        or message.get("function_call")
        or str(message.get("role", "")).lower() in {"tool", "function"}
    )


def _json_text(value: Any) -> str:
    """Convert a JSON-compatible value to compact text.

    Args:
        value: Value to render.

    Returns:
        Existing string text or compact JSON.
    """

    return value.strip() if isinstance(value, str) else _json(value)


def _json(value: Any) -> str:
    """Serialize values deterministically for stable dataset artifacts.

    Args:
        value: Value to serialize.

    Returns:
        Compact JSON text.
    """

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _text(value: Any) -> str:
    """Return a stripped text representation.

    Args:
        value: Value to convert.

    Returns:
        Stripped text.
    """

    return str(value or "").strip()
