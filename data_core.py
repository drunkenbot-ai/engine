from __future__ import annotations

import json
import re
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import PyPDF2


class OperationCancelled(RuntimeError):
    """Raised when a long-running operation is cancelled by the user."""


SUPPORTED_TEXT_SUFFIXES = {".txt", ".md", ".text"}
SUPPORTED_CODE_SUFFIXES = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".go": "go",
    ".rs": "rust",
    ".php": "php",
    ".rb": "ruby",
    ".swift": "swift",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".scala": "scala",
    ".r": "r",
    ".sql": "sql",
    ".sh": "bash",
    ".ps1": "powershell",
    ".html": "html",
    ".css": "css",
    ".xml": "xml",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".ini": "ini",
}


@dataclass
class Document:
    """Loaded training sample.

    Attributes:
        path: Original source path.
        text: Loaded or extracted sample text.
        kind: Sample type, usually ``prose`` or ``code``.
        language: Optional programming language label for code samples.
    """

    path: Path
    text: str
    kind: str = "prose"
    language: Optional[str] = None


def document_to_dict(document: Document) -> dict[str, Any]:
    """Convert a document to a JSON-friendly dictionary.

    Args:
        document: Document to serialize.

    Returns:
        JSON-friendly document dictionary.
    """

    return {
        "path": str(document.path),
        "text": document.text,
        "kind": document.kind,
        "language": document.language,
    }


def document_from_dict(value: dict[str, Any]) -> Document:
    """Load a document from a dictionary.

    Args:
        value: Serialized document.

    Returns:
        Document instance.
    """

    return Document(
        path=Path(value["path"]),
        text=str(value.get("text", "")),
        kind=str(value.get("kind", "prose")),
        language=value.get("language"),
    )


def file_sha256(path: Path) -> str:
    """Calculate a file SHA-256 digest.

    Args:
        path: File path.

    Returns:
        Hex digest.
    """

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fingerprint(path: Path, fast: bool = False, sample_bytes: int = 64 * 1024) -> str:
    """Calculate a file fingerprint.

    Args:
        path: File path.
        fast: When true, hash only sampled bytes and size metadata.
        sample_bytes: Bytes read from file head/tail in fast mode.

    Returns:
        Fingerprint hex digest.
    """

    if not fast:
        return file_sha256(path)
    sample_bytes = max(0, int(sample_bytes))

    stat = path.stat()
    size = stat.st_size
    digest = hashlib.blake2b(digest_size=20)
    digest.update(str(size).encode("utf-8"))
    if size <= 0:
        return f"fast:{digest.hexdigest()}"
    with path.open("rb") as file:
        head = file.read(sample_bytes)
        digest.update(head)
        if size > sample_bytes:
            file.seek(max(0, size - sample_bytes))
            digest.update(file.read(sample_bytes))
    return f"fast:{digest.hexdigest()}"


def supported_source_paths(input_dir: Path, code_training_mode: bool = False, include_source_code: bool = True) -> list[Path]:
    """Return supported source paths.

    Args:
        input_dir: Folder to scan.
        code_training_mode: Whether source-code files are supported.
        include_source_code: Whether to include source-code files.

    Returns:
        Sorted supported paths.

    Raises:
        FileNotFoundError: If the folder does not exist.
    """

    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")
    paths = [path for path in sorted(input_dir.rglob("*")) if path.is_file()]
    return [
        path
        for path in paths
        if path.suffix.lower() in SUPPORTED_TEXT_SUFFIXES | {".pdf", ".jsonl"}
        or (code_training_mode and include_source_code and path.suffix.lower() in SUPPORTED_CODE_SUFFIXES)
    ]


def clean_text(text: str, lowercase: bool = False) -> str:
    """Normalize prose text.

    Args:
        text: Raw text extracted from a document.
        lowercase: Whether to convert text to lowercase.

    Returns:
        Whitespace-normalized prose text.
    """

    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text.lower() if lowercase else text


def clean_code(text: str, lowercase: bool = False) -> str:
    """Normalize code while preserving structure.

    Args:
        text: Raw code text.
        lowercase: Whether to lowercase code. Usually false for code.

    Returns:
        Code text with line breaks and indentation retained.
    """

    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    text = text.strip()
    return text.lower() if lowercase else text


def read_pdf(path: Path) -> str:
    """Extract text from a PDF file.

    Args:
        path: PDF file path.

    Returns:
        Extracted text joined across pages.
    """

    chunks: list[str] = []
    with path.open("rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            chunks.append(page.extract_text() or "")
    return "\n".join(chunks)


def read_jsonl(path: Path) -> str:
    """Read text-like values from a JSONL file.

    Args:
        path: JSONL file path.

    Returns:
        Combined text from string rows or structured dataset fields.
    """

    chunks: list[str] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, str):
                chunks.append(value)
            else:
                text = _extract_structured_text(value, "instruction")
                if text:
                    chunks.append(text)
    return "\n".join(chunks)


def _iter_json_records(path: Path) -> list[Any]:
    """Read JSON or JSONL records from a file.

    Args:
        path: JSON or JSONL source file.

    Returns:
        List of decoded records.
    """

    if path.suffix.lower() == ".jsonl":
        records: list[Any] = []
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    records.append(json.loads(line))
        return records

    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in ("data", "examples", "items", "records", "rows"):
            nested = value.get(key)
            if isinstance(nested, list):
                return nested
        return [value]
    return [value]


def _role_name(value: Any) -> str:
    """Return a readable chat role name.

    Args:
        value: Raw role/from value.

    Returns:
        Normalized role label.
    """

    role = str(value or "").strip().lower()
    if role in {"human", "user", "prompt", "question"}:
        return "User"
    if role in {"gpt", "assistant", "bot", "model", "answer"}:
        return "Assistant"
    if role in {"system", "developer"}:
        return role.title()
    return role.title() if role else "Message"


def _format_message_list(messages: Any) -> str:
    """Format OpenAI/ShareGPT-style message rows.

    Args:
        messages: Message list from a structured dataset record.

    Returns:
        Human-readable transcript text.
    """

    if not isinstance(messages, list):
        return ""
    lines: list[str] = []
    for item in messages:
        if isinstance(item, str):
            content = item.strip()
            if content:
                lines.append(content)
            continue
        if not isinstance(item, dict):
            continue
        role = _role_name(item.get("role", item.get("from", item.get("speaker", item.get("author")))))
        content = item.get("content", item.get("value", item.get("text", item.get("message", ""))))
        if isinstance(content, list):
            content = json.dumps(content, ensure_ascii=False, separators=(",", ":"))
        content_text = str(content or "").strip()
        details: list[str] = []
        if content_text:
            details.append(content_text)
        # Do not discard structured tool calls or tool results.  JSON is used
        # deliberately so arguments and result payloads remain unambiguous.
        if item.get("tool_calls") is not None:
            details.append("tool_calls=" + json.dumps(item["tool_calls"], ensure_ascii=False, separators=(",", ":")))
        if item.get("tool_call_id") is not None:
            details.append("tool_call_id=" + str(item["tool_call_id"]))
        if item.get("name") is not None:
            details.append("name=" + str(item["name"]))
        if details:
            lines.append(f"{role}: {' '.join(details)}")
    return "\n".join(lines)


def _extract_structured_text(record: Any, kind: str) -> str:
    """Extract training text from a structured JSON record.

    Args:
        record: JSON value from a dataset file.
        kind: Target sample kind, usually conversation or instruction.

    Returns:
        Extracted sample text, or an empty string.
    """

    if isinstance(record, str):
        return record.strip()
    if isinstance(record, list):
        return _format_message_list(record)
    if not isinstance(record, dict):
        return ""

    for message_key in ("messages", "conversations", "dialogue", "utterances", "turns"):
        transcript = _format_message_list(record.get(message_key))
        if transcript:
            if record.get("tools") is not None:
                transcript = (
                    "Tools: " + json.dumps(record["tools"], ensure_ascii=False, separators=(",", ":"))
                    + "\n" + transcript
                )
            return transcript

    # OpenAI tool definitions are part of the training example context even
    # when the record has no textual messages.
    tools = record.get("tools")
    if tools is not None:
        tool_text = "Tools: " + json.dumps(tools, ensure_ascii=False, separators=(",", ":"))
        for key in ("messages", "tool_calls", "tool_results"):
            value = record.get(key)
            if value is not None and key != "messages":
                tool_text += f"\n{key}: " + json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        return tool_text

    instruction = str(record.get("instruction", "") or "").strip()
    user_input = str(record.get("input", "") or "").strip()
    output = str(
        record.get("output", record.get("response", record.get("answer", record.get("completion", "")))) or ""
    ).strip()
    if instruction or user_input:
        lines = []
        if instruction:
            lines.append(f"Instruction: {instruction}")
        if user_input:
            lines.append(f"Input: {user_input}")
        if output:
            lines.append(f"Response: {output}")
        return "\n".join(lines)

    prompt = str(record.get("prompt", record.get("question", "")) or "").strip()
    completion = str(record.get("completion", record.get("answer", record.get("response", ""))) or "").strip()
    if prompt or completion:
        if kind == "conversation":
            return "\n".join(part for part in (f"User: {prompt}" if prompt else "", f"Assistant: {completion}" if completion else "") if part)
        return "\n".join(part for part in (f"Prompt: {prompt}" if prompt else "", f"Completion: {completion}" if completion else "") if part)

    for key in ("text", "content", "body"):
        value = record.get(key)
        if value:
            return str(value).strip()
    return ""


def load_structured_json_documents(path: Path, kind: str, lowercase: bool = False) -> list[Document]:
    """Load conversation or instruction samples from JSON/JSONL files.

    Args:
        path: JSON/JSONL file or folder containing JSON/JSONL files.
        kind: Sample kind to assign, usually ``conversation`` or ``instruction``.
        lowercase: Whether to lowercase extracted text.

    Returns:
        Loaded structured dataset documents.

    Raises:
        FileNotFoundError: If the configured path does not exist.
        ValueError: If the file type is unsupported.
    """

    if kind not in {"conversation", "instruction", "tool_call"}:
        raise ValueError(f"Unsupported structured dataset kind: {kind}")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Structured dataset path does not exist: {path}")
    files = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.suffix.lower() in {".json", ".jsonl"})
    if not files:
        raise ValueError(f"No .json or .jsonl files found in {path}")

    documents: list[Document] = []
    for file_path in files:
        if file_path.suffix.lower() not in {".json", ".jsonl"}:
            raise ValueError(f"Unsupported structured dataset file: {file_path}")
        for index, record in enumerate(_iter_json_records(file_path), start=1):
            text = _extract_structured_text(record, kind)
            text = clean_code(text, lowercase=lowercase)
            if not text:
                continue
            documents.append(
                Document(
                    path=Path(f"{file_path}#{index}"),
                    text=text,
                    kind=kind,
                    language="local_json",
                )
            )
    return documents


def read_supported_document(
    path: Path,
    lowercase: bool = False,
    code_training_mode: bool = False,
    preserve_indentation: bool = True,
) -> Optional[Document]:
    """Read one supported document or source-code file.

    Args:
        path: Source file path.
        lowercase: Whether to lowercase loaded content.
        code_training_mode: Whether code-specific handling is enabled.
        preserve_indentation: Whether code line structure should be kept.

    Returns:
        Loaded document, or ``None`` when the file has no useful text.
    """

    suffix = path.suffix.lower()
    # Bundled code-training corpora may use .txt or .jsonl containers while
    # still being intended for code-aware preparation.  Classify those files
    # by their directory as well as by source-code extension.
    in_code_training_folder = any(
        part.lower() == "code_training" for part in path.parts
    )
    if code_training_mode and suffix in SUPPORTED_CODE_SUFFIXES:
        text = path.read_text(encoding="utf-8", errors="ignore")
        text = clean_code(text, lowercase=lowercase) if preserve_indentation else clean_text(text, lowercase=lowercase)
        if not text:
            return None
        return Document(path=path, text=text, kind="code", language=SUPPORTED_CODE_SUFFIXES[suffix])
    if suffix in SUPPORTED_TEXT_SUFFIXES:
        text = path.read_text(encoding="utf-8", errors="ignore")
    elif suffix == ".pdf":
        text = read_pdf(path)
    elif suffix == ".jsonl":
        text = read_jsonl(path)
    else:
        return None

    if in_code_training_folder and code_training_mode:
        text = clean_code(text, lowercase=lowercase)
        if not text:
            return None
        language = next(
            (
                language
                for extension, language in SUPPORTED_CODE_SUFFIXES.items()
                if path.stem.lower().startswith(extension.lstrip("."))
            ),
            None,
        )
        return Document(path=path, text=text, kind="code", language=language)

    text = clean_text(text, lowercase=lowercase)
    if not text:
        return None
    return Document(path=path, text=text)
