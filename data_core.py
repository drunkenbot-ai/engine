from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

import PyPDF2

from .tool_call_data import format_tool_call_record

LOGGER = logging.getLogger(__name__)
MAX_JSON_LOCATION_RANGES = 12
MAX_JSON_REASON_TYPES = 4
MAX_JSON_REASON_LENGTH = 160


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


@dataclass(frozen=True)
class JsonRecordIssue:
    """One malformed or schema-invalid JSON/JSONL record."""

    path: Path
    record_number: int
    reason: str

    def message(self) -> str:
        location = (
            f"line {self.record_number}"
            if self.path.suffix.lower() == ".jsonl"
            else f"record {self.record_number}"
        )
        return f"{self.path.name}, {location}: {self.reason}"


@dataclass
class JsonRecordDiagnostics:
    """Bounded per-file accounting for malformed or unusable JSON records."""

    path: Path
    invalid_record_count: int = 0
    location_ranges: list[tuple[int, int]] = field(default_factory=list)
    omitted_location_count: int = 0
    reason_counts: dict[str, int] = field(default_factory=dict)
    omitted_reason_count: int = 0

    @property
    def location_kind(self) -> str:
        return "line" if self.path.suffix.lower() == ".jsonl" else "record"

    def add(self, record_number: int, reason: str) -> None:
        """Record one issue without retaining one object or message per row."""

        reason = " ".join(str(reason).split())[:MAX_JSON_REASON_LENGTH]
        self.invalid_record_count += 1
        if record_number > 0:
            if (
                self.location_ranges
                and self.location_ranges[-1][1] + 1 == record_number
            ):
                start, _end = self.location_ranges[-1]
                self.location_ranges[-1] = (start, record_number)
            elif len(self.location_ranges) < MAX_JSON_LOCATION_RANGES:
                self.location_ranges.append((record_number, record_number))
            else:
                self.omitted_location_count += 1
        else:
            self.omitted_location_count += 1

        if reason in self.reason_counts:
            self.reason_counts[reason] += 1
        elif len(self.reason_counts) < MAX_JSON_REASON_TYPES:
            self.reason_counts[reason] = 1
        else:
            self.omitted_reason_count += 1

    def summary(self) -> str:
        """Return one bounded human-readable diagnostic sentence."""

        count = self.invalid_record_count
        noun = "record" if count == 1 else "records"
        locations = self._location_summary()
        reasons = ", ".join(
            f"{reason} ({reason_count:,})"
            for reason, reason_count in self.reason_counts.items()
        )
        if self.omitted_reason_count:
            reasons += (
                f", {self.omitted_reason_count:,} more issue"
                f"{'s' if self.omitted_reason_count != 1 else ''}"
            )
        return (
            f"{self.path.name}: {count:,} invalid {noun}"
            f"{f' at {locations}' if locations else ''}; reasons: {reasons}."
        )

    def message(self) -> str:
        """Compatibility alias for callers of the former issue object."""

        return self.summary()

    def to_jsonable(self) -> dict[str, Any]:
        """Return compact complete counts plus bounded location/reason previews."""

        return {
            "path": str(self.path),
            "filename": self.path.name,
            "location_kind": self.location_kind,
            "invalid_record_count": self.invalid_record_count,
            "location_ranges": [
                {"start": start, "end": end}
                for start, end in self.location_ranges
            ],
            "omitted_location_count": self.omitted_location_count,
            "reason_counts": dict(self.reason_counts),
            "omitted_reason_count": self.omitted_reason_count,
            "summary": self.summary(),
        }

    @classmethod
    def from_jsonable(
        cls,
        value: dict[str, Any],
        fallback_path: Optional[Path] = None,
    ) -> "JsonRecordDiagnostics":
        """Restore diagnostics from a cache or manifest payload."""

        path = Path(value.get("path") or fallback_path or value.get("filename") or "")
        diagnostics = cls(path=path)
        diagnostics.invalid_record_count = int(
            value.get("invalid_record_count", 0)
        )
        ranges = value.get("location_ranges") or []
        if not isinstance(ranges, list):
            raise TypeError("location_ranges must be a list")
        for item in ranges:
            if len(diagnostics.location_ranges) >= MAX_JSON_LOCATION_RANGES:
                break
            if (
                isinstance(item, dict)
                and item.get("start") is not None
                and item.get("end") is not None
            ):
                diagnostics.location_ranges.append(
                    (int(item["start"]), int(item["end"]))
                )
        diagnostics.omitted_location_count = int(
            value.get("omitted_location_count", 0)
        )
        reason_counts = value.get("reason_counts") or {}
        if not isinstance(reason_counts, dict):
            raise TypeError("reason_counts must be an object")
        for reason, count in reason_counts.items():
            if len(diagnostics.reason_counts) >= MAX_JSON_REASON_TYPES:
                break
            normalized_reason = " ".join(str(reason).split())[
                :MAX_JSON_REASON_LENGTH
            ]
            diagnostics.reason_counts[normalized_reason] = int(count)
        diagnostics.omitted_reason_count = int(
            value.get("omitted_reason_count", 0)
        )
        return diagnostics

    @classmethod
    def from_legacy_messages(
        cls,
        path: Path,
        messages: list[Any],
    ) -> "JsonRecordDiagnostics":
        """Compact an old cache's unbounded per-record message list."""

        diagnostics = cls(path=path)
        location_pattern = re.compile(
            r"(?:line|record)\s+(\d+)\s*:\s*(.*?)(?:\.)?$"
        )
        for message in messages:
            text = str(message)
            match = location_pattern.search(text)
            if match:
                diagnostics.add(int(match.group(1)), match.group(2))
            else:
                diagnostics.add(0, text)
        return diagnostics

    def _location_summary(self) -> str:
        values = []
        for start, end in self.location_ranges:
            values.append(str(start) if start == end else f"{start}-{end}")
        summary = ", ".join(values)
        if self.omitted_location_count:
            suffix = f"{self.omitted_location_count:,} more"
            summary = f"{summary}, {suffix}" if summary else suffix
        if not summary:
            return ""
        label = self.location_kind
        if len(self.location_ranges) != 1 or (
            self.location_ranges and self.location_ranges[0][0] != self.location_ranges[0][1]
        ) or self.omitted_location_count:
            label += "s"
        return f"{label} {summary}"


@dataclass
class StructuredDocumentLoad:
    """Structured documents plus per-record diagnostics and source files."""

    documents: list[Document]
    issues: list[JsonRecordDiagnostics]
    source_files: list[Path]

    @property
    def diagnostics(self) -> list[JsonRecordDiagnostics]:
        """Return bounded per-file issues under the explicit new name."""

        return self.issues


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


def load_jsonl_documents(
    path: Path,
    lowercase: bool = False,
    on_invalid: Optional[Callable[[str], None]] = None,
) -> list[Document]:
    """Load each JSONL record as an independent training document."""
    return load_jsonl_documents_with_diagnostics(
        path,
        lowercase=lowercase,
        on_invalid=on_invalid,
    ).documents


def load_jsonl_documents_with_diagnostics(
    path: Path,
    lowercase: bool = False,
    on_invalid: Optional[Callable[[str], None]] = None,
) -> StructuredDocumentLoad:
    """Load independent JSONL records with one bounded file diagnostic."""

    documents: list[Document] = []
    diagnostics = JsonRecordDiagnostics(path=path)
    record_seen = False
    for index, record, error in _iter_json_records_with_errors(path):
        record_seen = True
        if error:
            diagnostics.add(index, error)
            continue
        kind = _structured_record_kind(record)
        validation_error = _structured_record_error(record, kind)
        if validation_error:
            diagnostics.add(index, validation_error)
            continue
        text = clean_text(_extract_structured_text(record, kind), lowercase=lowercase)
        if not text:
            diagnostics.add(index, "record contains no extractable text")
            continue
        documents.append(
            Document(
                path=Path(f"{path}#{index}"),
                text=text,
                kind=kind,
                language="local_json",
            )
        )
    if not record_seen:
        diagnostics.add(0, "file contains no records")
    result_diagnostics = [diagnostics] if diagnostics.invalid_record_count else []
    _emit_json_diagnostics(result_diagnostics, on_invalid)
    return StructuredDocumentLoad(documents, result_diagnostics, [path])


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

    tool_call_text = format_tool_call_record(record)
    if tool_call_text:
        return tool_call_text
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


def load_structured_json_documents(
    path: Path,
    kind: str,
    lowercase: bool = False,
    on_invalid: Optional[Callable[[str], None]] = None,
) -> list[Document]:
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

    return load_structured_json_documents_with_diagnostics(
        path,
        kind,
        lowercase=lowercase,
        on_invalid=on_invalid,
    ).documents


def load_structured_json_documents_with_diagnostics(
    path: Path,
    kind: str,
    lowercase: bool = False,
    on_invalid: Optional[Callable[[str], None]] = None,
) -> StructuredDocumentLoad:
    """Load structured records while retaining source-level diagnostics."""

    if kind not in {"conversation", "instruction", "tool_call"}:
        raise ValueError(f"Unsupported structured dataset kind: {kind}")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Structured dataset path does not exist: {path}")
    files = structured_json_source_files(path)
    if not files:
        raise ValueError(f"No .json or .jsonl files found in {path}")

    documents: list[Document] = []
    diagnostics_by_file: list[JsonRecordDiagnostics] = []
    for file_path in files:
        if file_path.suffix.lower() not in {".json", ".jsonl"}:
            raise ValueError(f"Unsupported structured dataset file: {file_path}")
        diagnostics = JsonRecordDiagnostics(path=file_path)
        record_seen = False
        for index, record, error in _iter_json_records_with_errors(file_path):
            record_seen = True
            if error:
                diagnostics.add(index, error)
                continue
            validation_error = _structured_record_error(record, kind)
            if validation_error:
                diagnostics.add(index, validation_error)
                continue
            text = _extract_structured_text(record, kind)
            text = clean_code(text, lowercase=lowercase)
            if not text:
                diagnostics.add(index, "record contains no extractable text")
                continue
            documents.append(
                Document(
                    path=Path(f"{file_path}#{index}"),
                    text=text,
                    kind=kind,
                    language="local_json",
                )
            )
        if not record_seen:
            diagnostics.add(0, "file contains no records")
        if diagnostics.invalid_record_count:
            diagnostics_by_file.append(diagnostics)
    _emit_json_diagnostics(diagnostics_by_file, on_invalid)
    return StructuredDocumentLoad(documents, diagnostics_by_file, files)


def structured_json_source_files(path: Path) -> list[Path]:
    """Return the physical JSON/JSONL sources represented by a configured path."""

    path = Path(path)
    if path.is_file():
        return [path]
    return sorted(
        item
        for item in path.rglob("*")
        if item.suffix.lower() in {".json", ".jsonl"}
    )


def _emit_json_diagnostics(
    diagnostics: list[JsonRecordDiagnostics],
    on_invalid: Optional[Callable[[str], None]],
) -> None:
    for diagnostic in diagnostics:
        message = diagnostic.summary()
        if on_invalid is not None:
            on_invalid(message)
        else:
            LOGGER.warning("%s", message)


def _structured_record_kind(record: Any) -> str:
    if isinstance(record, dict) and any(
        key in record for key in ("tool_calls", "tool_results", "tools")
    ):
        return "tool_call"
    if isinstance(record, list) or (
        isinstance(record, dict)
        and any(
            key in record
            for key in (
                "messages",
                "conversations",
                "dialogue",
                "utterances",
                "turns",
            )
        )
    ):
        return "conversation"
    return "instruction"


def _structured_record_error(record: Any, kind: str) -> str | None:
    """Return a concise reason when a structured training record is unusable."""
    if not isinstance(record, (dict, list, str)):
        return f"expected a JSON object, array, or string, got {type(record).__name__}"
    if isinstance(record, str):
        return None if record.strip() else "empty string record"
    if isinstance(record, list):
        if not record:
            return "empty message list"
        if kind == "instruction":
            return "instruction records must be objects, not message arrays"
        return None
    if kind == "conversation":
        value = record.get("messages", record.get("conversations", record.get("dialogue",
                         record.get("utterances", record.get("turns")))))
        if value is not None and (not isinstance(value, list) or not value):
            return "conversation messages must be a non-empty array"
        if value is None and "role" in record and not any(
                record.get(key) for key in ("prompt", "question", "text", "body")
        ):
            return "standalone role/content object is not a conversation record"
        if value is None and not any(record.get(key) for key in ("prompt", "question", "text", "content", "body")):
            return "missing conversation messages or text fields"
    elif kind == "instruction":
        if not any(record.get(key) for key in ("instruction", "input", "output", "response",
                                                "answer", "completion", "prompt", "question", "text",
                                                "content", "body")):
            return "missing instruction/input/output or text fields"
    elif kind == "tool_call":
        messages = record.get("messages")
        if messages is not None and (not isinstance(messages, list) or not messages):
            return "tool_call messages must be a non-empty array"
        if not any(record.get(key) is not None for key in ("messages", "tools", "tool_calls",
                                                            "tool_results", "prompt", "text", "content")):
            return "missing tools, messages, tool_calls, or text fields"
    return None


def _iter_json_records_with_errors(
    path: Path,
) -> Iterator[tuple[int, Any, Optional[str]]]:
    """Yield (record number, record, error) while isolating malformed JSON rows."""
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    yield line_number, json.loads(line), None
                except json.JSONDecodeError as exc:
                    yield line_number, None, f"invalid JSON ({exc.msg})"
        return
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        yield 1, None, f"invalid JSON ({exc.msg})"
        return
    values = value if isinstance(value, list) else next(
        (value[key] for key in ("data", "examples", "items", "records", "rows")
         if isinstance(value, dict) and isinstance(value.get(key), list)),
        [value],
    )
    for index, record in enumerate(values, start=1):
        yield index, record, None


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
