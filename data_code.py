from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .data_core import (Document, SUPPORTED_CODE_SUFFIXES, clean_text, read_supported_document, load_structured_json_documents, supported_source_paths, OperationCancelled)

def is_code_like_line(line: str) -> bool:
    """Estimate whether a line appears to be source code.

    Args:
        line: Candidate text line.

    Returns:
        True when the line contains common code markers or dense syntax.
    """

    stripped = line.strip()
    if not stripped:
        return False
    code_markers = (
        "def ", "class ", "function ", "import ", "from ", "return ", "for ",
        "while ", "if ", "else:", "elif ", "try:", "except ", "public ",
        "private ", "protected ", "#include", "using ", "namespace ", "var ",
        "let ", "const ", "SELECT ", "INSERT ", "UPDATE ", "DELETE ",
    )
    if stripped.startswith(code_markers):
        return True
    symbol_count = sum(stripped.count(symbol) for symbol in "{}[]();=<>:+-*/")
    return symbol_count >= 3 or line.startswith(("    ", "\t"))


def guess_language(text: str, fallback: Optional[str] = None) -> Optional[str]:
    """Guess a programming language from code text.

    Args:
        text: Code sample text.
        fallback: Language to return when no heuristic matches.

    Returns:
        Guessed language name, fallback, or ``None``.
    """

    lowered = text.lower()
    if "def " in lowered or "import " in lowered or "self." in lowered:
        return "python"
    if "function " in lowered or "const " in lowered or "let " in lowered or "=>" in lowered:
        return "javascript"
    if "public class" in lowered or "system.out" in lowered:
        return "java"
    if "#include" in lowered or "std::" in lowered:
        return "cpp"
    if "select " in lowered and " from " in lowered:
        return "sql"
    return fallback


def extract_code_blocks_from_text(document: Document, preserve_indentation: bool = True) -> list[Document]:
    """Extract code-like blocks from prose/PDF text.

    Args:
        document: Source document whose text may contain code snippets.
        preserve_indentation: Whether extracted code should keep indentation.

    Returns:
        Code sample documents extracted from the source document.
    """

    lines = document.text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    blocks: list[Document] = []
    current: list[str] = []

    def flush() -> None:
        """Flush the current candidate block into ``blocks`` if code-like."""

        nonlocal current
        if len(current) >= 3:
            block = "\n".join(current)
            if sum(1 for line in current if is_code_like_line(line)) >= 2:
                cleaned = clean_code(block) if preserve_indentation else clean_text(block)
                blocks.append(
                    Document(
                        path=document.path,
                        text=cleaned,
                        kind="code",
                        language=guess_language(cleaned),
                    )
                )
        current = []

    for line in lines:
        if is_code_like_line(line):
            current.append(line)
        else:
            flush()
    flush()
    return blocks


def expand_code_documents(
    documents: list[Document],
    include_prose: bool = True,
    extract_code_blocks: bool = True,
    preserve_indentation: bool = True,
    should_stop: Optional[Callable[[], bool]] = None,
) -> list[Document]:
    """Expand documents for code-aware training.

    Args:
        documents: Loaded source documents.
        include_prose: Whether to keep prose documents.
        extract_code_blocks: Whether to extract code-like prose blocks.
        preserve_indentation: Whether to preserve code indentation.
        should_stop: Optional cancellation callback.

    Returns:
        Expanded document list.
    """

    expanded: list[Document] = []
    for document in documents:
        if should_stop and should_stop():
            raise OperationCancelled("Dataset preparation stopped by user.")
        if document.kind == "code":
            expanded.append(document)
            continue
        if include_prose:
            expanded.append(document)
        if extract_code_blocks:
            expanded.extend(extract_code_blocks_from_text(document, preserve_indentation=preserve_indentation))
    return expanded


def format_document_for_training(
    document: Document,
    generate_instruction_samples: bool = True,
    reasoning_sample_mode: str = "scaffold",
) -> str:
    """Format a document with tags for the training corpus.

    Args:
        document: Document to serialize.
        generate_instruction_samples: Whether code samples should include a
            simple instruction wrapper.
        reasoning_sample_mode: Instruction/reasoning style: none, scaffold, or detailed.

    Returns:
        Tagged training text for the document.
    """

    source = document.path.name
    if document.kind == "code":
        language = document.language or "unknown"
        if generate_instruction_samples:
            return format_code_instruction_sample(document, language, source, reasoning_sample_mode)
        return f"<code language=\"{language}\" source=\"{source}\">\n{document.text}\n</code>"
    if document.kind == "conversation":
        return f"<sample type=\"conversation\" source=\"{source}\">\n{document.text}\n</sample>"
    if document.kind == "instruction":
        return f"<sample type=\"instruction\" source=\"{source}\">\n{document.text}\n</sample>"
    return f"<sample type=\"prose\" source=\"{source}\">\n{document.text}\n</sample>"


def format_code_instruction_sample(document: Document, language: str, source: str, reasoning_sample_mode: str) -> str:
    """Format a code sample as an instruction/reasoning training example.

    Args:
        document: Code document.
        language: Programming language label.
        source: Source file name.
        reasoning_sample_mode: Instruction/reasoning style.

    Returns:
        Tagged training text.
    """

    task = infer_code_task(document, language)
    if reasoning_sample_mode == "none":
        return (
            f"<sample type=\"code\" language=\"{language}\" source=\"{source}\">\n"
            f"<instruction>{task}</instruction>\n"
            f"<answer>\n```{language}\n{document.text}\n```\n</answer>\n"
            f"</sample>"
        )
    if reasoning_sample_mode == "detailed":
        reasoning = (
            "1. Identify the goal implied by the file name, function names, and surrounding code.\n"
            "2. Inspect inputs, outputs, control flow, data structures, and error handling.\n"
            "3. Preserve language syntax, indentation, imports, and naming style.\n"
            "4. Produce the code first, then explain the important design choices and edge cases."
        )
        explanation = (
            "This sample teaches the model to connect a programming task with implementation details, "
            "syntax, structure, and a concise explanation."
        )
    else:
        reasoning = (
            "Understand the requested programming task, choose the relevant language patterns, "
            "preserve correct syntax, and provide the implementation."
        )
        explanation = "The answer contains the implementation that satisfies the task."
    return (
        f"<sample type=\"reasoning_code\" language=\"{language}\" source=\"{source}\">\n"
        f"<instruction>{task}</instruction>\n"
        f"<reasoning>\n{reasoning}\n</reasoning>\n"
        f"<answer>\n```{language}\n{document.text}\n```\n</answer>\n"
        f"<explanation>{explanation}</explanation>\n"
        f"</sample>"
    )


def infer_code_task(document: Document, language: str) -> str:
    """Infer a simple task instruction for a code sample.

    Args:
        document: Code document.
        language: Programming language label.

    Returns:
        Task instruction text.
    """

    stem = document.path.stem.replace("_", " ").replace("-", " ").strip()
    if stem and stem.lower() not in {"index", "main", "app"}:
        return f"Write or explain the {language} code for {stem}."
    return f"Write or explain this {language} code with correct syntax and structure."


def load_documents(
    input_dir: Path,
    lowercase: bool = False,
    max_workers: int = 4,
    code_training_mode: bool = False,
    include_prose: bool = True,
    include_source_code: bool = True,
    extract_code_blocks: bool = True,
    preserve_indentation: bool = True,
    progress: Optional[Callable[[Any], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> list[Document]:
    """Load supported files from a folder.

    Args:
        input_dir: Folder to scan recursively.
        lowercase: Whether to lowercase loaded content.
        max_workers: Maximum parallel file readers.
        code_training_mode: Enables code-aware loading and expansion.
        include_prose: Keeps prose documents in code-aware mode.
        include_source_code: Includes source-code files in code-aware mode.
        extract_code_blocks: Extracts code-like blocks from prose documents.
        preserve_indentation: Keeps code formatting where possible.
        progress: Optional callback receiving progress event dictionaries.
        should_stop: Optional callback returning true when loading should stop.

    Returns:
        Sorted list of loaded document samples.

    Raises:
        FileNotFoundError: If ``input_dir`` does not exist.
    """

    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")

    documents: list[Document] = []
    supported_paths = supported_source_paths(input_dir, code_training_mode=code_training_mode, include_source_code=include_source_code)
    if progress:
        progress({"message": f"Found {len(supported_paths)} supported files in {input_dir}.", "percent": 8})

    if not supported_paths:
        return documents

    worker_count = max(1, min(max_workers, len(supported_paths)))
    if progress:
        progress({"message": f"Reading files with {worker_count} worker(s).", "percent": 10})

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_map = {
            executor.submit(read_supported_document, path, lowercase, code_training_mode, preserve_indentation): path
            for path in supported_paths
        }
        for index, future in enumerate(as_completed(future_map), start=1):
            if should_stop and should_stop():
                for pending in future_map:
                    pending.cancel()
                raise OperationCancelled("Dataset preparation stopped by user.")
            path = future_map[future]
            percent = 10 + int(32 * index / max(len(supported_paths), 1))
            try:
                document = future.result()
            except Exception as exc:
                if progress:
                    progress({"message": f"Failed {path.name}: {exc}", "percent": percent})
                continue

            if document is None:
                if progress:
                    progress({"message": f"Skipped {path.name}: no readable text found.", "percent": percent})
                continue

            documents.append(document)
            if progress:
                progress({"message": f"Loaded {path.name}: {len(document.text):,} characters.", "percent": percent})

    if code_training_mode:
        documents = expand_code_documents(
            documents,
            include_prose=include_prose,
            extract_code_blocks=extract_code_blocks,
            preserve_indentation=preserve_indentation,
            should_stop=should_stop,
        )

    return sorted(documents, key=lambda document: (str(document.path), document.kind, document.language or ""))


def write_training_corpus(
    documents: list[Document],
    output_path: Path,
    code_training_mode: bool = False,
    generate_instruction_samples: bool = True,
    reasoning_sample_mode: str = "scaffold",
) -> None:
    """Write loaded samples into a tokenizer training corpus.

    Args:
        documents: Loaded document samples.
        output_path: Destination corpus text file.
        code_training_mode: Whether to use code/prose tags.
        generate_instruction_samples: Whether to wrap code samples with
            instruction text.
        reasoning_sample_mode: Instruction/reasoning style for code samples.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for doc in documents:
            if code_training_mode:
                file.write(
                    format_document_for_training(
                        doc,
                        generate_instruction_samples=generate_instruction_samples,
                        reasoning_sample_mode=reasoning_sample_mode,
                    )
                )
            else:
                file.write(doc.text)
            file.write("\n\n")


