from __future__ import annotations

import json
from pathlib import Path
from random import Random
from typing import Callable, Iterator, Optional

import numpy as np
import numpy.lib.format as npy_format
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
DEFAULT_CHAT_TEMPLATE = """{% for message in messages %}{{ '<bos>' if loop.first else '' }}{{ message['role'] | capitalize }}: {{ message['content'] }}{{ '<eos>' if loop.last else '\\n' }}{% endfor %}{% if add_generation_prompt %}{{ '\\nAssistant:' }}{% endif %}"""
MAX_TOKENIZER_LINE_CHARS = 8_192
# The Rust BPE trainer builds an in-memory pretoken frequency table sized to
# whatever corpus it is shown. Vocabulary quality saturates well before a
# multi-gigabyte corpus is fully consumed, so by default only a bounded,
# evenly-spread sample of the corpus is shown to the trainer. The full
# corpus is still encoded with the resulting tokenizer afterward -- only
# *training* the merges is sampled.
DEFAULT_TOKENIZER_TRAINING_MAX_BYTES = 2 * 1024 * 1024 * 1024  * 1024 * 1024 * 1024 # 12 GiB
TOKENIZER_SAMPLE_SEED = 1337
# Tokens are streamed to disk in fixed-size batches rather than accumulated
# into one giant Python list, so peak RAM during encoding stays roughly
# constant regardless of corpus size.
ENCODE_FLUSH_TOKEN_COUNT = 200_000
# Number of line-chunks encoded per tokenizer.encode_batch() call. The Rust
# tokenizers library parallelizes encode_batch() internally across CPU
# cores (via Rayon); calling encode() one string at a time from a Python
# loop -- the previous approach -- pays Python/Rust boundary overhead per
# call and never engages that internal parallelism, which dominates total
# time at billions-of-tokens scale.
ENCODE_BATCH_SIZE = 1_000
# Chunk size (in tokens) used when streaming raw token bytes into the final
# .npy file. Keeps the .bin -> .npy conversion step's RAM use flat too.
NPY_CONVERT_CHUNK_TOKENS = 1_000_000


def train_tokenizer(
    corpus_path: Path,
    output_path: Path,
    vocab_size: int = 8000,
    min_frequency: int = 2,
    should_stop: Optional[Callable[[], bool]] = None,
    max_training_bytes: Optional[int] = DEFAULT_TOKENIZER_TRAINING_MAX_BYTES,
) -> Tokenizer:
    """Train a byte-level BPE tokenizer.

    Args:
        corpus_path: Text corpus used for tokenizer training.
        output_path: Destination tokenizer JSON path.
        vocab_size: Target vocabulary size.
        min_frequency: Minimum token frequency for BPE merges.
        should_stop: Optional callback returning true when training should stop.
        max_training_bytes: Maximum corpus bytes shown to the BPE trainer.
            The trainer keeps a frequency table in memory sized to whatever it
            is shown, so on very large corpora only an evenly-spread sample
            up to this many bytes is used to fit merges. Pass ``None`` to
            disable sampling and train on the entire corpus. The full corpus
            is always encoded with the resulting tokenizer regardless of this
            setting.

    Returns:
        Trained tokenizer instance.
    """

    tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )
    corpus_size = corpus_path.stat().st_size
    sample_ratio = 1.0
    if max_training_bytes is not None and corpus_size > max_training_bytes:
        sample_ratio = max_training_bytes / corpus_size
    tokenizer.train_from_iterator(
        _iter_corpus_lines(corpus_path, should_stop, sample_ratio=sample_ratio),
        trainer=trainer,
    )
    tokenizer.post_processor = TemplateProcessing(
        single=f"{BOS_TOKEN} $A {EOS_TOKEN}",
        special_tokens=[
            (BOS_TOKEN, tokenizer.token_to_id(BOS_TOKEN)),
            (EOS_TOKEN, tokenizer.token_to_id(EOS_TOKEN)),
        ],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))
    save_tokenizer_package(tokenizer, output_path)
    return tokenizer


def save_tokenizer_package(
    tokenizer: Tokenizer,
    tokenizer_path: Path,
    model_max_length: Optional[int] = None,
) -> None:
    """Write standard tokenizer metadata beside the native tokenizer JSON."""

    tokenizer_path = Path(tokenizer_path)
    tokens = {
        "bos_token": BOS_TOKEN,
        "eos_token": EOS_TOKEN,
        "unk_token": UNK_TOKEN,
        "pad_token": PAD_TOKEN,
    }
    (tokenizer_path.parent / "special_tokens_map.json").write_text(
        json.dumps(tokens, indent=2) + "\n", encoding="utf-8"
    )
    config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "tokenizer_file": tokenizer_path.name,
        "model_max_length": int(model_max_length) if model_max_length else 1_000_000_000_000_000_000_000_000_000_000,
        "clean_up_tokenization_spaces": False,
        "add_bos_token": True,
        "add_eos_token": True,
        "chat_template": DEFAULT_CHAT_TEMPLATE,
        **tokens,
    }
    (tokenizer_path.parent / "tokenizer_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )


def _iter_corpus_lines(
    corpus_path: Path,
    should_stop: Optional[Callable[[], bool]],
    sample_ratio: float = 1.0,
) -> Iterator[str]:
    """Yield corpus lines and check for cancellation between chunks.

    Args:
        corpus_path: Text corpus path to read line by line.
        should_stop: Optional callback returning true when reading should stop.
        sample_ratio: Fraction of lines to keep, in ``(0.0, 1.0]``. Lines are
            kept via an independent per-line random draw (not a positional
            head-of-file cut), so the kept sample is spread evenly across the
            whole file rather than biased toward whatever content appears
            first.

    Raises:
        RuntimeError: If cancellation is requested.
    """

    sample_ratio = max(0.0, min(1.0, sample_ratio))
    sampling = sample_ratio < 1.0
    rng = Random(TOKENIZER_SAMPLE_SEED) if sampling else None

    with corpus_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if should_stop and should_stop():
                raise RuntimeError("Dataset preparation stopped by user.")
            if sampling and rng.random() > sample_ratio:
                continue

            for start in range(0, len(line), MAX_TOKENIZER_LINE_CHARS):
                chunk = line[start : start + MAX_TOKENIZER_LINE_CHARS]
                if chunk:
                    yield chunk


def load_tokenizer(path: Path) -> Tokenizer:
    """Load a tokenizer from disk.

    Args:
        path: Tokenizer JSON path.

    Returns:
        Loaded tokenizer.
    """

    return Tokenizer.from_file(str(path))


def token_id(tokenizer: Tokenizer, token: str) -> int:
    """Return the integer ID for a required special token.

    Args:
        tokenizer: Tokenizer to query.
        token: Token string to find.

    Returns:
        Token ID.

    Raises:
        ValueError: If the tokenizer does not contain the token.
    """

    value = tokenizer.token_to_id(token)
    if value is None:
        raise ValueError(f"Tokenizer is missing required token: {token}")
    return value


def missing_training_special_tokens(tokenizer: Tokenizer) -> list[str]:
    """Return special tokens missing from a tokenizer.

    Args:
        tokenizer: Tokenizer to inspect.

    Returns:
        Missing special token strings.
    """

    return [token for token in SPECIAL_TOKENS if tokenizer.token_to_id(token) is None]


def validate_training_tokenizer(tokenizer: Tokenizer) -> None:
    """Validate that a tokenizer can be used by the MicroGPT trainer.

    Args:
        tokenizer: Tokenizer to validate.

    Raises:
        ValueError: If required special tokens are missing.
    """

    missing = missing_training_special_tokens(tokenizer)
    if missing:
        raise ValueError(
            "Tokenizer is not compatible with Micro LLM Creator training. "
            f"Missing required special token(s): {', '.join(missing)}. "
            "Use a tokenizer created by this app, or import a tokenizer containing "
            "<pad>, <unk>, <bos>, and <eos>."
        )


def encode_text(tokenizer: Tokenizer, text: str) -> list[int]:
    """Encode text into token IDs.

    Args:
        tokenizer: Tokenizer used for encoding.
        text: Text to encode.

    Returns:
        List of token IDs.
    """

    token_ids: list[int] = []
    for start in range(0, len(text), MAX_TOKENIZER_LINE_CHARS):
        chunk = text[start : start + MAX_TOKENIZER_LINE_CHARS]
        if chunk:
            token_ids.extend(tokenizer.encode(chunk).ids)
    return token_ids


def encode_file(
    tokenizer: Tokenizer,
    corpus_path: Path,
    should_stop: Optional[Callable[[], bool]] = None,
) -> list[int]:
    """Encode a corpus file into token IDs without loading it all at once.

    Args:
        tokenizer: Tokenizer used for encoding.
        corpus_path: Text corpus path.
        should_stop: Optional callback returning true when encoding should stop.

    Returns:
        Token IDs for the corpus.

    Raises:
        RuntimeError: If cancellation is requested.
    """

    token_ids: list[int] = []
    with corpus_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if should_stop and should_stop():
                raise RuntimeError("Dataset preparation stopped by user.")
            for start in range(0, len(line), MAX_TOKENIZER_LINE_CHARS):
                chunk = line[start : start + MAX_TOKENIZER_LINE_CHARS]
                if chunk:
                    token_ids.extend(tokenizer.encode(chunk).ids)
    return token_ids


def token_dtype_for_vocab(vocab_size: int) -> np.dtype:
    """Pick the smallest unsigned integer dtype that can hold every token ID.

    Args:
        vocab_size: Tokenizer vocabulary size.

    Returns:
        ``uint16`` for the (overwhelmingly common) case of a vocab under
        65,536, otherwise ``uint32``.
    """

    return np.dtype(np.uint16) if vocab_size <= 65_535 else np.dtype(np.uint32)


def encode_file_to_bin(
    tokenizer: Tokenizer,
    corpus_path: Path,
    output_path: Path,
    dtype: np.dtype,
    should_stop: Optional[Callable[[], bool]] = None,
) -> int:
    """Encode a corpus file straight to a flat, headerless binary token file.

    Unlike ``encode_file``, this never accumulates the whole corpus as a
    Python list of ints in memory. Token IDs are buffered in small batches
    and flushed to disk as raw fixed-width integers (``dtype``), so peak RAM
    stays roughly constant no matter how large the corpus is. This is an
    intermediate format (no shape/dtype header) -- see ``encode_file_to_npy``
    for the version that produces a directly loadable ``.npy`` file.

    Line-chunks are also batched into groups of ``ENCODE_BATCH_SIZE`` and
    encoded with a single ``tokenizer.encode_batch()`` call per group rather
    than one ``tokenizer.encode()`` call per chunk -- the Rust tokenizer
    parallelizes ``encode_batch()`` across CPU cores internally, which
    matters enormously at multi-billion-token scale.

    Args:
        tokenizer: Tokenizer used for encoding.
        corpus_path: Text corpus path.
        output_path: Destination raw ``.bin`` path for the encoded token stream.
        dtype: Integer dtype to store each token ID as (see
            ``token_dtype_for_vocab``).
        should_stop: Optional callback returning true when encoding should stop.

    Returns:
        Total number of tokens written.

    Raises:
        RuntimeError: If cancellation is requested.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    buffer: list[int] = []
    pending_chunks: list[str] = []
    total_tokens = 0

    def flush_pending_chunks(sink) -> None:
        """Encode any buffered chunks as one batch and append their IDs."""

        nonlocal total_tokens
        if pending_chunks:
            for encoding in tokenizer.encode_batch(pending_chunks):
                buffer.extend(encoding.ids)
            pending_chunks.clear()
        if len(buffer) >= ENCODE_FLUSH_TOKEN_COUNT:
            np.asarray(buffer, dtype=dtype).tofile(sink)
            total_tokens += len(buffer)
            buffer.clear()

    with corpus_path.open("r", encoding="utf-8") as source, output_path.open("wb") as sink:
        for line in source:
            if should_stop and should_stop():
                raise RuntimeError("Dataset preparation stopped by user.")
            for start in range(0, len(line), MAX_TOKENIZER_LINE_CHARS):
                chunk = line[start : start + MAX_TOKENIZER_LINE_CHARS]
                if not chunk:
                    continue
                pending_chunks.append(chunk)
                if len(pending_chunks) >= ENCODE_BATCH_SIZE:
                    flush_pending_chunks(sink)
        flush_pending_chunks(sink)
        if buffer:
            np.asarray(buffer, dtype=dtype).tofile(sink)
            total_tokens += len(buffer)
    return total_tokens


def token_count_in_bin(path: Path, dtype: np.dtype) -> int:
    """Return how many tokens a raw ``.bin`` file holds, without loading it.

    Args:
        path: Raw token ``.bin`` file path.
        dtype: Integer dtype the tokens were stored as.

    Returns:
        Number of tokens in the file.
    """

    itemsize = np.dtype(dtype).itemsize
    return path.stat().st_size // itemsize


def load_token_memmap(path: Path, dtype: np.dtype) -> np.memmap:
    """Open a raw token ``.bin`` file as a read-only memory-mapped array.

    Args:
        path: Raw token ``.bin`` file path.
        dtype: Integer dtype the tokens were stored as.

    Returns:
        Read-only memory-mapped array of token IDs.
    """

    return np.memmap(path, dtype=dtype, mode="r")


def convert_bin_to_npy(bin_path: Path, npy_path: Path, dtype: np.dtype, token_count: int) -> None:
    """Convert a flat raw token ``.bin`` file into a self-describing ``.npy`` file.

    Only a small, fixed-size ``.npy`` header (built from ``dtype`` and
    ``token_count``, which ``encode_file_to_bin`` already gave us for free)
    needs to be known upfront. The token data itself is streamed across in
    fixed-size chunks via plain file reads/writes -- the full token array is
    never materialized in memory during conversion, regardless of how large
    the file is. The result is byte-for-byte a normal ``.npy`` file, loadable
    with ``numpy.load(path, mmap_mode="r")`` like any other.

    Args:
        bin_path: Source raw ``.bin`` file (as written by ``encode_file_to_bin``).
        npy_path: Destination ``.npy`` file path.
        dtype: Integer dtype the tokens were stored as.
        token_count: Number of tokens in ``bin_path``.
    """

    npy_path.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "descr": npy_format.dtype_to_descr(np.dtype(dtype)),
        "fortran_order": False,
        "shape": (token_count,),
    }
    chunk_bytes = NPY_CONVERT_CHUNK_TOKENS * np.dtype(dtype).itemsize
    with bin_path.open("rb") as source, npy_path.open("wb") as sink:
        npy_format.write_array_header_1_0(sink, header)
        while True:
            block = source.read(chunk_bytes)
            if not block:
                break
            sink.write(block)


def encode_file_to_npy(
    tokenizer: Tokenizer,
    corpus_path: Path,
    output_path: Path,
    dtype: np.dtype,
    should_stop: Optional[Callable[[], bool]] = None,
) -> int:
    """Encode a corpus file straight to a memory-map-friendly ``.npy`` file.

    Combines ``encode_file_to_bin`` (single-pass streaming encode, constant
    RAM) with ``convert_bin_to_npy`` (header + streamed byte copy, also
    constant RAM) so the whole corpus is never held in memory as a Python
    list or a full array at any point, while still producing a standard
    ``.npy`` file that ``numpy.load(path, mmap_mode="r")`` opens directly.

    Args:
        tokenizer: Tokenizer used for encoding.
        corpus_path: Text corpus path.
        output_path: Destination ``.npy`` path for the encoded token stream.
        dtype: Integer dtype to store each token ID as (see
            ``token_dtype_for_vocab``).
        should_stop: Optional callback returning true when encoding should stop.

    Returns:
        Total number of tokens written.

    Raises:
        RuntimeError: If cancellation is requested.
    """

    temp_bin_path = output_path.with_suffix(output_path.suffix + ".raw_tmp")
    try:
        token_count = encode_file_to_bin(tokenizer, corpus_path, temp_bin_path, dtype, should_stop=should_stop)
        convert_bin_to_npy(temp_bin_path, output_path, dtype, token_count)
    finally:
        temp_bin_path.unlink(missing_ok=True)
    return token_count
