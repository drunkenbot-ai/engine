from pathlib import Path
from engine.data import Document
from engine.dataset_corpus import _StreamingCorpusBuilder


def test_prefix_and_near_duplicate_filtering(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.txt"
    builder = _StreamingCorpusBuilder(
        corpus_path=corpus_path,
        filter_near_duplicates=True,
        max_prefix_repeats=5,  # low threshold for testing
    )

    # 1. Submit distinct high-entropy documents
    doc1 = Document(
        path=Path("doc1.txt"),
        text="Quantum mechanics explores the behavior of matter and light on atomic and subatomic scales.",
        kind="prose",
    )
    doc2 = Document(
        path=Path("doc2.txt"),
        text="General relativity describes gravitation as a geometric property of spacetime in physics.",
        kind="prose",
    )
    builder.submit(doc1)
    builder.submit(doc2)
    assert builder.stats.accepted_document_count == 2

    # 2. Submit repetitive prefix boilerplate (> 5 times)
    prefix_boilerplate = "You are a helpful, deep-thinking assistant. When presented with complex problems "
    for i in range(10):
        doc = Document(
            path=Path(f"prompt_{i}.txt"),
            text=f"{prefix_boilerplate} solve problem number {i} with distinct mathematical suffix {i*100}.",
            kind="prose",
        )
        builder.submit(doc)

    # Max prefix repeats is 5, so 5 accepted and 5 rejected as prefix duplicates
    assert builder.stats.prefix_duplicates_removed >= 4

    # 3. Submit high-similarity near-duplicates (swapping only a couple words in a template)
    template_base = (
        "Analyze the following synthetic 6-day daily OHLCV candlestick price action for Apple Inc. "
        "The open price is 150.00, high is 155.00, low is 149.00, and close is 154.00. "
        "The volume is 1,000,000 shares traded on the NASDAQ exchange with bullish momentum indicator."
    )
    doc_template_1 = Document(path=Path("stock_1.txt"), text=template_base, kind="prose")
    builder.submit(doc_template_1)

    # Near duplicate with only one number changed
    template_variant = template_base.replace("150.00", "151.00").replace("1,000,000", "1,000,500")
    doc_template_2 = Document(path=Path("stock_2.txt"), text=template_variant, kind="prose")
    builder.submit(doc_template_2)

    report = builder.close()
    assert report["prefix_duplicates_removed"] >= 4
    assert report["near_duplicates_removed"] >= 1 or report["duplicate_block_count"] >= 1
