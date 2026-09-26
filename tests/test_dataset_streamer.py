from pathlib import Path
from engine.dataset_streamer import PartitionedJsonlWriter, stream_dataset_to_partitions


def test_partitioned_jsonl_writer_respects_byte_limit(tmp_path: Path) -> None:
    # Use a small 10KB max_bytes limit to test rotation
    writer = PartitionedJsonlWriter(tmp_path, prefix="test_part", max_bytes=10 * 1024)

    # Write 50 records of ~500 bytes each
    sample_text = "The quick brown fox jumps over the lazy dog. " * 10
    for i in range(50):
        writer.write_record({"text": f"{sample_text} [Record {i}]", "meta": {"id": i}})

    manifest = writer.close()
    assert manifest["total_records"] == 50
    assert manifest["partitions_count"] >= 2
    assert (tmp_path / "manifest.json").exists()

    # Check each partition file
    for part_name in manifest["partition_files"]:
        part_file = tmp_path / part_name
        assert part_file.exists()
        assert part_file.stat().st_size <= 15 * 1024  # within bounded threshold


def test_stream_dataset_to_partitions_filters_and_deduplicates(tmp_path: Path) -> None:
    def sample_generator():
        # Yield unique records
        yield {"text": "Quantum entanglement occurs when a group of particles interact."}
        yield {"text": "General relativity describes gravity as the curvature of spacetime."}
        # Yield duplicate prefix
        yield {"text": "Quantum entanglement occurs when a group of particles interact and do something else."}
        # Yield short record
        yield {"text": "too short"}

    manifest = stream_dataset_to_partitions(
        records_iterable=sample_generator(),
        output_dir=tmp_path / "out",
        prefix="curated",
        target_tokens=1000,
    )
    assert manifest["total_records"] == 2
