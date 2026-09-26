import json
import tempfile
from pathlib import Path

from engine.dataset_scrubber import CorpusScrubber, scrub_jsonl_directory


def test_corpus_scrubber_boilerplate_and_minhash():
    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td) / "out"
        scrubber = CorpusScrubber(out_dir, prefix="test", max_prefix_repeats=2)

        # 1. Boilerplate prompt repeated 4 times:
        prompt = "You are a helpful, deep-thinking assistant. When presented with complex problems, think step by step. "
        for i in range(4):
            record = {"text": f"{prompt} Problem variation {i} detailed solution with mathematical derivations."}
            scrubber.scrub_record(record)

        # 2. Distinct document:
        distinct = {"text": "A completely different article about compiler optimization techniques in LLVM and Clang backends."}
        scrubber.scrub_record(distinct)

        # 3. Short record (< 60 chars):
        short = {"text": "Too short"}
        scrubber.scrub_record(short)

        report = scrubber.close()

        # Expected:
        # 2 of the 4 boilerplate records accepted (max_prefix_repeats=2), 2 dropped as prefix boilerplate
        # 1 distinct record accepted
        # 1 short record dropped
        assert report["total_scanned"] == 6
        assert report["prefix_boilerplate_dropped"] == 2
        assert report["near_duplicates_dropped"] == 1
        assert report["short_records_dropped"] == 1
        assert report["retained_records"] == 2


def test_scrub_jsonl_directory():
    with tempfile.TemporaryDirectory() as td:
        src_dir = Path(td) / "src"
        out_dir = Path(td) / "out"
        src_dir.mkdir(parents=True)

        topics = [
            "Quantum physics exploring wavefunction collapse in isolated atomic systems.",
            "Photosynthesis in deep marine phytoplankton under low incident light spectra.",
            "High throughput database indexes using B-trees and fractal cache hierarchies.",
            "History of maritime navigation techniques from early astrolabes to sextants.",
            "Atmospheric thermodynamics and condensation nuclei in cumulonimbus formation.",
        ]
        jsonl_file = src_dir / "sample.jsonl"
        with jsonl_file.open("w", encoding="utf-8") as f:
            for text in topics:
                line = json.dumps({"text": text})
                f.write(line + "\n")

        report = scrub_jsonl_directory(src_dir, out_dir, prefix="scrubbed")
        assert report["total_scanned"] == len(topics)
        assert report["retained_records"] == len(topics)
        assert (out_dir / "scrub_report.json").exists()
        assert len(report["writer_manifest"]["partition_files"]) >= 1
