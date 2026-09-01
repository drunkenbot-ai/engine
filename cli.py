from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from .config import DatasetConfig, ModelConfig, TrainingConfig
from .coordinator import run_coordinator_api
from .coordinator.artifacts import create_job_artifact_bundle
from .contracts import BackendKind
from .contracts.jobs import RuntimeSpec, TrainingJobSpec
from .evaluation import evaluate_checkpoint, normalize_prompts
from .export import export_hf_microgpt_package
from .dataset_build import build_dataset
from .training_orchestrator import train_from_dataset
from .tokenizer import load_tokenizer


def prepare(args: argparse.Namespace) -> None:
    """Prepare a dataset from command-line arguments.

    Args:
        args: Parsed command-line arguments for the prepare command.
    """

    def print_progress(event: object) -> None:
        """Print a progress event in CLI-friendly form.

        Args:
            event: Progress dictionary or message.
        """

        if isinstance(event, dict):
            message = event.get("message")
            percent = event.get("percent")
            prefix = f"[{percent:>3}%] " if percent is not None else ""
            if message:
                print(prefix + str(message))
        else:
            print(event)

    config = DatasetConfig(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        context_length=args.context_length,
        validation_split=args.validation_split,
        lowercase=False,
        max_workers=args.max_workers,
        code_training_mode=args.code_training_mode,
        include_prose=not args.exclude_prose,
        include_source_code=not args.exclude_source_code,
        extract_code_blocks=not args.no_extract_code_blocks,
        preserve_indentation=not args.no_preserve_indentation,
        generate_instruction_samples=not args.no_instruction_samples,
        reasoning_sample_mode=args.reasoning_sample_mode,
        prepare_mode=args.prepare_mode,
        tokenizer_strategy=args.tokenizer_strategy,
        tokenizer_path=Path(args.tokenizer_path) if args.tokenizer_path else None,
        dataset_stage=args.dataset_stage,
        conversation_datasets=[item.strip() for item in args.conversation_datasets.split(",") if item.strip()],
        conversation_sample_limit=args.conversation_sample_limit,
        tool_call_dataset_paths=[Path(path) for path in args.tool_call_dataset.split(",") if path.strip()],
        fast_scan_mode=args.fast_scan_mode,
        fast_scan_sample_bytes=args.fast_scan_sample_bytes,
        strict_duplicate_verification=args.strict_duplicate_verification,
    )
    result = build_dataset(config, progress=print_progress)
    print(
        f"Documents: {result.document_count} | Characters: {result.character_count} | "
        f"Tokens: {result.token_count} | Vocab: {result.vocab_size}"
    )
    print(f"Cache: reused {result.cached_file_count} file(s) | processed {result.processed_file_count} file(s)")


def train(args: argparse.Namespace) -> None:
    """Train a model from command-line arguments.

    Args:
        args: Parsed command-line arguments for the train command.
    """

    data_dir = Path(args.data_dir)
    tokenizer = load_tokenizer(data_dir / "tokenizer.json")

    model_config = ModelConfig(
        vocab_size=tokenizer.get_vocab_size(),
        context_length=args.context_length,
        embedding_size=args.embedding_size,
        head_count=args.head_count,
        layer_count=args.layer_count,
        dropout=args.dropout,
        norm_type=args.norm_type,
        position_encoding=args.position_encoding,
        mlp_type=args.mlp_type,
        rope_theta=args.rope_theta,
    )
    training_config = TrainingConfig(
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gradient_accumulation=args.gradient_accumulation,
        sample_stride=args.sample_stride,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        use_amp=args.use_amp,
        device=args.device,
        resume=not args.no_resume,
        resume_from_checkpoint=Path(args.resume_checkpoint) if args.resume_checkpoint else None,
        require_compatible_resume=not args.no_resume_safety,
    )
    result = train_from_dataset(data_dir, model_config, training_config)
    print(f"Saved model: {result.checkpoint_path}")
    print(f"Saved summary: {result.summary_path}")


def benchmark(args: argparse.Namespace) -> None:
    """Run benchmark prompts against a trained checkpoint.

    Args:
        args: Parsed command-line arguments for the benchmark command.
    """

    prompts = normalize_prompts(Path(args.prompts_file).read_text(encoding="utf-8") if args.prompts_file else args.prompts)
    result = evaluate_checkpoint(
        Path(args.model_dir),
        prompts,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=args.device,
        use_kv_cache=not args.no_kv_cache,
    )
    print(f"Benchmark prompts: {result.prompt_count}")
    print(f"Benchmark time: {result.total_seconds:.2f}s")
    print(f"Saved benchmark: {result.output_path}")


def export_hf(args: argparse.Namespace) -> None:
    """Export a MicroGPT checkpoint as an HF-style package.

    Args:
        args: Parsed command-line arguments for the export-hf command.
    """

    output = export_hf_microgpt_package(
        Path(args.model_dir),
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
    print(f"Saved HF-style MicroGPT package: {output}")


def coordinator_server(args: argparse.Namespace) -> None:
    """Run the coordinator HTTP API server.

    Args:
        args: Parsed command-line arguments for the coordinator command.
    """

    print(f"Coordinator API listening on http://{args.host}:{args.port}")
    print(f"Artifact root: {args.artifact_root}")
    run_coordinator_api(args.host, args.port, Path(args.artifact_root) if args.artifact_root else None)


def create_job_bundle(args: argparse.Namespace) -> None:
    """Create a portable job artifact bundle.

    Args:
        args: Parsed command-line arguments for the bundle command.
    """

    tokenizer = load_tokenizer(Path(args.dataset_dir) / "tokenizer.json")
    model_config = ModelConfig(
        vocab_size=tokenizer.get_vocab_size(),
        context_length=args.context_length,
        embedding_size=args.embedding_size,
        head_count=args.head_count,
        layer_count=args.layer_count,
        dropout=args.dropout,
    )
    training_config = TrainingConfig(
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        sample_stride=args.sample_stride,
        device=args.device,
    )
    job = TrainingJobSpec.local(Path(args.dataset_dir), model_config, training_config)
    if args.tags is None:
        default_tag = "gpu" if str(args.device).lower().startswith("cuda") else "cpu"
        tags = [default_tag]
    else:
        tags = [item.strip() for item in args.tags.split(",") if item.strip()]
    job.runtime = RuntimeSpec(
        backend=BackendKind.REMOTE_CLIENT,
        device=args.device,
        min_vram_gb=args.min_vram_gb,
        tags=tags,
    )
    bundle = create_job_artifact_bundle(
        job,
        artifact_root=Path(args.artifact_root) if args.artifact_root else None,
        base_url=args.base_url,
    )
    print(f"Created artifact bundle: {bundle}")
    print(f"Bundle URL: {job.metadata['artifact_bundle_url']}")
    print("Job JSON:")
    print(json.dumps(job.to_jsonable(), indent=2))


def worker_client(args: argparse.Namespace) -> None:
    """Run a remote worker client.

    Args:
        args: Parsed command-line arguments for the worker client command.
    """

    from .worker import WorkerClientConfig, run_worker_client

    labels = [item.strip() for item in args.labels.split(",") if item.strip()]
    config = WorkerClientConfig(
        coordinator_url=args.coordinator_url,
        worker_id=args.worker_id,
        device=args.device,
        labels=labels,
        heartbeat_interval_seconds=args.heartbeat_interval,
        execute_jobs=args.execute,
        claim_once=args.claim_once,
        workspace_dir=Path(args.workspace_dir),
    )
    print(f"Worker {config.worker_id} connecting to {config.coordinator_url}")
    run_worker_client(config)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser.

    Returns:
        Configured argument parser.
    """

    parser = argparse.ArgumentParser(description="Small LLM trainer backend")
    subparsers = parser.add_subparsers(required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Load documents and train tokenizer")
    prepare_parser.add_argument("--input_dir", required=True)
    prepare_parser.add_argument("--output_dir", required=True)
    prepare_parser.add_argument("--vocab_size", type=int, default=None)
    prepare_parser.add_argument("--min_frequency", type=int, default=2)
    prepare_parser.add_argument("--context_length", type=int, default=128)
    prepare_parser.add_argument("--validation_split", type=float, default=0.1)
    prepare_parser.add_argument("--max_workers", type=int, default=4)
    prepare_parser.add_argument("--code_training_mode", action="store_true")
    prepare_parser.add_argument("--exclude_prose", action="store_true")
    prepare_parser.add_argument("--exclude_source_code", action="store_true")
    prepare_parser.add_argument("--no_extract_code_blocks", action="store_true")
    prepare_parser.add_argument("--no_preserve_indentation", action="store_true")
    prepare_parser.add_argument("--no_instruction_samples", action="store_true")
    prepare_parser.add_argument(
        "--reasoning_sample_mode",
        choices=["none", "scaffold", "detailed"],
        default="scaffold",
    )
    prepare_parser.add_argument(
        "--prepare_mode",
        choices=["incremental", "full_rebuild", "force_reprocess"],
        default="incremental",
    )
    prepare_parser.add_argument(
        "--tokenizer_strategy",
        choices=["auto", "train_new", "reuse_dataset", "import_tokenizer"],
        default="auto",
    )
    prepare_parser.add_argument("--tokenizer_path", default=None)
    prepare_parser.add_argument(
        "--dataset_stage",
        choices=["base", "instruction", "conversation", "code", "tool_call"],
        default="base",
        help="Dataset purpose, including local tool-call fine-tune examples.",
    )
    prepare_parser.add_argument(
        "--tool_call_dataset",
        default="",
        help="Comma-separated JSON/JSONL paths containing OpenAI-style tool-call examples.",
    )
    prepare_parser.add_argument(
        "--conversation_datasets",
        default="",
        help="Comma-separated built-in online dataset IDs. TinyStories is for base; chat/instruction sets are for fine-tuning.",
    )
    prepare_parser.add_argument("--conversation_sample_limit", type=int, default=20000)
    prepare_parser.add_argument(
        "--fast_scan_mode",
        action="store_true",
        help="Use cheaper dataset fingerprints and cached preview stats for faster large-corpus scans.",
    )
    prepare_parser.add_argument(
        "--fast_scan_sample_bytes",
        type=int,
        default=64 * 1024,
        help="Bytes sampled from file head/tail for fast fingerprints.",
    )
    prepare_parser.add_argument(
        "--strict_duplicate_verification",
        action="store_true",
        help="In fast scan mode, re-hash only suspected duplicate groups with full SHA-256.",
    )
    prepare_parser.set_defaults(func=prepare)

    train_parser = subparsers.add_parser("train", help="Train a MicroGPT model")
    train_parser.add_argument("--data_dir", required=True)
    train_parser.add_argument("--output_dir", required=True)
    train_parser.add_argument("--epochs", type=int, default=5)
    train_parser.add_argument("--batch_size", type=int, default=16)
    train_parser.add_argument("--context_length", type=int, default=128)
    train_parser.add_argument("--embedding_size", type=int, default=256)
    train_parser.add_argument("--head_count", type=int, default=4)
    train_parser.add_argument("--layer_count", type=int, default=4)
    train_parser.add_argument("--dropout", type=float, default=0.1)
    train_parser.add_argument("--norm_type", choices=["layernorm", "rmsnorm"], default="layernorm")
    train_parser.add_argument("--position_encoding", choices=["learned", "rope"], default="learned")
    train_parser.add_argument("--mlp_type", choices=["gelu", "swiglu"], default="gelu")
    train_parser.add_argument("--rope_theta", type=float, default=10000.0)
    train_parser.add_argument("--learning_rate", type=float, default=3e-4)
    train_parser.add_argument("--gradient_accumulation", type=int, default=1)
    train_parser.add_argument("--sample_stride", type=int, default=1)
    train_parser.add_argument("--eval_interval", type=int, default=100)
    train_parser.add_argument("--save_interval", type=int, default=500)
    train_parser.add_argument("--use_amp", action="store_true")
    train_parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    train_parser.add_argument("--no_resume", action="store_true")
    train_parser.add_argument("--resume_checkpoint", default=None)
    train_parser.add_argument("--no_resume_safety", action="store_true")
    train_parser.set_defaults(func=train)

    benchmark_parser = subparsers.add_parser("benchmark", help="Run fixed prompts against a trained model")
    benchmark_parser.add_argument("--model_dir", required=True)
    benchmark_parser.add_argument("--prompts", default="")
    benchmark_parser.add_argument("--prompts_file", default=None)
    benchmark_parser.add_argument("--output_dir", default=None)
    benchmark_parser.add_argument("--max_new_tokens", type=int, default=128)
    benchmark_parser.add_argument("--temperature", type=float, default=0.7)
    benchmark_parser.add_argument("--top_k", type=int, default=50)
    benchmark_parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    benchmark_parser.add_argument("--no_kv_cache", action="store_true")
    benchmark_parser.set_defaults(func=benchmark)

    export_hf_parser = subparsers.add_parser("export-hf", help="Export a MicroGPT model as an HF-style package")
    export_hf_parser.add_argument("--model_dir", required=True)
    export_hf_parser.add_argument("--output_dir", default=None)
    export_hf_parser.set_defaults(func=export_hf)

    coordinator_parser = subparsers.add_parser("coordinator-server", help="Run the distributed training coordinator API")
    coordinator_parser.add_argument("--host", default="127.0.0.1")
    coordinator_parser.add_argument("--port", type=int, default=8765)
    coordinator_parser.add_argument("--artifact-root", default=None)
    coordinator_parser.set_defaults(func=coordinator_server)

    bundle_parser = subparsers.add_parser("create-job-bundle", help="Create a remote-worker dataset artifact bundle")
    bundle_parser.add_argument("--dataset-dir", required=True)
    bundle_parser.add_argument("--output-dir", required=True)
    bundle_parser.add_argument("--artifact-root", default=None)
    bundle_parser.add_argument("--base-url", default="/artifacts")
    bundle_parser.add_argument("--epochs", type=int, default=5)
    bundle_parser.add_argument("--batch-size", type=int, default=16)
    bundle_parser.add_argument("--context-length", type=int, default=128)
    bundle_parser.add_argument("--embedding-size", type=int, default=256)
    bundle_parser.add_argument("--head-count", type=int, default=4)
    bundle_parser.add_argument("--layer-count", type=int, default=4)
    bundle_parser.add_argument("--dropout", type=float, default=0.1)
    bundle_parser.add_argument("--learning-rate", type=float, default=3e-4)
    bundle_parser.add_argument("--sample-stride", type=int, default=1)
    bundle_parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    bundle_parser.add_argument("--min-vram-gb", type=float, default=None)
    bundle_parser.add_argument("--tags", default=None)
    bundle_parser.set_defaults(func=create_job_bundle)

    worker_parser = subparsers.add_parser("worker-client", help="Run a remote training worker client")
    worker_parser.add_argument("--coordinator-url", default="http://127.0.0.1:8765")
    worker_parser.add_argument("--worker-id", default=f"worker-{os.getpid()}")
    worker_parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    worker_parser.add_argument("--labels", default="gpu" if torch.cuda.is_available() else "cpu")
    worker_parser.add_argument("--heartbeat-interval", type=int, default=10)
    worker_parser.add_argument(
        "--workspace-dir",
        default=str(Path.home() / ".drunkenbot_ide" / "worker_workspace"),
    )
    worker_parser.add_argument("--claim-once", action="store_true")
    worker_parser.add_argument("--execute", action="store_true")
    worker_parser.set_defaults(func=worker_client)
    return parser


def main() -> None:
    """Run the command-line interface."""

    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
