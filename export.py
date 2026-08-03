from __future__ import annotations

import shutil
import subprocess
import json
from pathlib import Path
from typing import Optional

import torch

from .tokenizer import DEFAULT_CHAT_TEMPLATE


class ExportError(RuntimeError):
    """Raised when model export or quantization cannot be completed."""

    pass


def export_project_bundle(project_dir: Path, output_dir: Path) -> Path:
    """Create a portable model bundle.

    Args:
        project_dir: Trained model project folder.
        output_dir: Destination export folder.

    Returns:
        Export folder path.

    Raises:
        ExportError: If required model artifacts are missing.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    required = ["final_model.pt", "tokenizer.json", "training_summary.json"]
    for name in required:
        source = project_dir / name
        if not source.exists():
            raise ExportError(f"Missing required file for export: {source}")
        shutil.copy2(source, output_dir / name)
    for metadata_name in ("tokenizer_config.json", "special_tokens_map.json"):
        source = project_dir / metadata_name
        if source.exists():
            shutil.copy2(source, output_dir / metadata_name)
    optional_files = ["model_lineage.json", "dataset_summary.json"]
    for name in optional_files:
        source = project_dir / name
        if source.exists():
            shutil.copy2(source, output_dir / name)
    benchmarks = project_dir / "benchmarks"
    if benchmarks.exists():
        shutil.copytree(benchmarks, output_dir / "benchmarks", dirs_exist_ok=True)
    manifest = {
        "schema": "micro_llm_export_bundle",
        "project_dir": str(project_dir),
        "output_dir": str(output_dir),
        "files": sorted(path.name for path in output_dir.iterdir()),
    }
    (output_dir / "export_summary.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output_dir


def quantize_checkpoint(checkpoint_path: Path, output_path: Path, mode: str = "fp16") -> Path:
    """Create a smaller inference checkpoint.

    Args:
        checkpoint_path: Source PyTorch checkpoint path.
        output_path: Destination quantized checkpoint path.
        mode: Quantization mode. Currently only ``fp16`` is supported.

    Returns:
        Quantized checkpoint path.

    Raises:
        ExportError: If the checkpoint is missing or mode is unsupported.
    """
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    if not checkpoint_path.exists():
        raise ExportError(f"Checkpoint not found: {checkpoint_path}")

    mode = mode.lower()
    if mode not in {"fp16", "float16"}:
        raise ExportError("Only FP16 checkpoint quantization is currently supported for MicroGPT.")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict")
    if not state_dict:
        raise ExportError("Checkpoint does not contain model_state_dict.")

    checkpoint["model_state_dict"] = {
        key: value.half() if torch.is_floating_point(value) else value
        for key, value in state_dict.items()
    }
    checkpoint["quantization"] = {
        "mode": "fp16",
        "source": str(checkpoint_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    return output_path


def export_hf_microgpt_package(project_dir: Path, output_dir: Optional[Path] = None) -> Path:
    """Export a MicroGPT checkpoint as an HF-style local model package.

    Args:
        project_dir: Trained model project folder.
        output_dir: Optional destination folder. Defaults to ``project_dir/hf_model``.

    Returns:
        HF-style package folder.

    Raises:
        ExportError: If required files are missing.
    """

    project_dir = Path(project_dir)
    output_dir = Path(output_dir) if output_dir else project_dir / "hf_model"
    checkpoint_path = project_dir / "final_model.pt"
    tokenizer_path = project_dir / "tokenizer.json"
    summary_path = project_dir / "training_summary.json"
    for path in (checkpoint_path, tokenizer_path, summary_path):
        if not path.exists():
            raise ExportError(f"Missing required file for HF package: {path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_config = checkpoint.get("model_config")
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(model_config, dict) or not state_dict:
        raise ExportError("Checkpoint must contain model_config and model_state_dict.")

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, output_dir / "pytorch_model.bin")
    shutil.copy2(tokenizer_path, output_dir / "tokenizer.json")
    shutil.copy2(summary_path, output_dir / "training_summary.json")
    for optional in ("model_lineage.json", "dataset_summary.json"):
        source = project_dir / optional
        if source.exists():
            shutil.copy2(source, output_dir / optional)

    config = {
        "model_type": "microgpt",
        "architectures": ["MicroGPTForCausalLM"],
        "library_name": "micro-llm-creator",
        "llama_cpp_convertible": False,
        "vocab_size": model_config.get("vocab_size"),
        "n_positions": model_config.get("context_length"),
        "n_ctx": model_config.get("context_length"),
        "n_embd": model_config.get("embedding_size"),
        "n_head": model_config.get("head_count"),
        "n_layer": model_config.get("layer_count"),
        "dropout": model_config.get("dropout"),
        "bias": model_config.get("bias"),
        "norm_type": model_config.get("norm_type", "layernorm"),
        "position_encoding": model_config.get("position_encoding", "learned"),
        "mlp_type": model_config.get("mlp_type", "gelu"),
        "rope_theta": model_config.get("rope_theta", 10000.0),
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output_dir / "generation_config.json").write_text(
        json.dumps(
            {
                "max_new_tokens": 128,
                "temperature": 0.7,
                "top_k": 50,
                "do_sample": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "special_tokens_map.json").write_text(
        json.dumps(
            {
                "bos_token": "<bos>",
                "eos_token": "<eos>",
                "unk_token": "<unk>",
                "pad_token": "<pad>",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "tokenizer_file": "tokenizer.json",
                "bos_token": "<bos>",
                "eos_token": "<eos>",
                "unk_token": "<unk>",
                "pad_token": "<pad>",
                "model_max_length": model_config.get("context_length"),
                "chat_template": DEFAULT_CHAT_TEMPLATE,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(_hf_readme(config), encoding="utf-8")
    return output_dir


def export_llama_adapter_package(project_dir: Path, output_dir: Optional[Path] = None) -> Path:
    """Export a Llama-compatible state dict when the trained architecture matches Llama.

    This is a real tensor-name/layout adapter, not a relabelled MicroGPT
    checkpoint.  Classic-GPT models are rejected because their learned
    positions, LayerNorm, or GELU MLP cannot be represented faithfully by
    Llama/Qwen/Mistral loaders.
    """

    project_dir = Path(project_dir)
    output_dir = Path(output_dir) if output_dir else project_dir / "llama_model"
    checkpoint_path = project_dir / "final_model.pt"
    tokenizer_path = project_dir / "tokenizer.json"
    if not checkpoint_path.exists() or not tokenizer_path.exists():
        raise ExportError("Llama export requires final_model.pt and tokenizer.json.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("model_config")
    state = checkpoint.get("model_state_dict")
    if not isinstance(config, dict) or not isinstance(state, dict):
        raise ExportError("Checkpoint must contain model_config and model_state_dict.")
    required = {"norm_type": "rmsnorm", "position_encoding": "rope", "mlp_type": "swiglu"}
    incompatible = [f"{key}={config.get(key)!r}" for key, expected in required.items() if config.get(key) != expected]
    if config.get("bias", True):
        incompatible.append("bias=True")
    if int(config.get("attention_window", 0) or 0) != 0:
        incompatible.append("attention_window is enabled")
    if incompatible:
        raise ExportError(
            "Llama adapter requires the Llama-like architecture (RoPE, RMSNorm, SwiGLU, no bias, full attention). "
            "This checkpoint is incompatible: " + ", ".join(incompatible)
        )
    hidden = int(config["embedding_size"])
    heads = int(config["head_count"])
    kv_heads = int(config.get("kv_head_count") or heads)
    if config.get("attention_type") == "mqa":
        kv_heads = 1
    elif config.get("attention_type") == "gqa" and not config.get("kv_head_count"):
        kv_heads = max(1, heads // 2)
    head_dim = hidden // heads
    kv_hidden = kv_heads * head_dim
    adapted: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": state["token_embedding.weight"],
        "model.norm.weight": state["ln_f.weight"],
        "lm_head.weight": state["lm_head.weight"],
    }
    for layer in range(int(config["layer_count"])):
        source = f"blocks.{layer}"
        target = f"model.layers.{layer}"
        qkv = state[f"{source}.attn.c_attn.weight"]
        q, k, v = qkv.split((hidden, kv_hidden, kv_hidden), dim=0)
        adapted.update({
            f"{target}.input_layernorm.weight": state[f"{source}.ln_1.weight"],
            f"{target}.self_attn.q_proj.weight": q,
            f"{target}.self_attn.k_proj.weight": k,
            f"{target}.self_attn.v_proj.weight": v,
            f"{target}.self_attn.o_proj.weight": state[f"{source}.attn.c_proj.weight"],
            f"{target}.post_attention_layernorm.weight": state[f"{source}.ln_2.weight"],
            f"{target}.mlp.gate_proj.weight": state[f"{source}.mlp.w1.weight"],
            f"{target}.mlp.down_proj.weight": state[f"{source}.mlp.w2.weight"],
            f"{target}.mlp.up_proj.weight": state[f"{source}.mlp.w3.weight"],
        })
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(adapted, output_dir / "pytorch_model.bin")
    shutil.copy2(tokenizer_path, output_dir / "tokenizer.json")
    for name in ("tokenizer_config.json", "special_tokens_map.json"):
        source = project_dir / name
        if source.exists():
            shutil.copy2(source, output_dir / name)
    llama_config = {
        "model_type": "llama", "architectures": ["LlamaForCausalLM"],
        "vocab_size": int(config["vocab_size"]), "hidden_size": hidden,
        "intermediate_size": hidden * 4, "num_hidden_layers": int(config["layer_count"]),
        "num_attention_heads": heads, "num_key_value_heads": kv_heads,
        "max_position_embeddings": int(config["context_length"]),
        "rope_theta": float(config.get("rope_theta", 10000.0)),
        "rms_norm_eps": 1e-5, "hidden_act": "silu", "tie_word_embeddings": True,
        "bos_token_id": 2, "eos_token_id": 3, "pad_token_id": 0,
    }
    (output_dir / "config.json").write_text(json.dumps(llama_config, indent=2), encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# Llama-compatible MicroGPT export\n\n"
        "Weights were structurally adapted from a RoPE/RMSNorm/SwiGLU MicroGPT checkpoint. "
        "Load with Transformers `LlamaForCausalLM` or compatible Llama harnesses.\n",
        encoding="utf-8",
    )
    return output_dir


def _hf_readme(config: dict[str, object]) -> str:
    """Create README text for a MicroGPT HF-style package.

    Args:
        config: Exported config dictionary.

    Returns:
        README Markdown.
    """

    return (
        "# MicroGPT HF-Style Package\n\n"
        "This folder was exported by Micro LLM Creator. It uses a Hugging Face-style "
        "layout (`config.json`, `pytorch_model.bin`, `tokenizer.json`) for portability, "
        "but `model_type` is `microgpt` and it is not directly convertible by llama.cpp as "
        "a Llama/Mistral/Gemma model.\n\n"
        "Load it with Micro LLM Creator's MicroGPT code, or build a custom Transformers "
        "model class that understands this config and tensor naming.\n\n"
        f"- Block style: {config.get('norm_type')} / {config.get('position_encoding')} / {config.get('mlp_type')}\n"
        f"- Context length: {config.get('n_ctx')}\n"
        f"- Embedding size: {config.get('n_embd')}\n"
        f"- Layers: {config.get('n_layer')}\n"
        f"- Heads: {config.get('n_head')}\n"
    )


def find_llama_cpp_converter(llama_cpp_dir: Path) -> Path:
    """Find the llama.cpp Hugging Face to GGUF converter.

    Args:
        llama_cpp_dir: llama.cpp checkout folder or direct converter path.

    Returns:
        Path to the converter script.

    Raises:
        ExportError: If no converter is found.
    """

    llama_cpp_dir = Path(llama_cpp_dir)
    if llama_cpp_dir.is_file() and llama_cpp_dir.name == "convert_hf_to_gguf.py":
        return llama_cpp_dir
    if not str(llama_cpp_dir).strip() or str(llama_cpp_dir) == ".":
        raise ExportError(
            "Choose your local llama.cpp folder first. It should contain convert_hf_to_gguf.py."
        )
    if not llama_cpp_dir.exists():
        raise ExportError(f"llama.cpp path does not exist: {llama_cpp_dir}")
    if not llama_cpp_dir.is_dir():
        raise ExportError(f"llama.cpp path is not a folder: {llama_cpp_dir}")

    candidates = [
        llama_cpp_dir / "convert_hf_to_gguf.py",
        llama_cpp_dir / "convert" / "convert_hf_to_gguf.py",
        llama_cpp_dir / "examples" / "convert_hf_to_gguf.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = sorted(llama_cpp_dir.rglob("convert_hf_to_gguf.py"))
    if matches:
        return matches[0]

    searched = "\n".join(f"- {candidate}" for candidate in candidates)
    raise ExportError(
        "Could not find llama.cpp converter script.\n"
        "Expected a recent llama.cpp checkout containing convert_hf_to_gguf.py.\n"
        f"Searched:\n{searched}"
    )


def export_gguf_with_llama_cpp(project_dir: Path, llama_cpp_dir: Path, output_path: Path, outtype: str = "f16") -> Path:
    """Export a Hugging Face-compatible model through llama.cpp.

    Args:
        project_dir: Model project containing an ``hf_model`` folder.
        llama_cpp_dir: Local llama.cpp checkout folder.
        output_path: Destination GGUF file path.
        outtype: llama.cpp converter output type, usually f16 or f32.

    Returns:
        GGUF output path.

    Raises:
        ExportError: If converter or HF model folder is missing.
    """
    converter = find_llama_cpp_converter(llama_cpp_dir)

    hf_dir = project_dir / "hf_model"
    if not hf_dir.exists():
        raise ExportError(
            "GGUF export needs model_core/hf_model. Use Export HF Package first, "
            "but note MicroGPT packages are not llama.cpp-convertible unless llama.cpp "
            "has a matching MicroGPT converter/model implementation."
        )
    config_path = hf_dir / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("model_type") == "microgpt":
            raise ExportError(
                "This hf_model folder is a MicroGPT package, not a llama.cpp-supported "
                "Llama/Mistral/Gemma model. Real GGUF export needs a supported HF model "
                "architecture or a custom llama.cpp MicroGPT converter."
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if outtype not in {"f16", "f32", "bf16", "q8_0"}:
        raise ExportError(f"Unsupported GGUF outtype for converter: {outtype}")

    subprocess.run(
        ["python", str(converter), str(hf_dir), "--outfile", str(output_path), "--outtype", outtype],
        check=True,
    )
    return output_path

