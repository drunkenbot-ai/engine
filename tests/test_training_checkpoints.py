from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from engine.config import ModelConfig, TrainingConfig, dataclass_to_jsonable
from engine.microgpt_chat import MicroGPTChatSession
from engine.model import (
    LoRALinear,
    MicroGPT,
    apply_lora_adapters,
    merged_lora_state_dict,
)
from engine.training_impl import train_model


def _model_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=16,
        context_length=8,
        embedding_size=8,
        head_count=2,
        layer_count=1,
        dropout=0.0,
    )


def test_merged_lora_state_does_not_mutate_training_model() -> None:
    config = _model_config()
    model = MicroGPT(config)
    assert apply_lora_adapters(model, 2, 4.0, 0.0, "attention") > 0
    adapters = [module for module in model.modules() if isinstance(module, LoRALinear)]
    with torch.no_grad():
        for module in adapters:
            module.lora_b.fill_(0.1)

    merged = merged_lora_state_dict(model)

    assert adapters
    assert all(isinstance(module, LoRALinear) for module in adapters)
    assert not any(".lora_" in name or ".base." in name for name in merged)
    MicroGPT(config).load_state_dict(merged)


def test_lora_best_checkpoint_is_recommended_and_chat_loadable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _model_config()
    base_path = tmp_path / "base.pt"
    torch.save(
        {
            "model_config": dataclass_to_jsonable(config),
            "model_state_dict": MicroGPT(config).state_dict(),
        },
        base_path,
    )
    training = TrainingConfig(
        output_dir=tmp_path / "model",
        epochs=1,
        batch_size=1,
        learning_rate=1e-3,
        gradient_accumulation=1,
        sample_stride=8,
        warmup_steps=0,
        eval_interval=0,
        save_interval=0,
        use_amp=False,
        precision="fp32",
        device="cpu",
        training_mode="fine_tune",
        fine_tune_from_checkpoint=base_path,
        peft_method="lora",
        lora_rank=2,
        lora_alpha=4.0,
        lora_dropout=0.0,
        resume=False,
        early_stopping=False,
    )
    result = train_model(
        config,
        training,
        [index % config.vocab_size for index in range(48)],
        [index % config.vocab_size for index in range(24)],
        pad_token_id=-1,
    )
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    recommended = Path(summary["recommended_checkpoint_path"])
    resume_path = Path(summary["best_resume_checkpoint_path"])
    payload = torch.load(recommended, map_location="cpu")
    resume_payload = torch.load(resume_path, map_location="cpu")

    assert recommended.name == "checkpoint_best_val.pt"
    assert payload["artifact_type"] == "inference"
    assert payload["model_config"]
    assert payload["model_state_dict"]
    assert "adapter_state_dict" not in payload
    assert resume_payload["artifact_type"] == "resume"
    assert resume_payload["adapter_state_dict"]
    assert resume_payload["optimizer_state_dict"]

    tokenizer_path = training.output_dir / "tokenizer.json"
    tokenizer_path.write_text("{}", encoding="utf-8")

    class _Tokenizer:
        def token_to_id(self, _token: str) -> int:
            return 0

    monkeypatch.setattr(
        "engine.microgpt_chat.load_tokenizer",
        lambda _path: _Tokenizer(),
    )
    session = MicroGPTChatSession(recommended, device="cpu")
    assert isinstance(session.model, MicroGPT)

    with pytest.raises(ValueError, match="inference artifact"):
        train_model(
            config,
            replace(
                training,
                resume=True,
                resume_from_checkpoint=recommended,
            ),
            [index % config.vocab_size for index in range(48)],
            [index % config.vocab_size for index in range(24)],
            pad_token_id=-1,
        )
