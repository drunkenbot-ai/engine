from __future__ import annotations

from pathlib import Path

from engine.config import ModelConfig, TrainingConfig
from engine.training_impl import train_model
from engine.training_telemetry import TelemetryCadence


def test_cadence_selects_each_expensive_category_independently() -> None:
    now = [0.0]
    cadence = TelemetryCadence(1.0, 5.0, 10.0, clock=lambda: now[0])

    first = cadence.sample()
    now[0] = 0.5
    unsampled = cadence.sample()
    now[0] = 1.0
    lightweight = cadence.sample()
    now[0] = 5.0
    stability = cadence.sample()
    now[0] = 10.0
    preview = cadence.sample()

    assert (first.metrics, first.stability, first.preview) == (True, True, True)
    assert not any((unsampled.metrics, unsampled.stability, unsampled.preview))
    assert (lightweight.metrics, lightweight.stability, lightweight.preview) == (
        True,
        False,
        False,
    )
    assert (stability.metrics, stability.stability, stability.preview) == (
        True,
        True,
        False,
    )
    assert (preview.metrics, preview.stability, preview.preview) == (True, True, True)


def test_unsampled_optimizer_steps_skip_weight_norm_and_preview(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model_config = ModelConfig(
        vocab_size=16,
        context_length=8,
        embedding_size=8,
        head_count=2,
        layer_count=1,
        dropout=0.0,
    )
    training_config = TrainingConfig(
        output_dir=tmp_path,
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
        resume=False,
        early_stopping=False,
        telemetry_interval_seconds=3600.0,
        stability_metrics_interval_seconds=3600.0,
        preview_interval_seconds=3600.0,
    )
    calls = {"weight_norm": 0, "preview": 0}

    def weight_norm(_model) -> float:
        calls["weight_norm"] += 1
        return 1.0

    def preview(_tokens: list[int]) -> str:
        calls["preview"] += 1
        return "sample"

    monkeypatch.setattr("engine.training_impl._model_weight_norm", weight_norm)
    events: list[dict] = []
    result = train_model(
        model_config,
        training_config,
        [index % 16 for index in range(48)],
        [],
        pad_token_id=-1,
        progress=events.append,
        decode_preview=preview,
    )

    metric_events = [event for event in events if event.get("event_type") == "metrics"]
    assert len(metric_events) == 1
    assert metric_events[0]["step"] < 5
    assert metric_events[0]["step_seconds"] >= metric_events[0]["average_step_seconds"]
    assert calls == {"weight_norm": 1, "preview": 1}
    assert result.summary_path.exists()


def test_training_without_progress_skips_telemetry_sampling(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model_config = ModelConfig(
        vocab_size=16,
        context_length=8,
        embedding_size=8,
        head_count=2,
        layer_count=1,
        dropout=0.0,
    )
    training_config = TrainingConfig(
        output_dir=tmp_path,
        epochs=1,
        batch_size=1,
        sample_stride=8,
        warmup_steps=0,
        eval_interval=0,
        save_interval=0,
        use_amp=False,
        precision="fp32",
        device="cpu",
        resume=False,
        early_stopping=False,
    )
    calls = {"weight_norm": 0, "cpu": 0, "ram": 0}

    def counted(name, value):
        def sample(*_args):
            calls[name] += 1
            return value

        return sample

    monkeypatch.setattr(
        "engine.training_impl._model_weight_norm",
        counted("weight_norm", 1.0),
    )
    monkeypatch.setattr(
        "engine.training_impl.system_cpu_percent",
        counted("cpu", 1.0),
    )
    monkeypatch.setattr(
        "engine.training_impl.system_ram_percent",
        counted("ram", 1.0),
    )
    train_model(
        model_config,
        training_config,
        [index % 16 for index in range(32)],
        [],
        pad_token_id=-1,
        progress=None,
        decode_preview=lambda _tokens: "unused",
    )

    assert calls == {"weight_norm": 0, "cpu": 0, "ram": 0}


def test_stop_requested_during_validation_exports_stopped_checkpoint(
    tmp_path: Path,
) -> None:
    model_config = ModelConfig(
        vocab_size=16,
        context_length=8,
        embedding_size=8,
        head_count=2,
        layer_count=1,
        dropout=0.0,
    )
    training_config = TrainingConfig(
        output_dir=tmp_path,
        epochs=1,
        batch_size=1,
        sample_stride=8,
        warmup_steps=0,
        eval_interval=1,
        max_eval_batches=2,
        save_interval=0,
        use_amp=False,
        precision="fp32",
        device="cpu",
        resume=False,
        early_stopping=False,
    )
    stop_checks = 0
    events: list[dict] = []

    def should_stop() -> bool:
        nonlocal stop_checks
        stop_checks += 1
        return stop_checks >= 2

    result = train_model(
        model_config,
        training_config,
        [index % 16 for index in range(32)],
        [index % 16 for index in range(24)],
        pad_token_id=-1,
        progress=events.append,
        should_stop=should_stop,
    )

    assert result.stopped
    assert result.checkpoint_path.exists()
    assert events[-1]["event_type"] == "stop"
