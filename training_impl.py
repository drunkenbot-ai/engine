from __future__ import annotations
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Optional, Union
import numpy as np
import numpy.lib.format as npy_format
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from .config import ModelConfig, TrainingConfig, dataclass_to_jsonable
from .model import MicroGPT, apply_lora_adapters, freeze_non_lora_parameters, load_lora_state_dict, lora_parameter_count, lora_state_dict, merge_lora_adapters
try:
    import psutil
except ImportError:
    psutil = None
from .training_core import *
from .training_runtime import *
from .training_evaluation import *
from .training_resume import *

def train_model(model_config: ModelConfig, training_config: TrainingConfig, train_tokens: Union[list[int], np.ndarray], val_tokens: Union[list[int], np.ndarray], pad_token_id: int, progress: Optional[Callable[[Any], None]]=None, should_stop: Optional[Callable[[], bool]]=None, decode_preview: Optional[Callable[[list[int]], str]]=None) -> TrainingResult:
    """Train a MicroGPT model.
    Args:
        model_config: Architecture settings.
        training_config: Optimizer, device, and checkpoint settings.
        train_tokens: Training token stream.
        val_tokens: Validation token stream.
        pad_token_id: Token ID ignored by cross-entropy loss.
        progress: Optional callback receiving progress dictionaries.
        should_stop: Optional callback returning true when training should stop.
        decode_preview: Optional callback that decodes token IDs into a short text preview.
    Returns:
        Training result with checkpoint and summary paths.
    """
    model_config.validate()
    training_config.validate()
    set_seed(training_config.seed)
    if training_config.device.startswith('cuda') and torch.cuda.is_available():
        _configure_cuda_allocator()
    training_config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = training_config.output_dir / 'checkpoints'
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    emit_progress(progress, 'Building model...', 2)
    model = MicroGPT(model_config).to(training_config.device)
    model.enable_gradient_checkpointing(training_config.activation_checkpointing)
    if training_config.device.startswith('cuda') and torch.cuda.is_available():
        free_vram, total_vram = torch.cuda.mem_get_info()
        estimate = _estimated_training_vram_bytes(model, model_config, training_config)
        emit_progress(progress, f'VRAM preflight: estimated {estimate / 1024 ** 3:.2f} GB; currently free {free_vram / 1024 ** 3:.2f} GB of {total_vram / 1024 ** 3:.2f} GB.', 3, estimated_vram_gb=estimate / 1024 ** 3, free_vram_gb=free_vram / 1024 ** 3)
        if estimate > free_vram * 0.85:
            emit_progress(progress, '[WARN] Estimated training memory is close to available VRAM. Reduce micro-batch size, enable activation checkpointing, or use gradient accumulation.', 3)
    _release_cuda_cache()
    emit_progress(progress, 'Preparing token batches...', 4)
    loader_workers = max(0, int(training_config.data_loader_workers))
    pin_memory = training_config.device.startswith('cuda') and torch.cuda.is_available()
    loader_kwargs = {'num_workers': loader_workers, 'pin_memory': pin_memory, 'persistent_workers': loader_workers > 0}
    train_loader = DataLoader(TokenDataset(train_tokens, model_config.context_length, stride=training_config.sample_stride), batch_size=training_config.batch_size, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = None
    if len(val_tokens) > model_config.context_length:
        val_loader = DataLoader(TokenDataset(val_tokens, model_config.context_length), batch_size=training_config.batch_size, shuffle=False, drop_last=False, **loader_kwargs)
    global_step = 0
    start_epoch = 0
    final_train_loss = 0.0
    final_val_loss: Optional[float] = None
    best_val_loss: Optional[float] = None
    best_checkpoint_path: Optional[Path] = None
    early_stop_counter = 0
    early_stopped = False
    resume_path = training_config.resume_from_checkpoint if training_config.resume else None
    if resume_path is None and training_config.resume:
        resume_path = latest_checkpoint(checkpoints_dir)
    resume_checkpoint: Optional[dict[str, Any]] = None
    resume_compatibility: Optional[ResumeCompatibilityReport] = None
    if training_config.peft_method == 'lora':
        base_path = training_config.fine_tune_from_checkpoint
        if resume_path and Path(resume_path).exists():
            resume_checkpoint = torch.load(resume_path, map_location='cpu')
            checkpoint_base = resume_checkpoint.get('fine_tune_base_checkpoint')
            if checkpoint_base:
                base_path = Path(checkpoint_base)
        if base_path is None:
            raise ValueError('LoRA fine-tuning requires a base checkpoint.')
        base_path = Path(base_path)
        if not base_path.exists():
            raise FileNotFoundError(f'LoRA base checkpoint not found: {base_path}')
        emit_progress(progress, f'Loading LoRA base checkpoint: {base_path}', 5)
        base_checkpoint = torch.load(base_path, map_location='cpu')
        model.load_state_dict(base_checkpoint['model_state_dict'])
        wrapped = apply_lora_adapters(model, training_config.lora_rank, training_config.lora_alpha, training_config.lora_dropout, training_config.lora_target_modules)
        freeze_non_lora_parameters(model)
        emit_progress(progress, f'LoRA enabled: {wrapped} module(s), {lora_parameter_count(model):,} trainable adapter parameter(s).', 6)
    optimizer = make_optimizer(model, training_config)
    steps_per_epoch = max(math.ceil(len(train_loader) / training_config.gradient_accumulation), 1)
    total_steps = max(steps_per_epoch * training_config.epochs, 1)
    scheduler = make_scheduler(optimizer, total_steps, training_config)
    use_autocast, use_scaler, autocast_dtype = amp_settings(training_config)
    scaler = GradScaler('cuda', enabled=use_scaler)
    emit_progress(progress, f'Optimizer: {training_config.optimizer_name}, schedule: {training_config.scheduler_name}, precision: {training_config.precision}.', 5)
    if resume_path and Path(resume_path).exists():
        emit_progress(progress, f'Resuming from checkpoint: {resume_path}', 6)
        compatibility = resume_compatibility or check_resume_compatibility(Path(resume_path), model_config, training_config)
        for line in compatibility.info:
            emit_progress(progress, line, 6)
        for line in compatibility.warnings:
            emit_progress(progress, f'[WARN] {line}', 6)
        strict_resume_errors = list(compatibility.errors)
        if training_config.require_compatible_resume:
            if not compatibility.can_load_optimizer_state:
                strict_resume_errors.append('Safe resume requires matching optimizer state.')
            if not compatibility.can_load_scheduler_state:
                strict_resume_errors.append('Safe resume requires matching scheduler state.')
            if not compatibility.can_load_scaler_state:
                strict_resume_errors.append('Safe resume requires matching AMP scaler state.')
        if strict_resume_errors:
            message = 'Checkpoint is not compatible with the current training settings:\n' + '\n'.join((f'- {line}' for line in strict_resume_errors))
            raise ValueError(message)
        checkpoint = resume_checkpoint or torch.load(resume_path, map_location='cpu')
        if training_config.peft_method == 'lora' and 'adapter_state_dict' in checkpoint:
            load_lora_state_dict(model, checkpoint['adapter_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint and compatibility.can_load_optimizer_state:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scaler_state_dict' in checkpoint and use_scaler and compatibility.can_load_scaler_state:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        global_step = int(checkpoint.get('global_step', 0))
        start_epoch = min(int(checkpoint.get('epoch', 0)), training_config.epochs)
        final_train_loss = float(checkpoint.get('train_loss', 0.0))
        final_val_loss = checkpoint.get('val_loss')
        remaining_epochs = max(training_config.epochs - start_epoch, 1)
        total_steps = max(total_steps, global_step + steps_per_epoch * remaining_epochs, global_step + 1)
        scheduler = make_scheduler(optimizer, total_steps, training_config)
        if 'scheduler_state_dict' in checkpoint and compatibility.can_load_scheduler_state:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        emit_progress(progress, f'Checkpoint loaded at step {global_step}.', 8)
        _release_cuda_cache()
    elif training_config.training_mode == 'fine_tune' and training_config.fine_tune_from_checkpoint is not None and (training_config.peft_method != 'lora'):
        base_path = Path(training_config.fine_tune_from_checkpoint)
        if not base_path.exists():
            raise FileNotFoundError(f'Fine-tune base checkpoint not found: {base_path}')
        emit_progress(progress, f'Fine-tuning from base checkpoint: {base_path}', 6)
        compatibility = check_resume_compatibility(base_path, model_config, training_config)
        for line in compatibility.info:
            emit_progress(progress, line, 6)
        for line in compatibility.warnings:
            emit_progress(progress, f'[WARN] {line}', 6)
        if compatibility.errors:
            message = 'Fine-tune base checkpoint is not compatible with the current model settings:\n' + '\n'.join((f'- {line}' for line in compatibility.errors))
            raise ValueError(message)
        checkpoint = torch.load(base_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        emit_progress(progress, 'Base model weights loaded. Starting fresh fine-tune optimizer state.', 8)
    else:
        emit_progress(progress, 'Starting new training run.', 6)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    last_metric_time = perf_counter()
    step_time_window: list[float] = []
    for epoch in range(start_epoch, training_config.epochs):
        epoch_losses: list[float] = []
        epoch_batch_count = len(train_loader)
        for batch_index, (x, y) in enumerate(train_loader):
            if should_stop and should_stop():
                final_train_loss = sum(epoch_losses) / max(len(epoch_losses), 1) if epoch_losses else final_train_loss
                stopped_path = checkpoints_dir / f'checkpoint_stopped_step_{global_step}.pt'
                save_checkpoint(stopped_path, model, optimizer, scheduler, scaler, model_config, training_config, global_step, epoch, final_train_loss, final_val_loss)
                emit_progress(progress, f'Training stopped. Resume checkpoint saved: {stopped_path}', 100)
                summary_path = training_config.output_dir / 'training_summary.json'
                summary = {'model_config': dataclass_to_jsonable(model_config), 'training_config': dataclass_to_jsonable(training_config), 'final_train_loss': final_train_loss, 'final_val_loss': final_val_loss, 'total_steps': global_step, 'stopped': True, 'resume_checkpoint': str(stopped_path), 'parameters': sum((p.numel() for p in model.parameters()))}
                summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
                return TrainingResult(stopped_path, summary_path, final_train_loss, final_val_loss, stopped=True)
            x = x.to(training_config.device, non_blocking=pin_memory)
            y = y.to(training_config.device, non_blocking=pin_memory)
            with autocast('cuda', enabled=use_autocast, dtype=autocast_dtype):
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=pad_token_id)
                loss = loss / training_config.gradient_accumulation
            scaler.scale(loss).backward()
            should_step = (batch_index + 1) % training_config.gradient_accumulation == 0 or batch_index + 1 == epoch_batch_count
            if should_step:
                scaler.unscale_(optimizer)
                grad_norm_tensor = torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.max_grad_norm)
                grad_norm = float(grad_norm_tensor.item() if hasattr(grad_norm_tensor, 'item') else grad_norm_tensor)
                weight_norm = math.sqrt(sum((float(parameter.detach().float().norm(2).item()) ** 2 for parameter in model.parameters())))
                learning_rate = float(scheduler.get_last_lr()[0])
                update_ratio = learning_rate * grad_norm / max(weight_norm, 1e-12)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
                now = perf_counter()
                step_seconds = max(now - last_metric_time, 1e-09)
                last_metric_time = now
                step_time_window.append(step_seconds)
                step_time_window = step_time_window[-50:]
                average_step_seconds = sum(step_time_window) / max(len(step_time_window), 1)
                remaining_steps = max(total_steps - global_step, 0)
                eta_seconds = remaining_steps * average_step_seconds
                samples_seen = training_config.batch_size * training_config.gradient_accumulation
                tokens_seen = samples_seen * model_config.context_length
                vram_allocated_gb = None
                vram_reserved_gb = None
                gpu_memory_percent = None
                if training_config.device.startswith('cuda') and torch.cuda.is_available():
                    device_index = torch.cuda.current_device()
                    vram_allocated_gb = torch.cuda.memory_allocated(device_index) / 1024 ** 3
                    vram_reserved_gb = torch.cuda.memory_reserved(device_index) / 1024 ** 3
                    free_vram, total_vram = torch.cuda.mem_get_info(device_index)
                    gpu_memory_percent = 100.0 * (1.0 - free_vram / max(total_vram, 1))
                sample_text = None
                if decode_preview is not None:
                    try:
                        sample_text = decode_preview(x[0].detach().cpu().tolist())
                    except Exception:
                        sample_text = None
                current_progress = 8 + int(86 * min(global_step, total_steps) / max(total_steps, 1))
                emit_progress(progress, f'Epoch {epoch + 1}/{training_config.epochs}, step {global_step}/{total_steps}, loss {float(loss.item() * training_config.gradient_accumulation):.4f}', current_progress, epoch=epoch + 1, total_epochs=training_config.epochs, step=global_step, total_steps=total_steps, train_loss=float(loss.item() * training_config.gradient_accumulation), val_loss=final_val_loss, learning_rate=learning_rate, grad_norm=grad_norm, weight_norm=weight_norm, update_ratio=update_ratio, tokens_per_second=tokens_seen / step_seconds, samples_per_second=samples_seen / step_seconds, step_seconds=step_seconds, average_step_seconds=average_step_seconds, eta_seconds=eta_seconds, remaining_steps=remaining_steps, vram_allocated_gb=vram_allocated_gb, vram_reserved_gb=vram_reserved_gb, gpu_memory_percent=gpu_memory_percent, system_cpu_percent=system_cpu_percent(), system_ram_percent=system_ram_percent(), data_loader_workers=loader_workers, sample_text=sample_text)
                if val_loader is not None and training_config.eval_interval > 0 and (global_step % training_config.eval_interval == 0):
                    emit_progress(progress, f'Running validation at step {global_step}...', current_progress, epoch=epoch + 1, total_epochs=training_config.epochs, step=global_step, total_steps=total_steps, train_loss=float(loss.item() * training_config.gradient_accumulation), val_loss=final_val_loss, system_cpu_percent=system_cpu_percent(), system_ram_percent=system_ram_percent())
                    final_val_loss = evaluate(model, val_loader, training_config.device, pad_token_id, training_config.max_eval_batches, progress, should_stop, global_step, total_steps, current_progress)
                    emit_progress(progress, f'Validation loss at step {global_step}: {final_val_loss:.4f}', current_progress, epoch=epoch + 1, total_epochs=training_config.epochs, step=global_step, total_steps=total_steps, train_loss=epoch_losses[-1] if epoch_losses else None, val_loss=final_val_loss, system_cpu_percent=system_cpu_percent(), system_ram_percent=system_ram_percent())
                    if best_val_loss is None or final_val_loss < best_val_loss:
                        best_val_loss = final_val_loss
                        best_checkpoint_path = checkpoints_dir / 'checkpoint_best_val.pt'
                        save_checkpoint(best_checkpoint_path, model, optimizer, scheduler, scaler, model_config, training_config, global_step, epoch + 1, epoch_losses[-1] if epoch_losses else final_train_loss, final_val_loss)
                        emit_progress(progress, f'New best validation checkpoint: {best_checkpoint_path.name} ({best_val_loss:.4f}).', current_progress, checkpoint_quality='best_validation', best_val_loss=best_val_loss, best_checkpoint_path=str(best_checkpoint_path))
                        early_stop_counter = 0
                    elif training_config.early_stopping and best_val_loss is not None:
                        early_stop_counter += 1
                        if early_stop_counter >= training_config.early_stopping_patience:
                            reason = f'Early stopping: validation loss has not improved for {early_stop_counter} consecutive evaluation(s). Best val loss: {best_val_loss:.4f}, current: {final_val_loss:.4f}. Best checkpoint: {best_checkpoint_path}.'
                            emit_progress(progress, reason, current_progress)
                            early_stopped = True
                            break
                if training_config.save_interval > 0 and global_step % training_config.save_interval == 0:
                    save_checkpoint(checkpoints_dir / f'checkpoint_{global_step}.pt', model, optimizer, scheduler, scaler, model_config, training_config, global_step, epoch + 1, final_train_loss, final_val_loss)
                    emit_progress(progress, f'Saved checkpoint at step {global_step}.', current_progress)
            epoch_losses.append(float(loss.item() * training_config.gradient_accumulation))
        if early_stopped:
            break
        final_train_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        if val_loader is not None:
            final_val_loss = evaluate(model, val_loader, training_config.device, pad_token_id, training_config.max_eval_batches, progress, should_stop, global_step, total_steps, 8 + int(86 * (epoch + 1) / max(training_config.epochs, 1)))
            if best_val_loss is None or final_val_loss < best_val_loss:
                best_val_loss = final_val_loss
                best_checkpoint_path = checkpoints_dir / 'checkpoint_best_val.pt'
                save_checkpoint(best_checkpoint_path, model, optimizer, scheduler, scaler, model_config, training_config, global_step, epoch + 1, final_train_loss, final_val_loss)
                emit_progress(progress, f'New best validation checkpoint: {best_checkpoint_path.name} ({best_val_loss:.4f}).', 8 + int(86 * (epoch + 1) / max(training_config.epochs, 1)), checkpoint_quality='best_validation', best_val_loss=best_val_loss, best_checkpoint_path=str(best_checkpoint_path))
                early_stop_counter = 0
            elif training_config.early_stopping and best_val_loss is not None:
                early_stop_counter += 1
                if early_stop_counter >= training_config.early_stopping_patience:
                    reason = f'Early stopping at epoch {epoch + 1}: validation loss has not improved for {early_stop_counter} consecutive evaluation(s). Best val loss: {best_val_loss:.4f}, current: {final_val_loss:.4f}. Best checkpoint: {best_checkpoint_path}.'
                    emit_progress(progress, reason, 8 + int(86 * (epoch + 1) / max(training_config.epochs, 1)))
                    early_stopped = True
        print(f'epoch {epoch + 1}/{training_config.epochs}: train_loss={final_train_loss:.4f}')
        save_checkpoint(checkpoints_dir / f'checkpoint_epoch_{epoch + 1}.pt', model, optimizer, scheduler, scaler, model_config, training_config, global_step, epoch + 1, final_train_loss, final_val_loss)
        emit_progress(progress, f'Epoch {epoch + 1} complete. Checkpoint saved.', 8 + int(86 * (epoch + 1) / max(training_config.epochs, 1)), epoch=epoch + 1, total_epochs=training_config.epochs, step=global_step, total_steps=total_steps, train_loss=final_train_loss, val_loss=final_val_loss, system_cpu_percent=system_cpu_percent(), system_ram_percent=system_ram_percent())
        if early_stopped:
            break
    if training_config.peft_method == 'lora':
        adapter_path = training_config.output_dir / 'final_adapter.pt'
        save_checkpoint(adapter_path, model, optimizer, scheduler, scaler, model_config, training_config, global_step, training_config.epochs, final_train_loss, final_val_loss)
        merged_count = merge_lora_adapters(model)
        emit_progress(progress, f'Merged {merged_count} LoRA adapter module(s) into final model weights.', 96)
    checkpoint_path = training_config.output_dir / 'final_model.pt'
    save_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler, model_config, training_config, global_step, training_config.epochs, final_train_loss, final_val_loss)
    summary_path = training_config.output_dir / 'training_summary.json'
    summary = {'model_config': dataclass_to_jsonable(model_config), 'training_config': dataclass_to_jsonable(training_config), 'final_train_loss': final_train_loss, 'final_val_loss': final_val_loss, 'best_val_loss': best_val_loss, 'best_checkpoint_path': str(best_checkpoint_path) if best_checkpoint_path else None, 'recommended_checkpoint_path': str(best_checkpoint_path or checkpoint_path), 'total_steps': global_step, 'parameters': sum((p.numel() for p in model.parameters())), 'adapter_checkpoint': str(training_config.output_dir / 'final_adapter.pt') if training_config.peft_method == 'lora' else None, 'early_stopped': early_stopped}
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    emit_progress(progress, 'Training stopped early - validation loss converged.' if early_stopped else 'Training complete.', 100, epoch=training_config.epochs, total_epochs=training_config.epochs, step=global_step, total_steps=total_steps, train_loss=final_train_loss, val_loss=final_val_loss)
    return TrainingResult(checkpoint_path, summary_path, final_train_loss, final_val_loss)
from .training_resume import _release_cuda_cache, _configure_cuda_allocator, _estimated_training_vram_bytes
from .training_checkpoint import save_checkpoint
