import os
import sys
import time
import glob
import torch
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

try:
    from transformers import (
        BitsAndBytesConfig,
        TrainingArguments,
        Trainer,
        EarlyStoppingCallback,
    )
except ImportError:
    BitsAndBytesConfig = None
    TrainingArguments = None
    Trainer = None
    EarlyStoppingCallback = None

from training.src.model_adapters.base import BaseVLMAdapter
from training.src.lora import create_lora_config
from training.src.plotting import plot_training_curves
from training.src.dataset import VLMDataset, VLMDataCollator, DEFAULT_USER_PROMPT

try:
    from peft import get_peft_model, PeftModel, prepare_model_for_kbit_training
except ImportError:
    get_peft_model = None
    PeftModel = None
    prepare_model_for_kbit_training = None


def get_quantization_config(config: Dict[str, Any]) -> Optional[Any]:
    """Create BitsAndBytesConfig for 4-bit QLoRA."""
    q_cfg = config.get("quantization", {})
    if not q_cfg.get("enabled", False):
        return None

    if BitsAndBytesConfig is None:
        raise ImportError(
            "bitsandbytes is required for 4-bit QLoRA. Please run `pip install bitsandbytes`."
        )

    compute_dtype_str = q_cfg.get("compute_dtype", "bfloat16")
    if compute_dtype_str == "bfloat16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        compute_dtype = torch.bfloat16
    else:
        compute_dtype = torch.float16

    quant_config = BitsAndBytesConfig(
        load_in_4bit=(q_cfg.get("bits", 4) == 4),
        bnb_4bit_quant_type=q_cfg.get("quant_type", "nf4"),
        bnb_4bit_use_double_quant=q_cfg.get("double_quant", True),
        bnb_4bit_compute_dtype=compute_dtype,
    )
    return quant_config


def count_parameters(model: Any) -> Dict[str, Any]:
    """
    Count total, trainable, vision encoder, projector, and LLM parameters.
    """
    total_params = 0
    trainable_params = 0
    vision_trainable = 0
    projector_trainable = 0
    llm_trainable = 0

    for name, param in model.named_parameters():
        num_p = param.numel()
        total_params += num_p
        if param.requires_grad:
            trainable_params += num_p
            name_lower = name.lower()
            if "visual" in name_lower or "vision" in name_lower:
                vision_trainable += num_p
            elif "merger" in name_lower or "projector" in name_lower or "mlp" in name_lower:
                projector_trainable += num_p
            else:
                llm_trainable += num_p

    trainable_pct = (trainable_params / max(1, total_params)) * 100

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "trainable_percentage": round(trainable_pct, 4),
        "vision_trainable_parameters": vision_trainable,
        "projector_trainable_parameters": projector_trainable,
        "llm_trainable_parameters": llm_trainable,
    }


def print_parameter_summary(param_counts: Dict[str, Any]):
    print("=" * 60)
    print("      VLM LoRA PARAMETER COUNT SUMMARY")
    print("=" * 60)
    print(f"Total Parameters:               {param_counts['total_parameters']:,}")
    print(f"Trainable Parameters:           {param_counts['trainable_parameters']:,}")
    print(f"Trainable Percentage:           {param_counts['trainable_percentage']:.4f}%")
    print(f"  - Vision Encoder Trainable:   {param_counts['vision_trainable_parameters']:,}")
    print(f"  - Projector Trainable:        {param_counts['projector_trainable_parameters']:,}")
    print(f"  - LLM Trainable:              {param_counts['llm_trainable_parameters']:,}")
    print("=" * 60)


def _map_monitor_to_metric(monitor: str) -> Tuple[str, bool]:
    """Map an early-stopping monitor name to a Trainer metric and its direction."""
    if monitor == "val_loss":
        return "eval_loss", False
    if monitor == "loss":
        return "loss", False
    if monitor.endswith("_loss"):
        return monitor, False
    return monitor, True


class _EarlyStoppingRecorder(EarlyStoppingCallback):
    """EarlyStoppingCallback that records when and at which epoch it triggered."""

    def __init__(self, patience: int, threshold: Optional[float]):
        super().__init__(early_stopping_patience=patience, early_stopping_threshold=threshold)
        self.triggered = False
        self.trigger_epoch: Optional[float] = None

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        super().on_evaluate(args, state, control, metrics, **kwargs)
        if control.should_training_stop and not self.triggered:
            self.triggered = True
            self.trigger_epoch = state.epoch


def _find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Return the newest `checkpoint-<step>` directory under checkpoint_dir."""
    candidates = glob.glob(os.path.join(checkpoint_dir, "checkpoint-*"))
    if not candidates:
        return None
    candidates = [c for c in candidates if os.path.isdir(c)]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda p: int(os.path.basename(p).split("checkpoint-")[-1]),
    )


def train_vlm(
    adapter: BaseVLMAdapter,
    config: Dict[str, Any],
    experiment_name: str,
    train_manifest: str,
    val_manifest: str,
    resume: bool = False,
    smoke_test: bool = False,
    run_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Execute real VLM QLoRA fine-tuning via transformers.Trainer with native
    checkpointing, resume, early stopping, and CUDA OOM safety.
    """
    print(f"\n--- Initializing Fine-Tuning Run: {experiment_name} ---")
    if smoke_test:
        print("[SMOKE TEST] Fast verification training mode enabled (1 epoch, minimal samples).")

    # 0. Reserve CUDA Memory Overhead
    cuda_mem_fraction = config.get("training", {}).get("cuda_memory_fraction", 30 / 32)
    if torch.cuda.is_available():
        try:
            device_idx = torch.cuda.current_device()
            torch.cuda.set_per_process_memory_fraction(float(cuda_mem_fraction), device=device_idx)
            print(f"[CUDA MEMORY] Set per-process memory fraction to {float(cuda_mem_fraction):.4f} (~{float(cuda_mem_fraction)*100:.1f}%) on device {device_idx}")
        except Exception as e:
            print(f"[CUDA MEMORY] Note: Could not set memory fraction: {e}")

    if Trainer is None or TrainingArguments is None:
        raise ImportError(
            "transformers >= 4.41 is required for Trainer-based training. Please run `pip install transformers`."
        )

    checkpoint_dir = os.path.abspath(
        config.get("checkpoint", {}).get("output_dir",
                                         str((run_dir or Path("outputs") / experiment_name) / "checkpoints"))
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 1. Early stopping configuration (YAML + config.py overrides already merged by callers)
    es_cfg = config.get("early_stopping", {})
    early_stopping_enabled = es_cfg.get("enabled", True) and not smoke_test
    es_patience = es_cfg.get("patience", 3)
    es_threshold = es_cfg.get("stopping_threshold")
    es_monitor = es_cfg.get("monitor", "val_loss")
    es_mode = es_cfg.get("mode", "min")
    metric_for_best_model, greater_is_better = _map_monitor_to_metric(es_monitor)
    if es_mode == "min":
        greater_is_better = False
    elif es_mode == "max":
        greater_is_better = True

    if early_stopping_enabled:
        print(
            f"[EARLY STOPPING] Configured: monitor='{es_monitor}' (metric='{metric_for_best_model}'), "
            f"patience={es_patience}, threshold={es_threshold}"
        )
    else:
        print("[EARLY STOPPING] Disabled.")

    # 2. Quantization & Model Loading
    quant_config = get_quantization_config(config)
    torch_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

    print("Loading base model and processor...")
    model, processor = adapter.load_model_and_processor(
        quantization_config=quant_config,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else "cpu",
    )

    # 3. Configure PEFT LoRA
    strategy = config.get("adaptation", {}).get("strategy", "llm_projector")
    peft_config = create_lora_config(
        adapter=adapter,
        strategy=strategy,
        r=config.get("adaptation", {}).get("r", 16),
        lora_alpha=config.get("adaptation", {}).get("lora_alpha", 32),
        lora_dropout=config.get("adaptation", {}).get("lora_dropout", 0.05),
    )

    if quant_config is not None and prepare_model_for_kbit_training is not None:
        print("[TRAINING] Preparing 4-bit quantized model for k-bit LoRA training...")
        model = prepare_model_for_kbit_training(model)

    model = get_peft_model(model, peft_config)

    # 4. Print parameter counts
    param_counts = count_parameters(model)
    print_parameter_summary(param_counts)

    # 5. Datasets & collator
    train_cfg = config.get("training", {})
    user_prompt = config.get("data", {}).get("user_prompt", DEFAULT_USER_PROMPT)
    train_limit = 8 if smoke_test else None
    val_limit = 4 if smoke_test else None

    train_dataset = VLMDataset(train_manifest, user_prompt=user_prompt, max_items=train_limit)
    if len(train_dataset) == 0:
        raise ValueError(
            f"Training manifest is empty or missing: {train_manifest}. "
            "Please run `python training/scripts/prepare_dataset.py` first."
        )
    eval_dataset = VLMDataset(val_manifest, user_prompt=user_prompt, max_items=val_limit)
    data_collator = VLMDataCollator(processor)

    print(f"[DATA] Train samples: {len(train_dataset)} | Validation samples: {len(eval_dataset)}")
    print(f"[DATA] Collator max_length: {data_collator.max_length}")

    use_eval = early_stopping_enabled and len(eval_dataset) > 0

    # 6. Training arguments
    num_epochs = 1 if smoke_test else train_cfg.get("num_epochs", 5)
    lr = float(train_cfg.get("learning_rate", 2e-4))
    cuda_available = torch.cuda.is_available()
    fp16 = bool(train_cfg.get("fp16", False)) and cuda_available
    bf16 = bool(train_cfg.get("bf16", False)) and cuda_available
    gradient_checkpointing = bool(train_cfg.get("gradient_checkpointing", False))
    if gradient_checkpointing and hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=int(train_cfg.get("batch_size", 1)),
        per_device_eval_batch_size=int(train_cfg.get("batch_size", 1)),
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 8)),
        learning_rate=lr,
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.03)),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
        fp16=fp16,
        bf16=bf16,
        gradient_checkpointing=gradient_checkpointing,
        logging_steps=int(train_cfg.get("logging_steps", 10)),
        eval_strategy="steps" if use_eval else "no",
        eval_steps=int(train_cfg.get("eval_steps", 100)),
        save_strategy="steps",
        save_steps=int(train_cfg.get("save_steps", 100)),
        save_total_limit=int(config.get("checkpoint", {}).get("save_total_limit", 3)),
        load_best_model_at_end=use_eval,
        metric_for_best_model=metric_for_best_model if use_eval else None,
        greater_is_better=greater_is_better if use_eval else None,
        remove_unused_columns=False,
        prediction_loss_only=True,
        dataloader_pin_memory=False,
        report_to=[],
    )

    # 7. Build Trainer (with early stopping callback)
    es_callback = None
    callbacks = []
    if use_eval:
        es_callback = _EarlyStoppingRecorder(patience=es_patience, threshold=es_threshold)
        callbacks.append(es_callback)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if use_eval else None,
        callbacks=callbacks,
    )

    resume_ckpt = None
    if resume:
        resume_ckpt = _find_latest_checkpoint(checkpoint_dir)
        if resume_ckpt:
            print(f"[RESUME] Resuming from checkpoint: {resume_ckpt}")
        else:
            print("[RESUME] No existing checkpoint found. Starting fresh training run.")

    start_time = time.time()
    peak_vram_gb = 0.0

    try:
        if cuda_available:
            torch.cuda.reset_peak_memory_stats()

        trainer.train(resume_from_checkpoint=resume_ckpt)

        if cuda_available:
            vram_bytes = torch.cuda.max_memory_allocated()
            peak_vram_gb = max(peak_vram_gb, vram_bytes / (1024 ** 3))

    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "CUDA out of memory" in str(e):
            print("\n" + "!" * 60)
            print("         CUDA OUT OF MEMORY (OOM) DETECTED")
            print("!" * 60)
            print("Training was safely interrupted without corrupting the latest checkpoint.")
            print("Recommended Actions:")
            print("  1. Reduce image token budget (min_pixels / max_pixels)")
            print("  2. Reduce batch size or increase gradient_accumulation_steps")
            print("  3. Enable gradient_checkpointing in config YAML")
            print("!" * 60)
            if cuda_available:
                torch.cuda.empty_cache()
            sys.exit(1)
        else:
            raise e

    total_training_time = time.time() - start_time

    # 8. Export the best/last PEFT adapter to run_dir/adapter/
    final_model_dir = os.path.abspath(str((run_dir or Path("outputs") / experiment_name) / "adapter"))
    os.makedirs(final_model_dir, exist_ok=True)
    trainer.save_model(final_model_dir)
    if hasattr(processor, "save_pretrained"):
        processor.save_pretrained(final_model_dir)
    print(f"\nAdapter exported to: {final_model_dir}")

    # 9. Build training history from Trainer logs and plot curves
    history = {"train_loss": [], "val_loss": [], "learning_rate": []}
    best_epoch = None
    best_metric_value = None
    for log in trainer.state.log_history:
        if "eval_loss" in log:
            history["val_loss"].append(float(log["eval_loss"]))
            epoch = log.get("epoch")
            if best_metric_value is None:
                best_metric_value = log["eval_loss"]
                best_epoch = epoch
            elif greater_is_better and log["eval_loss"] > best_metric_value:
                best_metric_value = log["eval_loss"]
                best_epoch = epoch
            elif not greater_is_better and log["eval_loss"] < best_metric_value:
                best_metric_value = log["eval_loss"]
                best_epoch = epoch
        if "loss" in log and "eval_loss" not in log:
            history["train_loss"].append(float(log["loss"]))
            if "learning_rate" in log:
                history["learning_rate"].append(float(log["learning_rate"]))

    exp_plots_dir = os.path.abspath(str((run_dir or Path("outputs") / experiment_name) / "plots"))
    plot_training_curves(
        history=history,
        output_dir=exp_plots_dir,
        best_epoch=int(best_epoch) if best_epoch is not None else None,
        stopped_epoch=int(es_callback.trigger_epoch) if es_callback and es_callback.triggered else None,
    )

    print(f"\nTraining successfully finished in {total_training_time:.2f}s.")
    print(f"Final model adapter saved to: {final_model_dir}")

    early_stopping_summary = None
    if use_eval:
        early_stopping_summary = {
            "early_stopping_enabled": True,
            "monitored_metric": es_monitor,
            "metric_for_best_model": metric_for_best_model,
            "mode": es_mode,
            "patience": es_patience,
            "early_stopped": bool(es_callback.triggered),
            "stopped_epoch": int(es_callback.trigger_epoch) if es_callback.triggered else None,
            "best_epoch": int(best_epoch) if best_epoch is not None else None,
            "best_metric": round(best_metric_value, 6) if best_metric_value is not None else None,
        }

    return {
        "experiment": experiment_name,
        "model_key": adapter.model_key,
        "model_id": adapter.model_id,
        "param_counts": param_counts,
        "total_training_time_s": total_training_time,
        "peak_vram_gb": round(peak_vram_gb, 2),
        "final_model_dir": final_model_dir,
        "early_stopping": early_stopping_summary,
    }