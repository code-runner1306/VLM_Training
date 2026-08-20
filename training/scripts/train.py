import os
import sys
import yaml
import json
import argparse
from pathlib import Path

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

try:
    from config import config as pipeline_cfg
except ImportError:
    pipeline_cfg = None

from training.src.trainer import train_vlm
from training.src.model_factory import ModelFactory
from training.src.run_utils import (
    annotations_provenance,
    config_copy_path,
    create_run_dir,
    read_run_metadata,
    resolve_latest_run,
    write_run_metadata,
)
from training.scripts.evaluate import run_evaluation


def parse_args():
    parser = argparse.ArgumentParser(description="Train Vision-Language Model with QLoRA and Early Stopping.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML training configuration file.")
    parser.add_argument("--experiment", type=str, required=True, help="Experiment identifier name (e.g. qwen25vl-3b-v1).")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest saved checkpoint.")
    parser.add_argument("--no_eval", action="store_true", help="Skip post-training automated evaluation.")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast verification training (1 epoch, minimal steps).")
    parser.add_argument("--patience", type=int, default=None, help="Override early stopping patience.")
    parser.add_argument("--early-stopping-monitor", type=str, default=None, help="Override early stopping monitor metric (e.g. val_loss).")
    parser.add_argument("--no-early-stopping", action="store_true", help="Disable early stopping.")
    return parser.parse_args()


def main():
    args = parse_args()

    config_path = os.path.abspath(args.config)
    if not os.path.exists(config_path):
        print(f"[ERROR] Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 1. Initialize early stopping configuration from config.py defaults, YAML, and CLI overrides
    if "early_stopping" not in config:
        config["early_stopping"] = {}

    if pipeline_cfg is not None:
        cfg_mappings = [
            ("enabled", "early_stopping_enabled"),
            ("monitor", "early_stopping_monitor"),
            ("mode", "early_stopping_mode"),
            ("patience", "early_stopping_patience"),
            ("min_delta", "early_stopping_min_delta"),
            ("restore_best_weights", "early_stopping_restore_best_weights"),
            ("stopping_threshold", "early_stopping_stopping_threshold"),
            ("divergence_threshold", "early_stopping_divergence_threshold"),
        ]
        for key, attr in cfg_mappings:
            if key not in config["early_stopping"] and hasattr(pipeline_cfg, attr):
                config["early_stopping"][key] = getattr(pipeline_cfg, attr)

    # Apply CLI early stopping overrides if explicitly supplied
    if args.no_early_stopping:
        config["early_stopping"]["enabled"] = False
    if args.patience is not None:
        config["early_stopping"]["patience"] = args.patience
    if args.early_stopping_monitor is not None:
        config["early_stopping"]["monitor"] = args.early_stopping_monitor

    if "training" not in config:
        config["training"] = {}
    if pipeline_cfg is not None and hasattr(pipeline_cfg, "cuda_memory_fraction"):
        if "cuda_memory_fraction" not in config["training"]:
            config["training"]["cuda_memory_fraction"] = pipeline_cfg.cuda_memory_fraction

    model_key = config.get("model", {}).get("key", "qwen25vl_3b")
    adapter = ModelFactory.get_adapter(model_key, config)

    # Resolve run dir: new timestamped run, or latest run matching experiment+model for --resume
    if args.resume:
        resume_run = resolve_latest_run(experiment=args.experiment, model_key=model_key)
        if resume_run is not None:
            run_dir = resume_run
            print(f"[RESUME] Continuing run directory: {run_dir}")
        else:
            print(f"[RESUME] No prior run found for experiment '{args.experiment}' / model '{model_key}'. Starting fresh.")
            run_dir = create_run_dir(args.experiment)
    else:
        run_dir = create_run_dir(args.experiment)
    print(f"[RUN] Run directory: {run_dir}")

    train_manifest = os.path.abspath("artifacts/cotton_dataset/train_manifest.jsonl")
    val_manifest = os.path.abspath("artifacts/cotton_dataset/validation_manifest.jsonl")

    if not os.path.exists(train_manifest):
        print("[ERROR] Training manifest missing! Please run `python training/scripts/prepare_dataset.py` first.")
        sys.exit(1)

    # 2. Save run metadata with full provenance inside the run directory
    config_copy_path(run_dir, config_path)
    split_meta_path = Path("artifacts/cotton_dataset/split_metadata.json")
    split_metadata = {}
    if split_meta_path.exists():
        try:
            split_metadata = json.loads(split_meta_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    run_metadata = {
        "experiment": args.experiment,
        "model_key": model_key,
        "model_id": adapter.model_id,
        "config_file": config_path,
        "config_copy": str(run_dir / "config.yaml"),
        "adaptation_strategy": config.get("adaptation", {}).get("strategy"),
        "lora_rank": config.get("adaptation", {}).get("r"),
        "lora_alpha": config.get("adaptation", {}).get("lora_alpha"),
        "learning_rate": config.get("training", {}).get("learning_rate"),
        "epochs": config.get("training", {}).get("num_epochs"),
        "quantization": config.get("quantization", {}).get("quant_type"),
        "hardware_profile": config.get("hardware_profile"),
        "early_stopping_config": config.get("early_stopping", {}),
        "prompt_version": config.get("dataset", {}).get("prompt_version"),
        "teacher": config.get("dataset", {}).get("teacher_model"),
        "split_metadata": split_metadata,
    }
    run_metadata.update(annotations_provenance(Path("artifacts/cotton_dataset/annotations.jsonl")))
    write_run_metadata(run_dir, run_metadata)

    # 3. Run fine-tuning
    train_summary = train_vlm(
        adapter=adapter,
        config=config,
        experiment_name=args.experiment,
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        resume=args.resume,
        smoke_test=args.smoke_test,
        run_dir=run_dir,
    )

    # Update run metadata with timing, vram, parameter counts, and early stopping results
    run_metadata.update({
        "total_training_time_s": train_summary["total_training_time_s"],
        "peak_vram_gb": train_summary["peak_vram_gb"],
        "param_counts": train_summary["param_counts"],
        "early_stopping_result": train_summary.get("early_stopping"),
    })
    write_run_metadata(run_dir, run_metadata)

    # 4. Automatically run post-training test evaluation
    if not args.no_eval:
        print(f"\n--- Running Automated Post-Training Test Evaluation for {args.experiment} ---")
        run_evaluation(experiment_name=args.experiment, config=config, adapter=adapter, run_dir=run_dir)


if __name__ == "__main__":
    main()
