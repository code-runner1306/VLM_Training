import os
import sys
import yaml
import json
import argparse

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from training.src.model_factory import ModelFactory
from training.src.trainer import train_vlm
from training.scripts.evaluate import run_evaluation


def parse_args():
    parser = argparse.ArgumentParser(description="Train Vision-Language Model with QLoRA.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML training configuration file.")
    parser.add_argument("--experiment", type=str, required=True, help="Experiment identifier name (e.g. qwen25vl-3b-v1).")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest saved checkpoint.")
    parser.add_argument("--no_eval", action="store_true", help="Skip post-training automated evaluation.")
    return parser.parse_args()


def main():
    args = parse_args()

    config_path = os.path.abspath(args.config)
    if not os.path.exists(config_path):
        print(f"[ERROR] Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_key = config.get("model", {}).get("key", "qwen25vl_3b")
    adapter = ModelFactory.get_adapter(model_key, config)

    train_manifest = os.path.abspath("outputs/dataset/train_manifest.jsonl")
    val_manifest = os.path.abspath("outputs/dataset/validation_manifest.jsonl")

    if not os.path.exists(train_manifest):
        print("[ERROR] Training manifest missing! Please run `python training/scripts/prepare_dataset.py` first.")
        sys.exit(1)

    # 1. Save run metadata under outputs/experiments/<experiment>/run_metadata.json
    exp_output_dir = os.path.abspath(os.path.join("outputs", "experiments", args.experiment))
    os.makedirs(exp_output_dir, exist_ok=True)

    run_metadata = {
        "experiment": args.experiment,
        "model_key": model_key,
        "model_id": adapter.model_id,
        "config_file": config_path,
        "adaptation_strategy": config.get("adaptation", {}).get("strategy"),
        "lora_rank": config.get("adaptation", {}).get("r"),
        "lora_alpha": config.get("adaptation", {}).get("lora_alpha"),
        "learning_rate": config.get("training", {}).get("learning_rate"),
        "epochs": config.get("training", {}).get("num_epochs"),
        "quantization": config.get("quantization", {}).get("quant_type"),
        "hardware_profile": config.get("hardware_profile"),
    }

    with open(os.path.join(exp_output_dir, "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(run_metadata, f, indent=2)

    # 2. Run fine-tuning
    train_summary = train_vlm(
        adapter=adapter,
        config=config,
        experiment_name=args.experiment,
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        resume=args.resume,
    )

    # Update run metadata with timing/vram results
    run_metadata.update({
        "total_training_time_s": train_summary["total_training_time_s"],
        "peak_vram_gb": train_summary["peak_vram_gb"],
        "param_counts": train_summary["param_counts"],
    })
    with open(os.path.join(exp_output_dir, "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(run_metadata, f, indent=2)

    # 3. Automatically run post-training test evaluation
    if not args.no_eval:
        print(f"\n--- Running Automated Post-Training Test Evaluation for {args.experiment} ---")
        run_evaluation(experiment_name=args.experiment, config=config, adapter=adapter)


if __name__ == "__main__":
    main()
