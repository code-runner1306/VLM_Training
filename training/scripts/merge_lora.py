import os
import sys
import json
import torch
import argparse
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from training.src.run_utils import resolve_latest_run

try:
    from transformers import AutoModelForCausalLM, AutoProcessor
    from peft import PeftModel
except ImportError:
    pass


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter weights with base model and save locally.")
    parser.add_argument("--experiment", type=str, required=True, help="Experiment identifier name.")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for merged model.")
    args = parser.parse_args()

    run_dir = resolve_latest_run(experiment=args.experiment)
    if run_dir is None:
        print(f"[ERROR] No run directory found for experiment '{args.experiment}'.")
        sys.exit(1)

    adapter_dir = os.path.abspath(str(run_dir / "adapter"))
    if not os.path.exists(adapter_dir):
        print(f"[ERROR] Adapter directory not found: {adapter_dir}")
        sys.exit(1)

    out_dir = args.output_dir or os.path.abspath(str(run_dir / "merged"))

    print(f"--- Merging LoRA Adapter for {args.experiment} (run: {run_dir.name}) ---")
    print(f"Adapter Dir: {adapter_dir}")
    print(f"Output Dir:  {out_dir}")

    base_model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    meta_path = run_dir / "run_metadata.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            base_model_id = meta.get("model_id", base_model_id)

    print(f"Loading base model: {base_model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    processor = AutoProcessor.from_pretrained(base_model_id)

    print("Merging adapter weights into base model...")
    peft_model = PeftModel.from_pretrained(model, adapter_dir)
    merged_model = peft_model.merge_and_unload()

    print(f"Saving standalone merged model to {out_dir}...")
    os.makedirs(out_dir, exist_ok=True)
    merged_model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)

    print("[SUCCESS] Merged model export complete.")


if __name__ == "__main__":
    main()
