import os
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from training.src.run_utils import resolve_latest_run

try:
    from huggingface_hub import HfApi, create_repo
except ImportError:
    HfApi = None


def generate_model_card(experiment_name: str, base_model_id: str, repo_id: str) -> str:
    card = f"""---
license: apache-2.0
base_model: {base_model_id}
tags:
- vision-language
- vlm
- peft
- lora
- qlora
- cotton-disease
- agriculture
pipeline_tag: image-text-to-text
---

# Cotton Disease Diagnostic VLM Adapter (`{experiment_name}`)

This is a fine-tuned LoRA/QLoRA adapter for `{base_model_id}` trained on synthetic cotton plant leaf disease annotations.

## Model Summary

- **Base Model:** `{base_model_id}`
- **Repository:** `https://huggingface.co/{repo_id}`
- **Domain:** Agriculture / Cotton Leaf Disease Diagnosis
- **Training Method:** 4-bit QLoRA

## Usage

```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
import torch

base_model_id = "{base_model_id}"
adapter_id = "{repo_id}"

processor = AutoProcessor.from_pretrained(base_model_id)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = PeftModel.from_pretrained(model, adapter_id)
```

## Intended Use & Limitations

Intended for research and experimental visual plant disease diagnosis. Always verify field observations with agricultural extension specialists.
"""
    return card


def main():
    parser = argparse.ArgumentParser(description="Publish fine-tuned VLM LoRA adapter to Hugging Face Hub.")
    parser.add_argument("--experiment", type=str, required=True, help="Experiment identifier name.")
    parser.add_argument("--repo", type=str, required=True, help="Hugging Face repo ID (e.g. username/cotton-disease-vlm).")
    parser.add_argument("--private", action="store_true", help="Set Hugging Face repo as private.")
    args = parser.parse_args()

    if HfApi is None:
        print("[ERROR] huggingface_hub is required. Run `pip install huggingface-hub`.")
        sys.exit(1)

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("[ERROR] HF_TOKEN environment variable is missing. Please set HF_TOKEN before running.")
        sys.exit(1)

    run_dir = resolve_latest_run(experiment=args.experiment)
    if run_dir is None:
        print(f"[ERROR] No run directory found for experiment '{args.experiment}'.")
        sys.exit(1)

    adapter_dir = os.path.abspath(str(run_dir / "adapter"))
    if not os.path.exists(adapter_dir):
        print(f"[ERROR] Adapter directory not found: {adapter_dir}")
        sys.exit(1)

    print(f"--- Publishing Adapter '{args.experiment}' to Hugging Face Hub ---")
    print(f"Target Repo: https://huggingface.co/{args.repo}")

    api = HfApi(token=hf_token)
    create_repo(repo_id=args.repo, token=hf_token, private=args.private, exist_ok=True)

    # Auto-generate Model Card README.md
    base_model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    meta_path = run_dir / "run_metadata.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            base_model_id = meta.get("model_id", base_model_id)

    model_card_content = generate_model_card(args.experiment, base_model_id, args.repo)
    with open(os.path.join(adapter_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(model_card_content)

    print("Uploading adapter files...")
    api.upload_folder(
        folder_path=adapter_dir,
        repo_id=args.repo,
        repo_type="model",
    )

    print(f"[SUCCESS] Successfully published adapter to https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
