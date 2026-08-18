import os
import sys
import json
import yaml
import argparse
import torch
from typing import Dict, Any, List

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Import trainer first so transformers (and peft) are loaded in a safe order.
from training.src.trainer import get_quantization_config
from training.src.model_factory import ModelFactory
from training.src.evaluator import execute_test_evaluation
from training.src.dataset import DEFAULT_USER_PROMPT

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    Image = None
    HAS_PIL = False


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned VLM on held-out test set using real model inference.")
    parser.add_argument("--experiment", type=str, required=True, help="Experiment identifier name.")
    parser.add_argument("--test_manifest", type=str, default="outputs/dataset/test_manifest.jsonl", help="Path to test set manifest.")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for metrics and plots.")
    return parser.parse_args()


def load_fine_tuned_model(adapter: Any, config: Dict[str, Any], adapter_dir: str):
    """Load the base model from the local cache + the trained PEFT adapter."""
    if not os.path.isdir(adapter_dir):
        raise FileNotFoundError(
            f"Adapter directory not found: {adapter_dir}. Run training first (adapter is exported to models/<experiment>/)."
        )
    if PeftModel is None:
        raise ImportError("peft is required for adapter loading. Please run `pip install peft`.")

    quant_config = get_quantization_config(config)
    torch_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

    print("Loading base model (cache-first) and fine-tuned adapter...")
    model, processor = adapter.load_model_and_processor(
        quantization_config=quant_config,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else "cpu",
    )
    model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)
    model.eval()
    print(f"✓ Loaded adapter from {adapter_dir}")
    return model, processor


def generate_predictions(
    adapter: Any,
    config: Dict[str, Any],
    model: Any,
    processor: Any,
    test_manifest_path: str,
) -> List[Dict[str, Any]]:
    """Generate a real response per test image and parse it into prediction records."""
    if not HAS_PIL:
        raise ImportError("Pillow is required for image loading during evaluation.")

    pred_cfg = config.get("evaluation", {})
    max_new_tokens = int(pred_cfg.get("max_new_tokens", 512))
    do_sample = bool(pred_cfg.get("do_sample", False))
    temperature = float(pred_cfg.get("temperature", 0.7))
    top_p = float(pred_cfg.get("top_p", 0.9))
    num_beams = int(pred_cfg.get("num_beams", 1))
    user_prompt = config.get("data", {}).get("user_prompt", DEFAULT_USER_PROMPT)

    device = next(model.parameters()).device
    predictions: List[Dict[str, Any]] = []

    with open(test_manifest_path, "r", encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]

    print(f"\nGenerating predictions for {len(lines)} test images (real inference)...")
    for i, line in enumerate(lines, start=1):
        item = json.loads(line)
        image_path = item.get("image_path")
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"  [WARN] Could not load image {image_path}: {e}")
            continue

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": user_prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        gen_kwargs: Dict[str, Any] = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
        else:
            gen_kwargs["num_beams"] = num_beams

        with torch.no_grad():
            generated_ids = model.generate(**inputs, **gen_kwargs)
        generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
        raw_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        parsed_out = adapter.parse_generated_output(raw_text)
        predictions.append({
            "image_id": item.get("image_id"),
            "image_path": image_path,
            "ground_truth_disease": item.get("disease", "Unknown"),
            "predicted_disease": parsed_out.get("predicted_disease", "Unknown"),
            "raw_text": raw_text,
            "parsed_output": parsed_out,
        })
        print(
            f"  [{i}/{len(lines)}] GT={item.get('disease')!r} | Pred={parsed_out.get('predicted_disease')!r}"
        )

    return predictions


def run_evaluation(experiment_name: str, config: Dict[str, Any], adapter: Any):
    test_manifest_path = os.path.abspath("outputs/dataset/test_manifest.jsonl")
    if not os.path.exists(test_manifest_path):
        print(f"[ERROR] Test manifest not found: {test_manifest_path}")
        return

    exp_output_dir = os.path.abspath(os.path.join("outputs", "experiments", experiment_name))
    os.makedirs(exp_output_dir, exist_ok=True)

    print(f"\n--- Evaluating Experiment: {experiment_name} ---")
    print(f"Reading test manifest: {test_manifest_path}")

    adapter_dir = os.path.abspath(os.path.join("models", experiment_name))
    model, processor = load_fine_tuned_model(adapter, config, adapter_dir)

    predictions = generate_predictions(
        adapter=adapter,
        config=config,
        model=model,
        processor=processor,
        test_manifest_path=test_manifest_path,
    )

    if not predictions:
        print("[ERROR] No predictions generated. Nothing to evaluate.")
        return

    execute_test_evaluation(
        experiment_name=experiment_name,
        predictions=predictions,
        output_dir=exp_output_dir,
    )


def main():
    args = parse_args()
    exp_dir = os.path.abspath(os.path.join("outputs", "experiments", args.experiment))
    run_meta_path = os.path.join(exp_dir, "run_metadata.json")

    config: Dict[str, Any] = {}
    model_key = "qwen25vl_3b"
    if os.path.exists(run_meta_path):
        with open(run_meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            model_key = meta.get("model_key", "qwen25vl_3b")
            config_file = meta.get("config_file")
            if config_file and os.path.exists(config_file):
                with open(config_file, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f) or {}

    adapter = ModelFactory.get_adapter(model_key, config)
    run_evaluation(experiment_name=args.experiment, config=config, adapter=adapter)


if __name__ == "__main__":
    main()