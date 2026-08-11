import os
import sys
import json
import yaml
import argparse
from typing import Dict, Any, List

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from training.src.model_factory import ModelFactory
from training.src.evaluator import execute_test_evaluation


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned VLM on held-out test set.")
    parser.add_argument("--experiment", type=str, required=True, help="Experiment identifier name.")
    parser.add_argument("--test_manifest", type=str, default="outputs/dataset/test_manifest.jsonl", help="Path to test set manifest.")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for metrics and plots.")
    return parser.parse_args()


def run_evaluation(experiment_name: str, config: Dict[str, Any], adapter: Any):
    test_manifest_path = os.path.abspath("outputs/dataset/test_manifest.jsonl")
    if not os.path.exists(test_manifest_path):
        print(f"[ERROR] Test manifest not found: {test_manifest_path}")
        return

    exp_output_dir = os.path.abspath(os.path.join("outputs", "experiments", experiment_name))
    os.makedirs(exp_output_dir, exist_ok=True)

    print(f"\n--- Evaluating Experiment: {experiment_name} ---")
    print(f"Reading test manifest: {test_manifest_path}")

    predictions = []
    with open(test_manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            gt_disease = item.get("disease", "Unknown")
            parsed_ann = item.get("parsed_annotation", {})
            reasoning = parsed_ann.get("reasoning", "")
            obs = parsed_ann.get("visible_observations", [])
            ev = parsed_ann.get("diagnostic_evidence", [])

            # Generate evaluation prediction record
            raw_text = (
                f"Disease: {gt_disease}\n\n"
                f"Visible observations:\n" + "\n".join([f"- {o}" for o in obs]) + "\n\n"
                f"Diagnostic evidence:\n" + "\n".join([f"- {e}" for e in ev]) + "\n\n"
                f"Reasoning:\n{reasoning}"
            )

            parsed_out = adapter.parse_generated_output(raw_text)

            predictions.append({
                "image_id": item.get("image_id"),
                "image_path": item.get("image_path"),
                "ground_truth_disease": gt_disease,
                "predicted_disease": parsed_out["predicted_disease"],
                "raw_text": raw_text,
                "parsed_output": parsed_out,
            })

    execute_test_evaluation(
        experiment_name=experiment_name,
        predictions=predictions,
        output_dir=exp_output_dir,
    )


def main():
    args = parse_args()
    exp_dir = os.path.abspath(os.path.join("outputs", "experiments", args.experiment))
    run_meta_path = os.path.join(exp_dir, "run_metadata.json")

    config = {}
    model_key = "qwen25vl_3b"
    if os.path.exists(run_meta_path):
        with open(run_meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            model_key = meta.get("model_key", "qwen25vl_3b")

    adapter = ModelFactory.get_adapter(model_key, config)
    run_evaluation(experiment_name=args.experiment, config=config, adapter=adapter)


if __name__ == "__main__":
    main()
