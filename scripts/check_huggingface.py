import sys
import os
import argparse
from pathlib import Path

# Ensure root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vlm_annotation.src.annotation.hf_health import check_huggingface_environment_and_model


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-Flight Health Check for Local Hugging Face VLM Environment and Model.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="Requested Hugging Face model repository ID.")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"--- Running Hugging Face Pre-Flight Health Check ---")
    print(f"Target Model ID: {args.model}")
    print("Checking PyTorch, CUDA, transformers compatibility, and Hugging Face Hub model access...\n")

    ok, msg = check_huggingface_environment_and_model(model_id=args.model)
    print(msg)

    if ok:
        print("\n[PASSED] Pre-flight health check succeeded. Environment is ready for annotation.")
        sys.exit(0)
    else:
        print("\n[FAILED] Pre-flight health check failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
