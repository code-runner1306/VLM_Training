import sys
import os
import argparse
from pathlib import Path

# Ensure root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vlm_annotation.src.annotation.ollama_health import check_ollama_server_and_model


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-Flight Health Check for Local Ollama VLM Server and Model.")
    parser.add_argument("--host", type=str, default="http://127.0.0.1:11434", help="Ollama server host URL.")
    parser.add_argument("--model", type=str, default="qwen3-vl:8b", help="Requested Ollama VLM model identifier.")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"--- Running Ollama Pre-Flight Health Check ---")
    print(f"Server Host: {args.host}")
    print(f"Target Model: {args.model}")
    print("Checking reachability, model existence, vision processing, and JSON schema parsing...\n")

    ok, msg = check_ollama_server_and_model(host=args.host, model_name=args.model)
    print(msg)

    if ok:
        print("\n[PASSED] Pre-flight health check succeeded. Environment is ready for annotation.")
        sys.exit(0)
    else:
        print("\n[FAILED] Pre-flight health check failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
