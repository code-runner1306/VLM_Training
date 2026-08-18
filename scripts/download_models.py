"""Prefetch base models into the repository-local cache (models/base/).

Usage:
  python scripts/download_models.py --all
  python scripts/download_models.py --models Qwen/Qwen2.5-VL-3B-Instruct
  python scripts/download_models.py --all --force

Reads HF_TOKEN from the environment (or .env) unless --token is given.
"""

import argparse
import os
import sys

from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import config as pipeline_config
from training.src.model_cache import ensure_model_downloaded, is_model_cached, local_model_dir


def default_model_ids() -> list:
    """Default-pipeline model IDs: annotation teacher + training + scold models (deduped)."""
    ids = []
    ids.append(getattr(pipeline_config, "annotation_model", None))
    for item in getattr(pipeline_config, "training_models", []) or []:
        ids.append(item.get("model_id"))
    scold = getattr(pipeline_config, "scold_model", {}) or {}
    ids.append(scold.get("model_id"))
    seen = []
    for m in ids:
        if m and m not in seen:
            seen.append(m)
    return seen


def main():
    parser = argparse.ArgumentParser(description="Prefetch base models into models/base/.")
    parser.add_argument("--all", action="store_true", help="Download all default-pipeline model IDs from config.py.")
    parser.add_argument("--models", nargs="+", help="Explicit model IDs to download (e.g. --models Qwen/Qwen2.5-VL-3B-Instruct).")
    parser.add_argument("--token", default=None, help="Hugging Face token (defaults to HF_TOKEN env var).")
    parser.add_argument("--force", action="store_true", help="Re-download even if already cached.")
    args = parser.parse_args()

    if args.models:
        model_ids = args.models
    elif args.all:
        model_ids = default_model_ids()
    else:
        parser.error("Specify --all or --models <id> [<id> ...].")

    token = args.token or os.environ.get("HF_TOKEN")
    if token is None:
        print("[WARN] No HF_TOKEN found in env/.env. Gated models may fail to download.")

    results = {"cached": [], "downloaded": [], "failed": []}

    print(f"[MODEL PREFETCH] {len(model_ids)} model(s) to process")
    for i, model_id in enumerate(model_ids, start=1):
        print(f"\n[{i}/{len(model_ids)}] {model_id}")
        try:
            if is_model_cached(model_id) and not args.force:
                print(f"  - Already cached at {local_model_dir(model_id)} (skipping)")
                results["cached"].append(model_id)
            else:
                path = ensure_model_downloaded(model_id, token=token, force=args.force)
                print(f"  - Downloaded to {path}")
                results["downloaded"].append(model_id)
        except Exception as e:
            print(f"  - FAILED: {e}")
            results["failed"].append(model_id)

    print("\n=== SUMMARY ===")
    print(f"  Cached (skipped): {len(results['cached'])}")
    print(f"  Downloaded:       {len(results['downloaded'])}")
    print(f"  Failed:           {len(results['failed'])}")
    for model_id in results["failed"]:
        print(f"    - {model_id}")

    if results["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    load_dotenv()
    main()