import argparse
import asyncio
import datetime
import json
import logging
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vlm_annotation.src.annotation.checkpoint import CheckpointManager
from vlm_annotation.src.annotation.retry import RateLimiter, execute_with_retry
from vlm_annotation.src.annotation.validator import AnnotationValidator
from vlm_annotation.src.dataset import discover_dataset
from vlm_annotation.src.models.factory import create_vision_model

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/annotation.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("generate_annotations")


def load_config():
    config_path = Path(__file__).resolve().parent.parent / "vlm_annotation" / "config" / "models.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_annotation_prompt():
    prompt_path = Path(__file__).resolve().parent.parent / "vlm_annotation" / "prompts" / "annotation.txt"
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def load_disease_profile(disease_name: str) -> dict:
    profile_path = Path("outputs/disease_profiles") / f"{disease_name}.json"
    if profile_path.exists():
        with open(profile_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"disease": disease_name}


async def main():
    parser = argparse.ArgumentParser(description="Full Cotton Disease Dataset VLM Synthetic Annotation Pipeline.")
    parser.add_argument("--dataset-dir", type=str, default="dataset", help="Path to dataset root folder")
    parser.add_argument("--output-dir", type=str, default="outputs/annotations", help="Path to outputs directory")
    parser.add_argument("--provider", type=str, default="gemini", help="VLM Provider (gemini, nvidia, groq, openrouter)")
    parser.add_argument("--model", type=str, default="gemini-flash-latest", help="Model ID or name")
    parser.add_argument("--resume", action="store_true", help="Resume annotation, skipping existing image IDs")
    parser.add_argument("--start-index", type=int, default=0, help="Start image index for batch processing")
    parser.add_argument("--end-index", type=int, default=None, help="End image index for batch processing")
    parser.add_argument("--num-samples", type=int, default=None, help="Limit total number of images to annotate")
    parser.add_argument("--retry-failed", action="store_true", help="Re-process only items in failed.jsonl")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    config = load_config()
    model_cfg = None

    # Match model config from models.yaml or create dynamic config
    for m in config.get("models", []):
        if m.get("model") == args.model or m.get("name") == args.model:
            model_cfg = m
            break

    if not model_cfg:
        model_cfg = {
            "provider": args.provider,
            "model": args.model,
            "name": f"{args.provider}-{args.model}",
            "rate_limit": {"requests_per_minute": 30, "max_concurrency": 5}
        }

    try:
        model = create_vision_model(model_cfg)
    except Exception as e:
        logger.error(f"Failed to initialize VLM Model '{args.model}': {e}")
        sys.exit(1)

    rate_limit_cfg = model_cfg.get("rate_limit", {})
    limiter = RateLimiter(
        requests_per_minute=rate_limit_cfg.get("requests_per_minute", 30),
        max_concurrency=rate_limit_cfg.get("max_concurrency", 5)
    )

    checkpoint_mgr = CheckpointManager(
        output_file=str(out_dir / "annotations.jsonl"),
        failed_file=str(out_dir / "failed.jsonl")
    )

    validator = AnnotationValidator()
    prompt_template = load_annotation_prompt()

    # Discover images or load failed queue
    if args.retry_failed:
        logger.info("RETRY FAILED MODE: Loading items from failed.jsonl...")
        items = []
        failed_file = out_dir / "failed.jsonl"
        if failed_file.exists():
            with open(failed_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        items.append(data)
    else:
        logger.info(f"Scanning dataset at '{args.dataset_dir}'...")
        all_items, _ = discover_dataset(args.dataset_dir)
        total_found = len(all_items)
        if args.num_samples is not None:
            end_idx = min(args.start_index + args.num_samples, total_found)
        else:
            end_idx = args.end_index if args.end_index is not None else total_found
        items = all_items[args.start_index:end_idx]
        logger.info(f"Discovered {total_found} total images. Processing range [{args.start_index}:{end_idx}] ({len(items)} items).")

    completed_count = 0
    skipped_count = 0
    failed_count = 0
    total_latency_ms = 0.0
    start_time = time.monotonic()

    for idx, item in enumerate(items, start=1):
        image_id = item["image_id"] if isinstance(item, dict) else item.image_id
        image_path = item["image_path"] if isinstance(item, dict) else item.image_path
        relative_path = item["relative_path"] if isinstance(item, dict) else item.relative_path
        disease_name = item["disease_name"] if isinstance(item, dict) else item.disease_name

        if args.resume and checkpoint_mgr.is_completed(image_id):
            skipped_count += 1
            continue

        disease_profile = load_disease_profile(disease_name)
        formatted_prompt = prompt_template.replace(
            "{DISEASE_NAME}", disease_name
        ).replace(
            "{DISEASE_PROFILE_JSON}", json.dumps(disease_profile)
        ).replace(
            "{IMAGE_ID}", image_id
        ).replace(
            "{IMAGE_PATH}", relative_path
        )

        item_start = time.monotonic()
        try:
            response = await execute_with_retry(
                model.generate_annotation,
                image_path=image_path,
                disease_name=disease_name,
                prompt=formatted_prompt,
                disease_profile=disease_profile,
                rate_limiter=limiter,
                model_instance=model
            )

            latency_ms = (time.monotonic() - item_start) * 1000.0
            total_latency_ms += latency_ms

            if response.status == "success" and response.parsed_json:
                is_valid, quality_status, validation_msg = validator.validate(response.parsed_json, disease_name)

                record = {
                    "image_id": image_id,
                    "image_path": relative_path,
                    "disease": disease_name,
                    "quality_status": quality_status,
                    "parsed_annotation": response.parsed_json,
                    "raw_response": response.raw_response,
                    "teacher_model": model.model_id,
                    "teacher_provider": model.provider_name,
                    "prompt_version": "1.0",
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
                }

                checkpoint_mgr.save_annotation(record)
                completed_count += 1
            else:
                failed_count += 1
                fail_record = {
                    "image_id": image_id,
                    "image_path": relative_path,
                    "provider": model.provider_name,
                    "model": model.model_id,
                    "error": response.error_message or response.status,
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
                }
                checkpoint_mgr.save_failed(fail_record)

        except Exception as e:
            failed_count += 1
            fail_record = {
                "image_id": image_id,
                "image_path": relative_path,
                "provider": model.provider_name,
                "model": model.model_id,
                "error": str(e),
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
            }
            checkpoint_mgr.save_failed(fail_record)

        # Real-time CLI progress metrics
        elapsed_sec = time.monotonic() - start_time
        processed = completed_count + failed_count
        rpm = (processed / (elapsed_sec / 60.0)) if elapsed_sec > 0 else 0.0
        avg_lat = (total_latency_ms / processed / 1000.0) if processed > 0 else 0.0
        remaining = len(items) - (idx)
        eta_sec = (remaining / (rpm / 60.0)) if rpm > 0 else 0

        logger.info(
            f"Progress: [{idx}/{len(items)}] | Done: {completed_count} | Skipped: {skipped_count} | "
            f"Failed: {failed_count} | RPM: {rpm:.1f} | Avg Lat: {avg_lat:.2f}s | "
            f"429 Hits: {model.rate_limit_hits} | ETA: {int(eta_sec//60)}m {int(eta_sec%60)}s"
        )
        sys.stdout.flush()

    # Save model diagnostic counters to outputs/model_metrics.json
    metrics_file = Path("outputs/model_metrics.json")
    all_metrics = {}
    if metrics_file.exists():
        with open(metrics_file, "r", encoding="utf-8") as f:
            try:
                all_metrics = json.load(f)
            except Exception:
                pass
    all_metrics[model.model_id] = model.get_metrics()
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    logger.info(f"\n==========================================")
    logger.info(f"Annotation Run Complete for '{model.model_id}'!")
    logger.info(f"Total Completed: {completed_count} | Skipped: {skipped_count} | Failed: {failed_count}")
    logger.info(f"Cumulative Model Counters: {model.get_metrics()}")
    logger.info(f"==========================================")


if __name__ == "__main__":
    asyncio.run(main())
