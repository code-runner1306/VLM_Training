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
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import config
from vlm_annotation.src.annotation.checkpoint import CheckpointManager
from vlm_annotation.src.annotation.ollama_health import check_ollama_server_and_model
from vlm_annotation.src.annotation.hf_health import check_huggingface_environment_and_model
from vlm_annotation.src.annotation.retry import RateLimiter, execute_with_retry
from vlm_annotation.src.annotation.validator import AnnotationValidator
from vlm_annotation.src.dataset import discover_dataset
from vlm_annotation.src.models.factory import create_vision_model

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("generate_annotations")


def load_disease_profile(disease_name: str) -> dict:
    profile_path = Path("outputs/disease_profiles") / f"{disease_name}.json"
    if profile_path.exists():
        try:
            with open(profile_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def load_model_configs():
    config_path = Path(__file__).resolve().parent.parent / "vlm_annotation" / "config" / "models.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_prompt_template():
    prompt_path = Path(__file__).resolve().parent.parent / "vlm_annotation" / "prompts" / "annotation.txt"
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def save_benchmark_summary(output_dir: Path, metrics: dict, stats_data: dict, model_name: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "benchmark_summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"# VLM Speed & Throughput Benchmark Summary\n\n")
        f.write(f"- **Evaluated Model:** `{model_name}`\n")
        f.write(f"- **Total Benchmark Samples:** `{stats_data.get('total_processed', 0)}`\n")
        f.write(f"- **Successful Annotations:** `{metrics.get('total_successful', 0)}`\n")
        f.write(f"- **Failed Annotations:** `{metrics.get('total_failed', 0)}`\n")
        f.write(f"- **Average Speed:** `{metrics.get('avg_speed_sec_per_img', 0):.2f} sec/image` (`{metrics.get('throughput_img_per_min', 0):.2f} images/min`)\n")
        f.write(f"- **Total Benchmark Wall Time:** `{metrics.get('total_duration_sec', 0):.2f} seconds`\n")
        f.write(f"- **Estimated 20,000-Image Runtime:** `{stats_data.get('est_20k_hours', 0):.2f} hours`\n")


async def main():
    parser = argparse.ArgumentParser(description="Full Crop Disease Dataset VLM Synthetic Annotation Pipeline.")
    parser.add_argument("--dataset-dir", type=str, default=config.dataset_dir, help="Path to dataset root folder")
    parser.add_argument("--output-dir", type=str, default=None, help="Path to outputs directory")
    parser.add_argument("--provider", type=str, default="gemini", help="VLM Provider (gemini, huggingface, hf, ollama, nvidia, groq, openrouter)")
    parser.add_argument("--model", type=str, default="gemini-flash-latest", help="Model ID or name")
    parser.add_argument("--ollama-host", type=str, default="http://127.0.0.1:11434", help="Host URL for local Ollama server")
    parser.add_argument("--resume", action="store_true", help="Resume annotation, skipping existing image IDs")
    parser.add_argument("--start-index", type=int, default=0, help="Start image index for batch processing")
    parser.add_argument("--end-index", type=int, default=None, help="End image index for batch processing")
    parser.add_argument("--num-samples", type=int, default=None, help="Limit total number of images to annotate")
    parser.add_argument("--retry-failed", action="store_true", help="Re-process only items in failed.jsonl")
    parser.add_argument("--benchmark-speed", action="store_true", help="Run speed & throughput benchmark mode and estimate 20,000-image runtime")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast verification mode with 5 sample images")
    args = parser.parse_args()

    if args.smoke_test and args.num_samples is None:
        args.num_samples = 5
        logger.info("[SMOKE TEST] Capping annotation generation to 5 sample images.")

    # Determine provider-isolated output directory
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_model_tag = args.model.replace(":", "-").replace("/", "_")

    if args.output_dir:
        out_dir = Path(args.output_dir)
    elif args.provider.lower() in ["huggingface", "hf"]:
        out_dir = Path(f"outputs/annotations/huggingface/{clean_model_tag}/run_{timestamp_str}")
    elif args.provider.lower() == "ollama":
        out_dir = Path(f"outputs/annotations/ollama/{clean_model_tag}/run_{timestamp_str}")
    else:
        out_dir = Path(f"outputs/annotations/{args.provider}/{clean_model_tag}/run_{timestamp_str}")

    out_dir.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    # Pre-flight health checks
    if args.provider.lower() in ["huggingface", "hf"]:
        logger.info(f"Running Pre-Flight Health Check for Hugging Face model '{args.model}'...")
        ok, msg = check_huggingface_environment_and_model(model_id=args.model)
        if not ok:
            logger.error(msg)
            sys.exit(1)
        logger.info(msg)
    elif args.provider.lower() == "ollama":
        logger.info(f"Running Pre-Flight Health Check for Ollama model '{args.model}' at {args.ollama_host}...")
        ok, msg = check_ollama_server_and_model(host=args.ollama_host, model_name=args.model)
        if not ok:
            logger.error(msg)
            sys.exit(1)
        logger.info(msg)

    config = load_config()
    model_cfg = None

    for m in config.get("models", []):
        if m.get("model") == args.model or m.get("name") == args.model or (m.get("provider") == args.provider and args.provider == "ollama"):
            model_cfg = m
            break

    if not model_cfg:
        model_cfg = {
            "provider": args.provider,
            "model": args.model,
            "name": f"{args.provider}-{args.model}",
            "host": args.ollama_host,
            "rate_limit": {"requests_per_minute": 60 if args.provider == "ollama" else 30, "max_concurrency": 1 if args.provider == "ollama" else 5}
        }
    else:
        model_cfg["host"] = args.ollama_host
        model_cfg["model"] = args.model

    try:
        model = create_vision_model(model_cfg)
    except Exception as e:
        logger.error(f"Failed to initialize VLM Model '{args.model}': {e}")
        sys.exit(1)

    rate_limit_cfg = model_cfg.get("rate_limit", {})
    limiter = RateLimiter(
        requests_per_minute=rate_limit_cfg.get("requests_per_minute", 60 if args.provider == "ollama" else 30),
        max_concurrency=rate_limit_cfg.get("max_concurrency", 1 if args.provider == "ollama" else 5)
    )

    checkpoint_mgr = CheckpointManager(
        output_file=str(out_dir / "annotations.jsonl"),
        failed_file=str(out_dir / "failed.jsonl")
    )

    validator = AnnotationValidator()
    prompt_template = load_annotation_prompt()

    # Discover images
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

        end_idx = args.end_index if args.end_index is not None else total_found
        sliced_items = all_items[args.start_index:end_idx]

        if args.benchmark_speed:
            sample_size = args.num_samples or 20
            items = sliced_items[:sample_size]
            logger.info(f"[SPEED BENCHMARK MODE] Profiling throughput on next {len(items)} sample images...")
        elif args.resume and args.num_samples is not None:
            uncompleted = [
                item for item in sliced_items
                if not checkpoint_mgr.is_completed(item.image_id)
            ]
            items = uncompleted[:args.num_samples]
            logger.info(f"Discovered {total_found} total images. Processing next {len(items)} unannotated images.")
        elif args.num_samples is not None:
            items = sliced_items[:args.num_samples]
            logger.info(f"Discovered {total_found} total images. Selecting first {len(items)} images.")
        else:
            items = sliced_items
            logger.info(f"Discovered {total_found} total images. Processing range [{args.start_index}:{end_idx}] ({len(items)} items).")

    completed_count = 0
    skipped_count = 0
    failed_count = 0
    latencies_sec = []
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

            latency_s = time.monotonic() - item_start
            latencies_sec.append(latency_s)

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

        elapsed_sec = time.monotonic() - start_time
        processed = completed_count + failed_count
        rpm = (processed / (elapsed_sec / 60.0)) if elapsed_sec > 0 else 0.0
        avg_lat = float(np.mean(latencies_sec)) if latencies_sec else 0.0
        remaining = len(items) - (idx)
        eta_sec = (remaining / (rpm / 60.0)) if rpm > 0 else 0

        logger.info(
            f"Progress: [{idx}/{len(items)}] | Done: {completed_count} | Skipped: {skipped_count} | "
            f"Failed: {failed_count} | RPM: {rpm:.1f} | Avg Lat: {avg_lat:.2f}s | "
            f"ETA: {int(eta_sec//60)}m {int(eta_sec%60)}s"
        )
        sys.stdout.flush()

    total_runtime_s = time.monotonic() - start_time
    avg_lat_s = float(np.mean(latencies_sec)) if latencies_sec else 0.0
    median_lat_s = float(np.median(latencies_sec)) if latencies_sec else 0.0
    p95_lat_s = float(np.percentile(latencies_sec, 95)) if latencies_sec else 0.0
    throughput_ipm = (completed_count / (total_runtime_s / 60.0)) if total_runtime_s > 0 else 0.0
    est_20k_hrs = (20000 / (throughput_ipm * 60.0)) if throughput_ipm > 0 else 0.0

    stats_summary = {
        "provider": model.provider_name,
        "model": model.model_id,
        "host": args.ollama_host if args.provider == "ollama" else "cloud",
        "total_images": len(items),
        "successful": completed_count,
        "skipped": skipped_count,
        "failed": failed_count,
        "avg_latency_sec": round(avg_lat_s, 2),
        "median_latency_sec": round(median_lat_s, 2),
        "p95_latency_sec": round(p95_lat_s, 2),
        "images_per_min": round(throughput_ipm, 2),
        "total_runtime_sec": round(total_runtime_s, 2),
        "est_20k_hours": round(est_20k_hrs, 2),
    }

    save_run_statistics(out_dir, stats_summary)

    # Output run metadata and resource_metrics.json
    with open(out_dir / "resource_metrics.json", "w", encoding="utf-8") as f:
        json.dump(stats_summary, f, indent=2)

    with open(out_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "provider": args.provider,
            "model": args.model,
            "ollama_host": args.ollama_host,
            "concurrency": model_cfg.get("concurrency", 1),
            "output_dir": str(out_dir),
            "timestamp": timestamp_str,
            "speed_benchmark_mode": args.benchmark_speed,
        }, f, indent=2)

    if args.benchmark_speed:
        print("\n" + "=" * 60)
        print("          SPEED & THROUGHPUT BENCHMARK SUMMARY")
        print("=" * 60)
        print(f"Model:                    {model.model_id}")
        print(f"Images Tested:            {completed_count}")
        print(f"Average Latency:          {avg_lat_s:.2f} sec")
        print(f"Median Latency:           {median_lat_s:.2f} sec")
        print(f"P95 Latency:              {p95_lat_s:.2f} sec")
        print(f"Throughput:               {throughput_ipm:.1f} images/min")
        print(f"Estimated 20,000-Image Runtime: ~{est_20k_hrs:.1f} hours ({est_20k_hrs/24:.1f} days)")
        print("=" * 60 + "\n")
    else:
        logger.info(f"\n==========================================")
        logger.info(f"Annotation Run Complete for '{model.model_id}'!")
        logger.info(f"Total Completed: {completed_count} | Skipped: {skipped_count} | Failed: {failed_count}")
        logger.info(f"Saved run outputs to: {out_dir}")
        logger.info(f"==========================================")


if __name__ == "__main__":
    asyncio.run(main())
