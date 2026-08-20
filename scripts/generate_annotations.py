import argparse
import asyncio
import datetime
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vlm_annotation.src.annotation.checkpoint import CheckpointManager
from vlm_annotation.src.annotation.keys import load_gemini_keys, resolve_worker_count
from vlm_annotation.src.annotation.ollama_health import check_ollama_server_and_model
from vlm_annotation.src.annotation.hf_health import check_huggingface_environment_and_model
from vlm_annotation.src.annotation.parallel import (
    ParallelAnnotationError,
    aggregate_statistics,
    build_worker_command,
    build_worker_env,
    run_workers,
    seed_workers_for_resume,
    slice_chunks,
    write_run_metadata,
)
from vlm_annotation.src.annotation.store import (
    append_batch,
    compute_coverage,
    promote_merge,
    prune_worker_dirs,
    recompute_statistics,
    store_dir,
)
from vlm_annotation.src.annotation.retry import RateLimiter, execute_with_retry
from vlm_annotation.src.annotation.validator import AnnotationValidator
from vlm_annotation.src.dataset import discover_dataset
from vlm_annotation.src.models.factory import create_vision_model

load_dotenv()
Path("logs").mkdir(exist_ok=True)
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


def save_run_statistics(out_dir: Path, stats_data: dict):
    with open(out_dir / "statistics.json", "w", encoding="utf-8") as f:
        json.dump(stats_data, f, indent=2)

    with open(out_dir / "statistics.md", "w", encoding="utf-8") as f:
        f.write(f"# Annotation Run Statistics Summary\n\n")
        f.write(f"- **Provider / Model:** `{stats_data.get('provider')}` / `{stats_data.get('model')}`\n")
        f.write(f"- **Total Images:** `{stats_data.get('total_images')}`\n")
        f.write(f"- **Successful Annotations:** `{stats_data.get('successful')}`\n")
        f.write(f"- **Failed Annotations:** `{stats_data.get('failed')}`\n")
        f.write(f"- **Average Latency:** `{stats_data.get('avg_latency_sec', 0):.2f}s`\n")
        f.write(f"- **Median Latency:** `{stats_data.get('median_latency_sec', 0):.2f}s`\n")
        f.write(f"- **P95 Latency:** `{stats_data.get('p95_latency_sec', 0):.2f}s`\n")
        f.write(f"- **Throughput:** `{stats_data.get('images_per_min', 0):.1f} images/min`\n")


def run_parallel_annotation(args, run_dir: Path, timestamp_str: str):
    """Parallel Gemini annotation: one subprocess per key on a strict image slice, then promote to store."""
    keys = load_gemini_keys()
    worker_count = resolve_worker_count(keys, args.max_workers)
    logger.info(f"[PARALLEL] Found {len(keys)} Gemini key(s); running with {worker_count} worker(s).")

    # Discover images once for the global order & total count
    all_items, _ = discover_dataset(args.dataset_dir)
    total_found = len(all_items)
    logger.info(f"Discovered {total_found} total images at '{args.dataset_dir}'.")

    chunk_size = args.chunk_size or int(math.ceil(total_found / worker_count))
    chunks = slice_chunks(total_found, args.start_index, chunk_size, worker_count)
    store = store_dir(args.dataset_dir)

    commands = []
    envs = []
    active_chunks = []
    for n, (start, end) in enumerate(chunks):
        if start >= end:
            logger.info(f"[PARALLEL] Worker {n}: empty slice, skipping.")
            continue
        worker_dir = str(run_dir / f"worker_{n}")
        commands.append(
            build_worker_command(
                script_path=str(Path(__file__).resolve()),
                dataset_dir=args.dataset_dir,
                model=args.model,
                worker_dir=worker_dir,
                start_index=start,
                end_index=end,
                resume=args.resume,
                prompt_version=args.prompt_version,
                force_regenerate=args.force_regenerate,
            )
        )
        envs.append(build_worker_env(keys[n]))
        active_chunks.append((n, start, end))
        logger.info(f"[PARALLEL] Worker {n}: images [{start}:{end}) ({end - start} images).")

    if not commands:
        logger.error("No workers spawned: no images in the requested range.")
        raise RuntimeError("No parallel workers spawned: no images in the requested range.")

    # Resume seeding: carry already-completed image IDs from the canonical store into each worker checkpoint
    seed_workers_for_resume(run_dir, worker_count, store / "annotations.jsonl", args.resume, args.force_regenerate)

    exit_codes = run_workers(commands, envs, cwd=str(Path(__file__).resolve().parent.parent))
    failed_workers = [n for n, code in zip([c[0] for c in active_chunks], exit_codes) if code != 0]
    if failed_workers:
        logger.error(f"[PARALLEL] Workers failed with non-zero exit: {failed_workers}. Promoting partial results...")

    # Promote worker results into the canonical store and update traceability
    promote = promote_merge(
        store,
        run_dir,
        worker_count,
        provider="gemini",
        model=args.model,
        prompt_version=args.prompt_version,
        force_regenerate=args.force_regenerate,
    )
    coverage = compute_coverage(store, args.dataset_dir, provider="gemini", model=args.model)
    batch_stats = aggregate_statistics(run_dir, worker_count, provider="gemini", model=args.model)
    append_batch(store, {
        "timestamp": timestamp_str,
        "provider": "gemini",
        "model": args.model,
        "prompt_version": args.prompt_version,
        "start_index": args.start_index,
        "end_index": min(args.start_index + worker_count * chunk_size, total_found),
        "annotated": promote["annotated"],
        "failed": promote["failed"],
        "runtime_sec": batch_stats["total_runtime_sec"],
        "avg_latency_sec": batch_stats["avg_latency_sec"],
        "images_per_min": batch_stats["images_per_min"],
    })
    recompute_statistics(store, provider="gemini", model=args.model)
    write_run_metadata(run_dir, timestamp_str, worker_count, keys, chunks, args.model)
    prune_worker_dirs(run_dir, worker_count)

    print("\n" + "=" * 60)
    print("          PARALLEL ANNOTATION SUMMARY")
    print("=" * 60)
    print(f"Model:                 {args.model}")
    print(f"Images Annotated:      {promote['annotated']}")
    print(f"Failed:                {promote['failed']}")
    print(f"Throughput:            {batch_stats['images_per_min']:.1f} images/min")
    print(f"Total Runtime:         {batch_stats['total_runtime_sec']:.2f} sec")
    print(f"Store Coverage:        {coverage['annotated']}/{coverage['dataset_total']} "
          f"annotated, {coverage['missing']} missing, {coverage['failed']} failed")
    print("=" * 60 + "\n")

    if failed_workers:
        raise ParallelAnnotationError(f"Parallel annotation failed: workers {failed_workers} exited non-zero. Partial results promoted to {store}.")


async def main():
    parser = argparse.ArgumentParser(description="Full Cotton Disease Dataset VLM Synthetic Annotation Pipeline.")
    parser.add_argument("--dataset-dir", type=str, default="Cotton_dataset", help="Path to dataset root folder")
    parser.add_argument("--output-dir", type=str, default=None, help="Path to outputs directory")
    parser.add_argument("--provider", type=str, default="gemini", help="VLM Provider (gemini, huggingface, hf, ollama, nvidia, groq, openrouter)")
    parser.add_argument("--model", type=str, default="gemini-flash-latest", help="Model ID or name")
    parser.add_argument("--ollama-host", type=str, default="http://127.0.0.1:11434", help="Host URL for local Ollama server")
    parser.add_argument("--resume", action="store_true", help="Resume annotation, skipping existing image IDs")
    parser.add_argument("--start-index", type=int, default=0, help="Start image index for batch processing")
    parser.add_argument("--end-index", type=int, default=None, help="End image index for batch processing")
    parser.add_argument("--num-samples", type=int, default=None, help="Limit total number of images to annotate")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast verification mode with 5 sample images")
    parser.add_argument("--max-workers", type=int, default=None, help="Max parallel Gemini workers (default: min(keys, cpu_count, 4) or MAX_GEMINI_WORKERS env)")
    parser.add_argument("--chunk-size", type=int, default=None, help="Images per worker slice in parallel mode (default: covers the full dataset across workers)")
    parser.add_argument("--run-dir", type=str, default=None, help="Reuse an existing run directory instead of creating a new timestamped one (parallel mode)")
    parser.add_argument("--force-regenerate", action="store_true", help="Ignore existing store records and re-annotate with a new prompt version")
    parser.add_argument("--prompt-version", type=str, default="1.0", help="Prompt version tag written on annotation records")
    args = parser.parse_args()

    if args.smoke_test and args.num_samples is None:
        args.num_samples = 5
        logger.info("[SMOKE TEST] Capping annotation generation to 5 sample images.")

    # Parallel Gemini mode: auto-detected when >=2 Gemini keys are available.
    if args.provider.lower() == "gemini":
        keys = load_gemini_keys()
        worker_count = resolve_worker_count(keys, args.max_workers)
        if worker_count >= 2:
            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            clean_model_tag = args.model.replace(":", "-").replace("/", "_")
            if args.run_dir:
                run_dir = Path(args.run_dir)
                run_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"[PARALLEL] Reusing run directory: {run_dir}")
            else:
                run_dir = Path(f"outputs/annotations/gemini/{clean_model_tag}/run_{timestamp_str}")
                run_dir.mkdir(parents=True, exist_ok=True)
            Path("logs").mkdir(exist_ok=True)
            return run_parallel_annotation(args, run_dir, timestamp_str)

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
    logger.info(f"Scanning dataset at '{args.dataset_dir}'...")
    all_items, _ = discover_dataset(args.dataset_dir)
    total_found = len(all_items)

    end_idx = args.end_index if args.end_index is not None else total_found
    sliced_items = all_items[args.start_index:end_idx]

    if args.resume and args.num_samples is not None:
        uncompleted = [
            item for item in sliced_items
            if args.force_regenerate or not checkpoint_mgr.is_completed(item.image_id)
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

        if args.resume and not args.force_regenerate and checkpoint_mgr.is_completed(image_id):
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
                    "prompt_version": args.prompt_version,
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
            "prompt_version": args.prompt_version,
        }, f, indent=2)

    logger.info(f"\n==========================================")
    logger.info(f"Annotation Run Complete for '{model.model_id}'!")
    logger.info(f"Total Completed: {completed_count} | Skipped: {skipped_count} | Failed: {failed_count}")
    logger.info(f"Saved run outputs to: {out_dir}")
    logger.info(f"==========================================")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except ParallelAnnotationError as e:
        logger.error(str(e))
        sys.exit(1)
