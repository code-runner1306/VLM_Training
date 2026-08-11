import argparse
import asyncio
import json
import logging
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vlm_annotation.src.annotation.retry import RateLimiter, execute_with_retry
from vlm_annotation.src.dataset import ImageItem
from vlm_annotation.src.evaluation.benchmark import sample_benchmark_images
from vlm_annotation.src.evaluation.report import generate_benchmark_reports
from vlm_annotation.src.evaluation.scoring import BenchmarkEvaluator
from vlm_annotation.src.models.factory import create_vision_model

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark_models")


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


async def benchmark_single_model(
    m_cfg: Dict[str, Any],
    sample_items: list,
    prompt_template: str,
    evaluator: BenchmarkEvaluator,
    out_path: Path,
    resume: bool
) -> Optional[Dict[str, Any]]:
    if not m_cfg.get("enabled", True):
        logger.info(f"SKIPPED: Model '{m_cfg.get('name')}' is disabled in models.yaml")
        sys.stdout.flush()
        return None

    model_name = m_cfg.get("name", m_cfg.get("model"))
    provider_name = m_cfg.get("provider")

    logger.info(f"Starting parallel benchmark worker for: {model_name} ({provider_name})")
    sys.stdout.flush()

    try:
        model = create_vision_model(m_cfg)
    except Exception as e:
        logger.error(f"SKIPPED: {model_name} initialization failed: {e}")
        sys.stdout.flush()
        return None

    rate_limit_cfg = m_cfg.get("rate_limit", {})
    limiter = RateLimiter(
        requests_per_minute=rate_limit_cfg.get("requests_per_minute", 30),
        max_concurrency=rate_limit_cfg.get("max_concurrency", 5)
    )

    model_output_file = out_path / f"{model_name.replace('/', '_')}.jsonl"
    completed_ids = set()

    if model_output_file.exists() and resume:
        with open(model_output_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        completed_ids.add(record["image_id"])
                    except Exception:
                        pass
        logger.info(f"[{model_name}] Resuming: {len(completed_ids)}/{len(sample_items)} already completed.")
        sys.stdout.flush()

    start_bench_time = time.monotonic()

    with open(model_output_file, "a" if resume else "w", encoding="utf-8") as out_f:
        for idx, item in enumerate(sample_items, start=1):
            if item.image_id in completed_ids:
                continue

            disease_profile = load_disease_profile(item.disease_name)
            formatted_prompt = prompt_template.replace(
                "{DISEASE_NAME}", item.disease_name
            ).replace(
                "{DISEASE_PROFILE_JSON}", json.dumps(disease_profile)
            ).replace(
                "{IMAGE_ID}", item.image_id
            ).replace(
                "{IMAGE_PATH}", item.relative_path
            )

            try:
                response = await execute_with_retry(
                    model.generate_annotation,
                    image_path=item.image_path,
                    disease_name=item.disease_name,
                    prompt=formatted_prompt,
                    disease_profile=disease_profile,
                    rate_limiter=limiter,
                    model_instance=model
                )

                score_res = await evaluator.evaluate_annotation(
                    image_path=item.image_path,
                    disease_name=item.disease_name,
                    candidate_json=response.parsed_json,
                    raw_response=response.raw_response
                )

                record = {
                    "image_id": item.image_id,
                    "image_path": item.relative_path,
                    "disease": item.disease_name,
                    "provider": provider_name,
                    "model": model_name,
                    "status": response.status,
                    "latency_ms": response.latency_ms,
                    "score": score_res.total_score,
                    "parsed_annotation": response.parsed_json,
                    "raw_response": response.raw_response,
                    "judge_feedback": score_res.judge_feedback
                }

                out_f.write(json.dumps(record) + "\n")
                out_f.flush()

                logger.info(f"[{model_name}] [{idx}/{len(sample_items)}] {item.relative_path} | Latency: {response.latency_ms:.0f}ms | Score: {score_res.total_score:.1f} | 429 Hits: {model.rate_limit_hits}")
                sys.stdout.flush()

            except Exception as exc:
                logger.error(f"[{model_name}] [{idx}/{len(sample_items)}] Failed item {item.image_id}: {exc}")
                sys.stdout.flush()

    total_time_sec = time.monotonic() - start_bench_time

    # Calculate final aggregated statistics across ALL completed records in jsonl
    all_records = []
    if model_output_file.exists():
        with open(model_output_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        all_records.append(json.loads(line))
                    except Exception:
                        pass

    processed_total = len(all_records)
    successful_count = sum(1 for r in all_records if r.get("status") == "success")
    failed_count = processed_total - successful_count

    latencies = [r["latency_ms"] for r in all_records if "latency_ms" in r]
    scores = [r["score"] for r in all_records if "score" in r]

    avg_lat = statistics.mean(latencies) if latencies else 0.0
    med_lat = statistics.median(latencies) if latencies else 0.0
    p95_lat = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else avg_lat
    avg_score = statistics.mean(scores) if scores else 0.0
    validity_rate = (successful_count / processed_total * 100.0) if processed_total > 0 else 0.0
    rpm = (processed_total / (total_time_sec / 60.0)) if total_time_sec > 0 else 0.0

    metrics = model.get_metrics()
    summary = {
        "model": model_name,
        "provider": provider_name,
        "images_processed": processed_total,
        "successful_requests": successful_count,
        "failed_requests": failed_count,
        "rate_limit_hits": metrics["rate_limit_hits"],
        "json_parse_failures": metrics["json_parse_failures"],
        "json_validity_rate": round(validity_rate, 2),
        "average_latency_ms": round(avg_lat, 2),
        "median_latency_ms": round(med_lat, 2),
        "p95_latency_ms": round(p95_lat, 2),
        "requests_per_minute": round(rpm, 2),
        "overall_score": round(avg_score, 2)
    }

    logger.info(f"✓ Model '{model_name}' Completed! ({processed_total}/{len(sample_items)} images) | Score: {avg_score:.2f} | 429 Rate Limit Hits: {metrics['rate_limit_hits']} | Parse Errors: {metrics['json_parse_failures']}")
    sys.stdout.flush()
    return summary


async def main():
    parser = argparse.ArgumentParser(description="Run 200-image VLM Benchmark across configured models in parallel.")
    parser.add_argument("--dataset-dir", type=str, default="Cotton_dataset", help="Dataset root directory")
    parser.add_argument("--output-dir", type=str, default="outputs/benchmark", help="Benchmark output folder")
    parser.add_argument("--sample-count", type=int, default=200, help="Number of benchmark sample images")
    parser.add_argument("--provider", type=str, default=None, help="Specific provider to benchmark (gemini, ollama, nvidia, groq, openrouter)")
    parser.add_argument("--model", type=str, default=None, help="Specific model ID to benchmark")
    parser.add_argument("--ollama-host", type=str, default="http://127.0.0.1:11434", help="Ollama server host URL")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted benchmark")
    args = parser.parse_args()

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Pre-flight health checks
    if args.provider and args.provider.lower() in ["huggingface", "hf"]:
        from vlm_annotation.src.annotation.hf_health import check_huggingface_environment_and_model
        model_id = args.model or "Qwen/Qwen2.5-VL-7B-Instruct"
        logger.info(f"Running Pre-Flight Health Check for Hugging Face model '{model_id}'...")
        ok, msg = check_huggingface_environment_and_model(model_id=model_id)
        if not ok:
            logger.error(msg)
            sys.exit(1)
        logger.info(msg)
    elif args.provider and args.provider.lower() == "ollama":
        from vlm_annotation.src.annotation.ollama_health import check_ollama_server_and_model
        model_id = args.model or "qwen3-vl:8b"
        logger.info(f"Running Pre-Flight Health Check for Ollama model '{model_id}' at {args.ollama_host}...")
        ok, msg = check_ollama_server_and_model(host=args.ollama_host, model_name=model_id)
        if not ok:
            logger.error(msg)
            sys.exit(1)
        logger.info(msg)

    # 1. Sample benchmark images
    logger.info(f"Sampling {args.sample_count} benchmark images across dataset '{args.dataset_dir}'...")
    sample_items = sample_benchmark_images(
        dataset_dir=args.dataset_dir,
        target_count=args.sample_count,
        output_path=str(out_path / "benchmark_images.json")
    )
    logger.info(f"Sampled {len(sample_items)} benchmark images.")
    sys.stdout.flush()

    config = load_config()
    models_config = config.get("models", [])
    prompt_template = load_annotation_prompt()
    evaluator = BenchmarkEvaluator()

    if args.provider and args.model:
        selected_cfg = {
            "provider": args.provider,
            "model": args.model,
            "name": f"{args.provider}-{args.model.replace(':', '-')}",
            "host": args.ollama_host,
            "enabled": True,
            "rate_limit": {"requests_per_minute": 60 if args.provider == "ollama" else 30, "max_concurrency": 1 if args.provider == "ollama" else 5}
        }
        models_config = [selected_cfg]

    logger.info(f"\n==========================================")
    logger.info(f"Launching Parallel Benchmarking Workers for Enabled Models")
    logger.info(f"==========================================")
    sys.stdout.flush()

    tasks = [
        benchmark_single_model(m_cfg, sample_items, prompt_template, evaluator, out_path, args.resume)
        for m_cfg in models_config
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    model_summaries = []
    for res in results:
        if isinstance(res, dict):
            model_summaries.append(res)
        elif isinstance(res, Exception):
            logger.error(f"Worker task failed with exception: {res}")

    # 2. Generate Reports
    generate_benchmark_reports(model_summaries, output_dir=args.output_dir)
    logger.info(f"\n✓ Parallel Benchmark Complete! Final reports generated at '{args.output_dir}'.")
    sys.stdout.flush()


if __name__ == "__main__":
    asyncio.run(main())
