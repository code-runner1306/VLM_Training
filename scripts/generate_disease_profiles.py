import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vlm_annotation.src.annotation.retry import RateLimiter, execute_with_retry
from vlm_annotation.src.dataset import discover_dataset
from vlm_annotation.src.models.factory import create_vision_model

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_disease_profiles")


def load_model_configs():
    config_path = Path(__file__).resolve().parent.parent / "vlm_annotation" / "config" / "models.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_profile_prompt_template():
    prompt_path = Path(__file__).resolve().parent.parent / "vlm_annotation" / "prompts" / "disease_profile.txt"
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


async def main():
    parser = argparse.ArgumentParser(description="Generate and cache structured disease profiles.")
    parser.add_argument("--dataset-dir", type=str, default="dataset", help="Path to dataset directory")
    parser.add_argument("--output-dir", type=str, default="outputs/disease_profiles", help="Output directory for profiles")
    parser.add_argument("--provider", type=str, default="gemini", help="Provider for LLM/VLM profile generator")
    parser.add_argument("--model", type=str, default="gemini-flash-latest", help="Model ID")
    parser.add_argument("--force", action="store_true", help="Force regeneration of cached profiles")
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Scanning dataset at '{args.dataset_dir}'...")
    try:
        items, by_disease = discover_dataset(args.dataset_dir)
    except Exception as e:
        logger.error(f"Failed to scan dataset directory: {e}")
        sys.exit(1)

    diseases = list(by_disease.keys())
    logger.info(f"Discovered {len(diseases)} disease classes: {diseases}")

    configs = load_model_configs()
    model_cfg = None
    for m in configs.get("models", []):
        if m.get("model") == args.model or m.get("name") == args.model:
            model_cfg = m
            break

    if not model_cfg:
        model_cfg = {
            "provider": args.provider,
            "model": args.model,
            "name": f"{args.provider}-{args.model}",
            "rate_limit": {"requests_per_minute": 5, "max_concurrency": 1}
        }

    try:
        model = create_vision_model(model_cfg)
    except Exception as e:
        logger.error(f"Could not initialize provider '{args.provider}': {e}")
        sys.exit(1)

    rate_limit_cfg = model_cfg.get("rate_limit", {})
    limiter = RateLimiter(
        requests_per_minute=rate_limit_cfg.get("requests_per_minute", 5),
        max_concurrency=rate_limit_cfg.get("max_concurrency", 1)
    )

    prompt_template = get_profile_prompt_template()

    for disease_name in diseases:
        profile_file = output_path / f"{disease_name}.json"
        if profile_file.exists() and not args.force:
            # Check if valid non-empty profile file
            try:
                with open(profile_file, "r", encoding="utf-8") as pf:
                    data = json.load(pf)
                    if data and isinstance(data, dict) and "disease" in data:
                        logger.info(f"PROFILE CACHED: Skipping '{disease_name}' (found {profile_file})")
                        continue
            except Exception:
                pass

        logger.info(f"Generating disease profile for '{disease_name}'...")
        prompt = prompt_template.replace("{DISEASE_NAME}", disease_name)

        sample_item = by_disease[disease_name][0]
        try:
            response = await execute_with_retry(
                model.generate_annotation,
                image_path=sample_item.image_path,
                disease_name=disease_name,
                prompt=prompt,
                rate_limiter=limiter,
                model_instance=model
            )

            if response.status == "success" and response.parsed_json:
                profile_data = response.parsed_json
                profile_data["disease"] = disease_name
                with open(profile_file, "w", encoding="utf-8") as f:
                    json.dump(profile_data, f, indent=2)
                logger.info(f"✓ Saved profile for '{disease_name}' -> {profile_file}")
            else:
                logger.error(f"✗ Failed to generate profile for '{disease_name}': {response.error_message or response.status}")
        except Exception as exc:
            logger.error(f"✗ Exception generating profile for '{disease_name}': {exc}")

    metrics = model.get_metrics()
    logger.info(f"\nProfile Generation Metrics: {metrics}")


if __name__ == "__main__":
    asyncio.run(main())
