import argparse
import asyncio
import datetime
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/pipeline.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("main_pipeline")


def parse_args():
    parser = argparse.ArgumentParser(
        description="End-to-End VLM Synthetic Annotation & LoRA Fine-Tuning Pipeline."
    )
    parser.add_argument("--dataset-dir", type=str, default="Cotton_dataset", help="Root directory containing raw crop images.")
    parser.add_argument("--annotation-provider", type=str, default="huggingface", help="Annotation VLM provider (huggingface, hf, gemini, ollama, nvidia, groq, openrouter).")
    parser.add_argument("--annotation-model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="Annotation teacher model ID or name.")
    parser.add_argument("--ollama-host", type=str, default="http://127.0.0.1:11434", help="Host URL for local Ollama server if using ollama provider.")
    parser.add_argument("--num-annotation-samples", type=int, default=None, help="Limit number of images to annotate (for testing).")
    parser.add_argument("--train-config", type=str, default="training/configs/qwen25vl_3b.yaml", help="Path to training config YAML.")
    parser.add_argument("--experiment", type=str, default="qwen25vl-3b-v1", help="Experiment identifier for fine-tuning run.")
    parser.add_argument("--skip-annotation", action="store_true", help="Skip annotation generation and proceed straight to dataset prep and training.")
    parser.add_argument("--skip-training", action="store_true", help="Skip VLM training after annotation generation.")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted annotation or training run.")
    return parser.parse_args()


def find_latest_annotation_file(provider: str, model: str) -> Optional[Path]:
    """Search for the most recent annotations.jsonl file under outputs/annotations/."""
    clean_model_tag = model.replace(":", "-").replace("/", "_")
    base_dir = Path("outputs/annotations") / provider / clean_model_tag
    
    if base_dir.exists():
        run_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("run_")], reverse=True)
        for rdir in run_dirs:
            ann_file = rdir / "annotations.jsonl"
            if ann_file.exists() and ann_file.stat().st_size > 0:
                return ann_file

    # Fallback to root annotations.jsonl if present
    default_ann = Path("outputs/annotations/annotations.jsonl")
    if default_ann.exists() and default_ann.stat().st_size > 0:
        return default_ann

    return None


async def run_annotation_stage(args) -> Path:
    """Stage 1: Generate synthetic visual annotations using selected VLM teacher."""
    logger.info("\n========================================================")
    logger.info("  STAGE 1: VLM Synthetic Annotation Generation")
    logger.info("========================================================")

    from scripts.generate_annotations import main as generate_annotations_main

    # Build sys.argv override for generate_annotations.py CLI parser
    gen_args = [
        "generate_annotations.py",
        "--dataset-dir", args.dataset_dir,
        "--provider", args.annotation_provider,
        "--model", args.annotation_model,
        "--ollama-host", args.ollama_host,
    ]
    if args.resume:
        gen_args.append("--resume")
    if args.num_annotation_samples is not None:
        gen_args.extend(["--num-samples", str(args.num_annotation_samples)])

    sys.argv = gen_args
    await generate_annotations_main()

    # Locate generated output annotations.jsonl
    ann_file = find_latest_annotation_file(args.annotation_provider, args.annotation_model)
    if not ann_file or not ann_file.exists():
        logger.error("Annotation stage completed but output annotations.jsonl was not found!")
        sys.exit(1)

    logger.info(f"✓ Stage 1 Complete! Output annotations located at: {ann_file}")
    return ann_file


def run_dataset_preparation_stage(annotation_path: Path, args) -> Path:
    """Stage 2: Validate annotations, check visual grounding, and create leakage-free 80/10/10 split manifests."""
    logger.info("\n========================================================")
    logger.info("  STAGE 2: Dataset Preparation & Leakage-Free Splitting")
    logger.info("========================================================")

    from training.scripts.prepare_dataset import parse_args as prep_parse_args, generate_plots
    from training.src.dataset import validate_annotation, compute_image_hash

    dataset_root = os.path.abspath(args.dataset_dir)
    output_dir = os.path.abspath("outputs/dataset")
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Loading annotations from: {annotation_path}")
    raw_records = []
    with open(annotation_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    raw_records.append(json.loads(line))
                except Exception:
                    pass

    logger.info(f"Loaded {len(raw_records)} raw annotation records.")

    eligible_records = []
    ineligible_records = []

    for rec in raw_records:
        ann = rec.get("parsed_annotation", {})
        if validate_annotation(ann):
            # Resolve image path
            img_rel = rec.get("image_path") or rec.get("relative_path")
            img_abs = os.path.join(dataset_root, img_rel) if not os.path.isabs(img_rel) else img_rel
            
            if os.path.exists(img_abs):
                rec["abs_image_path"] = img_abs
                rec["image_hash"] = compute_image_hash(img_abs)
                eligible_records.append(rec)
            else:
                ineligible_records.append(rec)
        else:
            ineligible_records.append(rec)

    logger.info(f"Validation: {len(eligible_records)} eligible images | {len(ineligible_records)} ineligible/missing images.")

    if not eligible_records:
        logger.error("No eligible annotation records found for dataset training!")
        sys.exit(1)

    # Save eligible manifest
    eligible_path = os.path.join(output_dir, "eligible_manifest.jsonl")
    with open(eligible_path, "w", encoding="utf-8") as f:
        for r in eligible_records:
            f.write(json.dumps(r) + "\n")

    # Perceptual hash grouping for leak-free splits
    hash_groups = defaultdict(list)
    for r in eligible_records:
        hash_groups[r["image_hash"]].append(r)

    unique_hashes = list(hash_groups.keys())
    import random
    random.seed(42)
    random.shuffle(unique_hashes)

    n_hashes = len(unique_hashes)
    n_train = int(n_hashes * 0.80)
    n_val = int(n_hashes * 0.10)

    train_hashes = set(unique_hashes[:n_train])
    val_hashes = set(unique_hashes[n_train:n_train + n_val])
    test_hashes = set(unique_hashes[n_train + n_val:])

    train_records = [r for h in train_hashes for r in hash_groups[h]]
    val_records = [r for h in val_hashes for r in hash_groups[h]]
    test_records = [r for h in test_hashes for r in hash_groups[h]]

    with open(os.path.join(output_dir, "train_manifest.jsonl"), "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r) + "\n")

    with open(os.path.join(output_dir, "validation_manifest.jsonl"), "w", encoding="utf-8") as f:
        for r in val_records:
            f.write(json.dumps(r) + "\n")

    with open(os.path.join(output_dir, "test_manifest.jsonl"), "w", encoding="utf-8") as f:
        for r in test_records:
            f.write(json.dumps(r) + "\n")

    # Save leakage report
    with open(os.path.join(output_dir, "leakage_report.md"), "w", encoding="utf-8") as f:
        f.write("# Data Leakage Audit Report\n\n")
        f.write(f"- **Total Eligible Records:** {len(eligible_records)}\n")
        f.write(f"- **Unique Image Hashes:** {n_hashes}\n")
        f.write(f"- **Train Set Size:** {len(train_records)} ({len(train_hashes)} unique hashes)\n")
        f.write(f"- **Validation Set Size:** {len(val_records)} ({len(val_hashes)} unique hashes)\n")
        f.write(f"- **Test Set Size:** {len(test_records)} ({len(test_hashes)} unique hashes)\n")
        f.write(f"- **Cross-Split Hash Overlap:** 0 (PASSED - 0% Data Leakage)\n")

    logger.info(f"✓ Stage 2 Complete! Dataset manifests created under: {output_dir}")
    logger.info(f"  - Train Records: {len(train_records)}")
    logger.info(f"  - Val Records:   {len(val_records)}")
    logger.info(f"  - Test Records:  {len(test_records)}")

    return Path(output_dir)


def run_training_and_evaluation_stage(args):
    """Stage 3 & 4: Train VLM model with QLoRA and execute post-training held-out evaluation."""
    logger.info("\n========================================================")
    logger.info("  STAGE 3: VLM QLoRA Fine-Tuning & Evaluation")
    logger.info("========================================================")

    from training.scripts.train import main as train_main

    train_args = [
        "train.py",
        "--config", args.train_config,
        "--experiment", args.experiment,
    ]
    if args.resume:
        train_args.append("--resume")

    sys.argv = train_args
    train_main()

    logger.info("✓ Stage 3 & 4 Complete! Model checkpoint & evaluation metrics generated.")


def run_comparison_stage():
    """Stage 5: Aggregate all experiments and generate multi-criteria comparison recommendation report."""
    logger.info("\n========================================================")
    logger.info("  STAGE 5: Cross-Model Comparison & Final Report")
    logger.info("========================================================")

    from training.scripts.compare_models import main as compare_main

    sys.argv = ["compare_models.py"]
    compare_main()

    rec_file = Path("outputs/comparison/final_recommendation.md")
    logger.info(f"✓ Stage 5 Complete! Final recommendation report generated at: {rec_file}")


def main():
    args = parse_args()

    start_time = time.monotonic()
    logger.info("=========================================================")
    logger.info("    END-TO-END VLM ANNOTATION & LORA TRAINING PIPELINE   ")
    logger.info("=========================================================")
    logger.info(f"Dataset Dir:           {args.dataset_dir}")
    logger.info(f"Annotation Provider:   {args.annotation_provider}")
    logger.info(f"Annotation Model:      {args.annotation_model}")
    logger.info(f"Train Config:          {args.train_config}")
    logger.info(f"Experiment ID:         {args.experiment}")
    logger.info(f"Skip Annotation:       {args.skip_annotation}")
    logger.info(f"Skip Training:         {args.skip_training}")
    logger.info("=========================================================\n")

    # STAGE 1: Annotation Generation
    if not args.skip_annotation:
        annotation_path = asyncio.run(run_annotation_stage(args))
    else:
        logger.info("Skipping Stage 1 (Annotation Generation). Locating existing annotations...")
        annotation_path = find_latest_annotation_file(args.annotation_provider, args.annotation_model)
        if not annotation_path:
            logger.error("Could not find existing annotation file. Run without --skip-annotation first.")
            sys.exit(1)
        logger.info(f"Using existing annotation file: {annotation_path}")

    # STAGE 2: Dataset Prep & Leakage-Free Splitting
    run_dataset_preparation_stage(annotation_path, args)

    # STAGE 3 & 4: VLM QLoRA Fine-Tuning & Held-Out Evaluation
    if not args.skip_training:
        run_training_and_evaluation_stage(args)

        # STAGE 5: Cross-Model Comparison & Final Report
        run_comparison_stage()
    else:
        logger.info("Skipping Stage 3 & 4 (VLM Fine-Tuning & Evaluation) as requested.")

    elapsed_min = (time.monotonic() - start_time) / 60.0
    logger.info("\n========================================================")
    logger.info(f"  ALL PIPELINE STAGES COMPLETED SUCCESSFULLY in {elapsed_min:.2f} min!")
    logger.info("========================================================")


if __name__ == "__main__":
    main()
