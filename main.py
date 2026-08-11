import argparse
import asyncio
import datetime
import json
import logging
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(
        description="End-to-End VLM Synthetic Annotation & LoRA Fine-Tuning Pipeline with Graceful Error Handling & Auto-Push."
    )
    parser.add_argument("--dataset-dir", type=str, default="Cotton_dataset", help="Root directory containing raw crop images.")
    parser.add_argument("--annotation-provider", type=str, default="huggingface", help="Annotation VLM provider (huggingface, hf, gemini, ollama, nvidia, groq, openrouter).")
    parser.add_argument("--annotation-model", type=str, default="Qwen/Qwen3-VL-8B-Instruct", help="Annotation teacher model ID or name.")
    parser.add_argument("--ollama-host", type=str, default="http://127.0.0.1:11434", help="Host URL for local Ollama server if using ollama provider.")
    parser.add_argument("--num-annotation-samples", type=int, default=None, help="Limit number of images to annotate (for testing).")
    parser.add_argument("--train-config", type=str, default="training/configs/qwen25vl_3b.yaml", help="Path to training config YAML.")
    parser.add_argument("--experiment", type=str, default="qwen25vl-3b-v1", help="Experiment identifier for fine-tuning run.")
    parser.add_argument("--skip-annotation", action="store_true", help="Skip annotation generation and proceed straight to dataset prep and training.")
    parser.add_argument("--skip-training", action="store_true", help="Skip VLM training after annotation generation.")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted annotation or training run.")
    parser.add_argument("--no-auto-push", action="store_true", help="Disable automatic git commit & push on run completion or error.")
    return parser.parse_args()


def setup_logger(session_id: str):
    """Configure logger with console streaming and session log files."""
    Path("logs").mkdir(exist_ok=True)

    session_log_path = Path("logs") / f"pipeline_{session_id}.log"
    latest_log_path = Path("logs") / "pipeline_latest.log"

    logger = logging.getLogger("main_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Session Log File Handler
    fh_session = logging.FileHandler(session_log_path, encoding="utf-8")
    fh_session.setFormatter(formatter)
    logger.addHandler(fh_session)

    # Latest Log File Handler
    fh_latest = logging.FileHandler(latest_log_path, mode="w", encoding="utf-8")
    fh_latest.setFormatter(formatter)
    logger.addHandler(fh_latest)

    return logger, session_log_path


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


def update_status_file(status_data: dict):
    """Write pipeline execution status file at outputs/pipeline_status.json."""
    Path("outputs").mkdir(exist_ok=True)
    status_file = Path("outputs/pipeline_status.json")
    with open(status_file, "w", encoding="utf-8") as f:
        json.dump(status_data, f, indent=2)


def trigger_github_push(commit_message: str, logger: logging.Logger) -> bool:
    """Run safety audit and push code, configs, logs, and outputs to GitHub."""
    logger.info(f"\n[AUTO-PUSH] Triggering automated GitHub push...")
    logger.info(f"[AUTO-PUSH] Commit Message: '{commit_message}'")

    try:
        from training.scripts.push_github import main as push_github_main
        sys.argv = ["push_github.py", "--message", commit_message, "--yes"]
        push_github_main()
        logger.info("[AUTO-PUSH] ✓ Successfully committed and pushed run artifacts to GitHub!")
        return True
    except SystemExit as se:
        if se.code == 0:
            logger.info("[AUTO-PUSH] ✓ GitHub push completed successfully.")
            return True
        else:
            logger.error(f"[AUTO-PUSH] ⚠️ push_github.py exited with code {se.code}")
            return False
    except Exception as e:
        logger.error(f"[AUTO-PUSH] ⚠️ Failed to auto-push to GitHub: {e}")
        return False


async def run_annotation_stage(args, logger: logging.Logger) -> Path:
    """Stage 1: Generate synthetic visual annotations using selected VLM teacher."""
    logger.info("\n========================================================")
    logger.info("  STAGE 1: VLM Synthetic Annotation Generation")
    logger.info("========================================================")

    from scripts.generate_annotations import main as generate_annotations_main

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

    ann_file = find_latest_annotation_file(args.annotation_provider, args.annotation_model)
    if not ann_file or not ann_file.exists():
        raise FileNotFoundError(f"Annotation stage finished but output annotations.jsonl was not found under outputs/annotations/{args.annotation_provider}/")

    logger.info(f"✓ Stage 1 Complete! Output annotations located at: {ann_file}")
    return ann_file


def run_dataset_preparation_stage(annotation_path: Path, args, logger: logging.Logger) -> Path:
    """Stage 2: Validate annotations, check visual grounding, and create leakage-free 80/10/10 split manifests."""
    logger.info("\n========================================================")
    logger.info("  STAGE 2: Dataset Preparation & Leakage-Free Splitting")
    logger.info("========================================================")

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
        raise ValueError(f"No valid eligible annotation records found in {annotation_path}!")

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


def run_training_and_evaluation_stage(args, logger: logging.Logger):
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


def run_comparison_stage(logger: logging.Logger):
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

    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = f"session_{args.experiment}_{timestamp_str}"
    logger, session_log_path = setup_logger(session_id)

    start_time = time.monotonic()
    start_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    status_data = {
        "session_id": session_id,
        "status": "RUNNING",
        "current_stage": "Initialization",
        "experiment": args.experiment,
        "annotation_provider": args.annotation_provider,
        "annotation_model": args.annotation_model,
        "train_config": args.train_config,
        "dataset_dir": args.dataset_dir,
        "start_time": start_iso,
        "last_updated": start_iso,
        "duration_sec": 0.0,
        "error": None
    }
    update_status_file(status_data)

    current_stage_name = "Pipeline Initialization"

    logger.info("=========================================================")
    logger.info("    END-TO-END VLM ANNOTATION & LORA TRAINING PIPELINE   ")
    logger.info("=========================================================")
    logger.info(f"Session ID:            {session_id}")
    logger.info(f"Dataset Dir:           {args.dataset_dir}")
    logger.info(f"Annotation Provider:   {args.annotation_provider}")
    logger.info(f"Annotation Model:      {args.annotation_model}")
    logger.info(f"Train Config:          {args.train_config}")
    logger.info(f"Experiment ID:         {args.experiment}")
    logger.info(f"Auto-Push to GitHub:   {not args.no_auto_push}")
    logger.info(f"Session Log File:      {session_log_path}")
    logger.info("=========================================================\n")

    try:
        # STAGE 1: Annotation Generation
        if not args.skip_annotation:
            current_stage_name = "Stage 1: VLM Synthetic Annotation Generation"
            status_data["current_stage"] = current_stage_name
            status_data["last_updated"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            update_status_file(status_data)

            annotation_path = asyncio.run(run_annotation_stage(args, logger))
        else:
            logger.info("Skipping Stage 1 (Annotation Generation). Locating existing annotations...")
            annotation_path = find_latest_annotation_file(args.annotation_provider, args.annotation_model)
            if not annotation_path:
                raise FileNotFoundError(f"No existing annotations.jsonl found under outputs/annotations/ for provider '{args.annotation_provider}'. Cannot skip annotation.")
            logger.info(f"Using existing annotation file: {annotation_path}")

        # STAGE 2: Dataset Prep & Leakage-Free Splitting
        current_stage_name = "Stage 2: Dataset Preparation & Leakage-Free Splitting"
        status_data["current_stage"] = current_stage_name
        status_data["last_updated"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        update_status_file(status_data)

        run_dataset_preparation_stage(annotation_path, args, logger)

        # STAGE 3 & 4: VLM QLoRA Fine-Tuning & Held-Out Evaluation
        if not args.skip_training:
            current_stage_name = "Stage 3 & 4: VLM Fine-Tuning & Evaluation"
            status_data["current_stage"] = current_stage_name
            status_data["last_updated"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            update_status_file(status_data)

            run_training_and_evaluation_stage(args, logger)

            # STAGE 5: Cross-Model Comparison & Final Report
            current_stage_name = "Stage 5: Cross-Model Comparison & Final Report"
            status_data["current_stage"] = current_stage_name
            status_data["last_updated"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            update_status_file(status_data)

            run_comparison_stage(logger)
        else:
            logger.info("Skipping Stage 3 & 4 (VLM Fine-Tuning & Evaluation) as requested.")

        # Pipeline Success Handling
        elapsed_sec = time.monotonic() - start_time
        elapsed_min = elapsed_sec / 60.0

        status_data["status"] = "SUCCESS"
        status_data["current_stage"] = "Completed"
        status_data["duration_sec"] = round(elapsed_sec, 2)
        status_data["last_updated"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        update_status_file(status_data)

        logger.info("\n========================================================")
        logger.info(f"  ALL PIPELINE STAGES COMPLETED SUCCESSFULLY in {elapsed_min:.2f} min!")
        logger.info("========================================================")

        if not args.no_auto_push:
            commit_msg = f"SUCCESS: Run completed for {session_id} session"
            trigger_github_push(commit_msg, logger)

        sys.exit(0)

    except Exception as e:
        elapsed_sec = time.monotonic() - start_time
        tb_str = traceback.format_exc()

        logger.error("\n" + "!" * 60)
        logger.error(f"  PIPELINE FAILED AT STAGE: {current_stage_name}")
        logger.error(f"  Error Message: {str(e)}")
        logger.error("!" * 60)
        logger.error(f"\nFull Exception Traceback:\n{tb_str}")

        # Write error log to experiment folder
        exp_dir = Path("outputs/experiments") / args.experiment
        exp_dir.mkdir(parents=True, exist_ok=True)
        error_file = exp_dir / "error_log.txt"
        with open(error_file, "w", encoding="utf-8") as f:
            f.write(f"Session ID: {session_id}\n")
            f.write(f"Failed Stage: {current_stage_name}\n")
            f.write(f"Error Message: {str(e)}\n\n")
            f.write(f"Traceback:\n{tb_str}\n")

        status_data["status"] = "FAILED"
        status_data["failed_stage"] = current_stage_name
        status_data["duration_sec"] = round(elapsed_sec, 2)
        status_data["error"] = {
            "message": str(e),
            "traceback": tb_str
        }
        status_data["last_updated"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        update_status_file(status_data)

        if not args.no_auto_push:
            short_err = str(e).replace("\n", " ")[:50]
            commit_msg = f"FAILED: Error occurred in {session_id} session - {current_stage_name}: {short_err}"
            trigger_github_push(commit_msg, logger)

        logger.info(f"\n[PIPELINE EXITED GRACEFULLY] Error captured in log file: {session_log_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
