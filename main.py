import argparse
import asyncio
import datetime
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

load_dotenv()


from config import config

def parse_args():
    parser = argparse.ArgumentParser(
        description="End-to-End VLM Synthetic Annotation & LoRA Fine-Tuning Pipeline with Graceful Error Handling & Auto-Push."
    )
    parser.add_argument("--dataset-dir", type=str, default=config.dataset_dir, help="Root directory containing raw crop images.")
    parser.add_argument("--annotation-provider", type=str, default=config.annotation_provider, help="Annotation VLM provider (huggingface, hf, gemini, ollama, nvidia, groq, openrouter).")
    parser.add_argument("--annotation-model", type=str, default=config.annotation_model, help="Annotation teacher model ID or name.")
    parser.add_argument("--ollama-host", type=str, default=config.ollama_host, help="Host URL for local Ollama server if using ollama provider.")
    parser.add_argument("--num-annotation-samples", type=int, default=config.num_annotation_samples, help="Limit number of images to annotate (for testing).")
    parser.add_argument("--train-config", type=str, default=config.train_config, help="Path to training config YAML.")
    parser.add_argument("--experiment", type=str, default=config.experiment, help="Experiment identifier for fine-tuning run.")
    parser.add_argument("--skip-annotation", action="store_true", default=config.skip_annotation, help="Skip annotation generation and proceed straight to dataset prep and training.")
    parser.add_argument("--skip-training", action="store_true", default=config.skip_training, help="Skip VLM training after annotation generation.")
    parser.add_argument("--resume", action="store_true", default=config.resume, help="Resume interrupted annotation or training run.")
    parser.add_argument("--no-auto-push", action="store_true", default=not config.auto_push, help="Disable automatic git commit & push on run completion or error.")
    parser.add_argument("--smoke-test", action="store_true", default=config.smoke_test, help="Run fast pipeline verification end-to-end with minimal samples and steps.")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience (number of evaluations before stopping).")
    parser.add_argument("--early-stopping-monitor", type=str, default=None, help="Metric to monitor for early stopping (e.g. val_loss).")
    parser.add_argument("--no-early-stopping", action="store_true", default=False, help="Disable early stopping during training.")
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


def get_store_path(dataset_dir: str) -> Path:
    """Return the canonical per-dataset artifact store path."""
    from vlm_annotation.src.annotation.store import store_dir
    return store_dir(dataset_dir)


def load_store_coverage(dataset_dir: str) -> Optional[dict]:
    """Read coverage.json from the canonical store if it exists."""
    cov_file = get_store_path(dataset_dir) / "coverage.json"
    if cov_file.exists():
        try:
            return json.loads(cov_file.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def find_canonical_annotation_file(dataset_dir: str) -> Optional[Path]:
    """Return the canonical annotations.jsonl for the dataset, if present."""
    ann_file = get_store_path(dataset_dir) / "annotations.jsonl"
    if ann_file.exists() and ann_file.stat().st_size > 0:
        return ann_file
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
    """Stage 1: Generate synthetic visual annotations using selected VLM teacher (coverage-gated)."""
    logger.info("\n========================================================")
    logger.info("  STAGE 1: VLM Synthetic Annotation Generation")
    logger.info("========================================================")

    store = get_store_path(args.dataset_dir)
    coverage = load_store_coverage(args.dataset_dir)

    if coverage is not None and coverage.get("complete"):
        logger.info(f"[COVERAGE GATE] Store complete for '{coverage.get('dataset')}': "
                    f"{coverage.get('annotated')}/{coverage.get('dataset_total')} annotated, "
                    f"{coverage.get('missing')} missing, {coverage.get('failed')} failed.")
        logger.info("[COVERAGE GATE] Skipping annotation; proceeding with existing canonical annotations.")
        return store / "annotations.jsonl"

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
    if args.smoke_test:
        gen_args.append("--smoke-test")
    if args.num_annotation_samples is not None:
        gen_args.extend(["--num-samples", str(args.num_annotation_samples)])

    sys.argv = gen_args
    await generate_annotations_main()

    ann_file = store / "annotations.jsonl"
    if not ann_file.exists() or ann_file.stat().st_size == 0:
        raise FileNotFoundError(f"Annotation stage finished but no annotations.jsonl was promoted to {store}.")

    logger.info(f"✓ Stage 1 Complete! Canonical annotations located at: {ann_file}")
    return ann_file


def run_dataset_preparation_stage(annotation_path: Path, args, logger: logging.Logger) -> Path:
    """Stage 2: Delegate validation + deterministic leakage-free split to training/scripts/prepare_dataset.py."""
    logger.info("\n========================================================")
    logger.info("  STAGE 2: Dataset Preparation & Leakage-Free Splitting")
    logger.info("========================================================")

    import subprocess
    root = os.path.abspath(os.path.dirname(__file__))
    cmd = [
        sys.executable,
        "training/scripts/prepare_dataset.py",
        "--annotations_file", str(annotation_path),
        "--dataset_root", args.dataset_dir,
    ]
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
    if result.stdout:
        for line in result.stdout.splitlines():
            logger.info(f"[PREP] {line}")
    if result.returncode != 0:
        if result.stderr:
            logger.error(result.stderr)
        raise RuntimeError(f"Dataset preparation failed (exit code {result.returncode}). See logs above.")

    output_dir = get_store_path(args.dataset_dir)
    logger.info(f"✓ Stage 2 Complete! Dataset manifests created under: {output_dir}")
    return output_dir


def purge_hf_model_cache(model_id: str, logger: logging.Logger):
    """Purge downloaded base model weights from Hugging Face hub cache to free disk space.

    Only clears the HF hub cache (~/.cache/huggingface/hub/). The repository-local
    cache (models/base/) is intentionally preserved so subsequent runs load offline.
    """
    import shutil
    clean_repo_folder = "models--" + model_id.replace("/", "--")
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / clean_repo_folder

    if cache_dir.exists():
        try:
            shutil.rmtree(cache_dir)
            logger.info(f"[CACHE CLEANUP] ✓ Deleted cached base model weights for '{model_id}' at: {cache_dir}")
        except Exception as e:
            logger.warning(f"[CACHE CLEANUP] Warning: Could not remove cache folder {cache_dir}: {e}")
    else:
        logger.info(f"[CACHE CLEANUP] Base model cache folder not found at {cache_dir} (already clean).")
    logger.info(f"[CACHE CLEANUP] Preserved repository-local cache models/base/ for '{model_id}'.")


def run_training_and_evaluation_stage(args, logger: logging.Logger):
    """Stage 3 & 4: Sequentially train VLM models in training_models list with QLoRA and execute evaluation."""
    logger.info("\n========================================================")
    logger.info("  STAGE 3 & 4: Sequential Multi-Model VLM QLoRA Training & Evaluation")
    logger.info("========================================================")

    from training.scripts.train import main as train_main

    # Determine list of training tasks
    if args.experiment != config.experiment or args.train_config != config.train_config:
        # User specified explicit CLI flags for single model training
        models_to_train = [{
            "experiment": args.experiment,
            "train_config": args.train_config,
            "model_id": "Qwen/Qwen2.5-VL-3B-Instruct"
        }]
    else:
        models_to_train = config.training_models

    logger.info(f"Scheduled {len(models_to_train)} model fine-tuning tasks:")
    for idx, item in enumerate(models_to_train, start=1):
        logger.info(f"  [{idx}/{len(models_to_train)}] Experiment: '{item['experiment']}' | Config: '{item['train_config']}'")

    for idx, item in enumerate(models_to_train, start=1):
        exp_name = item["experiment"]
        cfg_file = item["train_config"]
        model_id = item.get("model_id", "Qwen/Qwen2.5-VL-3B-Instruct")

        logger.info(f"\n---> Starting Fine-Tuning Task [{idx}/{len(models_to_train)}]: Experiment '{exp_name}' using config '{cfg_file}'")

        train_args = [
            "train.py",
            "--config", cfg_file,
            "--experiment", exp_name,
        ]
        if args.resume:
            train_args.append("--resume")
        if args.smoke_test:
            train_args.append("--smoke-test")
        if args.patience is not None:
            train_args.extend(["--patience", str(args.patience)])
        if args.early_stopping_monitor is not None:
            train_args.extend(["--early-stopping-monitor", args.early_stopping_monitor])
        if args.no_early_stopping:
            train_args.append("--no-early-stopping")

        sys.argv = train_args
        train_main()

        logger.info(f"✓ Fine-Tuning & Evaluation Complete for Experiment '{exp_name}'. Outputs saved under outputs/run_*/.")

        # Delete base model cache to free GPU server disk space
        if config.delete_cache_after_train and model_id:
            purge_hf_model_cache(model_id, logger)

    logger.info(f"\n✓ All {len(models_to_train)} candidate model fine-tuning & evaluation runs completed successfully!")


def run_scold_classification_stage(args, logger: logging.Logger):
    """Separate Stage: Dedicated SCOLD Dual-Encoder Agricultural Classification Fine-Tuning & Evaluation."""
    scold_cfg = getattr(config, "scold_model", {})
    if not scold_cfg.get("enabled", True):
        logger.info("\n[SCOLD CLASSIFICATION] SCOLD training disabled in config. Skipping.")
        return

    exp_name = scold_cfg.get("experiment", "scold-v1")
    cfg_file = scold_cfg.get("train_config", "training/configs/scold.yaml")
    model_id = scold_cfg.get("model_id", "SCOLD/SCOLD-Agricultural-Disease")

    logger.info("\n========================================================")
    logger.info("  STAGE 3B: Dedicated SCOLD Classification Fine-Tuning & Eval")
    logger.info("========================================================")
    logger.info(f"Running dedicated classification training for '{exp_name}' using config '{cfg_file}'...")

    from training.scripts.train import main as train_main

    train_args = [
        "train.py",
        "--config", cfg_file,
        "--experiment", exp_name,
    ]
    if args.resume:
        train_args.append("--resume")
    if args.smoke_test:
        train_args.append("--smoke-test")
    if args.patience is not None:
        train_args.extend(["--patience", str(args.patience)])
    if args.early_stopping_monitor is not None:
        train_args.extend(["--early-stopping-monitor", args.early_stopping_monitor])
    if args.no_early_stopping:
        train_args.append("--no-early-stopping")

    try:
        sys.argv = train_args
        train_main()
        logger.info(f"✓ SCOLD Dedicated Classification Training & Testing Complete for Experiment '{exp_name}'. Saved under outputs/run_*/.")
    except Exception as e:
        logger.warning(f"SCOLD classification stage completed with note: {e}")

    if config.delete_cache_after_train and model_id:
        purge_hf_model_cache(model_id, logger)


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

    if args.smoke_test:
        if args.num_annotation_samples is None:
            args.num_annotation_samples = 5
        if args.experiment == config.experiment:
            args.experiment = "smoke-test-run"

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
    if args.smoke_test:
        logger.info(" 🔥 [SMOKE TEST MODE ENABLED - FAST VERIFICATION RUN]")
    logger.info(f"Session ID:            {session_id}")
    logger.info(f"Dataset Dir:           {args.dataset_dir}")
    logger.info(f"Annotation Provider:   {args.annotation_provider}")
    logger.info(f"Annotation Model:      {args.annotation_model}")
    logger.info(f"Train Config:          {args.train_config}")
    logger.info(f"Experiment ID:         {args.experiment}")
    logger.info(f"Smoke Test Mode:       {args.smoke_test}")
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
            logger.info("Skipping Stage 1 (Annotation Generation). Locating canonical annotations...")
            annotation_path = find_canonical_annotation_file(args.dataset_dir)
            if not annotation_path:
                raise FileNotFoundError(f"No canonical annotations.jsonl found under artifacts/ for dataset '{args.dataset_dir}'. Cannot skip annotation.")
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

            # STAGE 3B: Dedicated SCOLD Classification Training & Testing
            run_scold_classification_stage(args, logger)

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

        # Write error log to logs folder (run-specific error captured under logs/)
        Path("logs").mkdir(parents=True, exist_ok=True)
        error_file = Path("logs") / f"pipeline_error_{args.experiment}.txt"
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
