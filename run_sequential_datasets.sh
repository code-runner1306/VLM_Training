#!/usr/bin/env bash
# ==============================================================================
# Sequential Dual-Dataset VLM Annotation & LoRA Fine-Tuning Pipeline Runner
# Runs the full pipeline for Cotton_dataset and sugarcane_dataset sequentially,
# archives outputs, and pushes all results to GitHub upon completion.
# ==============================================================================

set -e  # Exit immediately if a command exits with a non-zero status

# Resolve root script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATASETS=("Cotton_dataset" "sugarcane_dataset")
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

mkdir -p logs outputs

echo "========================================================================"
echo "    STARTING SEQUENTIAL DUAL-DATASET PIPELINE EXECUTION"
echo "    Datasets:  ${DATASETS[*]}"
echo "    Timestamp: ${TIMESTAMP}"
echo "========================================================================"

overall_status=0

for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "========================================================================"
    echo "  RUNNING PIPELINE FOR DATASET: '${dataset}'"
    echo "========================================================================"

    # Check if dataset directory exists
    if [ ! -d "$dataset" ]; then
        echo "⚠️ Warning: Dataset directory '${dataset}' does not exist! Skipping."
        continue
    fi

    # Execute main pipeline for current dataset with --no-auto-push
    # Pass along any extra CLI flags passed to this script (e.g. --smoke-test)
    set +e
    python main.py --dataset-dir "$dataset" --no-auto-push "$@"
    run_exit_code=$?
    set -e

    if [ $run_exit_code -eq 0 ]; then
        echo "✓ Pipeline completed successfully for '${dataset}'"
    else
        echo "❌ Pipeline failed for '${dataset}' with exit code ${run_exit_code}!"
        overall_status=1
    fi

    # Archive outputs into dataset-isolated subfolder
    ARCHIVE_DIR="outputs/${dataset}"
    mkdir -p "$ARCHIVE_DIR"

    if [ -d "outputs/dataset" ]; then
        echo "Archiving dataset manifests to ${ARCHIVE_DIR}/dataset..."
        cp -r outputs/dataset "$ARCHIVE_DIR/" 2>/dev/null || true
    fi

    if [ -d "outputs/comparison" ]; then
        echo "Archiving evaluation comparison report to ${ARCHIVE_DIR}/comparison..."
        cp -r outputs/comparison "$ARCHIVE_DIR/" 2>/dev/null || true
    fi

    if [ -f "outputs/pipeline_status.json" ]; then
        cp outputs/pipeline_status.json "$ARCHIVE_DIR/pipeline_status.json" 2>/dev/null || true
    fi
done

echo ""
echo "========================================================================"
echo "  SEQUENTIAL RUNS FINISHED (Overall Status Code: ${overall_status})"
echo "========================================================================"

# Stage, commit, and push outputs to GitHub
if [ "$overall_status" -eq 0 ]; then
    COMMIT_MSG="SUCCESS: Sequential runs completed for Cotton_dataset and sugarcane_dataset"
else
    COMMIT_MSG="FAILED: Sequential runs completed with errors for dataset sweep"
fi

echo -e "\n[AUTO-PUSH] Triggering GitHub push for all dataset run results..."
python training/scripts/push_github.py --message "$COMMIT_MSG" --yes

if [ "$overall_status" -eq 0 ]; then
    echo "✓ All sequential dataset tasks and GitHub push finished successfully!"
    exit 0
else
    echo "⚠️ Pipeline finished with errors. Logs and available outputs pushed to GitHub."
    exit 1
fi
