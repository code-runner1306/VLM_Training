# ==============================================================================
# Sequential Dual-Dataset VLM Annotation & LoRA Fine-Tuning Pipeline Runner (PowerShell)
# Runs the full pipeline for Cotton_dataset and sugarcane_dataset sequentially,
# archives outputs, and pushes all results to GitHub upon completion.
# ==============================================================================

$ErrorActionPreference = "Continue"

# Resolve root script directory
if ($PSScriptRoot) {
    Set-Location $PSScriptRoot
}

$Datasets = @("Cotton_dataset", "sugarcane_dataset")
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

if (-not (Test-Path "logs")) { New-Item -ItemType Directory -Path "logs" | Out-Null }
if (-not (Test-Path "outputs")) { New-Item -ItemType Directory -Path "outputs" | Out-Null }

Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "    STARTING SEQUENTIAL DUAL-DATASET PIPELINE EXECUTION (PowerShell)" -ForegroundColor Cyan
Write-Host "    Datasets:  $($Datasets -join ', ')" -ForegroundColor Cyan
Write-Host "    Timestamp: $Timestamp" -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan

$OverallStatus = 0

foreach ($dataset in $Datasets) {
    Write-Host ""
    Write-Host "========================================================================" -ForegroundColor Yellow
    Write-Host "  RUNNING PIPELINE FOR DATASET: '$dataset'" -ForegroundColor Yellow
    Write-Host "========================================================================" -ForegroundColor Yellow

    if (-not (Test-Path $dataset)) {
        Write-Host "⚠️ Warning: Dataset directory '$dataset' does not exist! Skipping." -ForegroundColor Yellow
        continue
    }

    # Execute main pipeline for current dataset with --no-auto-push
    python main.py --dataset-dir $dataset --no-auto-push $args
    $runExitCode = $LASTEXITCODE

    if ($runExitCode -eq 0) {
        Write-Host "✓ Pipeline completed successfully for '$dataset'" -ForegroundColor Green
    } else {
        Write-Host "❌ Pipeline failed for '$dataset' with exit code $runExitCode!" -ForegroundColor Red
        $OverallStatus = 1
    }

    # Archive outputs into dataset-isolated subfolder
    $ArchiveDir = Join-Path "outputs" $dataset
    if (-not (Test-Path $ArchiveDir)) { New-Item -ItemType Directory -Path $ArchiveDir | Out-Null }

    if (Test-Path "outputs/dataset") {
        Write-Host "Archiving dataset manifests to $ArchiveDir/dataset..." -ForegroundColor Gray
        Copy-Item -Path "outputs/dataset" -Destination "$ArchiveDir/dataset" -Recurse -Force -ErrorAction SilentlyContinue
    }

    if (Test-Path "outputs/comparison") {
        Write-Host "Archiving evaluation comparison report to $ArchiveDir/comparison..." -ForegroundColor Gray
        Copy-Item -Path "outputs/comparison" -Destination "$ArchiveDir/comparison" -Recurse -Force -ErrorAction SilentlyContinue
    }

    if (Test-Path "outputs/pipeline_status.json") {
        Copy-Item -Path "outputs/pipeline_status.json" -Destination "$ArchiveDir/pipeline_status.json" -Force -ErrorAction SilentlyContinue
    }
}

Write-Host ""
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "  SEQUENTIAL RUNS FINISHED (Overall Status Code: $OverallStatus)" -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan

# Stage, commit, and push outputs to GitHub
if ($OverallStatus -eq 0) {
    $CommitMsg = "SUCCESS: Sequential runs completed for Cotton_dataset and sugarcane_dataset"
} else {
    $CommitMsg = "FAILED: Sequential runs completed with errors for dataset sweep"
}

Write-Host ""
Write-Host "[AUTO-PUSH] Triggering GitHub push for all dataset run results..." -ForegroundColor Magenta
python training/scripts/push_github.py --message $CommitMsg --yes

if ($OverallStatus -eq 0) {
    Write-Host "✓ All sequential dataset tasks and GitHub push finished successfully!" -ForegroundColor Green
    exit 0
} else {
    Write-Host "⚠️ Pipeline finished with errors. Logs and available outputs pushed to GitHub." -ForegroundColor Red
    exit 1
}
