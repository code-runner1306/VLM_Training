# Production-Grade VLM Synthetic Annotation Pipeline for Cotton Disease Dataset

An end-to-end Python pipeline for generating structured, visual-evidence-grounded synthetic annotations for cotton crop disease leaf and boll images. The pipeline evaluates multiple hosted Vision-Language Models (Google Gemini, Groq, NVIDIA NIM, OpenRouter) and generates structured JSON supervision dataset files suitable for fine-tuning 7B VLM models (such as Qwen2.5-VL).

---

## Key Features

- **Multi-Provider VLM Support**: Abstracted `VisionModel` interface supporting Google Gemini (`gemini-flash-lite-latest`, `gemini-flash-latest`), Groq (`qwen/qwen3.6-27b`), NVIDIA NIM (`meta/llama-3.2-11b-vision-instruct`), and OpenRouter.
- **Dynamic Dataset Discovery**: Recursively scans `.jpg`, `.jpeg`, `.png`, and `.webp` files, automatically extracting disease class labels from folder structures without hardcoding.
- **Disease Profile Generation & Caching**: Generates structured domain disease profiles once per class (`outputs/disease_profiles/{disease}.json`).
- **Parallel Stratified Benchmark**: Samples exactly 200 representative images across disease classes and evaluates enabled candidate VLMs concurrently using `asyncio.gather`.
- **Fault-Tolerant & Resumable Engine**: Flushes completed annotations immediately per image to `outputs/annotations/annotations.jsonl`. Supports `--resume`, `--start-index`, `--end-index`, and `--retry-failed`.
- **Per-Model Diagnostic Counters**: Tracks rate limit hits (HTTP 429 count), network errors, and JSON validation failures per model, saved to `outputs/model_metrics.json`.

---

## Installation & Setup

### 1. Create Virtual Environment with `uv`
```bash
# Create virtual environment
uv venv

# Activate on Windows PowerShell
.venv\Scripts\Activate.ps1

# Install dependencies
uv pip install -r requirements.txt
```

### 2. Configure API Keys
Create a `.env` file in the root directory (based on `.env.example`):
```env
GEMINI_API_KEY=AIzaSy-your-key-here
GROQ_API_KEY=gsk_your-key-here
NVIDIA_API_KEY=nvapi-your-key-here
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

---

## Full Execution Commands

### Step 1: Generate Disease Profiles
Discovers disease classes and generates domain profiles for all 18 cotton disease categories using Gemini:

```bash
uv run python -u scripts/generate_disease_profiles.py --dataset-dir Cotton_dataset --provider gemini --model gemini-flash-lite-latest
```

---

### Step 2: Run 200-Image Parallel VLM Benchmark
Evaluates configured candidate models concurrently on a stratified 200-image sample:

```bash
uv run python -u scripts/benchmark_models.py --dataset-dir Cotton_dataset --resume
```

---

### Step 3: Review Benchmark Leaderboard & Reports
Inspect the generated benchmark reports under `outputs/benchmark/`:
- **Markdown Leaderboard**: `outputs/benchmark/report.md`
- **CSV Summary**: `outputs/benchmark/final_report.csv`
- **Full JSON Report**: `outputs/benchmark/final_report.json`

---

### Step 4: Run Full Synthetic Annotation Pipeline (Gemini Flash-Lite)
Run full synthetic annotation across all ~20,000 dataset images using **Gemini Flash-Lite**:

```bash
uv run python -u scripts/generate_annotations.py --dataset-dir Cotton_dataset --provider gemini --model gemini-flash-lite-latest --resume
```

---

### Additional Useful CLI Options

#### Resume Interrupted Annotation Run
If your execution is stopped or interrupted, resume seamlessly without repeating work:
```bash
uv run python -u scripts/generate_annotations.py --dataset-dir Cotton_dataset --provider gemini --model gemini-flash-lite-latest --resume
```

#### Annotate a Specific Number of Sample Images
To process only a specific number of images (e.g., annotate exactly 500 images):
```bash
uv run python -u scripts/generate_annotations.py --dataset-dir Cotton_dataset --provider gemini --model gemini-flash-lite-latest --num-samples 500 --resume
```

#### Run Specific Batch Indices
To run a specific slice of images (e.g. index 0 to 5000):
```bash
uv run python -u scripts/generate_annotations.py --dataset-dir Cotton_dataset --provider gemini --model gemini-flash-lite-latest --start-index 0 --end-index 5000 --resume
```

#### Retry Only Failed Items
To re-process only items logged in `outputs/annotations/failed.jsonl`:
```bash
uv run python -u scripts/generate_annotations.py --dataset-dir Cotton_dataset --provider gemini --model gemini-flash-lite-latest --retry-failed
```

---

## Output Structure

All pipeline outputs are stored in `outputs/`:

```text
outputs/
├── disease_profiles/          # Structured domain profile JSON per disease class
├── benchmark/                 # Stratified sample list, per-model .jsonl outputs, and reports
│   ├── benchmark_images.json
│   ├── report.md
│   ├── final_report.csv
│   └── final_report.json
├── annotations/
│   ├── annotations.jsonl      # Complete machine-readable synthetic annotation dataset
│   └── failed.jsonl           # Failed items queue for retries
├── model_metrics.json         # Per-model 429 rate limit hit counts and error counters
└── usage.json                 # Cumulative token usage and request statistics
```

---

## Running Unit Tests

Run the full pytest suite (11 unit tests covering dataset discovery, validators, rate limiters, checkpoints, and providers):

```bash
uv run pytest tests/
```
