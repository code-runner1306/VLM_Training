# Production-Grade VLM Synthetic Annotation Pipeline for Cotton Disease Dataset

An end-to-end Python pipeline for generating structured, visual-evidence-grounded synthetic annotations for cotton crop disease leaf/boll images. The pipeline evaluates multiple hosted Vision-Language Models (NVIDIA NIM, Gemini, Groq, OpenRouter) and generates structured JSON supervision dataset files suitable for fine-tuning 7B VLM models (such as Qwen2.5-VL).

---

## Features

- **Multi-Provider VLM Support**: Abstracted `VisionModel` interface supporting NVIDIA NIM (Llama 3.2 90B Vision, Llama 4 Maverick), Google Gemini (Gemini 2.5 Flash, Flash-Lite), Groq (Llama 4 Scout), and OpenRouter.
- **Dynamic Dataset Discovery**: Recursively scans `.jpg`, `.jpeg`, `.png`, and `.webp` files, automatically extracting disease class labels from folder structures without hardcoded names.
- **Disease Profile Generation & Caching**: Generates structured domain disease profiles once per class (`outputs/disease_profiles/{disease}.json`).
- **Stratified 200-Image Benchmark**: Samples exactly 200 representative images across disease classes and evaluates all candidate VLMs on the exact same sample set.
- **Multi-Aspect Scoring & Teacher-as-Judge**: Evaluates outputs across Visual Observation (30%), Diagnostic Evidence (25%), Reasoning (20%), Hallucinations (15%), and Schema Reliability (10%).
- **Resumable & Fault-Tolerant Engine**: Flushes completed annotations immediately per image to `outputs/annotations/annotations.jsonl`. Supports `--resume`, `--start-index`, `--end-index`, and `--retry-failed`.
- **Per-Model Diagnostic Counters**: Tracks rate limit hits (HTTP 429 count), network errors, and JSON validation failures per model saved to `outputs/model_metrics.json`.

---

## Installation & Setup

1. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

2. **Configure API Keys**:
Create a `.env` file in the root directory (based on `.env.example`):
```env
NVIDIA_API_KEY=nvapi-your-key-here
GEMINI_API_KEY=AIzaSy-your-key-here
GROQ_API_KEY=gsk_your-key-here
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

---

## Execution Workflow

### Step 1: Generate Disease Profiles
Discovers disease classes and generates domain profiles:
```bash
python scripts/generate_disease_profiles.py --dataset-dir dataset --provider gemini --model gemini-2.5-flash
```

### Step 2: Run 200-Image VLM Benchmark
Evaluates enabled models on a stratified 200-image sample:
```bash
python scripts/benchmark_models.py --dataset-dir dataset --sample-count 200 --resume
```

### Step 3: Review Benchmark Results
Check generated benchmark reports:
- Markdown summary: `outputs/benchmark/report.md`
- CSV export: `outputs/benchmark/final_report.csv`
- Full JSON: `outputs/benchmark/final_report.json`

### Step 4: Run Full Annotation Pipeline
Run full synthetic annotation across all ~20,000 dataset images using the selected winning teacher model:
```bash
python scripts/generate_annotations.py --dataset-dir dataset --provider gemini --model gemini-2.5-flash --resume
```

### Step 5: Resume Interrupted Jobs or Retry Failures
If the job was interrupted or stopped:
```bash
python scripts/generate_annotations.py --dataset-dir dataset --resume
```

To re-process only items recorded in `outputs/annotations/failed.jsonl`:
```bash
python scripts/generate_annotations.py --retry-failed
```

---

## Output Structure

All outputs are saved under `outputs/`:
- `outputs/disease_profiles/`: Cached disease profiles JSON per class.
- `outputs/benchmark/`: `benchmark_images.json`, raw model responses (`.jsonl`), `report.md`, `final_report.csv`, `final_report.json`.
- `outputs/annotations/annotations.jsonl`: Machine-readable structured annotation dataset.
- `outputs/annotations/failed.jsonl`: Failed request queue for retry.
- `outputs/model_metrics.json`: Diagnostic rate-limit (429) hit counts and error metrics per model.
- `outputs/usage.json`: Token usage and request counts per provider/model.

---

## Running Unit Tests

To run the automated unit test suite with mock provider calls:
```bash
pytest tests/
```
