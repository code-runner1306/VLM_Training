# Production-Grade VLM Synthetic Annotation Pipeline & LoRA Training for Cotton Disease Dataset

An end-to-end Python pipeline for generating structured, visual-evidence-grounded synthetic annotations for cotton crop disease leaf and boll images, and fine-tuning/evaluating open-source Vision-Language Models (Qwen2.5-VL, Qwen3-VL).

---

## Key Features

- **Multi-Provider VLM Annotation Support**: Supports cloud hosted VLMs (Google Gemini, Groq, NVIDIA NIM, OpenRouter) and **local Hugging Face VLMs** (`Qwen/Qwen2.5-VL-7B-Instruct`, `Qwen/Qwen2.5-VL-3B-Instruct`).
- **Uniform Hugging Face Ecosystem**: Integrates `transformers` + `bitsandbytes` 4-bit `nf4` quantization across both annotation generation and LoRA model training. No third-party local server daemons needed.
- **Automated Pre-Flight Health Checks**: Validates PyTorch CUDA availability, `transformers` package compatibility, Hugging Face Hub model access, and vision inference before bulk processing.
- **Dynamic Dataset Discovery**: Recursively scans `.jpg`, `.jpeg`, `.png`, and `.webp` files, automatically extracting disease class labels from folder structures without hardcoding.
- **Fault-Tolerant & Resumable Engine**: Flushes completed annotations immediately per image. Supports `--resume`, `--start-index`, `--end-index`, `--retry-failed`, and `--benchmark-speed`.
- **Parallel Stratified Benchmark**: Samples 200 representative images and evaluates candidate VLMs (hosted and local Hugging Face) concurrently.
- **VLM LoRA Fine-Tuning & Evaluation**: Complete 4-bit QLoRA fine-tuning, deterministic 80/10/10 data splitting, held-out test evaluation, multi-criteria composite model ranking, and safe GitHub/Hugging Face release scripts.

---

## Installation & Setup

### 1. Create Virtual Environment & Install Dependencies
```bash
# Create virtual environment
python -m venv .venv

# Activate on Windows PowerShell
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys (for Cloud Hosted Models)
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=AIzaSy-your-key-here
GROQ_API_KEY=gsk_your-key-here
NVIDIA_API_KEY=nvapi-your-key-here
```

---

## Local Hugging Face VLM Annotation Workflow

### Step 1: Run Pre-Flight Health Check
Verify PyTorch CUDA capability, `transformers` library installation, and Hugging Face Hub model accessibility:
```bash
python scripts/check_huggingface.py --model Qwen/Qwen2.5-VL-7B-Instruct
```

### Step 2: Run 200-Image Benchmark (Local HF vs Cloud Hosted)
Evaluate the local Hugging Face VLM model on 200 sample images:
```bash
python scripts/benchmark_models.py --provider huggingface --model Qwen/Qwen2.5-VL-7B-Instruct --dataset-dir Cotton_dataset
```

### Step 3: Speed & Throughput Profiling Mode
Run a quick speed test to profile average/median/P95 latency, images/minute throughput, and estimate full 20,000-image runtime:
```bash
python scripts/generate_annotations.py --provider huggingface --model Qwen/Qwen2.5-VL-7B-Instruct --benchmark-speed --num-samples 50
```

### Step 4: Execute Full Annotation Pipeline with Resume
Run full synthetic annotation using local Hugging Face VLM (4-bit quantized):
```bash
python scripts/generate_annotations.py --provider huggingface --model Qwen/Qwen2.5-VL-7B-Instruct --dataset-dir Cotton_dataset --resume
```

---

## VLM LoRA Training & Evaluation Workflow

### 1. Prepare Dataset & Generate Leakage-Free 80/10/10 Split
```bash
python training/scripts/prepare_dataset.py
```
Output manifest and reports are saved to `outputs/dataset/`.

### 2. Train Candidate VLM Models (QLoRA)
```bash
# Train Qwen2.5-VL-3B
python training/scripts/train.py --config training/configs/qwen25vl_3b.yaml --experiment qwen25vl-3b-v1

# Train Qwen2.5-VL-7B
python training/scripts/train.py --config training/configs/qwen25vl_7b.yaml --experiment qwen25vl-7b-v1
```

### 3. Run Cross-Model Comparison & Generate Final Recommendation
```bash
python training/scripts/compare_models.py
```
Inspect `outputs/comparison/final_recommendation.md` for composite model rankings.

### 4. Safely Commit Results to GitHub
```bash
python training/scripts/push_github.py --dry-run --message "Add VLM training results"
```

---

## Unit Testing

Run unit tests covering the Hugging Face provider, pre-flight health check, and model factory:
```bash
python tests/test_huggingface.py
```
