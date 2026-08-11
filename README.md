# End-to-End VLM Synthetic Annotation Pipeline & LoRA Training for Cotton Disease Dataset

An all-in-one, production-grade Python pipeline for generating visual-grounding synthetic annotations across crop leaf/boll image datasets and fine-tuning/evaluating open-source Vision-Language Models (Qwen2.5-VL, Qwen3-VL).

---

## Quick Start: Single Automated Command (`main.py`)

To run synthetic annotation and 4-bit QLoRA training **back-to-back automatically in a single execution**:

```bash
python main.py --annotation-provider huggingface --annotation-model Qwen/Qwen2.5-VL-7B-Instruct --train-config training/configs/qwen25vl_3b.yaml --experiment qwen25vl-3b-v1
```

`main.py` performs all 5 pipeline stages sequentially:
1. **Pre-flight health check** & local Hugging Face model verification.
2. **Synthetic Annotation Generation** across all dataset images.
3. **Dataset Validation & 80/10/10 Split Preparation** (perceptual hash grouping, 0% data leakage).
4. **VLM QLoRA Fine-Tuning** & checkpoint saving under `models/`.
5. **Held-Out Test Set Evaluation & Cross-Model Comparison** reporting under `outputs/`.

---

## 📓 Google Colab 200-Image Benchmarking Notebook

A ready-to-run Google Colab notebook is available at [colab_benchmark.ipynb](file:///c:/Users/Mayank%20Mehta/Projects/PythonProjects/VLM_Training/colab_benchmark.ipynb) (or [notebooks/colab_benchmark.ipynb](file:///c:/Users/Mayank%20Mehta/Projects/PythonProjects/VLM_Training/notebooks/colab_benchmark.ipynb)).

It performs the complete cloud GPU benchmarking workflow:
1. Clones the `VLM_Training` repository from GitHub.
2. Unzips uploaded `Cotton_dataset.zip` directly into the Colab environment.
3. Installs dependencies and runs environment pre-flight verification.
4. Executes the 200-image benchmark on local Hugging Face models (`Qwen/Qwen2.5-VL-7B-Instruct` and `Qwen/Qwen2.5-VL-3B-Instruct`).
5. Uses `training/scripts/push_github.py` to push evaluation metrics, plots, and report files back to GitHub.

---

## 🛡️ Remote Server Execution, Graceful Error Handling & Auto-Push

When running long-running jobs on remote college/cloud servers, `main.py` guarantees **zero silent failures**:

1. **Session Logging**: Every execution generates a unique session ID (e.g. `session_qwen25vl-3b-v1_20260811_213500`) and streams logs to both console and `logs/pipeline_<session_id>.log`.
2. **Real-Time Status Tracking**: `outputs/pipeline_status.json` records live progress (`RUNNING`, `SUCCESS`, or `FAILED`), current stage, and execution timestamps.
3. **Automatic GitHub Notification on Error**:
   - If an unhandled exception or OOM occurs at any stage, `main.py` catches it, writes the full traceback to `outputs/experiments/<experiment>/error_log.txt`, and automatically runs `push_github.py` with:
     `FAILED: Error occurred in <session_id> session - <stage_name>: <error_summary>`
   - This pushes the error log to GitHub immediately, notifying you of the exact failure even if you are away from the remote terminal.
4. **Automatic GitHub Notification on Success**:
   - Upon completion, `main.py` pushes all logs, metrics, plots, and evaluation reports with:
     `SUCCESS: Run completed for <session_id> session`

*(Note: Pass `--no-auto-push` to disable automatic GitHub commits during local debugging).*

---

## Key Features

- **End-to-End Automation (`main.py`)**: Seamless back-to-back execution from raw images to fine-tuned VLM adapters and comparison reports.
- **Multi-Provider VLM Support**: Supports local Hugging Face `transformers` models (`Qwen/Qwen3-VL-8B-Instruct`, `OpenGVLab/InternVL2_5-8B`, `OpenGVLab/InternVL2_5-14B`, `Qwen/Qwen2.5-VL-7B-Instruct`), local Ollama (`qwen3-vl:8b`), and cloud hosted APIs (Google Gemini, Groq, NVIDIA NIM, OpenRouter).
- **Uniform Hugging Face Stack**: Shared `transformers` + `bitsandbytes` 4-bit `nf4` quantization ecosystem across both annotation generation and model training.
- **Leakage-Free Dataset Splitting**: Perceptual image MD5 hashing ensures identical images are placed strictly into the same split (0% train/val/test data leakage).
- **Comprehensive Evaluation & Safety**: Automated metrics (ECE, F1, Accuracy, Latency, VRAM), error analysis, cross-model composite ranking, and pre-commit Git porcelain safety scanner.

---

## Environment Setup

### 1. Create Virtual Environment & Install Dependencies
```bash
# Create virtual environment
python -m venv .venv

# Activate on Windows PowerShell
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables (Optional for Cloud APIs)
Create a `.env` file in the project root directory:
```env
GEMINI_API_KEY=AIzaSy-your-key-here
GROQ_API_KEY=gsk_your-key-here
NVIDIA_API_KEY=nvapi-your-key-here
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

---

## All Available Pipeline Commands

### 1. Automated Back-to-Back Pipeline (`main.py`)

#### Default End-to-End Execution (Hugging Face VLM)
```bash
python main.py
```

#### End-to-End Execution with Specific Models & Experiments
```bash
python main.py --annotation-provider huggingface --annotation-model Qwen/Qwen2.5-VL-7B-Instruct --train-config training/configs/qwen25vl_7b.yaml --experiment qwen25vl-7b-v1
```

#### Skip Annotation (Use Existing Annotations & Run Training Only)
```bash
python main.py --skip-annotation --train-config training/configs/qwen25vl_3b.yaml --experiment qwen25vl-3b-v1
```

#### Skip Training (Run Synthetic Annotation Generation Only)
```bash
python main.py --skip-training --annotation-provider huggingface --annotation-model Qwen/Qwen2.5-VL-7B-Instruct
```

#### Resume Interrupted Pipeline Execution
```bash
python main.py --resume
```

---

### 2. Pre-Flight Health Checks

#### Verify Hugging Face Environment & Model Access
```bash
python scripts/check_huggingface.py --model Qwen/Qwen2.5-VL-7B-Instruct
```

#### Verify Local Ollama Server & Model
```bash
python scripts/check_ollama.py --host http://127.0.0.1:11434 --model qwen3-vl:8b
```

---

### 3. Model Speed & Throughput Profiling

#### Profile Annotation Latency & Throughput (Estimate 20,000-Image Runtime)
```bash
python scripts/generate_annotations.py --provider huggingface --model Qwen/Qwen2.5-VL-7B-Instruct --benchmark-speed --num-samples 50
```

---

### 4. 200-Image Parallel Model Benchmarking

#### Benchmark All Enabled Models in `models.yaml`
```bash
python scripts/benchmark_models.py --dataset-dir Cotton_dataset --resume
```

#### Benchmark Qwen3-VL-8B (8B, Priority ⭐⭐⭐⭐⭐)
```bash
python scripts/benchmark_models.py --provider huggingface --model Qwen/Qwen3-VL-8B-Instruct --dataset-dir Cotton_dataset --sample-count 200 --resume
```

#### Benchmark InternVL3.5-8B / InternVL2.5-8B (8B, Priority ⭐⭐⭐⭐⭐)
```bash
python scripts/benchmark_models.py --provider huggingface --model OpenGVLab/InternVL2_5-8B --dataset-dir Cotton_dataset --sample-count 200 --resume
```

#### Benchmark InternVL3.5-14B / InternVL2.5-14B (14B, Priority ⭐⭐⭐⭐)
```bash
python scripts/benchmark_models.py --provider huggingface --model OpenGVLab/InternVL2_5-14B --dataset-dir Cotton_dataset --sample-count 200 --resume
```

#### Benchmark Qwen2.5-VL-7B (7B)
```bash
python scripts/benchmark_models.py --provider huggingface --model Qwen/Qwen2.5-VL-7B-Instruct --dataset-dir Cotton_dataset --sample-count 200 --resume
```

---

### 5. Standalone Synthetic Annotation Generation

#### Run Full Annotation with Local Hugging Face VLM
```bash
python scripts/generate_annotations.py --provider huggingface --model Qwen/Qwen2.5-VL-7B-Instruct --dataset-dir Cotton_dataset --resume
```

#### Run Full Annotation with Google Gemini Flash-Lite
```bash
python scripts/generate_annotations.py --provider gemini --model gemini-flash-lite-latest --dataset-dir Cotton_dataset --resume
```

#### Process First $N$ Sample Images
```bash
python scripts/generate_annotations.py --provider huggingface --model Qwen/Qwen2.5-VL-7B-Instruct --num-samples 500 --resume
```

#### Process Specific Batch Slices (Index 0 to 5000)
```bash
python scripts/generate_annotations.py --provider huggingface --model Qwen/Qwen2.5-VL-7B-Instruct --start-index 0 --end-index 5000 --resume
```

#### Retry Only Failed Annotations (`failed.jsonl`)
```bash
python scripts/generate_annotations.py --provider huggingface --model Qwen/Qwen2.5-VL-7B-Instruct --retry-failed
```

---

### 6. Dataset Preparation & Leakage-Free Splitting

#### Validate Annotations and Build 80/10/10 Split Manifests
```bash
python training/scripts/prepare_dataset.py --annotations_file outputs/annotations/huggingface/Qwen_Qwen2.5-VL-7B-Instruct/run_20260811_120000/annotations.jsonl --dataset_root Cotton_dataset
```

---

### 7. VLM QLoRA Fine-Tuning

#### Train Qwen2.5-VL-3B
```bash
python training/scripts/train.py --config training/configs/qwen25vl_3b.yaml --experiment qwen25vl-3b-v1
```

#### Train Qwen2.5-VL-7B
```bash
python training/scripts/train.py --config training/configs/qwen25vl_7b.yaml --experiment qwen25vl-7b-v1
```

#### Resume Interrupted Training Checkpoint
```bash
python training/scripts/train.py --config training/configs/qwen25vl_3b.yaml --experiment qwen25vl-3b-v1 --resume
```

---

### 8. Post-Training Evaluation & Comparison

#### Evaluate Fine-Tuned Model on Held-Out Test Set
```bash
python training/scripts/evaluate.py --config training/configs/qwen25vl_3b.yaml --experiment qwen25vl-3b-v1
```

#### Run Cross-Model Comparison & Generate Final Recommendation Report
```bash
python training/scripts/compare_models.py
```

---

### 9. Weight Merging, HF Publishing & GitHub Safety Audit

#### Merge LoRA Adapter Weights with Base VLM Weights
```bash
python training/scripts/merge_lora.py --base-model Qwen/Qwen2.5-VL-3B-Instruct --adapter-dir models/qwen25vl-3b-v1 --output-dir models/merged/qwen25vl-3b-v1
```

#### Publish Fine-Tuned Adapter to Hugging Face Hub
```bash
python training/scripts/push_huggingface.py --adapter-dir models/qwen25vl-3b-v1 --repo-id my-org/qwen25vl-cotton-adapter
```

#### Pre-Commit GitHub Safety Scanner (Prevents Accidental Weight Uploads)
```bash
python training/scripts/push_github.py --dry-run --message "Add VLM training pipeline updates"
```

---

### 10. Unit Test Suite

#### Run Unit Tests for Hugging Face Provider & Health Check
```bash
python tests/test_huggingface.py
```

#### Run Unit Tests for Ollama Provider & Health Check
```bash
python tests/test_ollama.py
```
