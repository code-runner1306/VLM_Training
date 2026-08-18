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

## ⚙️ Centralized Configuration (`config.py`)

All default pipeline parameters, model selections, batch settings, dataset paths, and remote auto-push preferences can be adjusted directly in **[config.py](file:///c:/Users/Mayank%20Mehta/Projects/PythonProjects/VLM_Training/config.py)**:

```python
# config.py
@dataclass
class PipelineConfig:
    dataset_dir: str = "Cotton_dataset"
    annotation_provider: str = "huggingface"       # Options: huggingface, gemini, ollama, nvidia, groq
    annotation_model: str = "Qwen/Qwen3-VL-8B-Instruct" # Default teacher model
    train_config: str = "training/configs/qwen25vl_3b.yaml"
    experiment: str = "qwen25vl-3b-v1"
    auto_push: bool = True                         # Auto-commit and push to GitHub on run completion or error
```

*(Any CLI flags passed during command execution will dynamically override these `config.py` defaults).*

---

## Key Features

- **End-to-End Automation (`main.py`)**: Seamless back-to-back execution from raw images to fine-tuned VLM adapters and comparison reports.
- **Multi-Provider VLM Support**: Supports local Hugging Face `transformers` models (`Qwen/Qwen3-VL-8B-Instruct`, `OpenGVLab/InternVL2_5-8B`, `OpenGVLab/InternVL2_5-14B`, `Qwen/Qwen2.5-VL-7B-Instruct`), local Ollama (`qwen3-vl:8b`), and cloud hosted APIs (Google Gemini, Groq, NVIDIA NIM, OpenRouter).
- **Uniform Hugging Face Stack**: Shared `transformers` + `bitsandbytes` 4-bit `nf4` quantization ecosystem across both annotation generation and model training.
- **Leakage-Free Dataset Splitting**: Perceptual image MD5 hashing ensures identical images are placed strictly into the same split (0% train/val/test data leakage).
- **Comprehensive Evaluation & Safety**: Automated metrics (ECE, F1, Accuracy, Latency, VRAM), error analysis, cross-model composite ranking, and pre-commit Git porcelain safety scanner.

---

## 📦 Local Base-Model Cache & Prefetch

Base models are cached **inside the repository** under `models/base/<org>__<name>/`
(e.g. `models/base/Qwen__Qwen2.5-VL-3B-Instruct/`) so training, annotation, and
evaluation never re-download from the Hub on repeat runs. `models/` is
git-ignored, so cached weights are never committed or pushed.

### Prefetch Models (Optional, Recommended Before Training)

```bash
# Download all default-pipeline models from config.py (annotation + training + scold)
python scripts/download_models.py --all

# Download specific models
python scripts/download_models.py --models Qwen/Qwen2.5-VL-3B-Instruct Qwen/Qwen3-VL-8B-Instruct

# Force re-download / bypass the cache
python scripts/download_models.py --all --force
```

`HF_TOKEN` is read from the environment or `.env` when `--token` is not given
(gated models such as Qwen2.5-VL require it).

### Cache-First Loading

Everywhere the pipeline loads a Hugging Face model/processor it resolves in this order:
1. `models/base/<org>__<name>/` (repository-local cache)
2. Hugging Face hub cache (`local_files_only=True`)
3. Fresh download into `models/base/` (then loaded from there)

`training/src/model_cache.py` implements the resolution helpers; annotation,
health checks, training adapters, and evaluation all use them. The `--force` flag
on `download_models.py` is the only way to invalidate a cached snapshot.

---

## Environment Setup

### 🚀 Direct Setup (Linux ML GPU Server & Local Development)

Simply create a virtual environment and run `pip install -r requirements.txt`. This will automatically fetch CUDA 12.1-enabled PyTorch, bitsandbytes, accelerate, and all multimodal training dependencies:

```bash
# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate       # On Linux / macOS
# .venv\Scripts\Activate.ps1   # On Windows PowerShell

# 2. Install all dependencies with PyTorch CUDA acceleration
pip install --upgrade pip
pip install -r requirements.txt
```

### 🔑 Configure Environment Variables (Optional for Cloud APIs)
Create a `.env` file in the project root directory if using cloud annotation providers:
```env
GEMINI_API_KEY=AIzaSy-your-key-here
GROQ_API_KEY=gsk_your-key-here
NVIDIA_API_KEY=nvapi-your-key-here
OPENROUTER_API_KEY=sk-or-v1-your-key-here
HF_TOKEN=hf_your-token-here
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

### 7. Real VLM QLoRA Fine-Tuning (transformers.Trainer)

Training uses a real gradient-based `transformers.Trainer` loop driven by the
YAML config: learning rate, weight decay, warmup ratio, scheduler, batch size,
gradient accumulation, max grad norm, bf16/fp16, gradient checkpointing, and
logging/eval/save steps. Checkpoints are written by the Trainer under
`checkpoints/<experiment>/` and support resume + early stopping.

#### Train Qwen2.5-VL-3B
```bash
python training/scripts/train.py --config training/configs/qwen25vl_3b.yaml --experiment qwen25vl-3b-v1
```

#### Train Qwen2.5-VL-7B
```bash
python training/scripts/train.py --config training/configs/qwen25vl_7b.yaml --experiment qwen25vl-7b-v1
```

#### Resume Interrupted Training from the Latest Checkpoint
```bash
python training/scripts/train.py --config training/configs/qwen25vl_3b.yaml --experiment qwen25vl-3b-v1 --resume
```

#### 1-Epoch Smoke Run (Fast Pipeline Validation)
```bash
python training/scripts/train.py --config training/configs/qwen25vl_3b.yaml --experiment qwen25vl-3b-v1 --smoke-test
```

#### Disable / Tune Early Stopping
```bash
python training/scripts/train.py --config training/configs/qwen25vl_3b.yaml --experiment qwen25vl-3b-v1 --no-early-stopping
python training/scripts/train.py --config training/configs/qwen25vl_3b.yaml --experiment qwen25vl-3b-v1 --patience 3 --early-stopping-monitor val_loss
```

The best (or last) PEFT adapter is exported to `models/<experiment>/` after
training, with a `run_metadata.json` summary (parameter counts, VRAM, timing,
early-stopping result).

---

### 8. Post-Training Evaluation & Comparison (Real Inference)

Evaluation loads the base model from the local cache, attaches the trained
`models/<experiment>/` adapter via `PeftModel.from_pretrained`, and generates a
real response per held-out test image using the training user prompt. Predictions
therefore come from actual model inference (never ground-truth passthrough).

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
