"""
Centralized Configuration File for VLM Annotation & LoRA Fine-Tuning Pipeline.

Modify parameters directly in this file to set your default pipeline preferences.
Any CLI flags passed during command execution will dynamically override these defaults.
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class PipelineConfig:
    # -------------------------------------------------------------------------
    # 1. Dataset & Directory Paths
    # -------------------------------------------------------------------------
    dataset_dir: str = "Cotton_dataset"
    output_dir: str = "outputs"
    log_dir: str = "logs"

    # -------------------------------------------------------------------------
    # 2. Synthetic Annotation Generation Settings (Single Model)
    # -------------------------------------------------------------------------
    annotation_provider: str = "huggingface"  # Options: huggingface, gemini, ollama, nvidia, groq, openrouter
    annotation_model: str = "Qwen/Qwen3-VL-8B-Instruct"  # Options: Qwen/Qwen3-VL-8B-Instruct, OpenGVLab/InternVL2_5-8B, OpenGVLab/InternVL2_5-14B, Qwen/Qwen2.5-VL-7B-Instruct
    ollama_host: str = "http://127.0.0.1:11434"
    num_annotation_samples: Optional[int] = None  # Set integer (e.g. 500) to test subset, or None for full dataset
    start_index: int = 0
    end_index: Optional[int] = None
    resume: bool = True
    retry_failed: bool = False
    benchmark_speed: bool = False

    # -------------------------------------------------------------------------
    # 3. VLM Fine-Tuning Candidate Models List (Sequential Multi-Model Training)
    # -------------------------------------------------------------------------
    training_models: List[Dict[str, Any]] = field(default_factory=lambda: [
        {
            "experiment": "scold-v1",
            "train_config": "training/configs/scold.yaml",
            "model_id": "SCOLD/SCOLD-Agricultural-Disease"
        },
        {
            "experiment": "qwen3vl-8b-v1",
            "train_config": "training/configs/qwen3vl.yaml",
            "model_id": "Qwen/Qwen3-VL-8B-Instruct"
        },
        {
            "experiment": "internvl25-8b-v1",
            "train_config": "training/configs/internvl_8b.yaml",
            "model_id": "OpenGVLab/InternVL2_5-8B"
        },
        {
            "experiment": "qwen25vl-3b-v1",
            "train_config": "training/configs/qwen25vl_3b.yaml",
            "model_id": "Qwen/Qwen2.5-VL-3B-Instruct"
        }
    ])

    delete_cache_after_train: bool = True  # Automatically delete downloaded base model weights from HF cache after training to save disk space

    train_config: str = "training/configs/qwen25vl_3b.yaml"
    experiment: str = "qwen25vl-3b-v1"

    seed: int = 42
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10

    # -------------------------------------------------------------------------
    # 4. Pipeline Execution & GitHub Remote Auto-Push Settings
    # -------------------------------------------------------------------------
    smoke_test: bool = False  # Enable fast end-to-end pipeline smoke test with minimal sample sizes and steps
    skip_annotation: bool = False
    skip_training: bool = False
    auto_push: bool = True  # Automatically stage, commit, and push outputs to GitHub on completion or error


# Global configuration object instance
config = PipelineConfig()
