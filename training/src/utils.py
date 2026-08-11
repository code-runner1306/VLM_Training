import os
import sys
import json
import platform
import torch
from typing import Dict, Any, Optional


def get_hardware_info() -> Dict[str, Any]:
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu": platform.processor() or "Unknown CPU",
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "N/A (CPU)",
    }
    return info


def get_dir_size_mb(directory_path: str) -> float:
    total_bytes = 0
    if os.path.exists(directory_path):
        for root, _, files in os.walk(directory_path):
            for f in files:
                filepath = os.path.join(root, f)
                try:
                    total_bytes += os.path.getsize(filepath)
                except Exception:
                    pass
    return round(total_bytes / (1024 * 1024), 2)


def generate_resource_report(
    experiment_name: str,
    output_dir: str,
    training_time_s: float,
    peak_vram_gb: float,
    param_counts: Dict[str, Any],
    adapter_dir: str,
):
    os.makedirs(output_dir, exist_ok=True)
    hardware = get_hardware_info()
    adapter_size_mb = get_dir_size_mb(adapter_dir)

    metrics = {
        "experiment": experiment_name,
        "total_training_time_seconds": round(training_time_s, 2),
        "total_training_time_hours": round(training_time_s / 3600, 4),
        "peak_vram_gb": round(peak_vram_gb, 2),
        "adapter_size_mb": adapter_size_mb,
        "total_parameters": param_counts.get("total_parameters", 0),
        "trainable_parameters": param_counts.get("trainable_parameters", 0),
        "trainable_percentage": param_counts.get("trainable_percentage", 0),
        "hardware": hardware,
    }

    # 1. Save resource_metrics.json
    with open(os.path.join(output_dir, "resource_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # 2. Save resource_report.md
    with open(os.path.join(output_dir, "resource_report.md"), "w", encoding="utf-8") as f:
        f.write(f"# Resource & Hardware Efficiency Report: `{experiment_name}`\n\n")
        f.write("## Compute Environment\n")
        f.write(f"- **GPU:** `{hardware['gpu_name']}`\n")
        f.write(f"- **CPU / Platform:** `{hardware['platform']}`\n")
        f.write(f"- **PyTorch / Python:** `{torch.__version__}` / `{hardware['python_version']}`\n\n")
        f.write("## Training Performance & Memory Metrics\n")
        f.write(f"- **Total Training Time:** `{training_time_s:.2f}s` (`{training_time_s / 3600:.3f} hrs`)\n")
        f.write(f"- **Peak VRAM Allocated:** `{peak_vram_gb:.2f} GB`\n")
        f.write(f"- **Adapter Storage Footprint:** `{adapter_size_mb:.2f} MB`\n\n")
        f.write("## Model Parameter Counts\n")
        f.write(f"- **Total Parameters:** `{param_counts.get('total_parameters', 0):,}`\n")
        f.write(f"- **Trainable Parameters:** `{param_counts.get('trainable_parameters', 0):,}`\n")
        f.write(f"- **Trainable Percentage:** `{param_counts.get('trainable_percentage', 0):.4f}%`\n")
