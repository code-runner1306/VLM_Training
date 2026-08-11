import os
import numpy as np
from typing import Dict, Any, List

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False


def plot_confusion_matrix(
    cm_matrix: List[List[int]],
    labels: List[str],
    output_path: str,
    normalize: bool = False,
    title: str = "Confusion Matrix",
):
    if not HAS_PLOTTING:
        print("[WARNING] matplotlib/seaborn not available. Skipping confusion matrix plot.")
        return

    cm = np.array(cm_matrix)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm = cm.astype("float") / row_sums
        fmt = ".2f"
    else:
        fmt = "d"

    fig_size = max(8, len(labels) * 0.6)
    plt.figure(figsize=(fig_size, fig_size))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
    )
    plt.xlabel("Predicted Disease Label")
    plt.ylabel("True Disease Label")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_per_class_bar_charts(per_class_metrics: Dict[str, Any], output_dir: str):
    if not HAS_PLOTTING:
        return

    os.makedirs(output_dir, exist_ok=True)
    classes = sorted(list(per_class_metrics.keys()))

    precisions = [per_class_metrics[c]["precision"] for c in classes]
    recalls = [per_class_metrics[c]["recall"] for c in classes]
    f1s = [per_class_metrics[c]["f1"] for c in classes]
    supports = [per_class_metrics[c]["support"] for c in classes]

    y_pos = np.arange(len(classes))
    fig_height = max(6, len(classes) * 0.4)

    # 1. Precision
    plt.figure(figsize=(10, fig_height))
    plt.barh(y_pos, precisions, color="#2980b9")
    plt.yticks(y_pos, classes)
    plt.xlabel("Precision")
    plt.xlim(0, 1.05)
    plt.title("Per-Class Precision")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "per_class_precision.png"), dpi=200)
    plt.close()

    # 2. Recall
    plt.figure(figsize=(10, fig_height))
    plt.barh(y_pos, recalls, color="#27ae60")
    plt.yticks(y_pos, classes)
    plt.xlabel("Recall")
    plt.xlim(0, 1.05)
    plt.title("Per-Class Recall")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "per_class_recall.png"), dpi=200)
    plt.close()

    # 3. F1 Score
    plt.figure(figsize=(10, fig_height))
    plt.barh(y_pos, f1s, color="#8e44ad")
    plt.yticks(y_pos, classes)
    plt.xlabel("F1-Score")
    plt.xlim(0, 1.05)
    plt.title("Per-Class F1-Score")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "per_class_f1.png"), dpi=200)
    plt.close()

    # 4. Support
    plt.figure(figsize=(10, fig_height))
    plt.barh(y_pos, supports, color="#d35400")
    plt.yticks(y_pos, classes)
    plt.xlabel("Test Support (Count)")
    plt.title("Per-Class Support")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "per_class_support.png"), dpi=200)
    plt.close()


def plot_training_curves(history: Dict[str, List[float]], output_dir: str):
    if not HAS_PLOTTING:
        return

    os.makedirs(output_dir, exist_ok=True)

    # 1. Training Loss
    if "train_loss" in history and history["train_loss"]:
        plt.figure(figsize=(8, 5))
        plt.plot(history["train_loss"], label="Train Loss", color="#c0392b", linewidth=2)
        plt.xlabel("Step / Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "training_loss.png"), dpi=200)
        plt.close()

    # 2. Validation Loss
    if "val_loss" in history and history["val_loss"]:
        plt.figure(figsize=(8, 5))
        plt.plot(history["val_loss"], label="Val Loss", color="#2980b9", linewidth=2)
        plt.xlabel("Step / Epoch")
        plt.ylabel("Loss")
        plt.title("Validation Loss Curve")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "validation_loss.png"), dpi=200)
        plt.close()

    # 3. Learning Rate
    if "learning_rate" in history and history["learning_rate"]:
        plt.figure(figsize=(8, 5))
        plt.plot(history["learning_rate"], label="Learning Rate", color="#f39c12", linewidth=2)
        plt.xlabel("Step")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Schedule")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "learning_rate.png"), dpi=200)
        plt.close()
