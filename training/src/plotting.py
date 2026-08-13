import os
import numpy as np
from typing import Dict, Any, List, Optional

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


def plot_training_curves(
    history: Dict[str, List[float]],
    output_dir: str,
    best_epoch: Optional[int] = None,
    stopped_epoch: Optional[int] = None,
):
    """
    Plot training/validation curves with best checkpoint and early stopping annotations.
    """
    if not HAS_PLOTTING:
        return

    os.makedirs(output_dir, exist_ok=True)

    has_train = "train_loss" in history and len(history["train_loss"]) > 0
    has_val = "val_loss" in history and len(history["val_loss"]) > 0

    # 1. Combined Train & Validation Loss Curve
    if has_train or has_val:
        plt.figure(figsize=(9, 5))
        epochs_train = list(range(1, len(history.get("train_loss", [])) + 1))
        epochs_val = list(range(1, len(history.get("val_loss", [])) + 1))

        if has_train:
            plt.plot(epochs_train, history["train_loss"], label="Train Loss", color="#c0392b", marker="o", linewidth=2)
        if has_val:
            plt.plot(epochs_val, history["val_loss"], label="Val Loss", color="#2980b9", marker="s", linewidth=2)

        # Draw best epoch line
        if best_epoch is not None:
            plt.axvline(x=best_epoch, color="#27ae60", linestyle="--", linewidth=1.8, label=f"Best Epoch ({best_epoch})")

        # Draw early stopping trigger line
        if stopped_epoch is not None:
            plt.axvline(x=stopped_epoch, color="#e67e22", linestyle=":", linewidth=2, label=f"Early Stopped ({stopped_epoch})")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training & Validation Loss Curves with Early Stopping")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "loss_curves.png"), dpi=200)
        plt.close()

    # 2. Individual Training Loss
    if has_train:
        plt.figure(figsize=(8, 5))
        plt.plot(history["train_loss"], label="Train Loss", color="#c0392b", linewidth=2)
        plt.xlabel("Epoch / Step")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "training_loss.png"), dpi=200)
        plt.close()

    # 3. Individual Validation Loss
    if has_val:
        plt.figure(figsize=(8, 5))
        plt.plot(history["val_loss"], label="Val Loss", color="#2980b9", linewidth=2)
        if best_epoch is not None:
            plt.axvline(x=best_epoch - 1, color="#27ae60", linestyle="--", label=f"Best Epoch ({best_epoch})")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Validation Loss Curve")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "validation_loss.png"), dpi=200)
        plt.close()

    # 4. Learning Rate Schedule
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
