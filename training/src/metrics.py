import json
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from collections import Counter

try:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        precision_recall_fscore_support,
        matthews_corrcoef,
        cohen_kappa_score,
        confusion_matrix,
    )
except ImportError:
    accuracy_score = None


def compute_classification_metrics(
    y_true: List[str],
    y_pred: List[str],
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics.
    """
    if labels is None:
        labels = sorted(list(set(y_true + y_pred)))

    if accuracy_score is None:
        # Fallback pure-python implementation if sklearn missing
        correct = sum([1 for gt, p in zip(y_true, y_pred) if gt == p])
        acc = correct / max(1, len(y_true))
        return {
            "overall": {"accuracy": acc, "count": len(y_true)},
            "per_class": {},
            "confusion_matrix": [],
        }

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    mcc = float(matthews_corrcoef(y_true, y_pred)) if len(set(y_true)) > 1 else 0.0
    kappa = float(cohen_kappa_score(y_true, y_pred)) if len(set(y_true)) > 1 else 0.0

    # Macro & Weighted F1
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0)

    # Per-class metrics
    p_class, r_class, f1_class, sup_class = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)

    per_class_metrics = {}
    for idx, label in enumerate(labels):
        correct_c = sum([1 for gt, p in zip(y_true, y_pred) if gt == label and p == label])
        incorrect_c = int(sup_class[idx]) - correct_c
        per_class_metrics[label] = {
            "precision": float(p_class[idx]),
            "recall": float(r_class[idx]),
            "f1": float(f1_class[idx]),
            "support": int(sup_class[idx]),
            "correct_predictions": correct_c,
            "incorrect_predictions": incorrect_c,
        }

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "overall": {
            "accuracy": round(acc, 4),
            "balanced_accuracy": round(bal_acc, 4),
            "macro_precision": round(float(p_macro), 4),
            "macro_recall": round(float(r_macro), 4),
            "macro_f1": round(float(f1_macro), 4),
            "weighted_f1": round(float(f1_weighted), 4),
            "micro_f1": round(float(f1_micro), 4),
            "matthews_corrcoef": round(mcc, 4),
            "cohen_kappa": round(kappa, 4),
            "total_samples": len(y_true),
        },
        "per_class": per_class_metrics,
        "labels": labels,
        "confusion_matrix": cm.tolist(),
    }


def compute_expected_calibration_error(
    confidences: List[float],
    correctness: List[int],
    n_bins: int = 10,
) -> float:
    """Calculate Expected Calibration Error (ECE)."""
    if not confidences or len(confidences) != len(correctness):
        return 0.0

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(confidences)

    conf_arr = np.array(confidences)
    corr_arr = np.array(correctness)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (conf_arr > bin_lower) & (conf_arr <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(corr_arr[in_bin])
            avg_confidence_in_bin = np.mean(conf_arr[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

    return round(float(ece), 4)
