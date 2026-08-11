import os
import json
import csv
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


def load_experiment_results(exp_dir: str) -> Optional[Dict[str, Any]]:
    metrics_dir = os.path.join(exp_dir, "metrics")
    reports_dir = os.path.join(exp_dir, "reports")
    meta_file = os.path.join(exp_dir, "run_metadata.json")

    clf_file = os.path.join(metrics_dir, "classification_metrics.json")
    expl_file = os.path.join(metrics_dir, "explanation_metrics.json")
    res_file = os.path.join(reports_dir, "resource_metrics.json")

    if not os.path.exists(clf_file):
        return None

    with open(clf_file, "r", encoding="utf-8") as f:
        clf_data = json.load(f)

    expl_data = {}
    if os.path.exists(expl_file):
        with open(expl_file, "r", encoding="utf-8") as f:
            expl_data = json.load(f)

    res_data = {}
    if os.path.exists(res_file):
        with open(res_file, "r", encoding="utf-8") as f:
            res_data = json.load(f)

    meta_data = {}
    if os.path.exists(meta_file):
        with open(meta_file, "r", encoding="utf-8") as f:
            meta_data = json.load(f)

    exp_name = os.path.basename(exp_dir)
    overall_clf = clf_data.get("overall", {})

    return {
        "experiment": exp_name,
        "model_key": meta_data.get("model_key", exp_name),
        "model_id": meta_data.get("model_id", "Unknown"),
        "adaptation_strategy": meta_data.get("adaptation_strategy", "llm_projector"),
        # Classification
        "accuracy": overall_clf.get("accuracy", 0.0),
        "balanced_accuracy": overall_clf.get("balanced_accuracy", 0.0),
        "macro_f1": overall_clf.get("macro_f1", 0.0),
        "weighted_f1": overall_clf.get("weighted_f1", 0.0),
        # Explanation
        "explanation_completeness": expl_data.get("explanation_completeness", 0.0),
        "visual_grounding_score": expl_data.get("visual_grounding_score", 0.0),
        "reasoning_consistency_score": expl_data.get("reasoning_consistency_score", 0.0),
        "unsupported_claim_rate": expl_data.get("unsupported_claim_rate", 0.0),
        "format_validity": expl_data.get("format_validity_percentage", 0.0),
        # Efficiency
        "total_training_time_s": res_data.get("total_training_time_seconds", 0.0),
        "peak_vram_gb": res_data.get("peak_vram_gb", 0.0),
        "adapter_size_mb": res_data.get("adapter_size_mb", 0.0),
        "trainable_params": res_data.get("trainable_parameters", 0),
        # Full raw metrics
        "classification_metrics": clf_data,
        "explanation_metrics": expl_data,
        "resource_metrics": res_data,
    }


def compute_composite_ranking(experiments: List[Dict[str, Any]], weights: Dict[str, float] = None) -> List[Dict[str, Any]]:
    if weights is None:
        weights = {
            "classification": 0.50,
            "explanation": 0.25,
            "grounding": 0.15,
            "efficiency": 0.10,
        }

    scored = []
    for exp in experiments:
        # Normalize scores to 0.0 - 1.0 range
        clf_score = (exp["accuracy"] + exp["macro_f1"] + exp["balanced_accuracy"]) / 3.0
        expl_score = (exp["explanation_completeness"] + (exp["format_validity"] / 100.0)) / 2.0
        grounding_score = max(0.0, exp["visual_grounding_score"] - exp["unsupported_claim_rate"])

        # Efficiency: lower VRAM & lower training time score higher
        vram_eff = 1.0 / max(1.0, exp["peak_vram_gb"] / 8.0)
        eff_score = min(1.0, vram_eff)

        total_score = (
            weights["classification"] * clf_score
            + weights["explanation"] * expl_score
            + weights["grounding"] * grounding_score
            + weights["efficiency"] * eff_score
        )

        item = dict(exp)
        item["composite_score"] = round(float(total_score) * 100, 2)
        scored.append(item)

    scored.sort(key=lambda x: x["composite_score"], reverse=True)
    for idx, item in enumerate(scored, 1):
        item["rank"] = idx

    return scored


def generate_comparison_plots(experiments: List[Dict[str, Any]], output_dir: str):
    if not HAS_PLOTTING:
        return

    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    names = [e["experiment"] for e in experiments]
    y_pos = np.arange(len(names))
    height = max(5, len(names) * 0.8)

    # 1. Accuracy comparison
    plt.figure(figsize=(9, height))
    plt.barh(y_pos, [e["accuracy"] for e in experiments], color="#2980b9")
    plt.yticks(y_pos, names)
    plt.xlabel("Accuracy")
    plt.xlim(0, 1.05)
    plt.title("Model Accuracy Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "model_accuracy_comparison.png"), dpi=200)
    plt.close()

    # 2. Macro F1 comparison
    plt.figure(figsize=(9, height))
    plt.barh(y_pos, [e["macro_f1"] for e in experiments], color="#8e44ad")
    plt.yticks(y_pos, names)
    plt.xlabel("Macro F1-Score")
    plt.xlim(0, 1.05)
    plt.title("Model Macro F1-Score Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "model_macro_f1_comparison.png"), dpi=200)
    plt.close()

    # 3. Overall Composite Score
    plt.figure(figsize=(9, height))
    plt.barh(y_pos, [e["composite_score"] for e in experiments], color="#27ae60")
    plt.yticks(y_pos, names)
    plt.xlabel("Composite Score (0 - 100)")
    plt.xlim(0, 105)
    plt.title("Model Composite Ranking Score")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "model_overall_score.png"), dpi=200)
    plt.close()


def generate_final_recommendation_report(ranked_experiments: List[Dict[str, Any]], output_path: str):
    if not ranked_experiments:
        return

    top_model = ranked_experiments[0]
    best_clf = max(ranked_experiments, key=lambda x: x["macro_f1"])
    best_expl = max(ranked_experiments, key=lambda x: x["visual_grounding_score"])
    best_eff = min(ranked_experiments, key=lambda x: x.get("peak_vram_gb", 999.0))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Final VLM Model Recommendation Report\n\n")
        f.write("## Overview of Findings\n\n")
        f.write(f"- **Overall Best Candidate Model:** `{top_model['experiment']}` (Composite Score: `{top_model['composite_score']}`/100)\n")
        f.write(f"- **Best Classification Candidate:** `{best_clf['experiment']}` (Macro F1: `{best_clf['macro_f1']:.4f}`)\n")
        f.write(f"- **Best Explanation Quality Candidate:** `{best_expl['experiment']}` (Visual Grounding: `{best_expl['visual_grounding_score']:.4f}`)\n")
        f.write(f"- **Best Efficiency / VRAM Trade-off:** `{best_eff['experiment']}` (Peak VRAM: `{best_eff['peak_vram_gb']:.2f} GB`)\n\n")
        
        f.write("## Data-Driven Justification\n\n")
        f.write(f"The model `{top_model['experiment']}` demonstrated superior balanced performance across diagnostic accuracy, visual evidence grounding, and computational resource requirements. ")
        f.write(f"It achieved an accuracy of `{top_model['accuracy']:.4f}` and a macro F1 of `{top_model['macro_f1']:.4f}` while maintaining a visual grounding score of `{top_model['visual_grounding_score']:.4f}`.\n\n")

        f.write("## Comparative Ranking Table\n\n")
        f.write("| Rank | Experiment | Model | Strategy | Accuracy | Macro F1 | Grounding | Peak VRAM | Composite Score |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for exp in ranked_experiments:
            f.write(f"| #{exp['rank']} | `{exp['experiment']}` | `{exp['model_key']}` | `{exp['adaptation_strategy']}` | `{exp['accuracy']:.4f}` | `{exp['macro_f1']:.4f}` | `{exp['visual_grounding_score']:.4f}` | `{exp['peak_vram_gb']:.2f}GB` | **{exp['composite_score']}** |\n")

        f.write("\n## Strategic Recommendation\n\n")
        f.write(f"We recommend selecting `{top_model['experiment']}` for production deployment and downstream integration into the cotton disease visual assistance system.\n")


def run_cross_model_comparison(experiments_root: str = "outputs/experiments", output_dir: str = "outputs/comparison"):
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(experiments_root):
        print(f"[WARNING] No experiments directory found at {experiments_root}.")
        return

    exp_dirs = [
        os.path.join(experiments_root, d)
        for d in os.listdir(experiments_root)
        if os.path.isdir(os.path.join(experiments_root, d))
    ]

    experiments = []
    for ed in exp_dirs:
        res = load_experiment_results(ed)
        if res:
            experiments.append(res)

    if not experiments:
        print("[WARNING] No completed experiment runs found to compare.")
        return

    ranked = compute_composite_ranking(experiments)

    # 1. Save model_comparison.json
    with open(os.path.join(output_dir, "model_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(ranked, f, indent=2)

    # 2. Save model_comparison.csv
    with open(os.path.join(output_dir, "model_comparison.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "experiment", "model_key", "strategy", "accuracy", "macro_f1", "grounding", "peak_vram_gb", "composite_score"])
        for exp in ranked:
            writer.writerow([exp["rank"], exp["experiment"], exp["model_key"], exp["adaptation_strategy"], exp["accuracy"], exp["macro_f1"], exp["visual_grounding_score"], exp["peak_vram_gb"], exp["composite_score"]])

    # 3. Save model_comparison.md
    with open(os.path.join(output_dir, "model_comparison.md"), "w", encoding="utf-8") as f:
        f.write("# Cross-Model Comparison Report\n\n")
        f.write("| Rank | Experiment | Accuracy | Balanced Acc | Macro F1 | Grounding | VRAM (GB) | Composite Score |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for exp in ranked:
            f.write(f"| #{exp['rank']} | `{exp['experiment']}` | {exp['accuracy']:.4f} | {exp['balanced_accuracy']:.4f} | {exp['macro_f1']:.4f} | {exp['visual_grounding_score']:.4f} | {exp['peak_vram_gb']:.2f} | **{exp['composite_score']}** |\n")

    # 4. Generate comparison plots
    generate_comparison_plots(ranked, output_dir)

    # 5. Generate final_recommendation.md
    generate_final_recommendation_report(ranked, os.path.join(output_dir, "final_recommendation.md"))

    print(f"\n--- Cross-Model Comparison Complete ---")
    print(f"Compared {len(ranked)} experiments.")
    print(f"Comparison report: {os.path.join(output_dir, 'model_comparison.md')}")
    print(f"Final recommendation report: {os.path.join(output_dir, 'final_recommendation.md')}")
