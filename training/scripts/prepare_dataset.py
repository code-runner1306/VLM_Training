import os
import sys
import json
import csv
import random
import argparse
from collections import Counter, defaultdict
import numpy as np

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from config import config
from training.src.dataset import validate_annotation, compute_image_hash

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare, validate, filter, and split VLM dataset.")
    parser.add_argument("--annotations_file", type=str, default="outputs/annotations/annotations.jsonl", help="Input annotations JSONL file.")
    parser.add_argument("--dataset_root", type=str, default=config.dataset_dir, help="Root directory containing raw images.")
    parser.add_argument("--output_dir", type=str, default="outputs/dataset", help="Output directory for manifests, stats, and plots.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting.")
    parser.add_argument("--train_ratio", type=float, default=0.80, help="Train set ratio.")
    parser.add_argument("--val_ratio", type=float, default=0.10, help="Validation set ratio.")
    parser.add_argument("--test_ratio", type=float, default=0.10, help="Test set ratio.")
    return parser.parse_args()


def generate_plots(output_dir, stats, eligible_records, ineligible_records):
    if not HAS_PLOTTING:
        print("[WARNING] matplotlib/seaborn not installed. Skipping plot generation.")
        return

    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    sns.set_theme(style="whitegrid", palette="muted")

    # 1. Class distribution (Eligible)
    plt.figure(figsize=(10, 6))
    class_counts = stats["class_distribution"]
    classes = list(class_counts.keys())
    counts = [class_counts[c] for c in classes]
    
    y_pos = np.arange(len(classes))
    plt.barh(y_pos, counts, align="center", color="#2b5c8f")
    plt.yticks(y_pos, classes)
    plt.xlabel("Number of Eligible Samples")
    plt.title("Eligible Dataset Class Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "class_distribution.png"), dpi=200)
    plt.close()

    # 2. Eligible vs Excluded
    plt.figure(figsize=(6, 6))
    plt.pie(
        [stats["eligible_images"], stats["excluded_images"]],
        labels=["Eligible", "Excluded"],
        autopct="%1.1f%%",
        colors=["#2ea44f", "#cb2431"],
        startangle=140,
    )
    plt.title("Dataset Eligibility Breakdown")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "eligible_vs_excluded.png"), dpi=200)
    plt.close()

    # 3. Excluded by reason
    plt.figure(figsize=(8, 5))
    reason_counts = stats["exclusion_reasons"]
    reasons = list(reason_counts.keys())
    r_counts = [reason_counts[r] for r in reasons]
    
    if r_counts and sum(r_counts) > 0:
        plt.bar(reasons, r_counts, color="#d93f0b")
        plt.xlabel("Exclusion Reason")
        plt.ylabel("Count")
        plt.title("Excluded Images by Reason")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "excluded_by_reason.png"), dpi=200)
    else:
        # Save placeholder if 0 excluded
        plt.text(0.5, 0.5, "Zero Excluded Images", ha="center", va="center")
        plt.savefig(os.path.join(plots_dir, "excluded_by_reason.png"), dpi=200)
    plt.close()

    # 4. Description length distribution
    plt.figure(figsize=(9, 5))
    desc_lengths = [len(rec["parsed_annotation"].get("reasoning", "")) for rec in eligible_records]
    if desc_lengths:
        plt.hist(desc_lengths, bins=30, color="#4a235a", edgecolor="white")
        plt.xlabel("Description Reasoning Length (characters)")
        plt.ylabel("Frequency")
        plt.title("Description Length Distribution")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "description_length_distribution.png"), dpi=200)
    plt.close()

    # 5. Class description coverage
    plt.figure(figsize=(10, 6))
    class_total = stats["class_total_scanned"]
    class_eligible = stats["class_distribution"]
    
    all_cls = sorted(list(set(list(class_total.keys()) + list(class_eligible.keys()))))
    cov_percentages = [(class_eligible.get(c, 0) / max(1, class_total.get(c, 1))) * 100 for c in all_cls]
    
    y_pos = np.arange(len(all_cls))
    plt.barh(y_pos, cov_percentages, align="center", color="#117a65")
    plt.yticks(y_pos, all_cls)
    plt.xlabel("Eligible Coverage (%)")
    plt.title("Class Description Coverage Rate")
    plt.xlim(0, 105)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "class_description_coverage.png"), dpi=200)
    plt.close()

    print(f"Generated 5 dataset plots under {plots_dir}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    annotations_file = os.path.abspath(args.annotations_file)
    dataset_root = os.path.abspath(args.dataset_root)

    print(f"--- Starting Dataset Preparation ---")
    print(f"Annotations source: {annotations_file}")
    print(f"Dataset root:       {dataset_root}")
    print(f"Output directory:   {args.output_dir}")

    if not os.path.exists(annotations_file):
        print(f"[ERROR] Annotations file not found: {annotations_file}")
        sys.exit(1)

    eligible_records = []
    ineligible_records = []
    exclusion_reasons = Counter()
    class_distribution = Counter()
    class_total_scanned = Counter()
    class_excluded = Counter()

    total_scanned = 0

    with open(annotations_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            total_scanned += 1
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                ineligible_records.append({
                    "line": line_num,
                    "eligible": False,
                    "reason": "malformed_annotation"
                })
                exclusion_reasons["malformed_annotation"] += 1
                continue

            disease_label = item.get("disease", "unknown")
            class_total_scanned[disease_label] += 1

            is_eligible, reason, cleaned_item = validate_annotation(item, dataset_root)

            if is_eligible:
                eligible_records.append(cleaned_item)
                class_distribution[cleaned_item["disease"]] += 1
            else:
                ineligible_records.append({
                    "image_id": item.get("image_id", ""),
                    "image_path": item.get("image_path", ""),
                    "disease": disease_label,
                    "eligible": False,
                    "reason": reason
                })
                exclusion_reasons[reason] += 1
                class_excluded[disease_label] += 1

    eligible_count = len(eligible_records)
    excluded_count = len(ineligible_records)
    eligibility_pct = (eligible_count / max(1, total_scanned)) * 100

    # Calculate description length stats
    desc_lengths = [len(rec["parsed_annotation"].get("reasoning", "")) for rec in eligible_records]
    if desc_lengths:
        avg_len = float(np.mean(desc_lengths))
        median_len = float(np.median(desc_lengths))
        min_len = int(np.min(desc_lengths))
        max_len = int(np.max(desc_lengths))
    else:
        avg_len = median_len = min_len = max_len = 0

    # 1. Save eligible_manifest.jsonl
    eligible_manifest_path = os.path.join(args.output_dir, "eligible_manifest.jsonl")
    with open(eligible_manifest_path, "w", encoding="utf-8") as f:
        for rec in eligible_records:
            f.write(json.dumps(rec) + "\n")

    # 2. Save dataset_eligibility.json & csv
    eligibility_summary = {
        "total_images": total_scanned,
        "eligible_images": eligible_count,
        "excluded_images": excluded_count,
        "eligibility_percentage": round(eligibility_pct, 2),
        "exclusion_reasons": dict(exclusion_reasons),
        "excluded_details": ineligible_records
    }

    with open(os.path.join(args.output_dir, "dataset_eligibility.json"), "w", encoding="utf-8") as f:
        json.dump(eligibility_summary, f, indent=2)

    with open(os.path.join(args.output_dir, "dataset_eligibility.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "image_path", "disease", "eligible", "reason"])
        for rec in ineligible_records:
            writer.writerow([
                rec.get("image_id", ""),
                rec.get("image_path", ""),
                rec.get("disease", ""),
                False,
                rec.get("reason", "")
            ])

    # 3. Save eligibility report markdown
    eligibility_report_path = os.path.join(args.output_dir, "dataset_eligibility_report.md")
    with open(eligibility_report_path, "w", encoding="utf-8") as f:
        f.write("# Dataset Eligibility Report\n\n")
        f.write(f"- **Total images scanned:** `{total_scanned}`\n")
        f.write(f"- **Valid eligible descriptions:** `{eligible_count}` (`{eligibility_pct:.2f}%`)\n")
        f.write(f"- **Excluded images:** `{excluded_count}` (`{100 - eligibility_pct:.2f}%`)\n\n")
        f.write("### Exclusion Reasons Breakdown\n\n")
        f.write("| Reason | Count |\n| --- | --- |\n")
        for reason, count in exclusion_reasons.items():
            f.write(f"| `{reason}` | `{count}` |\n")

    # 4. Save dataset statistics (json, csv, md)
    stats_data = {
        "total_images": total_scanned,
        "eligible_images": eligible_count,
        "excluded_images": excluded_count,
        "eligibility_percentage": round(eligibility_pct, 2),
        "class_total_scanned": dict(class_total_scanned),
        "class_distribution": dict(class_distribution),
        "class_excluded": dict(class_excluded),
        "exclusion_reasons": dict(exclusion_reasons),
        "description_length": {
            "average": round(avg_len, 2),
            "median": round(median_len, 2),
            "min": min_len,
            "max": max_len
        }
    }

    with open(os.path.join(args.output_dir, "dataset_statistics.json"), "w", encoding="utf-8") as f:
        json.dump(stats_data, f, indent=2)

    with open(os.path.join(args.output_dir, "dataset_statistics.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["disease", "total_scanned", "eligible_count", "excluded_count", "coverage_percentage"])
        for cls, total in class_total_scanned.items():
            el = class_distribution.get(cls, 0)
            ex = class_excluded.get(cls, 0)
            cov = (el / max(1, total)) * 100
            writer.writerow([cls, total, el, ex, f"{cov:.2f}%"])

    with open(os.path.join(args.output_dir, "dataset_statistics.md"), "w", encoding="utf-8") as f:
        f.write("# Dataset Statistics Summary\n\n")
        f.write("## Overall Metrics\n")
        f.write(f"- **Total Images:** {total_scanned}\n")
        f.write(f"- **Eligible Images:** {eligible_count} ({eligibility_pct:.2f}%)\n")
        f.write(f"- **Excluded Images:** {excluded_count}\n\n")
        f.write("## Description Length Statistics\n")
        f.write(f"- **Average Length:** {avg_len:.1f} chars\n")
        f.write(f"- **Median Length:** {median_len:.1f} chars\n")
        f.write(f"- **Min / Max Length:** {min_len} / {max_len} chars\n\n")
        f.write("## Per-Class Breakdown\n\n")
        f.write("| Disease Class | Scanned | Eligible | Excluded | Coverage |\n")
        f.write("| --- | --- | --- | --- | --- |\n")
        for cls, total in sorted(class_total_scanned.items()):
            el = class_distribution.get(cls, 0)
            ex = class_excluded.get(cls, 0)
            cov = (el / max(1, total)) * 100
            f.write(f"| `{cls}` | {total} | {el} | {ex} | {cov:.1f}% |\n")

    # Generate Plots
    generate_plots(args.output_dir, stats_data, eligible_records, ineligible_records)

    # 5. Deterministic Splitting & Data Leakage Prevention
    print(f"\n--- Splitting Dataset (Seed={args.seed}) ---")
    random.seed(args.seed)

    # Stratified grouped shuffle split by disease class and image hash to prevent leakage
    class_groups = defaultdict(lambda: defaultdict(list))
    for rec in eligible_records:
        img_hash = compute_image_hash(rec["image_path"]) or rec["image_id"]
        class_groups[rec["disease"]][img_hash].append(rec)

    train_set, val_set, test_set = [], [], []

    for disease, hash_dict in class_groups.items():
        unique_hashes = list(hash_dict.keys())
        random.shuffle(unique_hashes)
        n_hashes = len(unique_hashes)
        n_train_h = int(n_hashes * args.train_ratio)
        n_val_h = int(n_hashes * args.val_ratio)
        
        train_hashes = set(unique_hashes[:n_train_h])
        val_hashes = set(unique_hashes[n_train_h:n_train_h + n_val_h])
        test_hashes = set(unique_hashes[n_train_h + n_val_h:])

        for h, items in hash_dict.items():
            if h in train_hashes:
                train_set.extend(items)
            elif h in val_hashes:
                val_set.extend(items)
            else:
                test_set.extend(items)

    # Shuffle splits
    random.shuffle(train_set)
    random.shuffle(val_set)
    random.shuffle(test_set)

    def write_manifest(filepath, records):
        with open(filepath, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

    train_manifest_path = os.path.join(args.output_dir, "train_manifest.jsonl")
    val_manifest_path = os.path.join(args.output_dir, "validation_manifest.jsonl")
    test_manifest_path = os.path.join(args.output_dir, "test_manifest.jsonl")

    write_manifest(train_manifest_path, train_set)
    write_manifest(val_manifest_path, val_set)
    write_manifest(test_manifest_path, test_set)

    split_metadata = {
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "total_eligible": eligible_count,
        "train_count": len(train_set),
        "validation_count": len(val_set),
        "test_count": len(test_set),
    }

    with open(os.path.join(args.output_dir, "split_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(split_metadata, f, indent=2)

    # Data Leakage Checks
    print("Checking for duplicate image hashes and ID collisions across splits...")
    train_ids = {r["image_id"] for r in train_set}
    val_ids = {r["image_id"] for r in val_set}
    test_ids = {r["image_id"] for r in test_set}

    id_overlap_tv = train_ids.intersection(val_ids)
    id_overlap_tt = train_ids.intersection(test_ids)
    id_overlap_vt = val_ids.intersection(test_ids)

    # Hash checks
    train_hashes = {compute_image_hash(r["image_path"]): r["image_id"] for r in train_set if os.path.exists(r["image_path"])}
    val_hashes = {compute_image_hash(r["image_path"]): r["image_id"] for r in val_set if os.path.exists(r["image_path"])}
    test_hashes = {compute_image_hash(r["image_path"]): r["image_id"] for r in test_set if os.path.exists(r["image_path"])}

    hash_overlap_tv = set(train_hashes.keys()).intersection(set(val_hashes.keys())) - {None}
    hash_overlap_tt = set(train_hashes.keys()).intersection(set(test_hashes.keys())) - {None}
    hash_overlap_vt = set(val_hashes.keys()).intersection(set(test_hashes.keys())) - {None}

    leakage_report_path = os.path.join(args.output_dir, "leakage_report.md")
    with open(leakage_report_path, "w", encoding="utf-8") as f:
        f.write("# Data Leakage Report\n\n")
        f.write(f"- **Random Seed:** `{args.seed}`\n")
        f.write(f"- **Train Count:** `{len(train_set)}`\n")
        f.write(f"- **Validation Count:** `{len(val_set)}`\n")
        f.write(f"- **Test Count:** `{len(test_set)}`\n\n")
        f.write("## ID Collision Checks\n")
        f.write(f"- Train / Validation Overlap: `{len(id_overlap_tv)}` IDs\n")
        f.write(f"- Train / Test Overlap: `{len(id_overlap_tt)}` IDs\n")
        f.write(f"- Validation / Test Overlap: `{len(id_overlap_vt)}` IDs\n\n")
        f.write("## Image Hash Perceptual Leakage Checks\n")
        f.write(f"- Train / Validation Duplicate Hashes: `{len(hash_overlap_tv)}`\n")
        f.write(f"- Train / Test Duplicate Hashes: `{len(hash_overlap_tt)}`\n")
        f.write(f"- Validation / Test Duplicate Hashes: `{len(hash_overlap_vt)}`\n\n")
        if len(id_overlap_tv) == 0 and len(id_overlap_tt) == 0 and len(id_overlap_vt) == 0 and len(hash_overlap_tv) == 0 and len(hash_overlap_tt) == 0 and len(hash_overlap_vt) == 0:
            f.write("> [!NOTE]\n> **Data Leakage Check PASSED**: Zero duplicate image IDs or image hashes detected across train/val/test splits.\n")
        else:
            f.write("> [!WARNING]\n> **Potential Data Leakage Detected**: Duplicate images found across splits.\n")

    print(f"\n--- Dataset Preparation Complete ---")
    print(f"Eligible dataset manifest: {eligible_manifest_path} ({eligible_count} items)")
    print(f"Train manifest:           {train_manifest_path} ({len(train_set)} items)")
    print(f"Validation manifest:      {val_manifest_path} ({len(val_set)} items)")
    print(f"Test manifest:            {test_manifest_path} ({len(test_set)} items)")


if __name__ == "__main__":
    main()
