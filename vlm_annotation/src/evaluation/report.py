import csv
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List


def generate_benchmark_reports(
    model_summaries: List[Dict[str, Any]],
    output_dir: str = "outputs/benchmark"
):
    """
    Generate final_report.json, final_report.csv, and report.md from model summaries.
    Includes per-model rate limit hits, errors, latency, and composite scores.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Sort models by overall composite score descending
    sorted_summaries = sorted(model_summaries, key=lambda x: x.get("overall_score", 0.0), reverse=True)

    # 1. Write final_report.json
    json_path = out_path / "final_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sorted_summaries, f, indent=2)

    # 2. Write final_report.csv
    csv_path = out_path / "final_report.csv"
    fieldnames = [
        "model", "provider", "images_processed", "successful_requests", "failed_requests",
        "rate_limit_hits", "json_validity_rate", "average_latency_ms", "median_latency_ms",
        "p95_latency_ms", "requests_per_minute", "overall_score"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in sorted_summaries:
            row = {k: s.get(k, 0) for k in fieldnames}
            writer.writerow(row)

    # 3. Write report.md
    md_path = out_path / "report.md"
    lines = [
        "# VLM Benchmarking Final Report",
        "",
        "## Model Leaderboard",
        "",
        "| Rank | Model | Provider | Overall Score | Success Rate | 429 Rate Limit Hits | Avg Latency (s) | RPM |",
        "|---|---|---|---|---|---|---|---|"
    ]

    for idx, s in enumerate(sorted_summaries, start=1):
        name = s.get("model", "Unknown")
        provider = s.get("provider", "Unknown")
        score = s.get("overall_score", 0.0)
        validity = s.get("json_validity_rate", 0.0)
        rate_limits = s.get("rate_limit_hits", 0)
        avg_lat = round(s.get("average_latency_ms", 0.0) / 1000.0, 2)
        rpm = round(s.get("requests_per_minute", 0.0), 1)

        lines.append(
            f"| {idx} | {name} | {provider} | **{score:.1f}** | {validity:.1f}% | {rate_limits} | {avg_lat}s | {rpm} |"
        )

    lines.extend([
        "",
        "## Diagnostic & Error Counters per Model",
        ""
    ])

    for s in sorted_summaries:
        lines.append(f"### Model: {s.get('model')} ({s.get('provider')})")
        lines.append(f"- **Total Requests**: {s.get('images_processed')}")
        lines.append(f"- **Successful Requests**: {s.get('successful_requests')}")
        lines.append(f"- **Failed Requests**: {s.get('failed_requests')}")
        lines.append(f"- **Rate Limit Hits (429)**: {s.get('rate_limit_hits')}")
        lines.append(f"- **JSON Parsing Failures**: {s.get('json_parse_failures', 0)}")
        lines.append(f"- **Average Latency**: {s.get('average_latency_ms', 0):.2f} ms")
        lines.append(f"- **Median Latency**: {s.get('median_latency_ms', 0):.2f} ms")
        lines.append(f"- **P95 Latency**: {s.get('p95_latency_ms', 0):.2f} ms")
        lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
