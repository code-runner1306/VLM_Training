import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from vlm_annotation.src.dataset import discover_dataset


def normalize_dataset_name(dataset_dir: str) -> str:
    """Derive the canonical store folder name from a dataset directory path."""
    name = Path(dataset_dir).resolve().name
    return name.strip().lower().replace(" ", "_")


def store_dir(dataset_dir: str) -> Path:
    """Return the canonical per-dataset artifact store directory."""
    return Path("artifacts") / normalize_dataset_name(dataset_dir)


def read_jsonl(path: Path) -> List[dict]:
    records = []
    if not path.exists():
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def write_jsonl(path: Path, records: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def append_jsonl(path: Path, records: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def record_image_id(rec: dict) -> Optional[str]:
    return rec.get("image_id") or (rec.get("parsed_annotation") or {}).get("image_id")


def load_ids(path: Path) -> set:
    return {i for i in (record_image_id(r) for r in read_jsonl(path)) if i}


def compute_coverage(store: Path, dataset_dir: str, provider: str = "", model: str = "") -> dict:
    """Diff the dataset against the canonical store and write coverage.json.

    `complete` requires both zero missing images AND zero unresolved failures.
    """
    all_items, _ = discover_dataset(dataset_dir)
    all_ids = {it.image_id for it in all_items}
    annotated = load_ids(store / "annotations.jsonl")
    failed = load_ids(store / "failed.jsonl") - annotated
    missing = all_ids - annotated - failed

    coverage = {
        "dataset": normalize_dataset_name(dataset_dir),
        "provider": provider,
        "model": model,
        "dataset_total": len(all_ids),
        "annotated": len(annotated),
        "failed": len(failed),
        "missing": len(missing),
        "complete": len(missing) == 0 and len(failed) == 0,
    }
    write_json(store / "coverage.json", coverage)
    return coverage


def promote_merge(
    store: Path,
    run_dir: Path,
    worker_count: int,
    *,
    provider: str = "",
    model: str = "",
    prompt_version: str = "1.0",
    force_regenerate: bool = False,
) -> dict:
    """Promote per-worker annotations from a scratch run dir into the canonical store.

    Idempotent: existing store records are kept and win over duplicates. When
    `force_regenerate` is set, existing records are replaced by the new batch.
    """
    store.mkdir(parents=True, exist_ok=True)
    existing = [] if force_regenerate else read_jsonl(store / "annotations.jsonl")

    seen = set()
    merged = []
    for rec in existing:
        img_id = record_image_id(rec)
        if img_id:
            if img_id in seen:
                continue
            seen.add(img_id)
        merged.append(rec)

    new_annotated = 0
    for w in range(worker_count):
        for rec in read_jsonl(run_dir / f"worker_{w}" / "annotations.jsonl"):
            img_id = record_image_id(rec)
            if img_id:
                if img_id in seen:
                    continue
                seen.add(img_id)
                if force_regenerate:
                    rec["prompt_version"] = prompt_version
            new_annotated += 1
            merged.append(rec)

    merged.sort(key=lambda r: r.get("image_path") or "")
    write_jsonl(store / "annotations.jsonl", merged)

    failed = []
    for w in range(worker_count):
        failed.extend(read_jsonl(run_dir / f"worker_{w}" / "failed.jsonl"))
    append_jsonl(store / "failed.jsonl", failed)

    return {
        "annotated": new_annotated,
        "failed": len(failed),
        "total_in_store": len(merged),
    }


def append_batch(store: Path, batch: dict) -> None:
    append_jsonl(store / "batches.jsonl", [batch])


def recompute_statistics(store: Path, provider: str = "", model: str = "") -> dict:
    """Aggregate cumulative annotation statistics from batches.jsonl + coverage.json."""
    batches = read_jsonl(store / "batches.jsonl")
    annotated = sum(b.get("annotated", 0) for b in batches)
    failed = sum(b.get("failed", 0) for b in batches)
    runtime = sum(b.get("runtime_sec", 0) for b in batches)

    coverage = {}
    cov_path = store / "coverage.json"
    if cov_path.exists():
        coverage = json.loads(cov_path.read_text(encoding="utf-8"))

    stats = {
        "provider": provider or coverage.get("provider", ""),
        "model": model or coverage.get("model", ""),
        "batches": len(batches),
        "total_annotated": annotated,
        "total_failed": failed,
        "total_runtime_sec": round(runtime, 2),
        "annotated_in_store": coverage.get("annotated", 0),
        "dataset_total": coverage.get("dataset_total", 0),
        "complete": coverage.get("complete", False),
    }
    write_json(store / "statistics.json", stats)
    return stats


def prune_worker_dirs(run_dir: Path, worker_count: int) -> None:
    """Remove per-worker scratch dirs after a successful merge."""
    for w in range(worker_count):
        shutil.rmtree(run_dir / f"worker_{w}", ignore_errors=True)
