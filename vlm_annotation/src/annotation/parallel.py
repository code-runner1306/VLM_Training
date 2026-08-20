import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from vlm_annotation.src.annotation.keys import mask_key


class ParallelAnnotationError(RuntimeError):
    """Raised when one or more parallel annotation workers exit non-zero."""


def slice_chunks(total_images: int, start_index: int, chunk_size: int, worker_count: int) -> List[Tuple[int, int]]:
    """Return strict contiguous (start, end) slices for each worker.

    Worker N covers `[start_index + N*chunk_size, start_index + (N+1)*chunk_size)`,
    clamped to the global image list. Workers with an empty slice are still listed
    (callers should skip spawning them).
    """
    if total_images <= 0 or worker_count <= 0 or chunk_size <= 0:
        return []
    chunks: List[Tuple[int, int]] = []
    for n in range(worker_count):
        start = start_index + n * chunk_size
        end = min(start_index + (n + 1) * chunk_size, total_images)
        chunks.append((start, end))
    return chunks


def build_worker_command(
    script_path: str,
    dataset_dir: str,
    model: str,
    worker_dir: str,
    start_index: int,
    end_index: int,
    resume: bool = False,
    prompt_version: str = "1.0",
    force_regenerate: bool = False,
) -> List[str]:
    """Build the argv for a single worker subprocess."""
    cmd = [
        sys.executable,
        script_path,
        "--dataset-dir", dataset_dir,
        "--provider", "gemini",
        "--model", model,
        "--output-dir", worker_dir,
        "--start-index", str(start_index),
        "--end-index", str(end_index),
        "--prompt-version", prompt_version,
    ]
    if resume:
        cmd.append("--resume")
    if force_regenerate:
        cmd.append("--force-regenerate")
    return cmd


def build_worker_env(api_key: str) -> Dict[str, str]:
    """Child env: inherit parent, inject one GEMINI_API_KEY, force single-worker mode.

    `GEMINI_API_KEYS` is set to an empty string (rather than removed) because the
    worker calls `load_dotenv()` at import, which would otherwise reload the
    multi-key list from `.env` and re-trigger parallel mode recursively.
    """
    env = os.environ.copy()
    env["GEMINI_API_KEY"] = api_key
    env["GEMINI_API_KEYS"] = ""
    env["MAX_GEMINI_WORKERS"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _stream_pipe(pipe, worker_id: int, prefix: str = "[worker_{}] "):
    for line in pipe:
        sys.stdout.write(f"{prefix.format(worker_id)}{line}")
        sys.stdout.flush()


def run_workers(
    commands: Sequence[List[str]],
    envs: Sequence[Dict[str, str]],
    cwd: Optional[str] = None,
) -> List[int]:
    """Spawn all workers, stream their output live, and return exit codes."""
    procs = []
    threads = []
    for i, (cmd, env) in enumerate(zip(commands, envs)):
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
        )
        procs.append(proc)
        t = threading.Thread(target=_stream_pipe, args=(proc.stdout, i), daemon=True)
        t.start()
        threads.append(t)

    exit_codes = []
    for i, proc in enumerate(procs):
        proc.wait()
        if threads[i].is_alive():
            threads[i].join(timeout=1.0)
        exit_codes.append(proc.returncode)
    return exit_codes


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


def write_jsonl(path: Path, records: Sequence[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def merge_annotations(run_dir: Path, worker_count: int) -> List[dict]:
    """Merge per-worker annotations.jsonl: dedupe on image_id (first wins), sort by image_path.

    Records already present in run_dir/annotations.jsonl are kept first so reusing a
    run directory accumulates results across invocations instead of resetting them.
    """
    merged: List[dict] = []
    seen = set()

    existing = run_dir / "annotations.jsonl"
    if existing.exists():
        for rec in read_jsonl(existing):
            img_id = rec.get("image_id") or rec.get("parsed_annotation", {}).get("image_id")
            if img_id:
                if img_id in seen:
                    continue
                seen.add(img_id)
            merged.append(rec)

    for w in range(worker_count):
        wfile = run_dir / f"worker_{w}" / "annotations.jsonl"
        for rec in read_jsonl(wfile):
            img_id = rec.get("image_id") or rec.get("parsed_annotation", {}).get("image_id")
            if img_id:
                if img_id in seen:
                    continue
                seen.add(img_id)
            merged.append(rec)
    merged.sort(key=lambda r: r.get("image_path") or "")
    write_jsonl(run_dir / "annotations.jsonl", merged)
    return merged


def merge_failed(run_dir: Path, worker_count: int) -> List[dict]:
    """Concatenate per-worker failed.jsonl into run_dir/failed.jsonl (worker order)."""
    merged: List[dict] = []
    for w in range(worker_count):
        wfile = run_dir / f"worker_{w}" / "failed.jsonl"
        merged.extend(read_jsonl(wfile))
    write_jsonl(run_dir / "failed.jsonl", merged)
    return merged


def _read_stats(run_dir: Path, worker_count: int) -> List[dict]:
    stats = []
    for w in range(worker_count):
        sfile = run_dir / f"worker_{w}" / "statistics.json"
        if sfile.exists():
            try:
                with open(sfile, "r", encoding="utf-8") as f:
                    stats.append(json.load(f))
            except (json.JSONDecodeError, OSError):
                continue
    return stats


def aggregate_statistics(run_dir: Path, worker_count: int, provider: str = "gemini", model: str = "") -> dict:
    """Aggregate per-worker statistics.json into a combined summary for a single batch."""
    stats = _read_stats(run_dir, worker_count)
    total_images = sum(s.get("total_images", 0) for s in stats)
    successful = sum(s.get("successful", 0) for s in stats)
    failed = sum(s.get("failed", 0) for s in stats)
    skipped = sum(s.get("skipped", 0) for s in stats)
    total_runtime = sum(s.get("total_runtime_sec", 0) for s in stats)

    def weighted_avg(key: str):
        num = 0.0
        den = 0.0
        for s in stats:
            count = s.get("successful", 0)
            num += s.get(key, 0.0) * count
            den += count
        return round(num / den, 2) if den else 0.0

    avg_lat = weighted_avg("avg_latency_sec")
    median_lat = weighted_avg("median_latency_sec")
    p95_lat = weighted_avg("p95_latency_sec")
    throughput = (successful / (total_runtime / 60.0)) if total_runtime > 0 else 0.0

    summary = {
        "provider": provider,
        "model": model,
        "parallel_workers": worker_count,
        "total_images": total_images,
        "successful": successful,
        "skipped": skipped,
        "failed": failed,
        "avg_latency_sec": avg_lat,
        "median_latency_sec": median_lat,
        "p95_latency_sec": p95_lat,
        "images_per_min": round(throughput, 2),
        "total_runtime_sec": round(total_runtime, 2),
    }

    with open(run_dir / "statistics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(run_dir / "statistics.md", "w", encoding="utf-8") as f:
        f.write("# Annotation Run Statistics Summary\n\n")
        f.write(f"- **Provider / Model:** `{summary['provider']}` / `{summary['model']}`\n")
        f.write(f"- **Parallel Workers:** `{summary['parallel_workers']}`\n")
        f.write(f"- **Total Images:** `{summary['total_images']}`\n")
        f.write(f"- **Successful Annotations:** `{summary['successful']}`\n")
        f.write(f"- **Failed Annotations:** `{summary['failed']}`\n")
        f.write(f"- **Average Latency:** `{summary['avg_latency_sec']:.2f}s`\n")
        f.write(f"- **Median Latency:** `{summary['median_latency_sec']:.2f}s`\n")
        f.write(f"- **P95 Latency:** `{summary['p95_latency_sec']:.2f}s`\n")
        f.write(f"- **Throughput:** `{summary['images_per_min']:.1f} images/min`\n")

    return summary


def write_run_metadata(
    run_dir: Path,
    run_timestamp: str,
    worker_count: int,
    keys: Sequence[str],
    chunks: Sequence[Tuple[int, int]],
    model: str,
):
    metadata = {
        "provider": "gemini",
        "model": model,
        "parallel_workers": worker_count,
        "timestamp": run_timestamp,
        "worker_keys_masked": [mask_key(k) for k in keys[:worker_count]],
        "chunk_map": [
            {"worker": n, "start_index": s, "end_index": e}
            for n, (s, e) in enumerate(chunks)
        ],
    }
    with open(run_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def seed_workers_for_resume(
    run_dir: Path,
    worker_count: int,
    seed_file: Path,
    resume: bool,
    force_regenerate: bool = False,
):
    """Copy the canonical store annotations into each worker dir so CheckpointManager skips done images.

    Skip when resume is disabled or force-regenerate is requested.
    """
    if not resume or force_regenerate:
        return
    if seed_file is None or not seed_file.exists() or seed_file.stat().st_size == 0:
        return
    import shutil
    for w in range(worker_count):
        wdir = run_dir / f"worker_{w}"
        wdir.mkdir(parents=True, exist_ok=True)
        target = wdir / "annotations.jsonl"
        if not target.exists():
            shutil.copyfile(seed_file, target)