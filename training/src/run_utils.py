import datetime
import json
import os
from pathlib import Path
from typing import Optional


def create_run_dir(experiment: str) -> Path:
    """Create a self-contained run directory: outputs/run_<YYYYmmdd_HHMMSS>/."""
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("outputs") / f"run_{timestamp_str}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_run_metadata(run_dir: Path, metadata: dict) -> Path:
    meta_file = run_dir / "run_metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return meta_file


def read_run_metadata(run_dir: Path) -> dict:
    meta_file = run_dir / "run_metadata.json"
    if not meta_file.exists():
        return {}
    try:
        with open(meta_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def iter_run_dirs() -> list:
    """Return all outputs/run_* directories sorted newest-first."""
    outputs = Path("outputs")
    if not outputs.exists():
        return []
    return sorted(
        (d for d in outputs.iterdir() if d.is_dir() and d.name.startswith("run_")),
        key=lambda d: d.name,
        reverse=True,
    )


def resolve_latest_run(experiment: Optional[str] = None, model_key: Optional[str] = None) -> Optional[Path]:
    """Find the most recent run directory matching an experiment tag and/or model key.

    Matching is done via the `run_metadata.json` in each run dir.
    """
    for run_dir in iter_run_dirs():
        meta = read_run_metadata(run_dir)
        if experiment and meta.get("experiment") != experiment:
            continue
        if model_key and meta.get("model_key") != model_key:
            continue
        return run_dir
    return None


def annotations_provenance(annotations_file: Path) -> dict:
    """Compute SHA-256 + line count of the canonical annotations file."""
    import hashlib

    sha256 = None
    line_count = 0
    if annotations_file and annotations_file.exists():
        try:
            h = hashlib.sha256()
            with open(annotations_file, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
            sha256 = h.hexdigest()
            with open(annotations_file, "r", encoding="utf-8") as f:
                line_count = sum(1 for line in f if line.strip())
        except Exception:
            pass
    return {"annotations_source": str(annotations_file), "annotations_line_count": line_count, "annotations_sha256": sha256}


def config_copy_path(run_dir: Path, config_path: str) -> Path:
    """Copy a training YAML config into the run dir for self-containment."""
    import shutil

    target = run_dir / "config.yaml"
    if os.path.exists(config_path):
        shutil.copyfile(config_path, target)
    return target