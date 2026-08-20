import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from vlm_annotation.src.annotation.keys import (
    load_gemini_keys,
    mask_key,
    resolve_max_workers,
    resolve_worker_count,
)
from vlm_annotation.src.annotation.parallel import (
    ParallelAnnotationError,
    aggregate_statistics,
    build_worker_env,
    merge_annotations,
    merge_failed,
    seed_workers_for_resume,
    slice_chunks,
    write_run_metadata,
)


# ---------- 5.1 load_gemini_keys ----------

def test_keys_comma_split(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEYS", "key1,key2,key3")
    monkeypatch.setenv("GEMINI_API_KEY", "fallback")
    assert load_gemini_keys() == ["key1", "key2", "key3"]


def test_keys_whitespace_strip(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEYS", " key1 , key2 ,,key3 ")
    assert load_gemini_keys() == ["key1", "key2", "key3"]


def test_keys_fallback_to_single(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "single")
    assert load_gemini_keys() == ["single"]


def test_keys_empty_absent(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEYS", "")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert load_gemini_keys() == []


def test_keys_empty_multi_falls_back(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEYS", "  ,,")
    monkeypatch.setenv("GEMINI_API_KEY", "single")
    assert load_gemini_keys() == ["single"]


# ---------- 5.2 slice_chunks ----------

def test_chunks_offsets():
    chunks = slice_chunks(total_images=20000, start_index=5001, chunk_size=500, worker_count=3)
    assert chunks == [(5001, 5501), (5501, 6001), (6001, 6501)]


def test_chunks_final_partial():
    chunks = slice_chunks(total_images=6100, start_index=5001, chunk_size=500, worker_count=3)
    assert chunks == [(5001, 5501), (5501, 6001), (6001, 6100)]


def test_chunks_no_overlap_and_contiguous():
    total = 12000
    start = 0
    chunk = 500
    workers = 4
    chunks = slice_chunks(total, start, chunk, workers)
    assert len(chunks) == workers
    for idx, (s, e) in enumerate(chunks):
        assert s == start + idx * chunk
        assert e == s + chunk
    assert chunks[0][0] == 0
    assert chunks[-1][1] == chunk * workers


def test_chunks_empty_input():
    assert slice_chunks(0, 0, 500, 3) == []


# ---------- 5.3 resolve_worker_count / resolve_max_workers / mask_key ----------

def test_worker_cap_min(monkeypatch):
    monkeypatch.setenv("MAX_GEMINI_WORKERS", "4")
    with mock.patch("os.cpu_count", return_value=8):
        assert resolve_worker_count(["a", "b", "c", "d", "e"], max_workers=None) == 4


def test_worker_cap_cpu_limiting(monkeypatch):
    monkeypatch.setenv("MAX_GEMINI_WORKERS", "4")
    with mock.patch("os.cpu_count", return_value=2):
        assert resolve_worker_count(["a", "b", "c", "d", "e"], max_workers=None) == 2


def test_worker_cap_override_arg():
    with mock.patch("os.cpu_count", return_value=16):
        assert resolve_worker_count(["a", "b", "c"], max_workers=2) == 2


def test_worker_cap_no_keys():
    assert resolve_worker_count([], max_workers=4) == 0


def test_max_workers_env_invalid(monkeypatch):
    monkeypatch.setenv("MAX_GEMINI_WORKERS", "banana")
    assert resolve_max_workers() == 4


def test_mask_key():
    assert mask_key("0123456789abcdef") == "0123...cdef"
    assert mask_key("") == ""


# ---------- 5.4 merge / aggregation ----------

def _write_worker(run_dir, worker_id, records):
    wdir = run_dir / f"worker_{worker_id}"
    wdir.mkdir(parents=True, exist_ok=True)
    with open(wdir / "annotations.jsonl", "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def test_merge_ordering_and_dedupe(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    # Worker 0 completes out of file-name order intentionally
    _write_worker(run_dir, 0, [
        {"image_id": "b", "image_path": "Bacteria/leaf.jpg", "parsed_annotation": {"image_id": "b"}},
        {"image_id": "a", "image_path": "Aphid/leaf.jpg", "parsed_annotation": {"image_id": "a"}},
    ])
    # Worker 1 has a duplicate of image a plus a new one
    _write_worker(run_dir, 1, [
        {"image_id": "a", "image_path": "Aphid/leaf.jpg", "parsed_annotation": {"image_id": "a"}},
        {"image_id": "c", "image_path": "Zeta/leaf.jpg", "parsed_annotation": {"image_id": "c"}},
    ])

    merged = merge_annotations(run_dir, 2)

    assert [r["image_id"] for r in merged] == ["a", "b", "c"]
    assert [r["image_path"] for r in merged] == ["Aphid/leaf.jpg", "Bacteria/leaf.jpg", "Zeta/leaf.jpg"]
    out = run_dir / "annotations.jsonl"
    assert out.exists()
    lines = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 3


def test_merge_failed_concat(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    for wid in (0, 1):
        wdir = run_dir / f"worker_{wid}"
        wdir.mkdir()
        with open(wdir / "failed.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps({"image_id": f"f{wid}", "error": "boom"}) + "\n")
    merged = merge_failed(run_dir, 2)
    assert [r["image_id"] for r in merged] == ["f0", "f1"]


def test_aggregate_statistics(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    for wid, ok, fail, lat in [(0, 10, 2, 1.0), (1, 5, 1, 3.0)]:
        wdir = run_dir / f"worker_{wid}"
        wdir.mkdir()
        with open(wdir / "statistics.json", "w", encoding="utf-8") as f:
            json.dump({
                "total_images": ok + fail,
                "successful": ok,
                "failed": fail,
                "skipped": 0,
                "avg_latency_sec": lat,
                "median_latency_sec": lat,
                "p95_latency_sec": lat,
                "total_runtime_sec": 60.0,
            }, f)

    summary = aggregate_statistics(run_dir, 2, provider="gemini", model="m")
    assert summary["successful"] == 15
    assert summary["failed"] == 3
    assert summary["total_images"] == 18
    assert summary["parallel_workers"] == 2
    assert (run_dir / "statistics.json").exists()
    assert (run_dir / "statistics.md").exists()


def test_write_run_metadata_masks_keys(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_run_metadata(
        run_dir,
        run_timestamp="20260818",
        worker_count=2,
        keys=["0123456789abcdef", "fedcba9876543210"],
        chunks=[(5001, 5501), (5501, 6001)],
        model="gemini-flash",
    )
    md = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert md["worker_keys_masked"] == ["0123...cdef", "fedc...3210"]
    assert "0123456789abcdef" not in json.dumps(md)
    assert md["chunk_map"][0] == {"worker": 0, "start_index": 5001, "end_index": 5501}


def test_merge_preserves_existing_run_records(tmp_path):
    """Reusing a run dir accumulates: existing merged records survive re-merge."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with open(run_dir / "annotations.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"image_id": "a", "image_path": "Aphid/leaf.jpg", "parsed_annotation": {"image_id": "a"}}) + "\n")
        f.write(json.dumps({"image_id": "b", "image_path": "Bacteria/leaf.jpg", "parsed_annotation": {"image_id": "b"}}) + "\n")
    _write_worker(run_dir, 0, [
        {"image_id": "b", "image_path": "Bacteria/leaf.jpg", "parsed_annotation": {"image_id": "b"}},
        {"image_id": "c", "image_path": "Zeta/leaf.jpg", "parsed_annotation": {"image_id": "c"}},
    ])

    merged = merge_annotations(run_dir, 1)

    assert [r["image_id"] for r in merged] == ["a", "b", "c"]
    out = run_dir / "annotations.jsonl"
    lines = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 3


def test_seed_workers_prefers_own_run_history(tmp_path):
    """seed_workers_for_resume copies the canonical store seed into each worker dir."""
    run_dir = tmp_path / "run_001"
    run_dir.mkdir()
    seed_file = tmp_path / "store" / "annotations.jsonl"
    seed_file.parent.mkdir()
    with open(seed_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"image_id": "a", "image_path": "Aphid/leaf.jpg", "parsed_annotation": {"image_id": "a"}}) + "\n")

    seed_workers_for_resume(run_dir, worker_count=2, seed_file=seed_file, resume=True)

    for w in range(2):
        target = run_dir / f"worker_{w}" / "annotations.jsonl"
        assert target.exists()
        assert [json.loads(l)["image_id"] for l in target.read_text(encoding="utf-8").splitlines()] == ["a"]


def test_seed_workers_skips_when_resume_false(tmp_path):
    run_dir = tmp_path / "run_001"
    run_dir.mkdir()
    seed_file = tmp_path / "store" / "annotations.jsonl"
    seed_workers_for_resume(run_dir, worker_count=2, seed_file=seed_file, resume=False)
    assert not (run_dir / "worker_0").exists()


def test_seed_workers_skips_when_force_regenerate(tmp_path):
    run_dir = tmp_path / "run_001"
    run_dir.mkdir()
    seed_file = tmp_path / "store" / "annotations.jsonl"
    seed_file.parent.mkdir()
    seed_file.write_text("", encoding="utf-8")
    seed_workers_for_resume(run_dir, worker_count=2, seed_file=seed_file, resume=True, force_regenerate=True)
    assert not (run_dir / "worker_0").exists()


# ---------- 5.5 failure propagation / build_worker_env ----------

def test_build_worker_env_forces_single_mode(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEYS", "a,b,c")
    monkeypatch.setenv("MAX_GEMINI_WORKERS", "9")
    env = build_worker_env("workerkey")
    assert env["GEMINI_API_KEY"] == "workerkey"
    assert env["GEMINI_API_KEYS"] == ""
    assert env["MAX_GEMINI_WORKERS"] == "1"


def test_parallel_error_is_runtime_error():
    assert issubclass(ParallelAnnotationError, RuntimeError)


# ---------- 5.6 smoke integration: parallel path end-to-end with stub worker ----------

def test_parallel_run_workers_and_merge(tmp_path):
    """Fake worker subprocess (writes records, exits 0) exercises spawn+merge."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    worker_script = tmp_path / "fake_worker.py"
    worker_script.write_text(
        "import json, sys\n"
        "out = sys.argv[sys.argv.index('--output-dir') + 1]\n"
        "import os\n"
        "os.makedirs(out, exist_ok=True)\n"
        "with open(os.path.join(out, 'annotations.jsonl'), 'w', encoding='utf-8') as f:\n"
        "    for i in range(2):\n"
        "        f.write(json.dumps({'image_id': f'img{i}', 'image_path': f'd/dir{i}.jpg', 'parsed_annotation': {'image_id': f'img{i}'}}) + '\\n')\n",
        encoding="utf-8",
    )
    from vlm_annotation.src.annotation.parallel import run_workers

    commands = [[sys.executable, str(worker_script), "--output-dir", str(run_dir / "worker_0")]]
    envs = [build_worker_env("k")]
    codes = run_workers(commands, envs)
    assert codes == [0]

    merged = merge_annotations(run_dir, 1)
    assert len(merged) == 2
    assert (run_dir / "annotations.jsonl").exists()


def test_run_workers_failure_code(tmp_path):
    worker_script = tmp_path / "fail_worker.py"
    worker_script.write_text("import sys\nsys.exit(3)\n", encoding="utf-8")
    from vlm_annotation.src.annotation.parallel import run_workers

    codes = run_workers(
        [[sys.executable, str(worker_script)]],
        [build_worker_env("k")],
    )
    assert codes == [3]