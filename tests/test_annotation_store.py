import json
from pathlib import Path

from vlm_annotation.src.annotation.store import (
    append_batch,
    append_jsonl,
    compute_coverage,
    load_ids,
    normalize_dataset_name,
    promote_merge,
    prune_worker_dirs,
    read_jsonl,
    recompute_statistics,
    record_image_id,
    store_dir,
    write_jsonl,
)


def rec(image_id, path="Aphid/leaf.jpg", **extra):
    base = {"image_id": image_id, "image_path": path, "parsed_annotation": {"image_id": image_id}}
    base.update(extra)
    return base


def test_normalize_dataset_name():
    assert normalize_dataset_name("Cotton_dataset") == "cotton_dataset"
    assert normalize_dataset_name("My Dataset") == "my_dataset"


def test_store_dir_under_artifacts():
    assert store_dir("Cotton_dataset") == Path("artifacts") / "cotton_dataset"


def test_read_write_jsonl_roundtrip(tmp_path):
    p = tmp_path / "annotations.jsonl"
    write_jsonl(p, [rec("a"), rec("b")])
    assert [r["image_id"] for r in read_jsonl(p)] == ["a", "b"]


def test_append_jsonl(tmp_path):
    p = tmp_path / "annotations.jsonl"
    write_jsonl(p, [rec("a")])
    append_jsonl(p, [rec("b")])
    assert [r["image_id"] for r in read_jsonl(p)] == ["a", "b"]


def test_record_image_id_nested():
    assert record_image_id(rec("x")) == "x"
    assert record_image_id({"parsed_annotation": {"image_id": "y"}}) == "y"
    assert record_image_id({}) is None


def test_load_ids(tmp_path):
    p = tmp_path / "annotations.jsonl"
    write_jsonl(p, [rec("a"), rec("b"), rec("a")])
    assert load_ids(p) == {"a", "b"}


def test_promote_merge_first_wins(tmp_path):
    store = tmp_path / "store"
    run = tmp_path / "run"
    (run / "worker_0").mkdir(parents=True)
    write_jsonl(run / "worker_0" / "annotations.jsonl", [rec("a", path="A/1.jpg", prompt_version="1.0")])
    (run / "worker_1").mkdir()
    write_jsonl(run / "worker_1" / "annotations.jsonl", [rec("a", path="A/1.jpg", prompt_version="9.9"), rec("b")])
    write_jsonl(run / "worker_1" / "failed.jsonl", [{"image_id": "z"}])

    out = promote_merge(store, run, worker_count=2, provider="gemini", model="m", prompt_version="1.0")

    assert out["total_in_store"] == 2
    records = read_jsonl(store / "annotations.jsonl")
    assert {r["image_id"] for r in records} == {"a", "b"}
    a = next(r for r in records if r["image_id"] == "a")
    assert a["prompt_version"] == "1.0"
    assert [r["image_id"] for r in read_jsonl(store / "failed.jsonl")] == ["z"]


def test_promote_merge_force_regenerate(tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    write_jsonl(store / "annotations.jsonl", [rec("a", prompt_version="1.0")])
    run = tmp_path / "run"
    (run / "worker_0").mkdir(parents=True)
    write_jsonl(run / "worker_0" / "annotations.jsonl", [rec("a", prompt_version="2.0")])

    promote_merge(store, run, worker_count=1, force_regenerate=True, prompt_version="2.0")

    records = read_jsonl(store / "annotations.jsonl")
    assert len(records) == 1
    assert records[0]["prompt_version"] == "2.0"


def test_prune_worker_dirs(tmp_path):
    run = tmp_path / "run"
    for w in range(3):
        (run / f"worker_{w}").mkdir(parents=True)
    prune_worker_dirs(run, 3)
    assert not (run / "worker_0").exists()
    assert not (run / "worker_2").exists()


def test_append_batch_and_recompute_statistics(tmp_path):
    store = tmp_path / "store"
    append_batch(store, {"annotated": 10, "failed": 2, "runtime_sec": 60.0})
    append_batch(store, {"annotated": 5, "failed": 0, "runtime_sec": 30.0})

    stats = recompute_statistics(store, provider="gemini", model="m")
    assert stats["batches"] == 2
    assert stats["total_annotated"] == 15
    assert stats["total_failed"] == 2
    assert stats["total_runtime_sec"] == 90.0


def test_coverage_json_written(tmp_path, monkeypatch):
    from vlm_annotation.src import dataset as dataset_mod

    class FakeItem:
        def __init__(self, image_id):
            self.image_id = image_id
            self.image_path = f"{image_id}.jpg"
            self.relative_path = f"{image_id}.jpg"
            self.disease_name = "Aphid"

    monkeypatch.setattr(dataset_mod, "discover_dataset", lambda p: ([FakeItem("1"), FakeItem("2"), FakeItem("3")], []))
    monkeypatch.setattr("vlm_annotation.src.annotation.store.discover_dataset", lambda p: ([FakeItem("1"), FakeItem("2"), FakeItem("3")], []))

    store = tmp_path / "store"
    store.mkdir()
    write_jsonl(store / "annotations.jsonl", [rec("1"), rec("2")])
    write_jsonl(store / "failed.jsonl", [{"image_id": "3"}])

    cov = compute_coverage(store, "Cotton_dataset", provider="gemini", model="m")
    assert cov["dataset_total"] == 3
    assert cov["annotated"] == 2
    assert cov["failed"] == 1
    assert cov["missing"] == 0
    assert cov["complete"] is False
    assert json.loads((store / "coverage.json").read_text(encoding="utf-8")) == cov


def test_coverage_complete_requires_zero_failed(tmp_path, monkeypatch):
    from vlm_annotation.src import dataset as dataset_mod

    class FakeItem:
        def __init__(self, image_id):
            self.image_id = image_id
            self.image_path = f"{image_id}.jpg"
            self.relative_path = f"{image_id}.jpg"
            self.disease_name = "Aphid"

    monkeypatch.setattr(dataset_mod, "discover_dataset", lambda p: ([FakeItem("1")], []))
    monkeypatch.setattr("vlm_annotation.src.annotation.store.discover_dataset", lambda p: ([FakeItem("1")], []))

    store = tmp_path / "store"
    store.mkdir()
    write_jsonl(store / "annotations.jsonl", [rec("1")])
    write_jsonl(store / "failed.jsonl", [{"image_id": "x"}])

    cov = compute_coverage(store, "Cotton_dataset")
    assert cov["missing"] == 0
    assert cov["complete"] is False