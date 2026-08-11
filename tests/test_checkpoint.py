import json
import tempfile
from pathlib import Path
from vlm_annotation.src.annotation.checkpoint import CheckpointManager


def test_checkpoint_persistence_and_resume():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = Path(tmpdir) / "annotations.jsonl"
        fail_file = Path(tmpdir) / "failed.jsonl"

        mgr1 = CheckpointManager(str(out_file), str(fail_file))
        assert not mgr1.is_completed("img001")

        rec1 = {"image_id": "img001", "disease": "Disease_A", "parsed_annotation": {"image_id": "img001"}}
        mgr1.save_annotation(rec1)

        assert mgr1.is_completed("img001")
        assert out_file.exists()

        # Simulate interruption & resume
        mgr2 = CheckpointManager(str(out_file), str(fail_file))
        assert mgr2.is_completed("img001")
        assert not mgr2.is_completed("img002")
