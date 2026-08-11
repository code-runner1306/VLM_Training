import json
import logging
from pathlib import Path
from typing import Any, Dict, Set

logger = logging.getLogger("vlm_annotation.checkpoint")


class CheckpointManager:
    """Manages set of completed image IDs and appends valid annotations immediately to JSONL."""

    def __init__(self, output_file: str = "outputs/annotations/annotations.jsonl", failed_file: str = "outputs/annotations/failed.jsonl"):
        self.output_path = Path(output_file)
        self.failed_path = Path(failed_file)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.completed_ids: Set[str] = set()
        self._load_completed_ids()

    def _load_completed_ids(self):
        if self.output_path.exists():
            with open(self.output_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, start=1):
                    if line.strip():
                        try:
                            record = json.loads(line)
                            img_id = record.get("image_id") or record.get("parsed_annotation", {}).get("image_id")
                            if img_id:
                                self.completed_ids.add(img_id)
                        except Exception as e:
                            logger.warning(f"Malformed JSONL line {line_num} in {self.output_path}: {e}")

    def is_completed(self, image_id: str) -> bool:
        return image_id in self.completed_ids

    def save_annotation(self, record: Dict[str, Any]):
        """Append annotation immediately to disk and add image_id to completed set."""
        with open(self.output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()
        img_id = record.get("image_id") or record.get("parsed_annotation", {}).get("image_id")
        if img_id:
            self.completed_ids.add(img_id)

    def save_failed(self, record: Dict[str, Any]):
        """Append failed request metadata to failed.jsonl."""
        with open(self.failed_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()
