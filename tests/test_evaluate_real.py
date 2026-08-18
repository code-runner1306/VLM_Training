import json
import sys
from unittest.mock import MagicMock

import torch

sys.path.insert(0, ".")
from training.scripts.evaluate import generate_predictions

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

import pytest

pytestmark = pytest.mark.skipif(not HAS_PIL, reason="Pillow not installed")


class _DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))

    def generate(self, **kwargs):
        return torch.ones((1, 10), dtype=torch.long)


def _write_manifest(tmp_path):
    img_path = tmp_path / "img.png"
    Image.new("RGB", (10, 10), color="red").save(img_path)
    manifest = tmp_path / "test_manifest.jsonl"
    item = {
        "image_id": "img1",
        "image_path": str(img_path),
        "disease": "Bacterial_Blight",
        "parsed_annotation": {
            "visible_observations": ["leaf spot"],
            "diagnostic_evidence": ["brown spots"],
            "reasoning": "evidence-based",
        },
    }
    manifest.write_text(json.dumps(item) + "\n", encoding="utf-8")
    return manifest


def _fake_processor():
    processor = MagicMock()
    processor.apply_chat_template.return_value = "user prompt text"

    def fake_call(**kwargs):
        return {
            "input_ids": torch.ones((1, 5), dtype=torch.long),
            "attention_mask": torch.ones((1, 5), dtype=torch.long),
        }

    processor.side_effect = fake_call
    processor.__call__ = fake_call
    processor.batch_decode.return_value = ["The disease is Predicted_Disease_X"]
    return processor


def test_generate_predictions_uses_real_generation_not_gt_passthrough(tmp_path):
    manifest = _write_manifest(tmp_path)
    processor = _fake_processor()
    adapter = MagicMock()
    adapter.parse_generated_output.return_value = {
        "predicted_disease": "Predicted_Disease_X",
        "confidence": 0.8,
    }
    config = {
        "evaluation": {"max_new_tokens": 32, "do_sample": False, "num_beams": 1},
        "data": {"user_prompt": "What disease is this?"},
    }

    predictions = generate_predictions(
        adapter=adapter,
        config=config,
        model=_DummyModel(),
        processor=processor,
        test_manifest_path=str(manifest),
    )

    assert len(predictions) == 1
    pred = predictions[0]
    # Ground truth comes from the manifest, but the predicted disease MUST come
    # from model generation (not the ground truth) - the old oracle passthrough
    # produced predictions that always matched the ground truth.
    assert pred["ground_truth_disease"] == "Bacterial_Blight"
    assert pred["predicted_disease"] == "Predicted_Disease_X"
    assert pred["ground_truth_disease"] != pred["predicted_disease"]
    assert pred["raw_text"] == "The disease is Predicted_Disease_X"
    assert pred["parsed_output"]["predicted_disease"] == "Predicted_Disease_X"

    # Ensure the user prompt was turned into a generation prompt.
    call_args = processor.apply_chat_template.call_args
    assert call_args.kwargs["add_generation_prompt"] is True
    # And the adapter parsed the generated text (not a ground-truth reconstruction).
    adapter.parse_generated_output.assert_called_once_with("The disease is Predicted_Disease_X")


def test_generate_predictions_includes_batch_decode_output(tmp_path):
    manifest = _write_manifest(tmp_path)
    processor = _fake_processor()
    adapter = MagicMock()
    adapter.parse_generated_output.return_value = {"predicted_disease": "Unknown"}

    predictions = generate_predictions(
        adapter=adapter,
        config={},
        model=_DummyModel(),
        processor=processor,
        test_manifest_path=str(manifest),
    )
    assert predictions[0]["image_id"] == "img1"
    assert predictions[0]["ground_truth_disease"] == "Bacterial_Blight"