import sys
import json
import asyncio
import unittest
import torch
from unittest.mock import patch, MagicMock
from pathlib import Path

# Ensure root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vlm_annotation.src.models.huggingface import HuggingFaceVisionModel
from vlm_annotation.src.models.factory import create_vision_model
from vlm_annotation.src.annotation.hf_health import check_huggingface_environment_and_model


class TestHuggingFaceVisionModel(unittest.TestCase):

    def setUp(self):
        self.config = {
            "provider": "huggingface",
            "name": "hf-qwen2.5-vl-7b",
            "model": "Qwen/Qwen2.5-VL-7B-Instruct",
            "quantization": {"enabled": False},
        }

    @patch("vlm_annotation.src.models.huggingface.Qwen2_5_VLForConditionalGeneration")
    @patch("vlm_annotation.src.models.huggingface.AutoModelForCausalLM")
    @patch("vlm_annotation.src.models.huggingface.AutoProcessor")
    def test_factory_instantiation(self, mock_processor, mock_causal_lm, mock_qwen_vl):
        mock_processor.from_pretrained.return_value = MagicMock()
        mock_causal_lm.from_pretrained.return_value = MagicMock()
        mock_qwen_vl.from_pretrained.return_value = MagicMock()

        m = create_vision_model(self.config)
        self.assertIsInstance(m, HuggingFaceVisionModel)
        self.assertEqual(m.model_id, "Qwen/Qwen2.5-VL-7B-Instruct")

    @patch("vlm_annotation.src.models.huggingface.Image.open")
    @patch("vlm_annotation.src.models.huggingface.Qwen2_5_VLForConditionalGeneration")
    @patch("vlm_annotation.src.models.huggingface.AutoModelForCausalLM")
    @patch("vlm_annotation.src.models.huggingface.AutoProcessor")
    def test_successful_generate_annotation(self, mock_processor_cls, mock_causal_lm, mock_qwen_vl, mock_image_open):
        mock_img = MagicMock()
        mock_img.width = 800
        mock_img.height = 600
        mock_img.convert.return_value = mock_img
        mock_image_open.return_value = mock_img

        mock_proc_inst = MagicMock()
        mock_proc_inst.apply_chat_template.return_value = "Formatted prompt"
        mock_proc_inst.return_value = {"input_ids": torch.zeros((1, 10), dtype=torch.long)}
        json_output = json.dumps({
            "disease": "Bacterial_blight",
            "visible_observations": ["Angular leaf spots"],
            "diagnostic_evidence": ["Water-soaked lesions"],
            "reasoning": "Classic bacterial symptom"
        })
        mock_proc_inst.batch_decode.return_value = [json_output]
        mock_processor_cls.from_pretrained.return_value = mock_proc_inst

        mock_model_inst = MagicMock()
        mock_model_inst.generate.return_value = torch.zeros((1, 25), dtype=torch.long)
        mock_model_inst.device = "cpu"
        mock_qwen_vl.from_pretrained.return_value = mock_model_inst
        mock_causal_lm.from_pretrained.return_value = mock_model_inst

        model = HuggingFaceVisionModel("hf-qwen2.5-vl-7b", "Qwen/Qwen2.5-VL-7B-Instruct", self.config)

        response = asyncio.run(
            model.generate_annotation(
                image_path="fake/path.jpg",
                disease_name="Bacterial_blight",
                prompt="Analyze image",
            )
        )

        self.assertEqual(response.status, "success")
        self.assertIsNotNone(response.parsed_json)
        self.assertEqual(response.parsed_json["disease"], "Bacterial_blight")

    @patch("vlm_annotation.src.models.huggingface.Image.open")
    @patch("vlm_annotation.src.models.huggingface.AutoModel")
    @patch("vlm_annotation.src.models.huggingface.AutoTokenizer")
    @patch("vlm_annotation.src.models.huggingface.AutoProcessor")
    def test_internvl_chat_generation(self, mock_processor_cls, mock_tokenizer_cls, mock_auto_model_cls, mock_image_open):
        mock_processor_cls.from_pretrained.side_effect = Exception("No AutoProcessor for InternVL")
        mock_tokenizer_cls.from_pretrained.return_value = MagicMock()

        mock_img = MagicMock()
        mock_img.width = 800
        mock_img.height = 600
        mock_img.convert.return_value = mock_img
        mock_image_open.return_value = mock_img

        json_output = json.dumps({
            "disease": "Curl_virus",
            "visible_observations": ["Leaf curling"],
            "diagnostic_evidence": ["Vein thickening"],
            "reasoning": "Classic viral symptom"
        })

        mock_model_inst = MagicMock()
        mock_model_inst.chat.return_value = (json_output, None)
        mock_auto_model_cls.from_pretrained.return_value = mock_model_inst

        cfg = {
            "provider": "huggingface",
            "name": "hf-internvl3.5-8b",
            "model": "OpenGVLab/InternVL2_5-8B",
            "quantization": {"enabled": False},
        }

        model = HuggingFaceVisionModel("hf-internvl3.5-8b", "OpenGVLab/InternVL2_5-8B", cfg)

        response = asyncio.run(
            model.generate_annotation(
                image_path="fake/path.jpg",
                disease_name="Curl_virus",
                prompt="Analyze image",
            )
        )

        self.assertEqual(response.status, "success")
        self.assertIsNotNone(response.parsed_json)
        self.assertEqual(response.parsed_json["disease"], "Curl_virus")


class TestHuggingFaceHealthCheck(unittest.TestCase):

    @patch("vlm_annotation.src.annotation.hf_health.AutoProcessor")
    def test_health_check_success(self, mock_auto_processor):
        mock_auto_processor.from_pretrained.return_value = MagicMock()
        ok, msg = check_huggingface_environment_and_model("Qwen/Qwen2.5-VL-7B-Instruct")
        self.assertTrue(ok)
        self.assertIn("SUCCESS", msg)

    @patch("vlm_annotation.src.annotation.hf_health.AutoProcessor")
    def test_health_check_model_load_failure(self, mock_auto_processor):
        mock_auto_processor.from_pretrained.side_effect = Exception("Repository not found")
        ok, msg = check_huggingface_environment_and_model("NonExistent/Model")
        self.assertFalse(ok)
        self.assertIn("Failed to load AutoProcessor", msg)


if __name__ == "__main__":
    unittest.main()
