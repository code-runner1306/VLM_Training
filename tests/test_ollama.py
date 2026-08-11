import os
import sys
import json
import asyncio
import unittest
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path

# Ensure root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vlm_annotation.src.models.ollama import OllamaVisionModel
from vlm_annotation.src.models.factory import create_vision_model
from vlm_annotation.src.annotation.ollama_health import check_ollama_server_and_model


class TestOllamaVisionModel(unittest.TestCase):

    def setUp(self):
        self.config = {
            "provider": "ollama",
            "name": "ollama-qwen3-vl-8b",
            "model": "qwen3-vl:8b",
            "host": "http://127.0.0.1:11434",
            "timeout_seconds": 30,
            "options": {"temperature": 0.1, "top_p": 0.9},
            "generation": {"max_tokens": 1000},
        }
        self.model = OllamaVisionModel("ollama-qwen3-vl-8b", "qwen3-vl:8b", self.config)

    def test_factory_instantiation(self):
        m = create_vision_model(self.config)
        self.assertIsInstance(m, OllamaVisionModel)
        self.assertEqual(m.model_id, "qwen3-vl:8b")
        self.assertEqual(m.host, "http://127.0.0.1:11434")

    @patch("vlm_annotation.src.models.ollama.httpx.AsyncClient")
    @patch("vlm_annotation.src.models.ollama.OllamaVisionModel._encode_image")
    def test_successful_generate_annotation(self, mock_encode, mock_client_cls):
        mock_encode.return_value = "fake_base64_string"

        mock_post_res = MagicMock()
        mock_post_res.raise_for_status.return_value = None
        mock_post_res.json.return_value = {
            "response": json.dumps({
                "disease": "Alternaria_leaf_spot",
                "visible_observations": ["Target concentric rings"],
                "diagnostic_evidence": ["Target spots"],
                "reasoning": "Classic symptom pattern"
            })
        }

        mock_client_inst = AsyncMock()
        mock_client_inst.post.return_value = mock_post_res
        mock_client_inst.__aenter__.return_value = mock_client_inst
        mock_client_inst.__aexit__.return_value = None
        mock_client_cls.return_value = mock_client_inst

        response = asyncio.run(
            self.model.generate_annotation(
                image_path="fake/path.jpg",
                disease_name="Alternaria_leaf_spot",
                prompt="Analyze image",
            )
        )

        self.assertEqual(response.status, "success")
        self.assertIsNotNone(response.parsed_json)
        self.assertEqual(response.parsed_json["disease"], "Alternaria_leaf_spot")

    @patch("vlm_annotation.src.models.ollama.httpx.AsyncClient")
    @patch("vlm_annotation.src.models.ollama.OllamaVisionModel._encode_image")
    def test_json_parse_error(self, mock_encode, mock_client_cls):
        mock_encode.return_value = "fake_base64_string"

        mock_post_res = MagicMock()
        mock_post_res.raise_for_status.return_value = None
        mock_post_res.json.return_value = {"response": "invalid non json text"}

        mock_client_inst = AsyncMock()
        mock_client_inst.post.return_value = mock_post_res
        mock_client_inst.__aenter__.return_value = mock_client_inst
        mock_client_inst.__aexit__.return_value = None
        mock_client_cls.return_value = mock_client_inst

        response = asyncio.run(
            self.model.generate_annotation(
                image_path="fake/path.jpg",
                disease_name="Alternaria_leaf_spot",
                prompt="Analyze image",
            )
        )

        self.assertEqual(response.status, "json_parse_error")
        self.assertIsNone(response.parsed_json)


class TestOllamaHealthCheck(unittest.TestCase):

    @patch("vlm_annotation.src.annotation.ollama_health.httpx.get")
    def test_server_unreachable(self, mock_get):
        mock_get.side_effect = Exception("Connection refused")
        ok, msg = check_ollama_server_and_model("http://127.0.0.1:11434", "qwen3-vl:8b")
        self.assertFalse(ok)
        self.assertIn("not reachable", msg)

    @patch("vlm_annotation.src.annotation.ollama_health.httpx.get")
    def test_model_missing(self, mock_get):
        mock_res = MagicMock()
        mock_res.raise_for_status.return_value = None
        mock_res.json.return_value = {"models": [{"name": "llama3:8b"}]}
        mock_get.return_value = mock_res

        ok, msg = check_ollama_server_and_model("http://127.0.0.1:11434", "qwen3-vl:8b")
        self.assertFalse(ok)
        self.assertIn("ollama pull qwen3-vl:8b", msg)

    @patch("vlm_annotation.src.annotation.ollama_health.httpx.post")
    @patch("vlm_annotation.src.annotation.ollama_health.httpx.get")
    def test_health_check_success(self, mock_get, mock_post):
        mock_get_res = MagicMock()
        mock_get_res.raise_for_status.return_value = None
        mock_get_res.json.return_value = {"models": [{"name": "qwen3-vl:8b"}]}
        mock_get.return_value = mock_get_res

        mock_post_res = MagicMock()
        mock_post_res.raise_for_status.return_value = None
        mock_post_res.json.return_value = {"response": json.dumps({"disease": "Healthy", "reasoning": "No symptoms"})}
        mock_post.return_value = mock_post_res

        ok, msg = check_ollama_server_and_model("http://127.0.0.1:11434", "qwen3-vl:8b")
        self.assertTrue(ok)
        self.assertIn("SUCCESS", msg)


if __name__ == "__main__":
    unittest.main()
