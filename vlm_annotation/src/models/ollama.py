import base64
import json
import logging
import time
from typing import Any, Dict, Optional
import httpx
from vlm_annotation.src.models.base import ModelResponse, VisionModel, extract_json_from_text

logger = logging.getLogger("OllamaVisionModel")

JSON_SCHEMA_FORMAT = {
    "type": "object",
    "properties": {
        "disease": {"type": "string"},
        "visible_observations": {"type": "array", "items": {"type": "string"}},
        "affected_regions": {"type": "array", "items": {"type": "string"}},
        "color_characteristics": {"type": "array", "items": {"type": "string"}},
        "shape_characteristics": {"type": "array", "items": {"type": "string"}},
        "texture_characteristics": {"type": "array", "items": {"type": "string"}},
        "spatial_distribution": {"type": "string"},
        "severity": {"type": "string"},
        "diagnostic_evidence": {"type": "array", "items": {"type": "string"}},
        "reasoning": {"type": "string"},
        "uncertain_observations": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number"}
    },
    "required": ["disease", "visible_observations", "diagnostic_evidence", "reasoning"]
}


class OllamaVisionModel(VisionModel):
    """
    Provider implementation for local Ollama Vision-Language Models (e.g. Qwen3-VL 8B/4B/2B).
    """

    def __init__(self, provider_name: str, model_id: str, config: Dict[str, Any]):
        super().__init__(provider_name, model_id, config)
        self.host = config.get("host", "http://127.0.0.1:11434").rstrip("/")
        self.timeout = float(config.get("timeout_seconds", 120))
        self.think = config.get("think", False)
        
        # Generation options
        opts = config.get("options", {})
        gen = config.get("generation", {})
        self.options = {
            "temperature": float(opts.get("temperature", 0.1)),
            "top_p": float(opts.get("top_p", 0.9)),
            "num_predict": int(gen.get("max_tokens", opts.get("num_predict", 1000))),
        }

    def _encode_image(self, image_path: str) -> str:
        """Encode image file to base64 string."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    async def generate_annotation(
        self,
        image_path: str,
        disease_name: str,
        prompt: str,
        disease_profile: Optional[Dict[str, Any]] = None
    ) -> ModelResponse:
        self.total_requests += 1
        start_time = time.monotonic()

        try:
            b64_image = self._encode_image(image_path)
            
            payload = {
                "model": self.model_id,
                "prompt": prompt,
                "images": [b64_image],
                "stream": False,
                "format": JSON_SCHEMA_FORMAT,
                "options": self.options,
            }

            url = f"{self.host}/api/generate"
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                res_json = response.json()

            raw_text = res_json.get("response", "")
            latency_ms = (time.monotonic() - start_time) * 1000.0

            parsed_json = extract_json_from_text(raw_text)

            if parsed_json:
                self.successful_requests += 1
                self.total_latency_ms += latency_ms
                return ModelResponse(
                    provider=self.provider_name,
                    model_name=self.model_id,
                    raw_response=raw_text,
                    parsed_json=parsed_json,
                    latency_ms=latency_ms,
                    status="success"
                )
            else:
                self.json_parse_failures += 1
                self.failed_requests += 1
                return ModelResponse(
                    provider=self.provider_name,
                    model_name=self.model_id,
                    raw_response=raw_text,
                    parsed_json=None,
                    latency_ms=latency_ms,
                    status="json_parse_error",
                    error_message="Failed to parse valid JSON from Ollama response"
                )

        except httpx.ConnectError:
            latency_ms = (time.monotonic() - start_time) * 1000.0
            self.failed_requests += 1
            err_msg = f"Cannot connect to Ollama server at {self.host}. Please start Ollama."
            logger.error(err_msg)
            return ModelResponse(
                provider=self.provider_name,
                model_name=self.model_id,
                raw_response="",
                latency_ms=latency_ms,
                status="error",
                error_message=err_msg
            )
        except Exception as e:
            latency_ms = (time.monotonic() - start_time) * 1000.0
            self.failed_requests += 1
            err_msg = f"Ollama generation error: {str(e)}"
            logger.error(err_msg)
            return ModelResponse(
                provider=self.provider_name,
                model_name=self.model_id,
                raw_response="",
                latency_ms=latency_ms,
                status="error",
                error_message=err_msg
            )
