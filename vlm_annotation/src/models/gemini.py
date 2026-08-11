import json
import os
import re
import time
from typing import Any, Dict, Optional
from PIL import Image
from vlm_annotation.src.models.base import ModelResponse, VisionModel, extract_json_from_text

try:
    from google import genai
    from google.genai import types
    HAS_GOOGLE_GENAI = True
except ImportError:
    HAS_GOOGLE_GENAI = False

try:
    import google.generativeai as legacy_genai
    HAS_LEGACY_GENAI = True
except ImportError:
    HAS_LEGACY_GENAI = False


class GeminiVisionModel(VisionModel):
    """Provider implementation for Google Gemini Vision Models."""

    def __init__(self, provider_name: str, model_id: str, config: Dict[str, Any]):
        super().__init__(provider_name, model_id, config)
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")

        if HAS_GOOGLE_GENAI:
            self.client = genai.Client(api_key=api_key)
            self.mode = "google-genai"
        elif HAS_LEGACY_GENAI:
            legacy_genai.configure(api_key=api_key)
            self.model_client = legacy_genai.GenerativeModel(model_id)
            self.mode = "legacy-genai"
        else:
            raise ImportError("Neither google-genai nor google-generativeai is installed.")

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
            pil_img = Image.open(image_path).convert("RGB")
            raw_text = ""

            if self.mode == "google-genai":
                response = self.client.models.generate_content(
                    model=self.model_id,
                    contents=[pil_img, prompt],
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        response_mime_type="application/json"
                    )
                )
                raw_text = response.text or ""
            else:
                response = self.model_client.generate_content([pil_img, prompt])
                raw_text = response.text or ""

            latency_ms = (time.monotonic() - start_time) * 1000.0
            self.total_latency_ms += latency_ms

            parsed_json = extract_json_from_text(raw_text)
            if parsed_json is None:
                self.json_parse_failures += 1
                status = "json_parse_error"
            else:
                status = "success"
                self.successful_requests += 1

            return ModelResponse(
                provider=self.provider_name,
                model_name=self.model_id,
                raw_response=raw_text,
                parsed_json=parsed_json,
                latency_ms=round(latency_ms, 2),
                status=status
            )

        except Exception as e:
            err_str = str(e)
            is_transient = any(
                code in err_str.lower()
                for code in ["429", "rate limit", "quota", "resourceexhausted", "504", "503", "502", "500", "timeout", "deadline"]
            )
            if is_transient:
                raise e

            self.failed_requests += 1
            latency_ms = (time.monotonic() - start_time) * 1000.0
            return ModelResponse(
                provider=self.provider_name,
                model_name=self.model_id,
                raw_response="",
                latency_ms=round(latency_ms, 2),
                status="error",
                error_message=err_str
            )
