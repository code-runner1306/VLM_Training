import base64
import json
import os
import re
import time
from typing import Any, Dict, Optional
from vlm_annotation.src.dataset import validate_and_prepare_image
from vlm_annotation.src.models.base import ModelResponse, VisionModel, extract_json_from_text

try:
    from groq import AsyncGroq
    HAS_GROQ = True
except ImportError:
    AsyncGroq = None
    HAS_GROQ = False


class GroqVisionModel(VisionModel):
    """Provider implementation for Groq Hosted Vision Models."""

    def __init__(self, provider_name: str, model_id: str, config: Dict[str, Any]):
        super().__init__(provider_name, model_id, config)
        if not HAS_GROQ:
            raise ImportError("groq package is required for GroqVisionModel.")

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set.")

        timeout = float(config.get("timeout", 60.0))
        self.client = AsyncGroq(api_key=api_key, timeout=timeout)

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
            img_bytes = validate_and_prepare_image(image_path)
            b64_img = base64.b64encode(img_bytes).decode("utf-8")
            data_url = f"data:image/jpeg;base64,{b64_img}"

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}}
                    ]
                }
            ]

            response = await self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=0.1,
                max_tokens=3500
            )

            latency_ms = (time.monotonic() - start_time) * 1000.0
            self.total_latency_ms += latency_ms
            raw_text = response.choices[0].message.content or ""

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
                status=status,
                prompt_tokens=getattr(response.usage, "prompt_tokens", 0) if response.usage else 0,
                completion_tokens=getattr(response.usage, "completion_tokens", 0) if response.usage else 0
            )

        except Exception as e:
            err_str = str(e)
            is_transient = any(
                code in err_str.lower()
                for code in ["429", "rate limit", "504", "gateway timeout", "502", "503", "500", "timeout", "connection error"]
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
