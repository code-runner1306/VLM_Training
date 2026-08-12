import base64
import os
import time
from typing import Any, Dict, Optional
from vlm_annotation.src.dataset import validate_and_prepare_image
from vlm_annotation.src.models.base import ModelMemoryError, ModelResponse, VisionModel, extract_json_from_text

try:
    from openai import AsyncOpenAI
    HAS_OPENAI = True
except ImportError:
    AsyncOpenAI = None
    HAS_OPENAI = False


class OllamaVisionModel(VisionModel):
    """Provider implementation for locally served Ollama vision models (OpenAI-compatible API)."""

    def __init__(self, provider_name: str, model_id: str, config: Dict[str, Any]):
        super().__init__(provider_name, model_id, config)
        if not HAS_OPENAI:
            raise ImportError("openai package is required for OllamaVisionModel.")

        # Default to Ollama's local OpenAI-compatible endpoint if not specified.
        base_url = config.get("base_url", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"))
        api_key = config.get("api_key", os.getenv("OLLAMA_API_KEY", "ollama"))
        timeout = float(config.get("timeout", 300.0))
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.num_ctx = int(config.get("num_ctx", 2048))
        self.max_dimension = int(config.get("max_dimension", 640))
        self.max_tokens = int(config.get("max_tokens", 2000))

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
            img_bytes = validate_and_prepare_image(image_path, max_dimension=self.max_dimension)
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
                max_tokens=self.max_tokens,
                extra_body={"options": {"num_ctx": self.num_ctx}}
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
            err_lower = err_str.lower()

            # Memory errors are deterministic; they will not resolve on retry.
            if any(
                kw in err_lower
                for kw in ["requires more system memory", "out of memory", "not enough memory", "insufficient memory", "cuda out of memory", "allocation failed"]
            ):
                self.failed_requests += 1
                latency_ms = (time.monotonic() - start_time) * 1000.0
                return ModelResponse(
                    provider=self.provider_name,
                    model_name=self.model_id,
                    raw_response="",
                    latency_ms=round(latency_ms, 2),
                    status="error",
                    error_message=f"ModelMemoryError: {err_str}"
                )

            is_transient = any(
                code in err_lower
                for code in ["429", "rate limit", "504", "gateway timeout", "502", "503", "500", "timeout", "connection error", "connect"]
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

        return None
