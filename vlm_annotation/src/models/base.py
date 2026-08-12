import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


class ModelMemoryError(Exception):
    """Raised when the model runtime reports insufficient memory (OOM / KV-cache).

    Memory errors are deterministic and will not resolve on retry, so they must
    fail fast instead of entering a retry/backoff loop.
    """


@dataclass
class ModelResponse:
    provider: str
    model_name: str
    raw_response: str
    parsed_json: Optional[Dict[str, Any]] = None
    latency_ms: float = 0.0
    status: str = "success"  # success, error, rate_limited, json_parse_error
    error_message: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    retry_count: int = 0
    rate_limit_hits: int = 0


def _repair_json(text: str) -> Optional[Dict[str, Any]]:
    """Attempt to repair common malformed JSON from smaller VLMs (bad escapes, missing commas)."""
    repaired = text
    # Repair invalid backslash escape sequences (e.g. Windows paths: "C:\Users" -> "C:\\Users").
    # A backslash not followed by a valid JSON escape char gets doubled.
    repaired = re.sub(r'\\(?![\\"/bfnrtu]|u[0-9a-fA-F]{4})', r'\\\\', repaired)
    # Insert missing commas between object members. After a completed value (" } ]) a new
    # key looks like  "some_key":  so we insert a comma before it. Safe: does not touch
    # empty strings or array elements (they are not followed by a '":' key pattern).
    repaired = re.sub(r'(?<=["}\]])(?=\s*"[^"\n]+"\s*:)', ',', repaired)
    try:
        return json.loads(repaired)
    except Exception:
        pass
    return None


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Helper utility to extract and parse JSON from model responses including thinking models."""
    if not text or not text.strip():
        return None

    # 1. Strip out <think>...</think> reasoning tags
    clean_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # 2. Try direct json.loads
    try:
        return json.loads(clean_text)
    except Exception:
        pass

    # 3. Try markdown code fences ```json ... ```
    match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", clean_text)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            repaired = _repair_json(match.group(1))
            if repaired is not None:
                return repaired

    # 4. Try greedy match for outer JSON object {...}
    match = re.search(r"(\{[\s\S]*\})", clean_text)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            repaired = _repair_json(match.group(1))
            if repaired is not None:
                return repaired

    # 5. Repair the whole cleaned text directly
    repaired = _repair_json(clean_text)
    if repaired is not None:
        return repaired

    return None


class VisionModel(ABC):
    """Abstract Base Class for all VLM Providers."""

    def __init__(self, provider_name: str, model_id: str, config: Dict[str, Any]):
        self.provider_name = provider_name
        self.model_id = model_id
        self.config = config
        # Model-specific diagnostic counters
        self.total_requests: int = 0
        self.successful_requests: int = 0
        self.failed_requests: int = 0
        self.rate_limit_hits: int = 0
        self.json_parse_failures: int = 0
        self.total_latency_ms: float = 0.0

    @abstractmethod
    async def generate_annotation(
        self,
        image_path: str,
        disease_name: str,
        prompt: str,
        disease_profile: Optional[Dict[str, Any]] = None
    ) -> ModelResponse:
        """
        Generate grounded annotation for image.
        Returns ModelResponse containing raw text, parsed JSON dict, latency, and status.
        """
        pass

    def get_metrics(self) -> Dict[str, Any]:
        """Return cumulative counters for rate limits, errors, and throughput for this model."""
        avg_latency = (self.total_latency_ms / self.successful_requests) if self.successful_requests > 0 else 0.0
        return {
            "provider": self.provider_name,
            "model_id": self.model_id,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "rate_limit_hits": self.rate_limit_hits,
            "json_parse_failures": self.json_parse_failures,
            "average_latency_ms": round(avg_latency, 2)
        }
