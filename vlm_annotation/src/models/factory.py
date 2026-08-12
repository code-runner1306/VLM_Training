from typing import Any, Dict
from vlm_annotation.src.models.base import VisionModel
from vlm_annotation.src.models.gemini import GeminiVisionModel
from vlm_annotation.src.models.groq import GroqVisionModel
from vlm_annotation.src.models.nvidia_nim import NvidiaVisionModel
from vlm_annotation.src.models.ollama import OllamaVisionModel
from vlm_annotation.src.models.openrouter import OpenRouterVisionModel


def create_vision_model(model_config: Dict[str, Any]) -> VisionModel:
    provider = model_config.get("provider", "").lower()
    model_id = model_config.get("model", "")
    name = model_config.get("name", model_id)

    if provider == "ollama":
        return OllamaVisionModel(provider_name=name, model_id=model_id, config=model_config)
    elif provider == "nvidia":
        return NvidiaVisionModel(provider_name=name, model_id=model_id, config=model_config)
    elif provider == "gemini":
        return GeminiVisionModel(provider_name=name, model_id=model_id, config=model_config)
    elif provider == "groq":
        return GroqVisionModel(provider_name=name, model_id=model_id, config=model_config)
    elif provider == "openrouter":
        return OpenRouterVisionModel(provider_name=name, model_id=model_id, config=model_config)
    else:
        raise ValueError(f"Unsupported VLM provider: {provider}")
