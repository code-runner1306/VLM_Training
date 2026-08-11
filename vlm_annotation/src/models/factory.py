from typing import Any, Dict
from vlm_annotation.src.models.base import VisionModel


def create_vision_model(model_config: Dict[str, Any]) -> VisionModel:
    provider = model_config.get("provider", "").lower()
    model_id = model_config.get("model", "")
    name = model_config.get("name", model_id)

    if provider == "nvidia":
        from vlm_annotation.src.models.nvidia_nim import NvidiaVisionModel
        return NvidiaVisionModel(provider_name=name, model_id=model_id, config=model_config)
    elif provider == "gemini":
        from vlm_annotation.src.models.gemini import GeminiVisionModel
        return GeminiVisionModel(provider_name=name, model_id=model_id, config=model_config)
    elif provider == "groq":
        from vlm_annotation.src.models.groq import GroqVisionModel
        return GroqVisionModel(provider_name=name, model_id=model_id, config=model_config)
    elif provider == "openrouter":
        from vlm_annotation.src.models.openrouter import OpenRouterVisionModel
        return OpenRouterVisionModel(provider_name=name, model_id=model_id, config=model_config)
    elif provider == "ollama":
        try:
            from vlm_annotation.src.models.ollama import OllamaVisionModel
            return OllamaVisionModel(provider_name=name, model_id=model_id, config=model_config)
        except ImportError as e:
            raise ImportError(f"Failed to import OllamaVisionModel: {e}. Please ensure ollama package is installed.")
    elif provider in ["huggingface", "hf"]:
        from vlm_annotation.src.models.huggingface import HuggingFaceVisionModel
        return HuggingFaceVisionModel(provider_name=name, model_id=model_id, config=model_config)
    else:
        raise ValueError(f"Unsupported VLM provider: {provider}")
