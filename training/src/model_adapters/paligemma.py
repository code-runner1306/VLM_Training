import re
from typing import Dict, Any, Tuple, List, Optional
import torch
from training.src.model_adapters.base import BaseVLMAdapter
from training.src.model_cache import load_model_from_cache_or_hub


def load_from_pretrained_with_cache_check(cls, model_id: str, **kwargs):
    """
    Load a model or processor preferring the repo-local cache
    (models/base/<org>__<name>), then the HF hub cache, then a fresh download.
    """
    return load_model_from_cache_or_hub(cls, model_id, **kwargs)


class PaliGemmaAdapter(BaseVLMAdapter):
    """
    Adapter for Google PaliGemma / PaliGemma 2 models (e.g. google/paligemma2-10b-pt-448).
    Supports 448x448 resolution image processing and prefix-lm text generation.
    """

    def load_model_and_processor(
        self,
        quantization_config: Optional[Any] = None,
        torch_dtype: Optional[torch.dtype] = None,
        device_map: str = "auto",
    ) -> Tuple[Any, Any]:
        from transformers import AutoProcessor

        processor = load_from_pretrained_with_cache_check(
            AutoProcessor,
            self.model_id,
            trust_remote_code=True,
        )

        model_kwargs = {
            "torch_dtype": torch_dtype or torch.bfloat16,
            "device_map": device_map,
            "trust_remote_code": True,
        }

        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config

        model = None
        try:
            from transformers import PaliGemmaForConditionalGeneration
            model = load_from_pretrained_with_cache_check(
                PaliGemmaForConditionalGeneration,
                self.model_id,
                **model_kwargs,
            )
        except Exception:
            pass

        if model is None:
            from transformers import AutoModelForVision2Seq
            model = load_from_pretrained_with_cache_check(
                AutoModelForVision2Seq,
                self.model_id,
                **model_kwargs,
            )

        return model, processor

    def get_target_modules(self, strategy: str) -> List[str]:
        llm_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
        projector_modules = ["multi_modal_projector"]
        vision_modules = ["vision_tower"]

        if strategy == "llm_only":
            return llm_modules
        elif strategy == "llm_projector":
            return llm_modules + projector_modules
        elif strategy == "full_multimodal":
            return llm_modules + projector_modules + vision_modules
        else:
            raise ValueError(f"Unknown adaptation strategy: '{strategy}'")

    def prepare_inputs(self, processor: Any, images: List[Any], texts: List[str], device: Any) -> Dict[str, Any]:
        inputs = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
        )
        return {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    def parse_generated_output(self, generated_text: str) -> Dict[str, Any]:
        disease_match = re.search(r"Disease:\s*([^\n\r]+)", generated_text, re.IGNORECASE)
        predicted_disease = disease_match.group(1).strip() if disease_match else "Unknown"

        visible_obs = []
        obs_section = re.search(r"Visible observations:\s*(.*?)(?=\n\n|\n[A-Z]|\Z)", generated_text, re.DOTALL | re.IGNORECASE)
        if obs_section:
            visible_obs = [line.strip("- ").strip() for line in obs_section.group(1).strip().split("\n") if line.strip()]

        diagnostic_ev = []
        ev_section = re.search(r"Diagnostic evidence:\s*(.*?)(?=\n\n|\n[A-Z]|\Z)", generated_text, re.DOTALL | re.IGNORECASE)
        if ev_section:
            diagnostic_ev = [line.strip("- ").strip() for line in ev_section.group(1).strip().split("\n") if line.strip()]

        reasoning = ""
        reason_section = re.search(r"Reasoning:\s*(.*?)(?=\n\n|\n[A-Z]|\Z)", generated_text, re.DOTALL | re.IGNORECASE)
        if reason_section:
            reasoning = reason_section.group(1).strip()

        return {
            "predicted_disease": predicted_disease,
            "visible_observations": visible_obs,
            "diagnostic_evidence": diagnostic_ev,
            "reasoning": reasoning,
            "raw_text": generated_text,
        }
