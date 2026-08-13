import re
from typing import Dict, Any, Tuple, List, Optional
import torch
from training.src.model_adapters.base import BaseVLMAdapter


def load_from_pretrained_with_cache_check(cls, model_id: str, **kwargs):
    """
    Attempt to load a model or processor directly from local Hugging Face cache first (local_files_only=True).
    If the model is not found in local cache, fall back to downloading/loading from HF Hub (local_files_only=False).
    """
    try:
        return cls.from_pretrained(model_id, local_files_only=True, **kwargs)
    except Exception:
        return cls.from_pretrained(model_id, local_files_only=False, **kwargs)


class InternVLAdapter(BaseVLMAdapter):
    """
    Adapter for InternVL2 / InternVL2.5 models (e.g. OpenGVLab/InternVL2_5-8B).
    Optimized for high-resolution dynamic image tiling and fine-grained visual inspection.
    """

    def load_model_and_processor(
        self,
        quantization_config: Optional[Any] = None,
        torch_dtype: Optional[torch.dtype] = None,
        device_map: str = "auto",
    ) -> Tuple[Any, Any]:
        from transformers import AutoProcessor, AutoModelForCausalLM

        try:
            processor = load_from_pretrained_with_cache_check(
                AutoProcessor,
                self.model_id,
                trust_remote_code=True,
            )
        except Exception:
            from transformers import AutoTokenizer
            processor = load_from_pretrained_with_cache_check(
                AutoTokenizer,
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

        model = load_from_pretrained_with_cache_check(
            AutoModelForCausalLM,
            self.model_id,
            **model_kwargs,
        )

        return model, processor

    def get_target_modules(self, strategy: str) -> List[str]:
        llm_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        projector_modules = ["mlp1", "multi_modal_projector"]
        vision_modules = ["vision_model", "encoder"]

        if strategy == "llm_only":
            return llm_modules
        elif strategy == "llm_projector":
            return llm_modules + projector_modules
        elif strategy == "full_multimodal":
            return llm_modules + projector_modules + vision_modules
        else:
            raise ValueError(f"Unknown adaptation strategy: '{strategy}'")

    def prepare_inputs(self, processor: Any, images: List[Any], texts: List[str], device: Any) -> Dict[str, Any]:
        try:
            inputs = processor(
                text=texts,
                images=images,
                return_tensors="pt",
                padding=True,
            )
        except Exception:
            inputs = processor(
                text=texts,
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
