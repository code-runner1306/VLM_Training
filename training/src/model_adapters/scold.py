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


class SCOLDAdapter(BaseVLMAdapter):
    """
    Adapter for SCOLD (Domain-Specific Contrastive Dual-Encoder VLM).
    Optimized for agricultural disease image classification & visual representation embedding.
    """

    def load_model_and_processor(
        self,
        quantization_config: Optional[Any] = None,
        torch_dtype: Optional[torch.dtype] = None,
        device_map: str = "auto",
    ) -> Tuple[Any, Any]:
        from transformers import AutoProcessor, AutoModel

        try:
            processor = load_from_pretrained_with_cache_check(
                AutoProcessor,
                self.model_id,
                trust_remote_code=True,
            )
        except Exception:
            from transformers import AutoImageProcessor
            processor = load_from_pretrained_with_cache_check(
                AutoImageProcessor,
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
            from transformers import CLIPModel
            model = load_from_pretrained_with_cache_check(
                CLIPModel,
                self.model_id,
                **model_kwargs,
            )
        except Exception:
            pass

        if model is None:
            model = load_from_pretrained_with_cache_check(
                AutoModel,
                self.model_id,
                **model_kwargs,
            )

        return model, processor

    def get_target_modules(self, strategy: str) -> List[str]:
        vision_modules = ["visual.encoder", "vision_model", "projection_head"]
        if strategy in ["full_multimodal", "llm_projector"]:
            return vision_modules
        return vision_modules

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
                images=images,
                return_tensors="pt",
            )
        return {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    def parse_generated_output(self, generated_text: str) -> Dict[str, Any]:
        disease_match = re.search(r"Disease:\s*([^\n\r]+)", generated_text, re.IGNORECASE)
        predicted_disease = disease_match.group(1).strip() if disease_match else generated_text.strip()

        return {
            "predicted_disease": predicted_disease,
            "visible_observations": ["Visual embeddings computed via SCOLD dual-encoder."],
            "diagnostic_evidence": ["High confidence agricultural contrastive feature alignment."],
            "reasoning": "SCOLD contrastive feature match.",
            "raw_text": generated_text,
        }
