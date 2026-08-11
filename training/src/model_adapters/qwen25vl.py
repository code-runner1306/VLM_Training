import re
from typing import Dict, Any, Tuple, List, Optional
import torch
from training.src.model_adapters.base import BaseVLMAdapter


class Qwen25VLAdapter(BaseVLMAdapter):
    """
    Adapter for Qwen2.5-VL models (3B & 7B variants).
    """

    def load_model_and_processor(
        self,
        quantization_config: Optional[Any] = None,
        torch_dtype: Optional[torch.dtype] = None,
        device_map: str = "auto",
    ) -> Tuple[Any, Any]:
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        except ImportError:
            raise ImportError(
                "transformers >= 4.45 is required for Qwen2.5-VL models. "
                "Please run `pip install transformers>=4.45.0`."
            )

        min_pixels = self.config.get("image", {}).get("min_pixels", 256 * 28 * 28)
        max_pixels = self.config.get("image", {}).get("max_pixels", 1280 * 28 * 28)

        processor = AutoProcessor.from_pretrained(
            self.model_id,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            trust_remote_code=True,
        )

        model_kwargs = {
            "torch_dtype": torch_dtype or torch.bfloat16,
            "device_map": device_map,
            "trust_remote_code": True,
        }

        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            **model_kwargs,
        )

        return model, processor

    def get_target_modules(self, strategy: str) -> List[str]:
        llm_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
        projector_modules = ["merger"]
        vision_modules = ["visual.blocks"]

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
        """
        Extract predicted disease and breakdown from generated response text.
        """
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
