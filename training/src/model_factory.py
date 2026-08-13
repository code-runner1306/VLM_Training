import os
import yaml
from typing import Dict, Any, Type
from training.src.model_adapters.base import BaseVLMAdapter
from training.src.model_adapters.qwen25vl import Qwen25VLAdapter
from training.src.model_adapters.qwen3vl import Qwen3VLAdapter
from training.src.model_adapters.scold import SCOLDAdapter
from training.src.model_adapters.internvl import InternVLAdapter
from training.src.model_adapters.paligemma import PaliGemmaAdapter


class ModelFactory:
    """
    Factory for instantiating architecture-specific VLM adapters.
    """

    _ADAPTER_REGISTRY: Dict[str, Type[BaseVLMAdapter]] = {
        "Qwen25VLAdapter": Qwen25VLAdapter,
        "Qwen3VLAdapter": Qwen3VLAdapter,
        "SCOLDAdapter": SCOLDAdapter,
        "InternVLAdapter": InternVLAdapter,
        "PaliGemmaAdapter": PaliGemmaAdapter,
    }

    @classmethod
    def get_adapter(cls, model_key: str, run_config: Dict[str, Any]) -> BaseVLMAdapter:
        # Load registry config
        registry_path = os.path.join(
            os.path.dirname(__file__), "..", "configs", "models.yaml"
        )
        registry_path = os.path.abspath(registry_path)

        models_meta = {}
        if os.path.exists(registry_path):
            with open(registry_path, "r", encoding="utf-8") as f:
                registry_data = yaml.safe_load(f)
                models_meta = registry_data.get("models", {})

        model_meta = models_meta.get(model_key, {})
        model_id = run_config.get("model", {}).get("model_id") or model_meta.get("model_id", "Qwen/Qwen2.5-VL-3B-Instruct")
        adapter_class_name = model_meta.get("adapter_class", "Qwen25VLAdapter")

        adapter_class = cls._ADAPTER_REGISTRY.get(adapter_class_name, Qwen25VLAdapter)
        return adapter_class(model_key=model_key, model_id=model_id, config=run_config)
