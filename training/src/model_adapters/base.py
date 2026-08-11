from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List, Optional
import torch


class BaseVLMAdapter(ABC):
    """
    Abstract Base Adapter class for multi-architecture Vision-Language Models.
    Isolates model-specific API variations from the common trainer/evaluator engine.
    """

    def __init__(self, model_key: str, model_id: str, config: Dict[str, Any]):
        self.model_key = model_key
        self.model_id = model_id
        self.config = config

    @abstractmethod
    def load_model_and_processor(
        self,
        quantization_config: Optional[Any] = None,
        torch_dtype: Optional[torch.dtype] = None,
        device_map: str = "auto",
    ) -> Tuple[Any, Any]:
        """
        Load base model and processor/tokenizer.
        
        Returns:
            (model, processor)
        """
        pass

    @abstractmethod
    def get_target_modules(self, strategy: str) -> List[str]:
        """
        Return PEFT LoRA target module strings for specified strategy:
        - 'llm_only'
        - 'llm_projector'
        - 'full_multimodal'
        """
        pass

    @abstractmethod
    def prepare_inputs(self, processor: Any, images: List[Any], texts: List[str], device: Any) -> Dict[str, Any]:
        """
        Prepare batch tensor inputs for training or inference.
        """
        pass

    @abstractmethod
    def parse_generated_output(self, generated_text: str) -> Dict[str, Any]:
        """
        Parse raw model generation output into structured dictionary:
        - predicted_disease
        - visible_observations
        - diagnostic_evidence
        - reasoning
        """
        pass
