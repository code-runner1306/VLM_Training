from typing import List, Dict, Any, Optional
from training.src.model_adapters.base import BaseVLMAdapter

try:
    from peft import LoraConfig, TaskType
except ImportError:
    LoraConfig = None
    TaskType = None


def get_lora_target_modules(adapter: BaseVLMAdapter, strategy: str) -> List[str]:
    """
    Resolve target module names for specified adaptation strategy:
    - 'llm_only': LLM self-attention and MLP projections only.
    - 'llm_projector': LLM + multi-modal merger/projector.
    - 'full_multimodal': Vision Encoder + Projector + LLM.
    """
    return adapter.get_target_modules(strategy)


def create_lora_config(
    adapter: BaseVLMAdapter,
    strategy: str = "llm_projector",
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    bias: str = "none",
) -> Any:
    """
    Build PEFT LoraConfig instance for the target VLM adapter and strategy.
    """
    if LoraConfig is None:
        raise ImportError(
            "peft is required for LoRA training. Please run `pip install peft`."
        )

    target_modules = get_lora_target_modules(adapter, strategy)

    peft_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=TaskType.CAUSAL_LM,
    )

    return peft_config
