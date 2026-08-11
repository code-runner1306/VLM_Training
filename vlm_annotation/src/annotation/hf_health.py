import base64
import io
import json
from typing import Tuple
import torch
from PIL import Image

from vlm_annotation.src.models.base import extract_json_from_text

try:
    import transformers
    from transformers import AutoProcessor
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    AutoProcessor = None


def check_huggingface_environment_and_model(model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct") -> Tuple[bool, str]:
    """
    Pre-flight health check verifying:
    1. PyTorch & transformers library installation
    2. CUDA / GPU availability (or CPU fallback warning)
    3. Model & processor accessibility on Hugging Face Hub
    """
    if not HAS_TRANSFORMERS or AutoProcessor is None:
        return False, "ERROR: `transformers` library is not installed. Please run `pip install transformers`."

    cuda_available = torch.cuda.is_available()
    device_str = torch.cuda.get_device_name(0) if cuda_available else "CPU (No CUDA GPU detected)"

    # Test loading AutoProcessor to verify model ID reachability
    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:
        return False, f"ERROR: Failed to load AutoProcessor for Hugging Face model '{model_id}': {e}"

    msg = (
        f"SUCCESS: Pre-flight check passed for Hugging Face model '{model_id}'.\n"
        f"Compute Hardware: {device_str}\n"
        f"Transformers Version: {transformers.__version__}\n"
        f"PyTorch Version: {torch.__version__}"
    )
    return True, msg
