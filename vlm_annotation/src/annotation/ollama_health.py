import base64
import io
import json
from typing import Tuple
from PIL import Image
import httpx
from vlm_annotation.src.models.base import extract_json_from_text


def create_tiny_test_image_b64() -> str:
    """Generate 1x1 red PNG image base64 string for pre-flight health check."""
    img = Image.new("RGB", (1, 1), color="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def check_ollama_server_and_model(host: str, model_name: str, timeout: float = 10.0) -> Tuple[bool, str]:
    """
    Synchronous pre-flight health check verifying:
    1. Server reachability at host (/api/tags)
    2. Model existence
    3. Vision test image inference & JSON schema parsing
    """
    clean_host = host.rstrip("/")

    # 1. Server reachability check
    try:
        res = httpx.get(f"{clean_host}/api/tags", timeout=timeout)
        res.raise_for_status()
    except Exception:
        return False, f"ERROR: Ollama server is not reachable at {clean_host}\n\nStart Ollama and ensure it is running."

    # 2. Model existence check
    tags_json = res.json()
    models = [m.get("name", "") for m in tags_json.get("models", [])]
    
    # Allow matching qwen3-vl:8b or qwen3-vl:8b-latest or full tag
    model_found = any(
        m == model_name or m.startswith(f"{model_name}:") or m.startswith(model_name)
        for m in models
    )

    if not model_found:
        installed_str = ", ".join(models) if models else "None"
        return False, (
            f"ERROR: Model '{model_name}' is not available locally on Ollama.\n"
            f"Currently installed models: [{installed_str}]\n\n"
            f"Install it with:\n\n"
            f"  ollama pull {model_name}\n"
        )

    # 3. Vision + JSON parsing test request
    try:
        tiny_b64 = create_tiny_test_image_b64()
        payload = {
            "model": model_name,
            "prompt": "Identify the disease and return JSON with keys 'disease' and 'reasoning'.",
            "images": [tiny_b64],
            "stream": False,
            "format": "json",
            "options": {"num_predict": 100}
        }
        test_res = httpx.post(f"{clean_host}/api/generate", json=payload, timeout=30.0)
        test_res.raise_for_status()
        raw_text = test_res.json().get("response", "")
        
        parsed = extract_json_from_text(raw_text)
        if not parsed:
            return False, f"ERROR: Test request to model '{model_name}' returned malformed JSON response: {raw_text[:200]}"

    except Exception as e:
        return False, f"ERROR: Pre-flight test inference failed for model '{model_name}': {e}"

    return True, f"SUCCESS: Ollama server reachable at {clean_host} and model '{model_name}' is ready."
