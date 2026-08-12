import os
import json
import hashlib
from typing import Dict, Any, Tuple, Optional, List
from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None
    Dataset = object


def resolve_image_path(image_path: str, dataset_root: str) -> str:
    """Resolve image path relative to dataset root, handling OS path slashes."""
    normalized_path = image_path.replace("\\", "/")
    full_path = os.path.join(dataset_root, normalized_path)
    return os.path.abspath(full_path)


def compute_image_hash(filepath: str) -> Optional[str]:
    """Compute MD5 hash of an image file for duplicate detection."""
    try:
        hasher = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return None


def validate_annotation(item: Dict[str, Any], dataset_root: str) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Validate an annotation item from annotations.jsonl.
    
    Returns:
        (is_eligible: bool, reason: str, cleaned_annotation: Optional[Dict[str, Any]])
    """
    if not isinstance(item, dict):
        return False, "malformed_annotation", None

    image_path_raw = item.get("image_path")
    if not image_path_raw:
        return False, "unresolved_image_path", None

    full_image_path = resolve_image_path(image_path_raw, dataset_root)
    if not os.path.exists(full_image_path):
        return False, "missing_image", None

    try:
        with Image.open(full_image_path) as img:
            img.verify()
    except Exception:
        return False, "corrupt_image", None

    quality_status = str(item.get("quality_status", "")).lower()
    if quality_status in ["failed", "error", "rejected"]:
        return False, "failed_teacher_request", None

    disease = item.get("disease")
    if not disease or not isinstance(disease, str) or not disease.strip():
        return False, "invalid_annotation", None

    parsed = item.get("parsed_annotation")
    if not parsed or not isinstance(parsed, dict):
        return False, "missing_annotation", None

    visible_obs = parsed.get("visible_observations")
    diagnostic_ev = parsed.get("diagnostic_evidence")
    reasoning = parsed.get("reasoning")

    if not reasoning or not isinstance(reasoning, str) or not reasoning.strip():
        return False, "empty_description", None

    if not visible_obs or not isinstance(visible_obs, list) or len(visible_obs) == 0:
        return False, "empty_description", None

    if not diagnostic_ev or not isinstance(diagnostic_ev, list) or len(diagnostic_ev) == 0:
        return False, "empty_description", None

    cleaned_annotation = {
        "image_id": item.get("image_id", ""),
        "image_path": full_image_path,
        "relative_path": image_path_raw.replace("\\", "/"),
        "disease": disease.strip(),
        "quality_status": quality_status,
        "parsed_annotation": parsed,
        "teacher_model": item.get("teacher_model", "unknown"),
        "prompt_version": item.get("prompt_version", "1.0"),
    }

    return True, "eligible", cleaned_annotation


DEFAULT_USER_PROMPT = (
    "What disease is affecting this crop plant?\n\n"
    "Analyze the image and explain your diagnosis based only on visible evidence."
)


def format_vlm_conversation(
    item: Dict[str, Any],
    user_prompt: str = DEFAULT_USER_PROMPT,
) -> Dict[str, Any]:
    """
    Format synthetic annotation into a structured VLM User/Assistant conversation.
    """
    parsed = item.get("parsed_annotation", {})
    disease = item.get("disease", "Unknown")
    visible_obs = parsed.get("visible_observations", [])
    affected_regions = parsed.get("affected_regions", [])
    diagnostic_ev = parsed.get("diagnostic_evidence", [])
    reasoning = parsed.get("reasoning", "")
    uncertain_obs = parsed.get("uncertain_observations", [])

    obs_str = "\n".join([f"- {obs}" for obs in visible_obs]) if visible_obs else "- None noted"
    regions_str = "\n".join([f"- {r}" for r in affected_regions]) if affected_regions else "- None noted"
    evidence_str = "\n".join([f"- {ev}" for ev in diagnostic_ev]) if diagnostic_ev else "- None noted"
    uncertain_str = "\n".join([f"- {u}" for u in uncertain_obs]) if uncertain_obs else ""

    assistant_reply = (
        f"Disease: {disease}\n\n"
        f"Visible observations:\n{obs_str}\n\n"
        f"Affected regions:\n{regions_str}\n\n"
        f"Diagnostic evidence:\n{evidence_str}\n\n"
        f"Reasoning:\n{reasoning}"
    )

    if uncertain_str:
        assistant_reply += f"\n\nUncertainty:\n{uncertain_str}"

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": item["image_path"]},
                {"type": "text", "text": user_prompt},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": assistant_reply},
            ],
        },
    ]

    return {
        "image_id": item.get("image_id"),
        "image_path": item["image_path"],
        "disease": disease,
        "messages": conversation,
        "raw_assistant_reply": assistant_reply,
    }


class VLMDataset(Dataset):
    """PyTorch Dataset for Crop Disease VLM Fine-Tuning."""

    def __init__(
        self,
        manifest_path: str,
        user_prompt: str = DEFAULT_USER_PROMPT,
    ):
        self.manifest_path = manifest_path
        self.user_prompt = user_prompt
        self.items: List[Dict[str, Any]] = []

        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.items.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        formatted = format_vlm_conversation(item, user_prompt=self.user_prompt)
        return formatted


class VLMDataCollator:
    """Multi-modal Data Collator for HuggingFace / Custom Trainer."""

    def __init__(self, processor: Any, max_length: int = 2048):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        images = []
        texts = []

        for item in batch:
            image_path = item["image_path"]
            try:
                img = Image.open(image_path).convert("RGB")
            except Exception as e:
                raise RuntimeError(f"Failed loading image {image_path}: {e}")
            images.append(img)

            if hasattr(self.processor, "apply_chat_template"):
                formatted_text = self.processor.apply_chat_template(
                    item["messages"], tokenize=False, add_generation_prompt=False
                )
            else:
                formatted_text = item["raw_assistant_reply"]
            texts.append(formatted_text)

        inputs = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        labels = inputs["input_ids"].clone()
        if hasattr(self.processor, "tokenizer") and self.processor.tokenizer.pad_token_id is not None:
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels

        return inputs
