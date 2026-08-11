import hashlib
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class ImageItem:
    image_id: str
    image_path: str
    relative_path: str
    disease_name: str


def generate_image_id(relative_path: str) -> str:
    """Generate a deterministic SHA-256 hash ID based on normalized relative path."""
    normalized_path = relative_path.replace("\\", "/").strip().lower()
    return hashlib.sha256(normalized_path.encode("utf-8")).hexdigest()[:16]


def discover_dataset(dataset_dir: str) -> Tuple[List[ImageItem], Dict[str, List[ImageItem]]]:
    """
    Recursively discover all supported images in dataset directory.
    Discovers disease class names automatically from subfolder titles.
    """
    dataset_path = Path(dataset_dir).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    items: List[ImageItem] = []
    by_disease: Dict[str, List[ImageItem]] = {}

    for root, _, files in os.walk(dataset_path):
        for file in sorted(files):
            ext = Path(file).suffix.lower()
            if ext in SUPPORTED_EXTENSIONS:
                full_path = Path(root) / file
                rel_path = full_path.relative_to(dataset_path)

                # Determine disease name from immediate parent folder relative to dataset root
                parts = rel_path.parts
                if len(parts) > 1:
                    disease_name = parts[0]
                else:
                    disease_name = "Unknown"

                img_id = generate_image_id(str(rel_path))
                item = ImageItem(
                    image_id=img_id,
                    image_path=str(full_path),
                    relative_path=str(rel_path),
                    disease_name=disease_name
                )
                items.append(item)

                if disease_name not in by_disease:
                    by_disease[disease_name] = []
                by_disease[disease_name].append(item)

    return items, by_disease


def validate_and_prepare_image(
    image_path: str,
    max_dimension: int = 1536,
    quality: int = 85
) -> bytes:
    """
    Validate that an image can be opened, handle color channels/transparency,
    resize while preserving aspect ratio if larger than max_dimension, and encode to JPEG bytes.
    """
    try:
        with Image.open(image_path) as img:
            img.verify()
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            width, height = img.size

            if width > max_dimension or height > max_dimension:
                if width >= height:
                    new_w = max_dimension
                    new_h = int(height * (max_dimension / width))
                else:
                    new_h = max_dimension
                    new_w = int(width * (max_dimension / height))
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=quality)
            return buffer.getvalue()
    except Exception as e:
        raise ValueError(f"Corrupted or invalid image file {image_path}: {str(e)}")
