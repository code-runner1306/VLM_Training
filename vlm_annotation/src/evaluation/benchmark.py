import json
import math
import random
from pathlib import Path
from typing import Dict, List
from vlm_annotation.src.dataset import ImageItem, discover_dataset


def sample_benchmark_images(
    dataset_dir: str,
    target_count: int = 200,
    seed: int = 42,
    output_path: str = "outputs/benchmark/benchmark_images.json"
) -> List[ImageItem]:
    """
    Perform stratified sampling across all disease classes to select target_count representative images.
    Saves and returns deterministic image sample list.
    """
    output_file = Path(output_path)
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return [ImageItem(**item) for item in data]

    items, by_disease = discover_dataset(dataset_dir)
    diseases = sorted(by_disease.keys())
    if not diseases:
        raise ValueError(f"No disease classes discovered in dataset directory {dataset_dir}")

    per_class_target = math.ceil(target_count / len(diseases))
    sampled_items: List[ImageItem] = []

    rng = random.Random(seed)
    for disease in diseases:
        class_items = sorted(by_disease[disease], key=lambda x: x.image_id)
        if len(class_items) <= per_class_target:
            sampled_items.extend(class_items)
        else:
            sampled = rng.sample(class_items, per_class_target)
            sampled_items.extend(sampled)

    # Trim to exact target count if oversampled
    if len(sampled_items) > target_count:
        sampled_items = rng.sample(sampled_items, target_count)

    sampled_items = sorted(sampled_items, key=lambda x: x.image_id)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump([item.__dict__ for item in sampled_items], f, indent=2)

    return sampled_items
