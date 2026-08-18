import logging
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger("model_cache")

try:
    from huggingface_hub import snapshot_download

    HAS_HUB = True
except ImportError:
    HAS_HUB = False
    snapshot_download = None


def sanitize_model_name(model_id: str) -> str:
    """Convert a Hugging Face repo id into a filesystem-safe directory name."""
    return model_id.replace("/", "__").replace(":", "-")


def models_base_dir() -> Path:
    """Return the repository-local base-model cache root (models/base/)."""
    return Path("models") / "base"


def local_model_dir(model_id: str) -> Path:
    """Resolve the local cache path for a model id: models/base/<org>__<name>/."""
    return models_base_dir() / sanitize_model_name(model_id)


def is_model_cached(model_id: str) -> bool:
    """Return True if a valid model snapshot exists in the local cache."""
    return (local_model_dir(model_id) / "config.json").is_file()


def ensure_model_downloaded(
    model_id: str,
    token: Optional[str] = None,
    force: bool = False,
) -> Path:
    """Download a model snapshot into models/base/<org>__<name>/ if not cached.

    Returns the local cache path for the model.
    """
    cache_path = local_model_dir(model_id)
    if is_model_cached(model_id) and not force:
        logger.info(f"[MODEL CACHE] '{model_id}' already cached at {cache_path}")
        return cache_path

    if snapshot_download is None:
        raise ImportError(
            "huggingface_hub is required to download models. Run `pip install huggingface-hub`."
        )

    cache_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"[MODEL CACHE] Downloading '{model_id}' into {cache_path} ...")
    snapshot_download(
        repo_id=model_id,
        local_dir=str(cache_path),
        token=token,
        force_download=force,
    )
    if not is_model_cached(model_id):
        raise RuntimeError(
            f"Model '{model_id}' was downloaded but no config.json was found under {cache_path}."
        )
    logger.info(f"[MODEL CACHE] ✓ '{model_id}' cached at {cache_path}")
    return cache_path


def load_model_from_cache_or_hub(
    cls: Callable[..., Any],
    model_id: str,
    token: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """Instantiate ``cls.from_pretrained(...)`` preferring the repo-local cache.

    Resolution order:
      1. models/base/<org>__<name>  (repo-local cache)
      2. Hugging Face hub cache      (local_files_only=True)
      3. Download into models/base/  then load from there.
    """
    cache_path = local_model_dir(model_id)
    if is_model_cached(model_id):
        return cls.from_pretrained(str(cache_path), local_files_only=True, **kwargs)

    try:
        return cls.from_pretrained(model_id, local_files_only=True, **kwargs)
    except Exception:
        ensure_model_downloaded(model_id, token=token)
        return cls.from_pretrained(str(cache_path), local_files_only=True, **kwargs)