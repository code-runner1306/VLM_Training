import os
from typing import List, Optional


def load_gemini_keys() -> List[str]:
    """Return Gemini API keys for annotation.

    Precedence: comma-separated `GEMINI_API_KEYS` (stripped) wins; otherwise
    falls back to the single `GEMINI_API_KEY`. Returns an empty list when
    neither variable is set.
    """
    multi = os.getenv("GEMINI_API_KEYS")
    if multi:
        keys = [k.strip() for k in multi.split(",") if k.strip()]
        if keys:
            return keys

    single = os.getenv("GEMINI_API_KEY")
    if single:
        return [single.strip()]

    return []


def mask_key(key: str) -> str:
    """Mask an API key for logs/metadata: keep first 4 + last 4 chars."""
    if not key:
        return ""
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:4]}...{key[-4:]}"


def resolve_max_workers(default: int = 4) -> int:
    """Resolve worker cap override from `MAX_GEMINI_WORKERS` env var."""
    raw = os.getenv("MAX_GEMINI_WORKERS")
    if not raw:
        return default
    try:
        value = int(raw.strip())
        return value if value > 0 else default
    except ValueError:
        return default


def resolve_worker_count(keys: List[str], max_workers: Optional[int] = None) -> int:
    """Compute the number of parallel workers.

    `worker_count = min(len(keys), cpu_count, max_workers)`. When max_workers
    is None it is resolved from `MAX_GEMINI_WORKERS` env (default 4).
    """
    if not keys:
        return 0
    cap = max_workers if max_workers is not None else resolve_max_workers()
    import os as _os
    cpu = _os.cpu_count() or 1
    return min(len(keys), max(1, cpu), cap)