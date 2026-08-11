import json
from pathlib import Path
from typing import Any, Dict


class UsageTracker:
    def __init__(self, usage_file: str = "outputs/usage.json"):
        self.file_path = Path(usage_file)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.usage_data: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if self.file_path.exists():
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def record_usage(self, provider: str, model: str, prompt_tokens: int = 0, completion_tokens: int = 0, requests: int = 1):
        if provider not in self.usage_data:
            self.usage_data[provider] = {}
        if model not in self.usage_data[provider]:
            self.usage_data[provider][model] = {
                "requests": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }

        m = self.usage_data[provider][model]
        m["requests"] += requests
        m["prompt_tokens"] += prompt_tokens
        m["completion_tokens"] += completion_tokens
        m["total_tokens"] += (prompt_tokens + completion_tokens)

        self._save()

    def _save(self):
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(self.usage_data, f, indent=2)
