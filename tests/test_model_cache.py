import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from training.src import model_cache


def test_sanitize_model_name(monkeypatch):
    assert model_cache.sanitize_model_name("Qwen/Qwen2.5-VL-3B-Instruct") == "Qwen__Qwen2.5-VL-3B-Instruct"
    assert model_cache.sanitize_model_name("Org:model/name") == "Org-model__name"


def test_local_model_dir_resolution(monkeypatch, tmp_path):
    monkeypatch.setattr(model_cache, "models_base_dir", lambda: tmp_path / "models" / "base")
    result = model_cache.local_model_dir("Qwen/Qwen2.5-VL-3B-Instruct")
    assert result == tmp_path / "models" / "base" / "Qwen__Qwen2.5-VL-3B-Instruct"


def test_is_model_cached_checks_config_json(monkeypatch, tmp_path):
    monkeypatch.setattr(model_cache, "models_base_dir", lambda: tmp_path / "models" / "base")
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    assert not model_cache.is_model_cached(model_id)

    cache_dir = model_cache.local_model_dir(model_id)
    cache_dir.mkdir(parents=True)
    assert not model_cache.is_model_cached(model_id)

    (cache_dir / "config.json").write_text(json.dumps({"architectures": ["Qwen2_5_VLForConditionalGeneration"]}))
    assert model_cache.is_model_cached(model_id)


def test_ensure_model_downloaded_skips_when_cached(monkeypatch, tmp_path):
    monkeypatch.setattr(model_cache, "models_base_dir", lambda: tmp_path / "models" / "base")
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    cache_dir = model_cache.local_model_dir(model_id)
    cache_dir.mkdir(parents=True)
    (cache_dir / "config.json").write_text("{}")

    with patch("training.src.model_cache.snapshot_download") as mock_download:
        result = model_cache.ensure_model_downloaded(model_id, token="tok")
    mock_download.assert_not_called()
    assert result == cache_dir


def test_ensure_model_downloaded_downloads_on_miss(monkeypatch, tmp_path):
    monkeypatch.setattr(model_cache, "models_base_dir", lambda: tmp_path / "models" / "base")
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    cache_dir = model_cache.local_model_dir(model_id)

    def fake_download(repo_id, local_dir, token=None, force_download=False):
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        (Path(local_dir) / "config.json").write_text("{}")

    with patch("training.src.model_cache.snapshot_download", side_effect=fake_download) as mock_download:
        result = model_cache.ensure_model_downloaded(model_id, token="tok", force=True)
    mock_download.assert_called_once()
    assert mock_download.call_args.kwargs["local_dir"] == str(cache_dir)
    assert mock_download.call_args.kwargs["token"] == "tok"
    assert result == cache_dir


def test_ensure_model_downloaded_raises_when_incomplete(monkeypatch, tmp_path):
    monkeypatch.setattr(model_cache, "models_base_dir", lambda: tmp_path / "models" / "base")
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    with patch("training.src.model_cache.snapshot_download", return_value=None):
        try:
            model_cache.ensure_model_downloaded(model_id, token="tok")
        except RuntimeError as exc:
            assert "no config.json" in str(exc)
        else:
            raise AssertionError("Expected RuntimeError for incomplete download")


def test_load_model_from_cache_or_hub_prefers_local_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(model_cache, "models_base_dir", lambda: tmp_path / "models" / "base")
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    cache_dir = model_cache.local_model_dir(model_id)
    cache_dir.mkdir(parents=True)
    (cache_dir / "config.json").write_text("{}")

    loaded_from = {}

    class FakeCls:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            loaded_from["path"] = path
            loaded_from["local_files_only"] = kwargs.get("local_files_only")
            return object()

    result = model_cache.load_model_from_cache_or_hub(FakeCls, model_id, trust_remote_code=True)
    assert loaded_from["path"] == str(cache_dir)
    assert loaded_from["local_files_only"] is True
    assert result is not None


def test_load_model_from_cache_or_hub_falls_back_to_hub_then_download(monkeypatch, tmp_path):
    monkeypatch.setattr(model_cache, "models_base_dir", lambda: tmp_path / "models" / "base")
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    calls = []

    def fake_download(repo_id, local_dir, token=None, force_download=False):
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        (Path(local_dir) / "config.json").write_text("{}")

    class FakeCls:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append((path, kwargs.get("local_files_only")))
            if path == model_id:
                raise RuntimeError("not in hub cache")
            return "loaded-from-cache"

    with patch("training.src.model_cache.snapshot_download", side_effect=fake_download):
        result = model_cache.load_model_from_cache_or_hub(FakeCls, model_id, token="tok")

    assert result == "loaded-from-cache"
    assert calls[0] == (model_id, True)
    assert calls[1][0] == str(model_cache.local_model_dir(model_id))