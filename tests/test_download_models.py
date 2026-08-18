import sys
from unittest.mock import patch

sys.path.insert(0, ".")
import scripts.download_models as dm


def test_default_model_ids_from_config(monkeypatch):
    class FakeConfig:
        annotation_model = "Ann/Model"
        training_models = [
            {"model_id": "Train/A"},
            {"model_id": "Train/B"},
        ]
        scold_model = {"model_id": "Scold/C"}

    monkeypatch.setattr(dm, "pipeline_config", FakeConfig())
    assert dm.default_model_ids() == ["Ann/Model", "Train/A", "Train/B", "Scold/C"]


def test_default_model_ids_dedupes(monkeypatch):
    class FakeConfig:
        annotation_model = "Shared/X"
        training_models = [{"model_id": "Shared/X"}, {"model_id": "Train/B"}]
        scold_model = {"model_id": "Scold/C"}

    monkeypatch.setattr(dm, "pipeline_config", FakeConfig())
    assert dm.default_model_ids() == ["Shared/X", "Train/B", "Scold/C"]


def test_main_skips_cached_models(monkeypatch):
    monkeypatch.setattr(dm, "is_model_cached", lambda mid, **k: True)
    mock_download = patch.object(dm, "ensure_model_downloaded")
    with mock_download as m:
        monkeypatch.setattr(sys, "argv", ["download_models.py", "--models", "Qwen/Qwen2.5-VL-3B-Instruct"])
        dm.main()
    m.assert_not_called()


def test_main_downloads_missing_models(monkeypatch):
    monkeypatch.setattr(dm, "is_model_cached", lambda mid, **k: False)
    mock_download = patch.object(dm, "ensure_model_downloaded", return_value="/tmp/models/base/Qwen__Qwen2.5-VL-3B-Instruct")
    with mock_download as m:
        monkeypatch.setattr(sys, "argv", ["download_models.py", "--models", "Qwen/Qwen2.5-VL-3B-Instruct"])
        dm.main()
    m.assert_called_once()
    assert m.call_args.args[0] == "Qwen/Qwen2.5-VL-3B-Instruct"


def test_main_all_uses_default_model_ids(monkeypatch):
    monkeypatch.setattr(dm, "default_model_ids", lambda: ["M/1", "M/2"])
    monkeypatch.setattr(dm, "is_model_cached", lambda mid, **k: False)
    mock_download = patch.object(dm, "ensure_model_downloaded", return_value="/tmp/cache")
    with mock_download as m:
        monkeypatch.setattr(sys, "argv", ["download_models.py", "--all"])
        dm.main()
    assert m.call_count == 2
    assert m.call_args_list[0].args[0] == "M/1"
    assert m.call_args_list[1].args[0] == "M/2"


def test_main_exits_on_failure(monkeypatch):
    monkeypatch.setattr(dm, "is_model_cached", lambda mid, **k: False)
    with patch.object(dm, "ensure_model_downloaded", side_effect=RuntimeError("boom")):
        monkeypatch.setattr(sys, "argv", ["download_models.py", "--models", "Broken/X"])
        try:
            dm.main()
        except SystemExit as exc:
            assert exc.code == 1
        else:
            raise AssertionError("Expected SystemExit(1) when a model download fails")