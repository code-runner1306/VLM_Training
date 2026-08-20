import sys
from unittest.mock import MagicMock

sys.path.insert(0, ".")
import training.src.trainer as t


def _base_config():
    return {
        "model": {"key": "qwen25vl_3b"},
        "quantization": {"enabled": False},
        "adaptation": {"strategy": "llm_projector", "r": 16, "lora_alpha": 32, "lora_dropout": 0.05},
        "training": {
            "num_epochs": 2,
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "batch_size": 1,
            "gradient_accumulation_steps": 4,
            "max_grad_norm": 1.0,
            "warmup_ratio": 0.03,
            "lr_scheduler_type": "cosine",
            "bf16": False,
            "fp16": False,
            "logging_steps": 5,
            "eval_steps": 10,
            "save_steps": 10,
            "gradient_checkpointing": False,
            "cuda_memory_fraction": 0.9,
        },
        "early_stopping": {"enabled": True, "patience": 2, "monitor": "val_loss", "mode": "min"},
        "checkpoint": {"save_total_limit": 2},
        "data": {"user_prompt": "test prompt"},
    }


def _fake_adapter():
    adapter = MagicMock()
    adapter.model_key = "qwen25vl_3b"
    adapter.model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    adapter.load_model_and_processor.return_value = (MagicMock(), MagicMock())
    return adapter


class FakeTrainingArguments:
    last = None

    def __init__(self, **kwargs):
        FakeTrainingArguments.last = kwargs


class FakeDataset:
    def __len__(self):
        return 10

    def __getitem__(self, idx):
        return {}


class FakeTrainer:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.callbacks = kwargs.get("callbacks", [])
        self.model = kwargs["model"]
        self.state = MagicMock()
        self.state.log_history = []
        self.resumed_from = None
        self.saved_to = None
        FakeTrainer.instances.append(self)

    def train(self, resume_from_checkpoint=None):
        self.resumed_from = resume_from_checkpoint

    def save_model(self, out_dir):
        self.saved_to = out_dir


def _last_trainer():
    return FakeTrainer.instances[-1]


def _patch_trainer_deps(monkeypatch):
    FakeTrainingArguments.last = None
    FakeTrainer.instances = []
    monkeypatch.setattr(t, "TrainingArguments", FakeTrainingArguments)
    monkeypatch.setattr(t, "Trainer", FakeTrainer)
    monkeypatch.setattr(t, "VLMDataset", MagicMock(side_effect=lambda *a, **k: FakeDataset()))
    monkeypatch.setattr(t, "VLMDataCollator", MagicMock(return_value=MagicMock(max_length=2048)))
    monkeypatch.setattr(t, "get_peft_model", MagicMock(side_effect=lambda model, cfg: model))
    monkeypatch.setattr(t, "plot_training_curves", MagicMock())


def test_train_vlm_builds_training_arguments_from_config(monkeypatch, tmp_path):
    _patch_trainer_deps(monkeypatch)
    result = t.train_vlm(
        adapter=_fake_adapter(),
        config=_base_config(),
        experiment_name="exp-v1",
        train_manifest=str(tmp_path / "train.jsonl"),
        val_manifest=str(tmp_path / "val.jsonl"),
    )

    args = FakeTrainingArguments.last
    assert args["num_train_epochs"] == 2
    assert args["learning_rate"] == 1e-4
    assert args["weight_decay"] == 0.01
    assert args["per_device_train_batch_size"] == 1
    assert args["gradient_accumulation_steps"] == 4
    assert args["max_grad_norm"] == 1.0
    assert args["warmup_ratio"] == 0.03
    assert args["lr_scheduler_type"] == "cosine"
    assert args["eval_strategy"] == "steps"
    assert args["load_best_model_at_end"] is True
    assert args["metric_for_best_model"] == "eval_loss"
    assert args["greater_is_better"] is False
    assert args["save_strategy"] == "steps"
    assert args["save_total_limit"] == 2
    assert args["remove_unused_columns"] is False
    assert "checkpoints" in args["output_dir"] and "exp-v1" in args["output_dir"]

    trainer = _last_trainer()
    assert trainer.saved_to.endswith(os_sep_join("outputs", "exp-v1", "adapter"))

    assert result["experiment"] == "exp-v1"
    assert "param_counts" in result
    assert "total_training_time_s" in result
    assert "peak_vram_gb" in result
    assert result["early_stopping"]["early_stopping_enabled"] is True
    assert result["early_stopping"]["monitored_metric"] == "val_loss"


def test_train_vlm_registers_early_stopping_callback(monkeypatch, tmp_path):
    _patch_trainer_deps(monkeypatch)
    t.train_vlm(
        adapter=_fake_adapter(),
        config=_base_config(),
        experiment_name="exp-v1",
        train_manifest=str(tmp_path / "train.jsonl"),
        val_manifest=str(tmp_path / "val.jsonl"),
    )
    trainer = _last_trainer()
    assert len(trainer.callbacks) == 1
    assert isinstance(trainer.callbacks[0], t._EarlyStoppingRecorder)


def test_train_vlm_resume_uses_latest_checkpoint(monkeypatch, tmp_path):
    _patch_trainer_deps(monkeypatch)
    ckpt = str(tmp_path / "checkpoints" / "exp-v1" / "checkpoint-100")
    monkeypatch.setattr(t, "_find_latest_checkpoint", MagicMock(return_value=ckpt))

    t.train_vlm(
        adapter=_fake_adapter(),
        config=_base_config(),
        experiment_name="exp-v1",
        train_manifest=str(tmp_path / "train.jsonl"),
        val_manifest=str(tmp_path / "val.jsonl"),
        resume=True,
    )
    assert _last_trainer().resumed_from == ckpt


def test_train_vlm_no_early_stopping_disables_eval(monkeypatch, tmp_path):
    _patch_trainer_deps(monkeypatch)
    config = _base_config()
    config["early_stopping"]["enabled"] = False

    t.train_vlm(
        adapter=_fake_adapter(),
        config=config,
        experiment_name="exp-v1",
        train_manifest=str(tmp_path / "train.jsonl"),
        val_manifest=str(tmp_path / "val.jsonl"),
    )
    args = FakeTrainingArguments.last
    assert args["eval_strategy"] == "no"
    assert args["load_best_model_at_end"] is False
    assert args["metric_for_best_model"] is None
    assert len(_last_trainer().callbacks) == 0


def test_train_vlm_smoke_test_sets_single_epoch(monkeypatch, tmp_path):
    _patch_trainer_deps(monkeypatch)
    t.train_vlm(
        adapter=_fake_adapter(),
        config=_base_config(),
        experiment_name="exp-v1",
        train_manifest=str(tmp_path / "train.jsonl"),
        val_manifest=str(tmp_path / "val.jsonl"),
        smoke_test=True,
    )
    assert FakeTrainingArguments.last["num_train_epochs"] == 1


import os


def os_sep_join(*parts):
    return os.path.join(*parts)