import os
import tempfile
import torch
import torch.nn as nn
from pathlib import Path
from training.src.early_stopping import EarlyStopping
from training.src.checkpoint import CheckpointManager


class SimpleDummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


def test_early_stopping_min_mode_triggers_patience():
    es = EarlyStopping(monitor="val_loss", mode="min", patience=2, min_delta=0.01, verbose=False)

    assert not es.step({"val_loss": 0.50}, epoch=1, step=100)
    assert es.best_score == 0.50
    assert es.best_epoch == 1
    assert es.counter == 0

    assert not es.step({"val_loss": 0.40}, epoch=2, step=200)
    assert es.best_score == 0.40
    assert es.best_epoch == 2
    assert es.counter == 0

    # No improvement (0.42 > 0.40) -> counter=1
    assert not es.step({"val_loss": 0.42}, epoch=3, step=300)
    assert es.counter == 1
    assert not es.should_stop

    # No improvement (0.45 > 0.40) -> counter=2 -> should stop!
    assert es.step({"val_loss": 0.45}, epoch=4, step=400)
    assert es.counter == 2
    assert es.should_stop
    assert es.stopped_epoch == 4
    assert "Patience exceeded" in es.stop_reason


def test_early_stopping_max_mode_triggers_patience():
    es = EarlyStopping(monitor="val_macro_f1", mode="max", patience=3, min_delta=0.005, verbose=False)

    assert not es.step({"val_macro_f1": 0.75}, epoch=1, step=100)
    assert es.best_score == 0.75

    assert not es.step({"val_macro_f1": 0.85}, epoch=2, step=200)
    assert es.best_score == 0.85

    assert not es.step({"val_macro_f1": 0.84}, epoch=3, step=300)
    assert es.counter == 1

    assert not es.step({"val_macro_f1": 0.83}, epoch=4, step=400)
    assert es.counter == 2

    # Third failure to improve -> stops
    assert es.step({"val_macro_f1": 0.82}, epoch=5, step=500)
    assert es.counter == 3
    assert es.should_stop
    assert es.best_score == 0.85
    assert es.best_epoch == 2


def test_early_stopping_auto_mode_inference():
    es_loss = EarlyStopping(monitor="val_loss", mode="auto", verbose=False)
    assert es_loss.mode == "min"

    es_err = EarlyStopping(monitor="validation_error_rate", mode="auto", verbose=False)
    assert es_err.mode == "min"

    es_f1 = EarlyStopping(monitor="val_macro_f1", mode="auto", verbose=False)
    assert es_f1.mode == "max"

    es_acc = EarlyStopping(monitor="accuracy", mode="auto", verbose=False)
    assert es_acc.mode == "max"


def test_early_stopping_min_delta_filtering():
    es = EarlyStopping(monitor="val_loss", mode="min", patience=2, min_delta=0.05, verbose=False)

    es.step({"val_loss": 0.50}, epoch=1, step=100)
    assert es.best_score == 0.50

    # 0.48 is smaller than 0.50, but delta (0.02) < min_delta (0.05) -> counted as non-improvement
    es.step({"val_loss": 0.48}, epoch=2, step=200)
    assert es.counter == 1
    assert es.best_score == 0.50

    # 0.42 improves beyond 0.50 - 0.05 = 0.45 -> reset counter
    es.step({"val_loss": 0.42}, epoch=3, step=300)
    assert es.counter == 0
    assert es.best_score == 0.42


def test_early_stopping_stopping_threshold():
    es = EarlyStopping(monitor="val_loss", mode="min", stopping_threshold=0.10, verbose=False)

    assert not es.step({"val_loss": 0.30}, epoch=1, step=100)
    assert not es.should_stop

    # Reaching stopping threshold (0.08 <= 0.10) triggers immediate stop
    assert es.step({"val_loss": 0.08}, epoch=2, step=200)
    assert es.should_stop
    assert "Stopping threshold" in es.stop_reason


def test_early_stopping_divergence_threshold_and_nan():
    # Divergence test
    es_div = EarlyStopping(monitor="val_loss", mode="min", divergence_threshold=10.0, verbose=False)
    assert es_div.step({"val_loss": 15.5}, epoch=1, step=100)
    assert es_div.should_stop
    assert "diverged" in es_div.stop_reason

    # NaN test
    es_nan = EarlyStopping(monitor="val_loss", mode="min", verbose=False)
    assert es_nan.step({"val_loss": float("nan")}, epoch=1, step=100)
    assert es_nan.should_stop
    assert "NaN" in es_nan.stop_reason


def test_early_stopping_state_dict_serialization_and_resume():
    es1 = EarlyStopping(monitor="val_loss", mode="min", patience=3, min_delta=0.01, verbose=False)
    es1.step({"val_loss": 0.50}, epoch=1, step=100)
    es1.step({"val_loss": 0.40}, epoch=2, step=200)
    es1.step({"val_loss": 0.42}, epoch=3, step=300)
    assert es1.counter == 1

    state = es1.state_dict()

    # Create new instance and load state
    es2 = EarlyStopping(monitor="val_loss", mode="min", patience=3, verbose=False)
    es2.load_state_dict(state)

    assert es2.best_score == 0.40
    assert es2.best_epoch == 2
    assert es2.counter == 1
    assert len(es2.history) == 3

    # Step on resumed instance
    es2.step({"val_loss": 0.43}, epoch=4, step=400)
    assert es2.counter == 2
    assert not es2.should_stop

    es2.step({"val_loss": 0.44}, epoch=5, step=500)
    assert es2.counter == 3
    assert es2.should_stop


def test_metric_extraction_with_nested_and_alias_keys():
    es = EarlyStopping(monitor="overall.accuracy", mode="max", verbose=False)
    metrics_nested = {"overall": {"accuracy": 0.945, "count": 100}}
    assert es.extract_metric_value(metrics_nested) == 0.945

    es_alias = EarlyStopping(monitor="val_loss", mode="min", verbose=False)
    assert es_alias.extract_metric_value({"eval_loss": 0.22}) == 0.22
    assert es_alias.extract_metric_value({"loss_val": 0.18}) == 0.18

    # Scalar input
    assert es_alias.extract_metric_value(0.15) == 0.15


def test_checkpoint_manager_best_checkpoint_and_restoration():
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_mgr = CheckpointManager(checkpoint_dir=tmpdir, max_to_keep=2)
        model = SimpleDummyModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Save step checkpoints
        ckpt_mgr.save_checkpoint(
            model=model,
            processor=None,
            optimizer=optimizer,
            scheduler=None,
            step=100,
            epoch=1,
            training_config={"lr": 1e-3},
            metrics={"val_loss": 0.5},
        )
        ckpt_mgr.save_checkpoint(
            model=model,
            processor=None,
            optimizer=optimizer,
            scheduler=None,
            step=200,
            epoch=2,
            training_config={"lr": 1e-3},
            metrics={"val_loss": 0.4},
        )
        ckpt_mgr.save_checkpoint(
            model=model,
            processor=None,
            optimizer=optimizer,
            scheduler=None,
            step=300,
            epoch=3,
            training_config={"lr": 1e-3},
            metrics={"val_loss": 0.6},
        )

        # Checkpoint-100 should be pruned because max_to_keep=2
        assert not os.path.exists(os.path.join(tmpdir, "checkpoint-100"))
        assert os.path.exists(os.path.join(tmpdir, "checkpoint-200"))
        assert os.path.exists(os.path.join(tmpdir, "checkpoint-300"))

        # Save dedicated best checkpoint
        best_path = ckpt_mgr.save_best_checkpoint(
            model=model,
            processor=None,
            optimizer=optimizer,
            scheduler=None,
            step=200,
            epoch=2,
            training_config={"lr": 1e-3},
            metrics={"val_loss": 0.4},
            best_score=0.4,
        )

        assert os.path.exists(best_path)
        assert ckpt_mgr.get_best_checkpoint() == best_path

        # Add more step checkpoints and verify checkpoint-best is never pruned
        ckpt_mgr.save_checkpoint(
            model=model,
            processor=None,
            optimizer=optimizer,
            scheduler=None,
            step=400,
            epoch=4,
            training_config={"lr": 1e-3},
            metrics={"val_loss": 0.7},
        )
        assert os.path.exists(best_path)

        # Test weight restoration
        restored = ckpt_mgr.restore_best_weights(model)
        assert restored


def test_pipeline_config_early_stopping_integration():
    from config import PipelineConfig

    custom_cfg = PipelineConfig(
        early_stopping_enabled=True,
        early_stopping_monitor="val_macro_f1",
        early_stopping_mode="max",
        early_stopping_patience=5,
        early_stopping_min_delta=0.002,
        early_stopping_restore_best_weights=True,
    )

    es = EarlyStopping(
        monitor=custom_cfg.early_stopping_monitor,
        mode=custom_cfg.early_stopping_mode,
        patience=custom_cfg.early_stopping_patience,
        min_delta=custom_cfg.early_stopping_min_delta,
        restore_best_weights=custom_cfg.early_stopping_restore_best_weights,
        verbose=False,
    )

    assert es.monitor == "val_macro_f1"
    assert es.mode == "max"
    assert es.patience == 5
    assert es.min_delta == 0.002
    assert es.restore_best_weights is True

