import math
import logging
from typing import Dict, Any, Optional, Union

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early Stopping monitor to stop training when a monitored metric has stopped improving.
    Supports best checkpoint tracking, weight restoration, divergence detection, and state serialization.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        mode: str = "min",
        patience: int = 3,
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
        baseline: Optional[float] = None,
        stopping_threshold: Optional[float] = None,
        divergence_threshold: Optional[float] = None,
        verbose: bool = True,
    ):
        """
        Args:
            monitor: Quantity to be monitored (e.g. 'val_loss', 'val_macro_f1', 'loss').
            mode: One of {'min', 'max', 'auto'}.
                  In 'min' mode, training stops when the quantity monitored has stopped decreasing.
                  In 'max' mode, training stops when the quantity monitored has stopped increasing.
                  In 'auto' mode, direction is inferred from the name of the monitor metric.
            patience: Number of evaluations with no improvement after which training will be stopped.
            min_delta: Minimum change in the monitored quantity to qualify as an improvement.
            restore_best_weights: Whether to restore model weights from the epoch with the best value.
            baseline: Baseline value for the monitored quantity. Training won't stop before reaching baseline.
            stopping_threshold: Absolute value threshold to stop training immediately if met.
            divergence_threshold: Absolute value threshold to stop training immediately if diverged.
            verbose: Verbosity mode (prints progress to stdout/logger).
        """
        self.monitor = monitor
        self.patience = max(1, int(patience))
        self.min_delta = abs(float(min_delta))
        self.restore_best_weights = restore_best_weights
        self.baseline = float(baseline) if baseline is not None else None
        self.stopping_threshold = float(stopping_threshold) if stopping_threshold is not None else None
        self.divergence_threshold = float(divergence_threshold) if divergence_threshold is not None else None
        self.verbose = verbose

        # Determine optimization mode
        if mode == "auto":
            if any(term in monitor.lower() for term in ["loss", "error", "err"]):
                self.mode = "min"
            else:
                self.mode = "max"
        elif mode in ["min", "max"]:
            self.mode = mode
        else:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of ['min', 'max', 'auto'].")

        # Internal state
        self.best_score: Optional[float] = None
        self.best_epoch: Optional[int] = None
        self.best_step: Optional[int] = None
        self.best_checkpoint_path: Optional[str] = None
        self.counter: int = 0
        self.should_stop: bool = False
        self.stopped_epoch: Optional[int] = None
        self.stopped_step: Optional[int] = None
        self.stop_reason: Optional[str] = None
        self.history: list[Dict[str, Any]] = []

    def _is_improvement(self, current: float) -> bool:
        """Check if current score improves upon best_score considering min_delta and baseline."""
        if self.best_score is None:
            if self.baseline is not None:
                if self.mode == "min" and current > self.baseline:
                    return False
                if self.mode == "max" and current < self.baseline:
                    return False
            return True

        if self.mode == "min":
            return current < (self.best_score - self.min_delta)
        else:
            return current > (self.best_score + self.min_delta)

    def extract_metric_value(self, metrics: Union[Dict[str, Any], float, int]) -> Optional[float]:
        """Extract monitored float metric from scalar or dictionary (including nested keys)."""
        if isinstance(metrics, (int, float)):
            return float(metrics)

        if not isinstance(metrics, dict):
            return None

        # Direct key match
        if self.monitor in metrics:
            val = metrics[self.monitor]
            return float(val) if isinstance(val, (int, float)) else None

        # Case-insensitive match or alias handling
        for k, v in metrics.items():
            if k.lower() == self.monitor.lower():
                return float(v) if isinstance(v, (int, float)) else None

        # Handle nested structures like metrics["overall"]["accuracy"] or metrics["loss"]
        if "." in self.monitor:
            parts = self.monitor.split(".")
            curr = metrics
            for p in parts:
                if isinstance(curr, dict) and p in curr:
                    curr = curr[p]
                else:
                    return None
            return float(curr) if isinstance(curr, (int, float)) else None

        # Common fallback aliases
        aliases = {
            "val_loss": ["eval_loss", "validation_loss", "loss_val", "loss"],
            "val_accuracy": ["eval_accuracy", "validation_accuracy", "accuracy", "val_acc", "acc"],
            "val_macro_f1": ["macro_f1", "val_f1", "f1_macro", "eval_macro_f1"],
            "loss": ["train_loss", "training_loss"],
        }
        for alias in aliases.get(self.monitor, []):
            if alias in metrics:
                val = metrics[alias]
                return float(val) if isinstance(val, (int, float)) else None

        return None

    def step(
        self,
        metrics: Union[Dict[str, Any], float, int],
        epoch: int,
        step: int,
        model: Optional[Any] = None,
        processor: Optional[Any] = None,
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        ckpt_manager: Optional[Any] = None,
        training_config: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Evaluate current metrics against patience and thresholds, saving best checkpoint when improved.

        Args:
            metrics: Current step/epoch metrics dictionary or scalar value.
            epoch: Current training epoch (1-indexed).
            step: Current global training step.
            model: Optional model instance for saving best checkpoint / restoring weights.
            processor: Optional processor/tokenizer instance.
            optimizer: Optional optimizer.
            scheduler: Optional LR scheduler.
            ckpt_manager: Optional CheckpointManager instance for best checkpoint persistence.
            training_config: Optional configuration dictionary.

        Returns:
            bool: True if early stopping criteria is triggered and training should halt.
        """
        if self.should_stop:
            return True

        current = self.extract_metric_value(metrics)
        if current is None:
            if self.verbose:
                print(f"[EARLY STOPPING] Warning: Monitored metric '{self.monitor}' not found in metrics: {metrics}")
            return False

        # Check for NaN / Inf divergence
        if math.isnan(current) or math.isinf(current):
            self.should_stop = True
            self.stopped_epoch = epoch
            self.stopped_step = step
            self.stop_reason = f"Monitored metric '{self.monitor}' became NaN or Inf."
            if self.verbose:
                print(f"\n[EARLY STOPPING] ⚠️ {self.stop_reason} Halting training.")
            return True

        # Check divergence threshold
        if self.divergence_threshold is not None:
            diverged = (self.mode == "min" and current >= self.divergence_threshold) or (
                self.mode == "max" and current <= self.divergence_threshold
            )
            if diverged:
                self.should_stop = True
                self.stopped_epoch = epoch
                self.stopped_step = step
                self.stop_reason = (
                    f"Monitored metric '{self.monitor}' diverged past threshold: "
                    f"{current:.4f} (threshold: {self.divergence_threshold:.4f})."
                )
                if self.verbose:
                    print(f"\n[EARLY STOPPING] ⚠️ {self.stop_reason} Halting training.")
                return True

        # Record history
        record = {
            "epoch": epoch,
            "step": step,
            "metric": self.monitor,
            "value": current,
            "counter": self.counter,
        }

        # Check improvement
        if self._is_improvement(current):
            prev_best = self.best_score
            self.best_score = current
            self.best_epoch = epoch
            self.best_step = step
            self.counter = 0
            record["is_best"] = True

            if self.verbose:
                if prev_best is None:
                    print(
                        f"[EARLY STOPPING] Metric '{self.monitor}' initialized to {current:.4f} at epoch {epoch} (step {step})."
                    )
                else:
                    direction = "decreased" if self.mode == "min" else "increased"
                    delta = abs(current - prev_best)
                    print(
                        f"[EARLY STOPPING] Metric '{self.monitor}' {direction} from {prev_best:.4f} to {current:.4f} (Δ={delta:.4f}). Best score updated!"
                    )

            # Persist best checkpoint via CheckpointManager if available
            if ckpt_manager is not None and model is not None:
                save_metrics = metrics if isinstance(metrics, dict) else {self.monitor: current}
                saved_path = ckpt_manager.save_best_checkpoint(
                    model=model,
                    processor=processor,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=step,
                    epoch=epoch,
                    training_config=training_config or {},
                    metrics=save_metrics,
                    best_score=current,
                    early_stopping_state=self.state_dict(),
                )
                self.best_checkpoint_path = saved_path
        else:
            self.counter += 1
            record["is_best"] = False

            if self.verbose:
                best_str = f"{self.best_score:.4f}" if self.best_score is not None else "N/A"
                print(
                    f"[EARLY STOPPING] Counter: {self.counter}/{self.patience} "
                    f"(no improvement in '{self.monitor}' for {self.counter} evaluation{'s' if self.counter > 1 else ''}, best: {best_str} at epoch {self.best_epoch})."
                )

            # Check patience exhaustion
            if self.counter >= self.patience:
                self.should_stop = True
                self.stopped_epoch = epoch
                self.stopped_step = step
                self.stop_reason = (
                    f"Patience exceeded: '{self.monitor}' did not improve for {self.patience} consecutive evaluations "
                    f"(best: {self.best_score:.4f} at epoch {self.best_epoch}, step {self.best_step})."
                )
                if self.verbose:
                    print(f"\n[EARLY STOPPING] ⏹️  {self.stop_reason} Triggering early stop.")

        # Check stopping threshold milestone
        if not self.should_stop and self.stopping_threshold is not None:
            milestone_met = (self.mode == "min" and current <= self.stopping_threshold) or (
                self.mode == "max" and current >= self.stopping_threshold
            )
            if milestone_met:
                self.should_stop = True
                self.stopped_epoch = epoch
                self.stopped_step = step
                self.stop_reason = (
                    f"Stopping threshold milestone met: {current:.4f} "
                    f"({'<=' if self.mode == 'min' else '>='} {self.stopping_threshold:.4f})."
                )
                if self.verbose:
                    print(f"\n[EARLY STOPPING] 🎯 {self.stop_reason} Training goal achieved.")

        self.history.append(record)

        # If early stopping triggered and restore_best_weights enabled
        if self.should_stop and self.restore_best_weights:
            if ckpt_manager is not None and model is not None:
                ckpt_manager.restore_best_weights(model=model, processor=processor)

        return self.should_stop

    def restore_weights_if_needed(self, model: Any, ckpt_manager: Any, processor: Optional[Any] = None) -> bool:
        """Explicitly restore best weights into model if restore_best_weights is True."""
        if self.restore_best_weights and ckpt_manager is not None and model is not None:
            return ckpt_manager.restore_best_weights(model=model, processor=processor)
        return False

    def state_dict(self) -> Dict[str, Any]:
        """Serialize internal state for checkpointing and resuming."""
        return {
            "monitor": self.monitor,
            "mode": self.mode,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "restore_best_weights": self.restore_best_weights,
            "baseline": self.baseline,
            "stopping_threshold": self.stopping_threshold,
            "divergence_threshold": self.divergence_threshold,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "best_step": self.best_step,
            "best_checkpoint_path": self.best_checkpoint_path,
            "counter": self.counter,
            "should_stop": self.should_stop,
            "stopped_epoch": self.stopped_epoch,
            "stopped_step": self.stopped_step,
            "stop_reason": self.stop_reason,
            "history": self.history,
        }

    def load_state_dict(self, state: Dict[str, Any]):
        """Restore internal state from serialized checkpoint dictionary."""
        self.monitor = state.get("monitor", self.monitor)
        self.mode = state.get("mode", self.mode)
        self.patience = state.get("patience", self.patience)
        self.min_delta = state.get("min_delta", self.min_delta)
        self.restore_best_weights = state.get("restore_best_weights", self.restore_best_weights)
        self.baseline = state.get("baseline", self.baseline)
        self.stopping_threshold = state.get("stopping_threshold", self.stopping_threshold)
        self.divergence_threshold = state.get("divergence_threshold", self.divergence_threshold)
        self.best_score = state.get("best_score", self.best_score)
        self.best_epoch = state.get("best_epoch", self.best_epoch)
        self.best_step = state.get("best_step", self.best_step)
        self.best_checkpoint_path = state.get("best_checkpoint_path", self.best_checkpoint_path)
        self.counter = state.get("counter", 0)
        self.should_stop = state.get("should_stop", False)
        self.stopped_epoch = state.get("stopped_epoch", None)
        self.stopped_step = state.get("stopped_step", None)
        self.stop_reason = state.get("stop_reason", None)
        self.history = state.get("history", [])

    def get_summary(self) -> Dict[str, Any]:
        """Return structured summary dictionary of early stopping results."""
        return {
            "early_stopping_enabled": True,
            "monitored_metric": self.monitor,
            "mode": self.mode,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "early_stopped": self.should_stop,
            "stopped_epoch": self.stopped_epoch,
            "stopped_step": self.stopped_step,
            "stop_reason": self.stop_reason,
            "best_score": round(self.best_score, 6) if self.best_score is not None else None,
            "best_epoch": self.best_epoch,
            "best_step": self.best_step,
            "best_checkpoint_path": self.best_checkpoint_path,
            "total_evaluations": len(self.history),
        }
