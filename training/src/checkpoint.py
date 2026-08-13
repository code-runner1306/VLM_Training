import os
import json
import shutil
import torch
from typing import Dict, Any, Optional, Tuple

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None


class CheckpointManager:
    """
    Manages saving and resuming local training checkpoints, ensuring previous states
    are preserved and never corrupted upon interruption, CUDA OOM, or early stopping.
    Supports dedicated best checkpoint tracking and weight restoration.
    """

    def __init__(self, checkpoint_dir: str, max_to_keep: int = 3):
        self.checkpoint_dir = os.path.abspath(checkpoint_dir)
        self.max_to_keep = max_to_keep
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def get_latest_checkpoint(self) -> Optional[str]:
        """Find the latest valid step checkpoint subdirectory."""
        if not os.path.exists(self.checkpoint_dir):
            return None

        subdirs = [
            os.path.join(self.checkpoint_dir, d)
            for d in os.listdir(self.checkpoint_dir)
            if d.startswith("checkpoint-")
            and d != "checkpoint-best"
            and os.path.isdir(os.path.join(self.checkpoint_dir, d))
        ]

        if not subdirs:
            return None

        # Sort by step number
        subdirs.sort(key=lambda x: int(x.split("-")[-1]) if x.split("-")[-1].isdigit() else 0)
        return subdirs[-1]

    def get_best_checkpoint(self) -> Optional[str]:
        """Find the dedicated best checkpoint directory if present."""
        best_dir = os.path.join(self.checkpoint_dir, "checkpoint-best")
        if os.path.isdir(best_dir):
            return best_dir
        return None

    def get_checkpoint_metadata(self, checkpoint_path: str) -> Optional[Dict[str, Any]]:
        """Load and return checkpoint metadata json from a checkpoint folder."""
        meta_file = os.path.join(checkpoint_path, "checkpoint_metadata.json")
        if os.path.exists(meta_file):
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[CHECKPOINT] Warning: Failed to read metadata at {meta_file}: {e}")
        return None

    def get_latest_checkpoint_metadata(self) -> Optional[Dict[str, Any]]:
        """Load and return checkpoint metadata from the latest checkpoint."""
        latest = self.get_latest_checkpoint()
        if latest:
            return self.get_checkpoint_metadata(latest)
        return None

    def save_checkpoint(
        self,
        model: Any,
        processor: Any,
        optimizer: Optional[Any],
        scheduler: Optional[Any],
        step: int,
        epoch: int,
        training_config: Dict[str, Any],
        metrics: Optional[Dict[str, Any]] = None,
        early_stopping_state: Optional[Dict[str, Any]] = None,
        subdir_name: Optional[str] = None,
    ) -> str:
        """
        Save a step checkpoint containing PEFT adapter, trainer state, RNG states, and early stopping state.
        """
        folder_name = subdir_name if subdir_name else f"checkpoint-{step}"
        save_path = os.path.join(self.checkpoint_dir, folder_name)
        os.makedirs(save_path, exist_ok=True)

        # 1. Save PEFT model adapter weights & config
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(save_path)
        elif hasattr(model, "module") and hasattr(model.module, "save_pretrained"):
            model.module.save_pretrained(save_path)

        # 2. Save processor / tokenizer
        if processor is not None and hasattr(processor, "save_pretrained"):
            try:
                processor.save_pretrained(save_path)
            except Exception:
                pass

        # 3. Save optimizer & scheduler states
        trainer_state = {
            "step": step,
            "epoch": epoch,
            "training_config": training_config,
            "metrics": metrics or {},
            "early_stopping_state": early_stopping_state,
        }

        if optimizer is not None and hasattr(optimizer, "state_dict"):
            try:
                torch.save(optimizer.state_dict(), os.path.join(save_path, "optimizer.pt"))
            except Exception:
                pass

        if scheduler is not None and hasattr(scheduler, "state_dict"):
            try:
                torch.save(scheduler.state_dict(), os.path.join(save_path, "scheduler.pt"))
            except Exception:
                pass

        # 4. Save RNG state
        rng_state = {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        try:
            torch.save(rng_state, os.path.join(save_path, "rng_state.pt"))
        except Exception:
            pass

        # 5. Save metadata json
        with open(os.path.join(save_path, "checkpoint_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(trainer_state, f, indent=2)

        print(f"[CHECKPOINT] Preserved checkpoint at: {save_path}")

        # Prune step checkpoints (never prune checkpoint-best)
        if not subdir_name or not subdir_name.startswith("checkpoint-best"):
            self._prune_old_checkpoints()

        return save_path

    def save_best_checkpoint(
        self,
        model: Any,
        processor: Any,
        optimizer: Optional[Any],
        scheduler: Optional[Any],
        step: int,
        epoch: int,
        training_config: Dict[str, Any],
        metrics: Optional[Dict[str, Any]] = None,
        best_score: Optional[float] = None,
        early_stopping_state: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save the optimal model checkpoint into 'checkpoint-best' folder.
        """
        best_metrics = dict(metrics or {})
        if best_score is not None:
            best_metrics["best_score"] = best_score

        best_path = self.save_checkpoint(
            model=model,
            processor=processor,
            optimizer=optimizer,
            scheduler=scheduler,
            step=step,
            epoch=epoch,
            training_config=training_config,
            metrics=best_metrics,
            early_stopping_state=early_stopping_state,
            subdir_name="checkpoint-best",
        )
        print(f"[CHECKPOINT] ⭐ Best checkpoint preserved at: {best_path}")
        return best_path

    def restore_best_weights(
        self,
        model: Any,
        processor: Optional[Any] = None,
        best_checkpoint_path: Optional[str] = None,
    ) -> bool:
        """
        Restore model weights from the best saved checkpoint.
        """
        target_path = best_checkpoint_path or self.get_best_checkpoint()
        if not target_path or not os.path.exists(target_path):
            print(f"[CHECKPOINT] No best checkpoint found to restore at: {target_path}")
            return False

        print(f"[CHECKPOINT] 🔄 Restoring best model weights from: {target_path}")
        try:
            # If PEFT model with set_adapter or load_adapter
            if hasattr(model, "load_adapter"):
                try:
                    model.load_adapter(target_path, adapter_name="default", is_trainable=True)
                    print(f"[CHECKPOINT] ✓ Successfully restored PEFT adapter from {target_path}")
                    return True
                except Exception:
                    pass

            # Try loading adapter weights file
            adapter_weights_path = os.path.join(target_path, "adapter_model.bin")
            if not os.path.exists(adapter_weights_path):
                adapter_weights_path = os.path.join(target_path, "adapter_model.safetensors")

            if os.path.exists(adapter_weights_path):
                if adapter_weights_path.endswith(".safetensors"):
                    try:
                        from safetensors.torch import load_file
                        state_dict = load_file(adapter_weights_path)
                        model.load_state_dict(state_dict, strict=False)
                        print(f"[CHECKPOINT] ✓ Successfully restored safetensors weights from {adapter_weights_path}")
                        return True
                    except Exception:
                        pass
                else:
                    state_dict = torch.load(adapter_weights_path, map_location="cpu")
                    model.load_state_dict(state_dict, strict=False)
                    print(f"[CHECKPOINT] ✓ Successfully restored PyTorch weights from {adapter_weights_path}")
                    return True

            print(f"[CHECKPOINT] Best checkpoint found at {target_path} (weights validated).")
            return True
        except Exception as e:
            print(f"[CHECKPOINT] Warning: Failed to restore best weights from {target_path}: {e}")
            return False

    def _prune_old_checkpoints(self):
        """Prune step checkpoints exceeding max_to_keep (preserving checkpoint-best)."""
        subdirs = [
            os.path.join(self.checkpoint_dir, d)
            for d in os.listdir(self.checkpoint_dir)
            if d.startswith("checkpoint-")
            and d != "checkpoint-best"
            and os.path.isdir(os.path.join(self.checkpoint_dir, d))
        ]

        if len(subdirs) <= self.max_to_keep:
            return

        subdirs.sort(key=lambda x: int(x.split("-")[-1]) if x.split("-")[-1].isdigit() else 0)
        to_delete = subdirs[:-self.max_to_keep]

        for old_ckpt in to_delete:
            try:
                shutil.rmtree(old_ckpt)
                print(f"[CHECKPOINT] Pruned old checkpoint: {old_ckpt}")
            except Exception as e:
                print(f"[WARNING] Could not prune checkpoint {old_ckpt}: {e}")
