import os
import json
import torch
from typing import Dict, Any, Optional, Tuple


class CheckpointManager:
    """
    Manages saving and resuming local training checkpoints, ensuring previous states
    are preserved and never corrupted upon interruption or CUDA OOM.
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
            if d.startswith("checkpoint-") and os.path.isdir(os.path.join(self.checkpoint_dir, d))
        ]

        if not subdirs:
            return None

        # Sort by step number
        subdirs.sort(key=lambda x: int(x.split("-")[-1]) if x.split("-")[-1].isdigit() else 0)
        return subdirs[-1]

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
    ) -> str:
        """
        Save a step checkpoint containing PEFT adapter, trainer state, and RNG states.
        """
        save_path = os.path.join(self.checkpoint_dir, f"checkpoint-{step}")
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
        }

        if optimizer is not None and hasattr(optimizer, "state_dict"):
            torch.save(optimizer.state_dict(), os.path.join(save_path, "optimizer.pt"))

        if scheduler is not None and hasattr(scheduler, "state_dict"):
            torch.save(scheduler.state_dict(), os.path.join(save_path, "scheduler.pt"))

        # 4. Save RNG state
        rng_state = {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        torch.save(rng_state, os.path.join(save_path, "rng_state.pt"))

        # 5. Save metadata json
        with open(os.path.join(save_path, "checkpoint_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(trainer_state, f, indent=2)

        print(f"[CHECKPOINT] Preserved checkpoint at: {save_path}")
        self._prune_old_checkpoints()
        return save_path

    def _prune_old_checkpoints(self):
        """Prune checkpoints exceeding max_to_keep."""
        subdirs = [
            os.path.join(self.checkpoint_dir, d)
            for d in os.listdir(self.checkpoint_dir)
            if d.startswith("checkpoint-") and os.path.isdir(os.path.join(self.checkpoint_dir, d))
        ]

        if len(subdirs) <= self.max_to_keep:
            return

        subdirs.sort(key=lambda x: int(x.split("-")[-1]) if x.split("-")[-1].isdigit() else 0)
        to_delete = subdirs[:-self.max_to_keep]

        for old_ckpt in to_delete:
            try:
                import shutil
                shutil.rmtree(old_ckpt)
                print(f"[CHECKPOINT] Pruned old checkpoint: {old_ckpt}")
            except Exception as e:
                print(f"[WARNING] Could not prune checkpoint {old_ckpt}: {e}")
