"""Unit tests for BrainManager (reset to baseline, rollback, and checkpoints)."""

import tempfile
from pathlib import Path
import torch
import numpy as np

from brain_manager import BrainManager
from fly_brain import FlyBrain
from connectome_loader import get_or_build_connectome
import config


def test_brain_manager_lifecycle() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        best_path = tmp_path / "brain_weights_best.pt"
        prev_path = tmp_path / "brain_weights_previous.pt"
        base_path = tmp_path / "brain_weights_baseline.pt"
        checkpoints_dir = tmp_path / "checkpoints"

        manager = BrainManager(
            active_weights_path=best_path,
            previous_weights_path=prev_path,
            baseline_weights_path=base_path,
            checkpoints_dir=checkpoints_dir,
        )

        # 1. Reset to baseline from clean state
        success = manager.reset_to_baseline()
        assert success, "Reset to baseline must succeed!"
        assert best_path.exists(), "best_weights_path must be created!"
        assert base_path.exists(), "baseline_weights_path must be created!"

        info = manager.get_checkpoint_info(best_path)
        assert best_path.exists() is True
        assert info["type"] == "biological_baseline"
        assert info["best_fitness"] == 0.0

        # 2. Simulate training and session backup
        connectome = get_or_build_connectome()
        brain = FlyBrain(connectome)
        # Mutate weights slightly
        params = brain.get_plastic_parameters()
        brain.set_plastic_parameters(params + 0.5)
        brain.save_weights(
            best_path,
            metadata={"generation": 5, "best_fitness": 350.5, "type": "trained", "saved_at": 1000.0},
        )

        # Check status after training
        info_after_train = manager.get_checkpoint_info(best_path)
        assert info_after_train["best_fitness"] == 350.5
        assert info_after_train["generation"] == 5

        # 3. Simulate another reset (should archive trained weights)
        success_reset_2 = manager.reset_to_baseline()
        assert success_reset_2 is True
        assert len(manager.list_archived_checkpoints()) >= 1
        assert prev_path.exists()

        # Check that previous weights captured the 350.5 fitness
        prev_info = manager.get_checkpoint_info(prev_path)
        assert prev_info["best_fitness"] == 350.5

        # 4. Rollback to previous session
        rollback_success = manager.rollback_to_previous()
        assert rollback_success is True

        # Best should now be restored to 350.5
        info_after_rollback = manager.get_checkpoint_info(best_path)
        assert info_after_rollback["best_fitness"] == 350.5
        assert info_after_rollback["generation"] == 5


if __name__ == "__main__":
    print("Running BrainManager unit tests...")
    test_brain_manager_lifecycle()
    print("test_brain_manager_lifecycle passed.")
    print("ALL BRAIN MANAGER TESTS PASSED SUCCESSFULLY!")
