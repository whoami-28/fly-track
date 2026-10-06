"""Integration tests for Closed-Loop Orchestrator and Evolutionary Trainer (Module 5)."""

from pathlib import Path
import tempfile
import numpy as np

import config
from main import PolytrackOrchestrator
from trainer import EvolutionaryTrainer


def test_orchestrator_step_and_hud() -> None:
    orch = PolytrackOrchestrator(use_synthetic=True)
    try:
        orch.step()  # Warmup PyTorch and OpenCV kernels
        frame, retina_out, brain_out, telem = orch.step()
        assert frame is not None
        assert retina_out is not None
        assert brain_out is not None
        assert "latency_total_ms" in telem
        assert telem["latency_total_ms"] < 40.0
        assert set(brain_out.actions.keys()) == {"w", "a", "s", "d"}

        # Render HUD
        hud = orch.render_hud_overlay(frame, retina_out, brain_out, telem)
        assert hud is not None
        assert hud.shape == (640, 1024, 3)
        assert hud.dtype == np.uint8
    finally:
        orch.harness.inputs.release_all()


def test_trainer_single_generation() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        weights_path = Path(tmp_dir) / "test_weights.pt"
        trainer = EvolutionaryTrainer(
            population_size=4,
            max_eval_frames=30,  # Fast test
            use_synthetic=True,
            output_weights_path=weights_path,
        )

        gen_summary = trainer.train_generation(gen_idx=1)
        assert gen_summary["generation"] == 1
        assert "mean_fitness" in gen_summary
        assert "max_fitness" in gen_summary
        assert weights_path.exists(), "Trainer must save best weights checkpoint!"


if __name__ == "__main__":
    print("Running System Integration tests...")
    test_orchestrator_step_and_hud()
    print("test_orchestrator_step_and_hud passed.")
    test_trainer_single_generation()
    print("test_trainer_single_generation passed.")
    print("ALL SYSTEM INTEGRATION TESTS PASSED SUCCESSFULLY!")
