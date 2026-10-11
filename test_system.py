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


def test_is_on_track_sky_tolerance() -> None:
    """Ensure bright blue sky does NOT trigger menu detection."""
    from game_harness import GameStateDetector, ScreenCapture
    detector = GameStateDetector()

    # Generate realistic racing frame (has blue sky at top and road at bottom)
    racing_frame = ScreenCapture.generate_synthetic_polytrack_frame(800, 600, is_falling=False, biome="summer")
    is_racing, reason = detector.is_on_track(racing_frame)
    assert is_racing is True, f"Racing frame with blue sky was falsely flagged as menu: {reason}"

    # Pure dark navy menu frame (simulating PolyTrack level selector)
    menu_frame = np.full((600, 800, 3), (45, 25, 20), dtype=np.uint8)  # BGR dark navy blue
    is_menu_racing, menu_reason = detector.is_on_track(menu_frame)
    assert is_menu_racing is False, f"Dark navy menu frame was not detected: {menu_reason}"


def test_trainer_mutant_overtaking_champion() -> None:
    """Ensure that if a mutant outperforms Candidate 0, it is promoted to Champion."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        weights_path = Path(tmp_dir) / "test_champion_weights.pt"
        trainer = EvolutionaryTrainer(
            population_size=4,
            max_eval_frames=20,
            use_synthetic=True,
            output_weights_path=weights_path,
        )

        # Mock evaluate_candidate so Mutant #1 beats Champion (Cand 0)
        original_eval = trainer.evaluate_candidate

        def mock_eval(params, candidate_idx=0, is_champion=False):
            if is_champion:
                return 100.0, {"fitness": 100.0, "survival_frames": 20, "is_alive": True, "death_reason": "survived", "avg_speed": 0.1, "speed_index": 10.0, "corridor_detected_frames": 10, "centered_frames": 10, "overtime_extensions": 0, "max_frames_reached": 20}
            elif candidate_idx == 1:
                return 250.0, {"fitness": 250.0, "survival_frames": 20, "is_alive": True, "death_reason": "survived", "avg_speed": 0.5, "speed_index": 50.0, "corridor_detected_frames": 20, "centered_frames": 20, "overtime_extensions": 0, "max_frames_reached": 20}
            else:
                return 80.0, {"fitness": 80.0, "survival_frames": 20, "is_alive": True, "death_reason": "survived", "avg_speed": 0.05, "speed_index": 5.0, "corridor_detected_frames": 5, "centered_frames": 5, "overtime_extensions": 0, "max_frames_reached": 20}

        trainer.evaluate_candidate = mock_eval
        summary = trainer.train_generation(gen_idx=1)

        assert summary["max_fitness"] == 250.0
        assert trainer.best_fitness == 250.0
        assert trainer.best_generation == 1
        assert weights_path.exists()

