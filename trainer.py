"""Evolutionary Strategies (ES) Trainer for Drosophila Plastic Layers.

This module implements the training loop of Module 5:
- Optimizes plastic weights of the Input Adapter [retina -> LPTC] and Motor Decoder [DN -> WASD].
- Uses Natural Evolution Strategies (NES) with antithetic perturbation sampling.
- Fitness function evaluates survival time, forward speed (optic flow VS), and track progression.
- Automatically resets Polytrack between candidate evaluations.
- Persists best model weights to data/brain_weights_best.pt.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

import config
from fly_brain import FlyBrain
from game_harness import PolytrackHarness, ScreenCapture
from retina import DrosophilaRetina

logger = logging.getLogger("FlyTrainer")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


class EvolutionaryTrainer:
    """Evolution Strategies (ES) optimizer for Drosophila plastic brain layers."""

    def __init__(
        self,
        population_size: int = 12,
        mutation_sigma: float = 0.05,
        learning_rate: float = 0.02,
        max_eval_frames: int = 350,
        use_synthetic: bool = True,
        output_weights_path: Path = config.DATA_DIR / "brain_weights_best.pt",
    ) -> None:
        self.pop_size = population_size
        self.sigma = mutation_sigma
        self.lr = learning_rate
        self.max_frames = max_eval_frames
        self.use_synthetic = use_synthetic
        self.output_path = output_weights_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Baseline components
        self.brain = FlyBrain()
        self.retina = DrosophilaRetina()
        self.harness = PolytrackHarness(fallback_synthetic=use_synthetic)

        # Base parameters (mean vector theta)
        self.theta = self.brain.get_plastic_parameters().copy()
        self.n_params = len(self.theta)

        self.best_fitness: float = -float("inf")
        self.best_params: np.ndarray = self.theta.copy()
        self.history: List[Dict[str, Any]] = []

    def evaluate_candidate(
        self,
        candidate_params: np.ndarray,
        candidate_idx: int = 0,
    ) -> Tuple[float, Dict[str, float]]:
        """Evaluate a single parameter vector in the Polytrack environment."""
        self.brain.set_plastic_parameters(candidate_params)
        self.brain.reset()
        self.retina.reset()

        if not self.use_synthetic:
            self.harness.inputs.reset_game()
            time.sleep(0.3)  # Brief pause after reset

        total_frames = 0
        cumulative_forward_speed = 0.0
        consecutive_falls = 0
        is_alive = True

        t0 = time.perf_counter()
        while total_frames < self.max_frames and is_alive:
            total_frames += 1

            # 1. Capture screen
            if self.use_synthetic:
                phase = total_frames * 0.05
                is_falling = (total_frames > 250) and (candidate_idx % 2 == 1)
                frame_bgr = ScreenCapture.generate_synthetic_polytrack_frame(800, 600, is_falling=is_falling)
                shift = int(np.sin(phase) * 30.0)
                M = np.float32([[1, 0, shift], [0, 1, 0]])
                frame_bgr = cv2.warpAffine(frame_bgr, M, (800, 600), borderMode=cv2.BORDER_REFLECT)
            else:
                frame_bgr, _ = self.harness.capture.grab_bgr()
                is_falling, _ = self.harness.state_detector.detect_fall(frame_bgr)

            # 2. Retina processing
            retina_out = self.retina.process(frame_bgr)

            # 3. Brain SNN inference
            brain_out = self.brain.forward_frame(retina_out.stimulation_currents)

            # 4. Actuation
            if not self.use_synthetic:
                self.harness.inputs.set_actions(
                    w=brain_out.actions["w"],
                    a=brain_out.actions["a"],
                    s=brain_out.actions["s"],
                    d=brain_out.actions["d"],
                )

            # Accumulate reward metrics
            forward_thrust = brain_out.dn_spike_rates.get("DNp01_L", 0.0) + brain_out.dn_spike_rates.get("DNp01_R", 0.0)
            optic_flow_speed = max(0.0, retina_out.lptc_vs_forward)
            cumulative_forward_speed += float(forward_thrust + optic_flow_speed)

            if is_falling:
                consecutive_falls += 1
                if consecutive_falls >= 5:
                    is_alive = False
            else:
                consecutive_falls = 0

        # Release keys safely
        if not self.use_synthetic:
            self.harness.inputs.release_all()

        survival_fraction = total_frames / float(self.max_frames)
        avg_speed = cumulative_forward_speed / max(1, total_frames)

        # Composite fitness function:
        # Fitness = (Survival Time) * 100 + (Average Forward Speed) * 50 - (Fall Penalty)
        fall_penalty = 50.0 if not is_alive else 0.0
        fitness = (survival_fraction * 100.0) + (avg_speed * 50.0) - fall_penalty

        metrics = {
            "fitness": fitness,
            "survival_frames": total_frames,
            "survival_fraction": survival_fraction,
            "avg_speed": avg_speed,
            "completed": bool(is_alive and total_frames >= self.max_frames),
        }
        return fitness, metrics

    def train_generation(self, gen_idx: int) -> Dict[str, Any]:
        """Execute one generation of Evolution Strategies with antithetic sampling."""
        t_gen_start = time.perf_counter()
        half_pop = self.pop_size // 2

        # Sample perturbations (half_pop vectors)
        noise = np.random.randn(half_pop, self.n_params).astype(np.float32)

        # Antithetic population: [+eps, -eps]
        candidates = []
        noises = []
        for i in range(half_pop):
            eps = noise[i]
            candidates.append(self.theta + self.sigma * eps)
            noises.append(eps)
            candidates.append(self.theta - self.sigma * eps)
            noises.append(-eps)

        fitnesses = []
        all_metrics = []

        print(f"\n--- Generation {gen_idx:3d} (Evaluating {len(candidates)} candidates) ---")
        for idx, cand in enumerate(candidates):
            fit, met = self.evaluate_candidate(cand, candidate_idx=idx)
            fitnesses.append(fit)
            all_metrics.append(met)
            print(
                f"  Candidate {idx + 1:2d}/{len(candidates)}: "
                f"Fitness = {fit:6.1f} | Survival: {met['survival_frames']:3d} frames | "
                f"Speed: {met['avg_speed']:.2f}"
            )

            # Track global best
            if fit > self.best_fitness:
                self.best_fitness = fit
                self.best_params = cand.copy()
                self.brain.set_plastic_parameters(cand)
                self.brain.save_weights(self.output_path)
                print(f"  >>> [NEW BEST] Saved checkpoint to {self.output_path} (Fitness: {fit:.1f})")

        # Rank-normalized fitness weights for ES update
        fit_arr = np.array(fitnesses)
        normalized_fit = (fit_arr - np.mean(fit_arr)) / (np.std(fit_arr) + 1e-6)

        # ES Gradient approximation: grad = sum(F_i * eps_i) / (P * sigma)
        grad = np.zeros(self.n_params, dtype=np.float32)
        for i in range(len(candidates)):
            grad += normalized_fit[i] * noises[i]
        grad /= float(len(candidates) * self.sigma)

        # Parameter update
        self.theta += self.lr * grad

        t_gen_duration = time.perf_counter() - t_gen_start
        gen_summary = {
            "generation": gen_idx,
            "mean_fitness": float(np.mean(fit_arr)),
            "max_fitness": float(np.max(fit_arr)),
            "best_fitness_ever": float(self.best_fitness),
            "generation_time_sec": float(t_gen_duration),
        }
        self.history.append(gen_summary)

        print(
            f"Generation {gen_idx} Complete in {t_gen_duration:.1f}s | "
            f"Mean: {gen_summary['mean_fitness']:.1f} | Max: {gen_summary['max_fitness']:.1f} | "
            f"Best Ever: {self.best_fitness:.1f}\n"
        )
        return gen_summary

    def run(self, num_generations: int = 10) -> None:
        """Run full evolutionary optimization loop."""
        print("=" * 65)
        print("DROSOPHILA PLASTIC LAYER EVOLUTIONARY TRAINER")
        print(f"Population Size: {self.pop_size} | Parameter Dim: {self.n_params:,}")
        print(f"Generations: {num_generations} | Mode: {'SYNTHETIC' if self.use_synthetic else 'LIVE DESKTOP POLYTRACK'}")
        print(f"Target Checkpoint: {self.output_path}")
        print("=" * 65)

        try:
            for g in range(1, num_generations + 1):
                self.train_generation(g)
        finally:
            self.harness.inputs.release_all()
            print(f"\nTraining finished. Global best fitness: {self.best_fitness:.1f}")
            print(f"Best weights preserved at: {self.output_path}\n")


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Evolutionary Trainer for Drosophila Polytrack Brain")
    parser.add_argument("--generations", type=int, default=5, help="Number of ES generations")
    parser.add_argument("--population", type=int, default=8, help="Population size")
    parser.add_argument("--live", action="store_true", help="Train on live Desktop Polytrack window instead of synthetic")
    parser.add_argument("--max-frames", type=int, default=250, help="Max evaluation frames per episode")

    args = parser.parse_args()

    trainer = EvolutionaryTrainer(
        population_size=args.population,
        max_eval_frames=args.max_frames,
        use_synthetic=not args.live,
    )
    trainer.run(num_generations=args.generations)


if __name__ == "__main__":
    main()
