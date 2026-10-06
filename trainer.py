"""Champion-Anchored Elitist Evolutionary Strategy (ES) Trainer for Drosophila Plastic Brain Layers.

This module implements the training loop of Module 5:
- Optimizes plastic weights of the Input Adapter [retina -> LPTC] and Motor Decoder [DN -> WASD].
- Implements strict Champion Elitism (μ + λ): Candidate 0 is ALWAYS the current Champion.
- Subsequent generations mutate strictly around the Best Point rather than a drifting mean vector.
- Degeneration across subsequent generations is mathematically prevented.
- Detects the Polytrack in-game respawn banner ("Нажмите R / Enter..."):
  Immediately terminates the attempt upon seeing the banner, resetting the game without wasting frames.
- Persists best weights monotonically to data/brain_weights_best.pt with full checkpoint metadata.
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
    """Champion-Anchored Elitist (μ + λ) Evolution Strategies optimizer.

    Guarantees monotonic best-point preservation:
    1. Candidate 0 is ALWAYS the current Champion (best_params).
    2. Candidates 1..N are perturbations (mutants) generated directly around best_params.
    3. If any mutant achieves fitness > best_fitness, it is registered as the new BEST POINT,
       persisted to data/brain_weights_best.pt, and becomes the new anchor.
    4. If no mutant improves upon the champion, the champion is strictly preserved, and mutation
       step size sigma adapts (shrinks to exploit, or widens to explore).
    5. Degeneration to worse policies across subsequent generations is mathematically impossible.
    """

    def __init__(
        self,
        population_size: int = config.EVOLUTION_POPULATION_SIZE,
        mutation_sigma: float = config.EVOLUTION_MUTATION_SIGMA,
        max_eval_frames: int = 350,
        use_synthetic: bool = True,
        output_weights_path: Path = config.BEST_BRAIN_WEIGHTS_FILE,
    ) -> None:
        self.pop_size = max(2, population_size)
        self.base_sigma = mutation_sigma
        self.sigma = mutation_sigma
        self.max_frames = max_eval_frames
        self.use_synthetic = use_synthetic
        self.output_path = output_weights_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Baseline neural and environment components
        self.brain = FlyBrain()
        self.retina = DrosophilaRetina()
        self.harness = PolytrackHarness(fallback_synthetic=use_synthetic)

        # Baseline parameter extraction
        self.n_params = len(self.brain.get_plastic_parameters())
        self.best_params: np.ndarray = self.brain.get_plastic_parameters().copy()
        self.best_fitness: float = -float("inf")
        self.best_generation: int = 0
        self.stagnation_counter: int = 0
        self.history: List[Dict[str, Any]] = []

        # Attempt to resume from existing saved checkpoint
        self._load_existing_checkpoint()

    def _load_existing_checkpoint(self) -> None:
        """Load prior best point if existing checkpoint file is present."""
        if self.output_path.exists():
            try:
                ckpt = torch.load(self.output_path, map_location="cpu")
                if isinstance(ckpt, dict) and "input_adapter" in ckpt:
                    self.brain.load_weights(self.output_path)
                    self.best_params = self.brain.get_plastic_parameters().copy()
                    saved_fit = ckpt.get("best_fitness")
                    saved_gen = ckpt.get("generation", 0)
                    if saved_fit is not None:
                        self.best_fitness = float(saved_fit)
                        self.best_generation = int(saved_gen)
                        logger.info(
                            f"Resumed prior Champion from {self.output_path}: "
                            f"Fitness = {self.best_fitness:.2f} (from Gen {self.best_generation})"
                        )
            except Exception as e:
                logger.warning(f"Could not restore checkpoint from {self.output_path}: {e}")

    def _backup_current_weights(self) -> None:
        """Create automatic backup of current brain weights before training modifies them."""
        if not self.output_path.exists():
            return
        try:
            import shutil

            # 1. Update immediate rollback pointer: data/brain_weights_previous.pt
            shutil.copy2(self.output_path, config.PREVIOUS_BRAIN_WEIGHTS_FILE)
            logger.info(f"Created immediate rollback backup at {config.PREVIOUS_BRAIN_WEIGHTS_FILE}")

            # 2. Archive to historical checkpoints folder: data/checkpoints/brain_weights_...
            config.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            fit_val = self.best_fitness if self.best_fitness != -float("inf") else 0.0
            fit_str = f"{fit_val:.1f}".replace("-", "neg")
            archive_name = f"brain_weights_{timestamp_str}_gen{self.best_generation}_fit{fit_str}.pt"
            archive_path = config.CHECKPOINTS_DIR / archive_name
            shutil.copy2(self.output_path, archive_path)
            logger.info(f"Archived previous session checkpoint to {archive_path}")
        except Exception as e:
            logger.warning(f"Could not backup existing weights: {e}")

    def evaluate_candidate(
        self,
        candidate_params: np.ndarray,
        candidate_idx: int = 0,
        is_champion: bool = False,
    ) -> Tuple[float, Dict[str, Any]]:
        """Evaluate a single parameter vector in Polytrack.

        Terminates the attempt immediately if:
        1. Car falls off track (Laplacian variance collapse)
        2. Car crashes and Polytrack respawn banner appears on screen ("Нажмите R / Enter...")
        """
        self.brain.set_plastic_parameters(candidate_params)
        self.brain.reset()
        self.retina.reset()

        if not self.use_synthetic:
            self.harness.inputs.reset_game()
            time.sleep(0.35)  # Pause for game track reset

        total_frames = 0
        cumulative_forward_speed = 0.0
        positive_forward_frames = 0
        reverse_frames = 0
        consecutive_falls = 0
        is_alive = True
        death_reason = "survived"
        max_prompt_score = 0.0

        while total_frames < self.max_frames and is_alive:
            total_frames += 1

            # ---------------------------------------------------------
            # 1. Capture screen & detect game terminal state
            # ---------------------------------------------------------
            if self.use_synthetic:
                phase = total_frames * 0.04
                is_falling = (total_frames > 220) and (candidate_idx % 2 == 1 and not is_champion)
                is_respawn = (total_frames > 180) and (candidate_idx % 3 == 0 and not is_champion)
                frame_bgr = ScreenCapture.generate_synthetic_polytrack_frame(
                    800, 600, is_falling=is_falling, has_respawn_banner=is_respawn, biome="summer"
                )
                shift = int(np.sin(phase) * 30.0)
                M = np.float32([[1, 0, shift], [0, 1, 0]])
                frame_bgr = cv2.warpAffine(frame_bgr, M, (800, 600), borderMode=cv2.BORDER_REFLECT)
                is_terminal = is_falling or is_respawn
                prompt_score = 1.0 if is_respawn else 0.1
            else:
                frame_bgr, _ = self.harness.capture.grab_bgr()
                is_terminal, is_falling, is_respawn, lap_var, prompt_score = (
                    self.harness.state_detector.detect_fall_or_crash(frame_bgr)
                )

            max_prompt_score = max(max_prompt_score, prompt_score)

            # TERMINATION CHECK 1: Respawn banner ends attempt IMMEDIATELY!
            if is_respawn:
                is_alive = False
                death_reason = "respawn_banner"
                break

            # TERMINATION CHECK 2: Falling off track (variance collapse or floor color)
            if is_falling:
                consecutive_falls += 1
                if consecutive_falls >= 4:
                    is_alive = False
                    death_reason = "fall_off_track"
                    break
            else:
                consecutive_falls = 0

            # ---------------------------------------------------------
            # 2. Retina processing (Optical Flow & Ommatidia)
            # ---------------------------------------------------------
            retina_out = self.retina.process(frame_bgr)
            vs_forward = float(retina_out.lptc_vs_forward)

            # ---------------------------------------------------------
            # 3. Brain SNN inference
            # ---------------------------------------------------------
            brain_out = self.brain.forward_frame(retina_out.stimulation_currents, vs_forward=vs_forward)

            # Track directional movement
            if self.use_synthetic:
                # In synthetic mode, pressing W simulates forward velocity
                if brain_out.actions["w"]:
                    sim_speed = 0.45
                    positive_forward_frames += 1
                    cumulative_forward_speed += sim_speed
                else:
                    still_speed = 0.0
            else:
                if vs_forward > 0.02:
                    positive_forward_frames += 1
                    cumulative_forward_speed += vs_forward
                elif vs_forward < -0.01:
                    reverse_frames += 1

            # TERMINATION CHECK 3: Anti-Reverse & Anti-Stagnation
            # If the car starts going backwards or stands still for too long, abort!
            if total_frames == 45 and not self.use_synthetic:
                if positive_forward_frames < 6 or reverse_frames > 15:
                    is_alive = False
                    death_reason = "stagnation_or_reverse"
                    break

            if reverse_frames >= 25 and not self.use_synthetic:
                is_alive = False
                death_reason = "reversing"
                break

            # ---------------------------------------------------------
            # 4. Actuation (WASD dispatch)
            # ---------------------------------------------------------
            if not self.use_synthetic:
                self.harness.inputs.set_actions(
                    w=brain_out.actions["w"],
                    a=brain_out.actions["a"],
                    s=brain_out.actions["s"],
                    d=brain_out.actions["d"],
                )

        # Release keys safely after candidate attempt
        if not self.use_synthetic:
            self.harness.inputs.release_all()
            if not is_alive:
                self.harness.inputs.reset_game()

        # -------------------------------------------------------------
        # 5. Composite Fitness Scoring: Forward Progress Optimization
        # -------------------------------------------------------------
        forward_ratio = positive_forward_frames / max(1, total_frames)
        avg_speed = cumulative_forward_speed / max(1, total_frames)

        # Distance & Speed rewards
        distance_score = float(cumulative_forward_speed * 12.0)
        speed_score = float(avg_speed * 80.0)

        # Survival points: ONLY rewarded if actively driving forward!
        # Standing still or reversing gives 0 survival points.
        survival_fraction = total_frames / float(self.max_frames)
        survival_score = float(survival_fraction * 80.0 * forward_ratio)

        # Penalties:
        crash_penalty = 40.0 if not is_alive else 0.0
        reverse_penalty = 50.0 if (death_reason in ["reversing", "stagnation_or_reverse"] or reverse_frames > 12) else 0.0

        fitness = distance_score + speed_score + survival_score - crash_penalty - reverse_penalty

        metrics = {
            "fitness": float(fitness),
            "survival_frames": total_frames,
            "survival_fraction": float(survival_fraction),
            "positive_forward_frames": positive_forward_frames,
            "reverse_frames": reverse_frames,
            "forward_ratio": float(forward_ratio),
            "avg_speed": float(avg_speed),
            "is_alive": bool(is_alive),
            "death_reason": death_reason,
            "max_prompt_score": float(max_prompt_score),
            "is_champion": is_champion,
        }
        return fitness, metrics

    def train_generation(self, gen_idx: int) -> Dict[str, Any]:
        """Execute one generation of Champion-Anchored Elitist Evolution."""
        t_gen_start = time.perf_counter()

        # Build candidate population:
        # Candidate 0 is STRICTLY the Current Champion (Elitism)
        candidates: List[np.ndarray] = [self.best_params.copy()]
        is_champ_flags: List[bool] = [True]

        # Remaining candidates: Mutants generated around the Current Champion
        num_mutants = self.pop_size - 1
        half_mutants = num_mutants // 2

        # Antithetic perturbations around best_params: [+eps, -eps]
        for _ in range(half_mutants):
            eps = np.random.randn(self.n_params).astype(np.float32)
            candidates.append(self.best_params + self.sigma * eps)
            is_champ_flags.append(False)
            candidates.append(self.best_params - self.sigma * eps)
            is_champ_flags.append(False)

        # If odd population size, add one standard perturbation
        while len(candidates) < self.pop_size:
            eps = np.random.randn(self.n_params).astype(np.float32)
            candidates.append(self.best_params + self.sigma * eps)
            is_champ_flags.append(False)

        fitnesses: List[float] = []
        all_metrics: List[Dict[str, Any]] = []

        print(f"\n=================================================================")
        print(f"--- GENERATION {gen_idx:3d} (Evaluating {len(candidates)} candidates | sigma={self.sigma:.4f}) ---")
        print(f"=================================================================")

        for idx, (cand, is_champ) in enumerate(zip(candidates, is_champ_flags)):
            cand_label = "CHAMPION" if is_champ else f"Mutant #{idx}"
            fit, met = self.evaluate_candidate(cand, candidate_idx=idx, is_champion=is_champ)
            fitnesses.append(fit)
            all_metrics.append(met)

            term_tag = f"[{met['death_reason']}]" if not met["is_alive"] else "[SURVIVED]"
            print(
                f"  Candidate {idx + 1:2d}/{len(candidates)} ({cand_label:10s}): "
                f"Fitness = {fit:6.1f} | Frames: {met['survival_frames']:3d}/{self.max_frames} | "
                f"Speed: {met['avg_speed']:.2f} {term_tag}"
            )

        # Identify generation best candidate
        best_cand_idx = int(np.argmax(fitnesses))
        best_cand_fit = float(fitnesses[best_cand_idx])

        # -------------------------------------------------------------
        # STRICT BEST POINT REGISTRATION & MONOTONIC UPDATE
        # -------------------------------------------------------------
        if best_cand_fit > self.best_fitness + 0.05:
            old_best = self.best_fitness
            self.best_fitness = best_cand_fit
            self.best_params = candidates[best_cand_idx].copy()
            self.best_generation = gen_idx
            self.stagnation_counter = 0

            # Immediately persist weights with full checkpoint metadata
            self.brain.set_plastic_parameters(self.best_params)
            checkpoint_data = {
                "input_adapter": self.brain.input_adapter.state_dict(),
                "motor_decoder": self.brain.motor_decoder.state_dict(),
                "best_fitness": self.best_fitness,
                "generation": gen_idx,
                "candidate_idx": best_cand_idx,
                "metrics": all_metrics[best_cand_idx],
                "saved_at": time.time(),
            }
            torch.save(checkpoint_data, self.output_path)

            print(f"\n  *****************************************************************")
            print(f"  >>> [NEW BEST POINT REGISTERED!] Candidate #{best_cand_idx + 1}")
            print(f"      Fitness: {best_cand_fit:.2f} (Surpassed prior: {old_best:.2f}, +{best_cand_fit - old_best:.2f})")
            print(f"      Saved weights & metadata to: {self.output_path}")
            print(f"  *****************************************************************\n")
            # Slightly expand sigma if improving
            self.sigma = min(0.12, self.sigma * 1.02)
        else:
            self.stagnation_counter += 1
            print(
                f"\n  [CHAMPION PRESERVED] No mutant exceeded Champion "
                f"(Best Ever remains: {self.best_fitness:.2f} from Gen {self.best_generation})."
            )
            # Adaptive step-size: shrink sigma to perform finer exploitation around champion
            if self.stagnation_counter in [2, 4, 6]:
                self.sigma = max(0.012, self.sigma * 0.85)
                print(f"  [ADAPTIVE ES] Plateau detected ({self.stagnation_counter} gens). Refining sigma -> {self.sigma:.4f}")
            elif self.stagnation_counter >= 8:
                # Scout pulse: wide exploration while Champion (Cand 0) still guarantees safety
                self.sigma = min(0.10, self.base_sigma * 1.5)
                print(f"  [ADAPTIVE ES] Scout exploration pulse triggered -> sigma {self.sigma:.4f}")

        t_gen_duration = time.perf_counter() - t_gen_start
        gen_summary = {
            "generation": gen_idx,
            "mean_fitness": float(np.mean(fitnesses)),
            "max_fitness": best_cand_fit,
            "best_fitness_ever": float(self.best_fitness),
            "generation_time_sec": float(t_gen_duration),
            "sigma": float(self.sigma),
            "stagnation": self.stagnation_counter,
        }
        self.history.append(gen_summary)

        print(
            f"Generation {gen_idx} finished in {t_gen_duration:.1f}s | "
            f"Gen Mean: {gen_summary['mean_fitness']:.1f} | Best Ever: {self.best_fitness:.1f}\n"
        )
        return gen_summary

    def run(self, num_generations: int = 10) -> None:
        """Run full evolutionary optimization loop."""
        print("=" * 68)
        print("DROSOPHILA PLASTIC LAYER ELITIST EVOLUTIONARY TRAINER")
        print(f"Population Size: {self.pop_size} (1 Champion + {self.pop_size - 1} Mutants)")
        print(f"Plastic Parameters: {self.n_params:,}")
        print(f"Generations: {num_generations} | Mode: {'SYNTHETIC' if self.use_synthetic else 'LIVE DESKTOP POLYTRACK'}")
        print(f"Active Checkpoint: {self.output_path}")
        print("Current Best Fitness Anchor: " f"{self.best_fitness:.2f} (Gen {self.best_generation})")
        print("=" * 68)

        # Automatic backup of current weights before training modifies them
        self._backup_current_weights()

        # If baseline fitness is unknown, evaluate champion once before starting
        if self.best_fitness == -float("inf"):
            print("\nEvaluating initial Champion baseline...")
            base_fit, base_met = self.evaluate_candidate(self.best_params, candidate_idx=0, is_champion=True)
            self.best_fitness = base_fit
            self.brain.set_plastic_parameters(self.best_params)
            torch.save(
                {
                    "input_adapter": self.brain.input_adapter.state_dict(),
                    "motor_decoder": self.brain.motor_decoder.state_dict(),
                    "best_fitness": self.best_fitness,
                    "generation": 0,
                    "metrics": base_met,
                    "saved_at": time.time(),
                },
                self.output_path,
            )
            print(f"Initial Champion Fitness Established: {self.best_fitness:.2f}\n")

        try:
            for g in range(1, num_generations + 1):
                self.train_generation(g)
        finally:
            self.harness.inputs.release_all()
            print(f"\nTraining session complete.")
            print(f"Global Best Fitness Ever: {self.best_fitness:.2f} (Registered in Gen {self.best_generation})")
            print(f"Best model weights firmly preserved at: {self.output_path}\n")


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Elitist Trainer for Drosophila Polytrack SNN")
    parser.add_argument("--generations", type=int, default=5, help="Number of ES generations")
    parser.add_argument("--population", type=int, default=config.EVOLUTION_POPULATION_SIZE, help="Population size")
    parser.add_argument("--live", action="store_true", help="Train on live Desktop Polytrack window instead of synthetic")
    parser.add_argument("--max-frames", type=int, default=300, help="Max evaluation frames per episode")
    parser.add_argument("--sigma", type=float, default=config.EVOLUTION_MUTATION_SIGMA, help="Mutation sigma")

    args = parser.parse_args()

    trainer = EvolutionaryTrainer(
        population_size=args.population,
        mutation_sigma=args.sigma,
        max_eval_frames=args.max_frames,
        use_synthetic=not args.live,
    )
    trainer.run(num_generations=args.generations)


if __name__ == "__main__":
    main()
