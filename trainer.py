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
from boundary_detector import BoundaryPerceptionOutput, TrackBoundaryDetector
from fly_brain import FlyBrain
from game_harness import PolytrackHarness, ScreenCapture
from neural_overlay import NeuralOverlayController
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
        max_eval_frames: int = 900,
        use_synthetic: bool = True,
        output_weights_path: Path = config.BEST_BRAIN_WEIGHTS_FILE,
        enable_overtime: bool = True,
        max_overtime_cap: int = 2500,
        show_overlay: bool = True,
    ) -> None:
        self.pop_size = max(2, population_size)
        self.base_sigma = mutation_sigma
        self.sigma = mutation_sigma
        self.max_frames = max_eval_frames
        self.enable_overtime = enable_overtime
        self.max_overtime_cap = max_overtime_cap
        self.use_synthetic = use_synthetic
        self.show_overlay = show_overlay
        self.output_path = output_weights_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Baseline neural, visual, and environment components
        self.brain = FlyBrain()
        self.retina = DrosophilaRetina()
        self.boundary_detector = TrackBoundaryDetector()
        self.harness = PolytrackHarness(fallback_synthetic=use_synthetic)

        # In-game neural network overlay
        self.overlay: Optional[NeuralOverlayController] = None
        if self.show_overlay and not self.use_synthetic:
            target_hwnd = getattr(getattr(self.harness, "capture", None), "hwnd", 0)
            self.overlay = NeuralOverlayController(target_hwnd=target_hwnd)
            self.overlay.start()

        # Baseline parameter extraction
        self.n_params = len(self.brain.get_plastic_parameters())
        self.best_params: np.ndarray = self.brain.get_plastic_parameters().copy()
        self.best_fitness: float = -float("inf")
        self.best_generation: int = 0
        self.stagnation_counter: int = 0
        self.history: List[Dict[str, Any]] = []

        # Behavioral niches for speciation & diversity preservation
        self.niches = ["speedrunner", "centrist", "tactician"]
        self.niche_anchors: Dict[str, np.ndarray] = {n: self.best_params.copy() for n in self.niches}
        self.niche_best_fitness: Dict[str, float] = {n: -float("inf") for n in self.niches}

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
                    for n in self.niches:
                        self.niche_anchors[n] = self.best_params.copy()
                    saved_fit = ckpt.get("best_fitness")
                    saved_gen = ckpt.get("generation", 0)
                    if saved_fit is not None:
                        self.best_fitness = float(saved_fit)
                        self.best_generation = int(saved_gen)
                        for n in self.niches:
                            self.niche_best_fitness[n] = self.best_fitness
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
        niche: str = "speedrunner",
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
            self.harness.ensure_window_focus()
            self.harness.inputs.reset_game()
            time.sleep(0.35)  # Pause for game track reset

        total_frames = 0
        current_frame_limit = self.max_frames
        overtime_extensions = 0
        cumulative_forward_speed = 0.0
        positive_forward_frames = 0
        reverse_frames = 0
        consecutive_falls = 0
        consecutive_respawns = 0
        consecutive_curb_breach = 0
        consecutive_curb_beached = 0
        consecutive_stuck_frames = 0
        centered_frames = 0
        curb_riding_frames = 0
        corridor_detected_frames = 0
        bottleneck_frames = 0
        bottleneck_decel_frames = 0
        is_alive = True
        death_reason = "survived"
        max_prompt_score = 0.0

        while total_frames < current_frame_limit and is_alive:
            total_frames += 1

            # Dynamic Overtime / Endurance Extension:
            # If candidate reaches current frame limit and is actively driving forward (>= 40% forward motion):
            if self.enable_overtime and (total_frames >= current_frame_limit) and is_alive:
                if positive_forward_frames > total_frames * 0.40:
                    if current_frame_limit < self.max_overtime_cap:
                        current_frame_limit = min(self.max_overtime_cap, current_frame_limit + 250)
                        overtime_extensions += 1
                        print(f"      [OVERTIME +250fr] Муха уверенно едет! Заезд продлен -> {current_frame_limit} кадров...")

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
                boundary_out = self.boundary_detector.detect(frame_bgr)
            else:
                frame_bgr, _ = self.harness.capture.grab_bgr()
                boundary_out = self.boundary_detector.detect(frame_bgr)
                is_terminal, is_falling, is_respawn, lap_var, prompt_score = (
                    self.harness.state_detector.detect_fall_or_crash(
                        frame_bgr, corridor_detected=boundary_out.corridor_detected
                    )
                )

            max_prompt_score = max(max_prompt_score, prompt_score)

            # TERMINATION CHECKS (Warmup grace period: first 60 frames (~1.0s) are protected)
            # In the first 60 frames, the car literally just reset and is accelerating away from the start grid!
            if total_frames > 60:
                # CHECK 1: Persistent in-game respawn banner ("Нажмите R / Enter...")
                if is_respawn:
                    consecutive_respawns += 1
                    if consecutive_respawns >= 2:
                        is_alive = False
                        death_reason = "respawn_banner"
                        break
                else:
                    consecutive_respawns = 0

                # CHECK 2: Curb breach (car has fully crossed over outer curb)
                if getattr(boundary_out, "is_curb_breached", False):
                    consecutive_curb_breach += 1
                    if consecutive_curb_breach >= 3:
                        is_alive = False
                        death_reason = "curb_breached"
                        break
                else:
                    consecutive_curb_breach = 0

                # CHECK 3: Falling off track (never triggers on bridges/viaducts)
                is_off_track = is_falling and not boundary_out.corridor_detected
                if is_off_track:
                    consecutive_falls += 1
                    if consecutive_falls >= 4:
                        is_alive = False
                        death_reason = "fall_off_track"
                        break
                else:
                    consecutive_falls = 0

            # ---------------------------------------------------------
            # 2. Track Boundary / Curbstone Perception (Centering Reflex)
            # ---------------------------------------------------------
            if boundary_out.corridor_detected:
                corridor_detected_frames += 1
                if abs(boundary_out.lateral_offset) <= 0.25:
                    centered_frames += 1
                elif abs(boundary_out.lateral_offset) >= 0.80:
                    curb_riding_frames += 1

            # ---------------------------------------------------------
            # 3. Retina processing (Optical Flow & Ommatidia)
            # ---------------------------------------------------------
            retina_out = self.retina.process(frame_bgr)
            vs_forward = float(retina_out.lptc_vs_forward)

            # ---------------------------------------------------------
            # 4. Brain SNN inference (with Optomotor Centering Reflex)
            # ---------------------------------------------------------
            brain_out = self.brain.forward_frame(
                retina_out.stimulation_currents,
                vs_forward=vs_forward,
                boundary=boundary_out,
            )

            # Update in-game neural overlay if active
            if self.overlay:
                curb_st = (
                    "breached"
                    if getattr(boundary_out, "is_curb_breached", False)
                    else ("beached" if getattr(boundary_out, "is_curb_beached", False) else "ok")
                )
                self.overlay.update(
                    actions=brain_out.actions,
                    action_probs=brain_out.action_probabilities,
                    dn_rates=brain_out.dn_spike_rates,
                    heading_deg=brain_out.compass_heading_deg,
                    vs_forward=retina_out.lptc_vs_forward,
                    hs_left=retina_out.lptc_hs_left,
                    hs_right=retina_out.lptc_hs_right,
                    lateral_offset=boundary_out.lateral_offset,
                    fitness=cumulative_forward_speed * 100.0,
                    generation=self.best_generation,
                    fps=30.0,
                    curb_status=curb_st,
                )

            # Track bottleneck navigation & tactical trail deceleration
            if getattr(boundary_out, "bottleneck_ahead", False) or getattr(boundary_out, "constriction_risk", 0.0) > 0.35:
                bottleneck_frames += 1
                if brain_out.actions["s"] or not brain_out.actions["w"]:
                    bottleneck_decel_frames += 1

            # Check HUD timer activity (verifies game is active and not frozen)
            hud_delta = 0.0
            if not self.use_synthetic:
                hud_delta = self.harness.state_detector.check_hud_activity(frame_bgr)

            # Real forward movement detection:
            # Requires TRUE optical flow expansion (vs_forward > 0.001) in track coordinates.
            # NOTE: hud_delta is the game's stopwatch clock and must NOT be used for car movement,
            # as the clock ticks even when the car is overturned or sitting motionless in grass!
            is_moving_forward = False
            if self.use_synthetic:
                if brain_out.actions["w"]:
                    is_moving_forward = True
                    cumulative_forward_speed += 0.45
            else:
                has_visual_motion = (vs_forward > 0.0005)
                if has_visual_motion:
                    is_moving_forward = True
                    cumulative_forward_speed += max(0.0, vs_forward)
                    consecutive_stuck_frames = 0
                    consecutive_curb_beached = 0
                else:
                    # Car is visually motionless or spinning wheels without translation!
                    if brain_out.actions["w"]:
                        consecutive_stuck_frames += 1
                        # If car is on the curb and motionless while holding throttle, terminate in 12 frames (~0.3s)
                        if getattr(boundary_out, "is_curb_beached", False):
                            consecutive_curb_beached += 1
                            if total_frames > 60 and consecutive_curb_beached >= 12:
                                is_alive = False
                                death_reason = "curb_beached"
                                break
                        else:
                            consecutive_curb_beached = 0

                        # General motionless stuck (~2.5 seconds motionless)
                        if total_frames > 90 and consecutive_stuck_frames >= 75:
                            is_alive = False
                            death_reason = "stuck_motionless"
                            break
                    else:
                        consecutive_stuck_frames = 0
                        consecutive_curb_beached = 0

            if is_moving_forward:
                positive_forward_frames += 1

            # Only flag true idle standing if the car literally never touched throttle
            # and HUD timer never started after 180 frames (3 full seconds — accounts for reset lag)
            if total_frames == 180 and not self.use_synthetic:
                if positive_forward_frames < 5:
                    is_alive = False
                    death_reason = "did_not_start"
                    break

            # ---------------------------------------------------------
            # 5. Actuation (WASD dispatch)
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
        # 6. Composite Fitness Scoring: Progress Odometry & Behavioral Niches
        # -------------------------------------------------------------
        # True forward translation distance (odometry from optic flow)
        total_distance = float(cumulative_forward_speed * 100.0)
        # Average pace in optical flow units
        average_pace = float((cumulative_forward_speed / max(1, total_frames)) * 1000.0)

        centering_bonus = float(centered_frames * 0.5)
        curb_penalty = float(curb_riding_frames * 1.5)
        crash_penalty = 50.0 if not is_alive and death_reason != "survived" else 0.0
        stuck_penalty = 80.0 if death_reason in ("stuck_on_wall", "stuck_motionless", "curb_beached", "curb_breached") else 0.0

        # Behavioral niche shaping
        if niche == "centrist":
            # Reward staying inside the asphalt corridor and severely penalize touching curbs
            raw_fitness = (
                total_distance * 0.8
                + (centered_frames * 1.2)
                - (curb_riding_frames * 3.0)
                - crash_penalty
                + (positive_forward_frames * 0.2)
            )
        elif niche == "tactician":
            # Reward safe clearance from walls, high survival, and proactive trail braking in bottlenecks
            decel_bonus = float(bottleneck_decel_frames * 2.0)
            tactical_centering = float(centered_frames * 1.0)
            raw_fitness = (
                total_distance * 1.2
                + tactical_centering
                + decel_bonus
                - (curb_riding_frames * 2.5)
                - crash_penalty
                + (positive_forward_frames * 0.25)
            )
        elif niche == "drifter":
            # Reward aggressive steering, allows curb riding as apex touching
            raw_fitness = (
                total_distance * 1.0
                + (curb_riding_frames * 0.8)
                + (positive_forward_frames * 0.4)
                - (curb_riding_frames * 0.2)
                - crash_penalty
            )
        else:
            # Default "speedrunner": pure distance * pace with centering guidance
            raw_fitness = (
                total_distance * 1.5
                + average_pace * 2.0
                + centering_bonus
                - curb_penalty
                - crash_penalty
            )

        # Breakthrough Sector Milestone: reward conquering the viaduct bottleneck sector (D > 420)
        breakthrough_bonus = 0.0
        if total_distance > 420.0 and is_alive:
            breakthrough_bonus = 400.0
            raw_fitness += breakthrough_bonus

        if death_reason == "did_not_start":
            fitness = 0.0
        elif death_reason in ("stuck_on_wall", "stuck_motionless"):
            # Penalize heavily: spinning wheels while motionless or upside down
            fitness = max(1.0, (total_distance * 0.2) - stuck_penalty)
        elif death_reason == "fall_off_track":
            # Falling off elevated track onto grass ends run and devalues non-road time
            fitness = max(1.0, (raw_fitness * 0.4) - 40.0)
        else:
            fitness = max(1.0, raw_fitness)

        avg_spd = float(cumulative_forward_speed / max(1, total_frames))
        metrics = {
            "fitness": float(fitness),
            "survival_frames": total_frames,
            "survival_fraction": float(total_frames / float(self.max_frames)),
            "positive_forward_frames": positive_forward_frames,
            "centered_frames": centered_frames,
            "curb_riding_frames": curb_riding_frames,
            "corridor_detected_frames": corridor_detected_frames,
            "bottleneck_frames": bottleneck_frames,
            "bottleneck_decel_frames": bottleneck_decel_frames,
            "breakthrough_bonus": float(breakthrough_bonus),
            "total_distance": total_distance,
            "avg_speed": avg_spd,
            "speed_index": float(avg_spd * 1000.0),
            "average_pace": average_pace,
            "niche": niche,
            "overtime_extensions": overtime_extensions,
            "max_frames_reached": current_frame_limit,
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
        cand_niches: List[str] = ["speedrunner"]

        # Remaining candidates: Mutants partitioned across niches
        remaining_slots = self.pop_size - 1
        niche_assignments = [self.niches[i % len(self.niches)] for i in range(remaining_slots)]

        for niche in self.niches:
            niche_count = sum(1 for n in niche_assignments if n == niche)
            half = niche_count // 2
            anchor = self.niche_anchors[niche]
            for _ in range(half):
                eps = np.random.randn(self.n_params).astype(np.float32)
                candidates.append(anchor + self.sigma * eps)
                is_champ_flags.append(False)
                cand_niches.append(niche)
                candidates.append(anchor - self.sigma * eps)
                is_champ_flags.append(False)
                cand_niches.append(niche)
            if niche_count % 2 == 1:
                eps = np.random.randn(self.n_params).astype(np.float32)
                candidates.append(anchor + self.sigma * eps)
                is_champ_flags.append(False)
                cand_niches.append(niche)

        fitnesses: List[float] = []
        all_metrics: List[Dict[str, Any]] = []

        print(f"\n=================================================================")
        print(f"--- GENERATION {gen_idx:3d} (Evaluating {len(candidates)} candidates | sigma={self.sigma:.4f}) ---")
        print(f"=================================================================")

        for idx, (cand, is_champ, cand_niche) in enumerate(zip(candidates, is_champ_flags, cand_niches)):
            cand_label = "CHAMPION" if is_champ else f"Mutant #{idx} ({cand_niche[:5]})"
            try:
                fit, met = self.evaluate_candidate(cand, candidate_idx=idx, is_champion=is_champ, niche=cand_niche)
            except TypeError:
                fit, met = self.evaluate_candidate(cand, candidate_idx=idx, is_champion=is_champ)
            fitnesses.append(fit)
            all_metrics.append(met)

            term_tag = f"[{met['death_reason']}]" if not met["is_alive"] else "[SURVIVED]"
            ot_tag = f" (OT:{met['overtime_extensions']}x)" if met.get("overtime_extensions", 0) > 0 else ""
            dist_val = met.get("total_distance", 0.0)
            print(
                f"  Candidate {idx + 1:2d}/{len(candidates)} ({cand_label:14s}): "
                f"Fitness = {fit:6.1f} | Dist: {dist_val:5.1f} | Frames: {met['survival_frames']:3d}/{met['max_frames_reached']} | "
                f"Speed: {met['speed_index']:.1f} | Corridor: {met['corridor_detected_frames']:3d}fr "
                f"| Centered: {met['centered_frames']:3d}fr {term_tag}{ot_tag}"
            )

        # Update niche anchors with their respective best performers
        for niche in self.niches:
            niche_indices = [i for i, n in enumerate(cand_niches) if n == niche]
            if niche_indices:
                best_niche_local_idx = max(niche_indices, key=lambda i: fitnesses[i])
                best_niche_fit = fitnesses[best_niche_local_idx]
                if best_niche_fit > self.niche_best_fitness[niche]:
                    self.niche_best_fitness[niche] = best_niche_fit
                    self.niche_anchors[niche] = candidates[best_niche_local_idx].copy()

        # Champion is always Candidate 0
        champ_fit = float(fitnesses[0])
        best_cand_idx = int(np.argmax(fitnesses))
        best_cand_fit = float(fitnesses[best_cand_idx])

        # A mutant won if it achieved higher fitness than the Champion in this generation
        mutant_won = (best_cand_idx != 0) and (best_cand_fit > champ_fit + 0.05)
        new_alltime_record = best_cand_fit > self.best_fitness + 0.05

        # -------------------------------------------------------------
        # STRICT BEST POINT REGISTRATION & MONOTONIC UPDATE
        # -------------------------------------------------------------
        if mutant_won or new_alltime_record:
            old_best = self.best_fitness
            self.best_fitness = max(self.best_fitness, best_cand_fit)
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
                "niche": cand_niches[best_cand_idx],
                "metrics": all_metrics[best_cand_idx],
                "saved_at": time.time(),
            }
            torch.save(checkpoint_data, self.output_path)

            cand_tag = f"Mutant #{best_cand_idx} [{cand_niches[best_cand_idx]}]" if best_cand_idx != 0 else "CHAMPION"
            print(f"\n  *****************************************************************")
            print(f"  >>> [НОВЫЙ ЧЕМПИОН ЗАРЕГИСТРИРОВАН!] Кандидат #{best_cand_idx + 1} ({cand_tag})")
            print(f"      Фитнес: {best_cand_fit:.1f} (Превзошел Чемпиона этого поколения: {champ_fit:.1f}, +{best_cand_fit - champ_fit:.1f})")
            print(f"      Веса обновлены и сохранены в: {self.output_path.name}")
            print(f"  *****************************************************************\n")
            # Slightly expand sigma if improving
            self.sigma = min(0.12, self.sigma * 1.02)
        else:
            self.stagnation_counter += 1
            print(
                f"\n  [ЧЕМПИОН СОХРАНЕН] Ни один мутант не превзошел Чемпиона "
                f"(Результат Чемпиона: {champ_fit:.1f} | Лучший за все время: {self.best_fitness:.1f})."
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
        print(f"Base Episode Duration: {self.max_frames} frames (Overtime max: {self.max_overtime_cap})")
        print(f"Generations: {num_generations} | Mode: {'SYNTHETIC' if self.use_synthetic else 'LIVE DESKTOP POLYTRACK'}")
        print(f"Active Checkpoint: {self.output_path.name}")
        print("Current Best Fitness Anchor: " f"{self.best_fitness:.1f} (Gen {self.best_generation})")
        print("=" * 68)

        # Automatic backup of current weights before training modifies them
        self._backup_current_weights()

        # Check menu status advisory at session start (non-blocking)
        if not self.use_synthetic:
            try:
                chk_frame, _ = self.harness.capture.grab_bgr()
                is_rc, rc_reason = self.harness.state_detector.is_on_track(chk_frame)
                if not is_rc:
                    print(f"\n  [ВНИМАНИЕ] Похоже, игра не на трассе ({rc_reason}). Убедитесь, что нажали 'Играть' в PolyTrack!\n")
            except Exception:
                pass

        # Baseline calibration:
        # If fitness is unknown OR if in live mode and saved fitness came from synthetic mode (> 1000):
        needs_live_baseline = not self.use_synthetic and (self.best_fitness > 1000.0 or self.best_fitness == -float("inf"))
        if self.best_fitness == -float("inf") or needs_live_baseline:
            mode_tag = "в живой игре PolyTrack" if not self.use_synthetic else "на синтетическом треке"
            print(f"\n[КАЛИБРОВКА] Измерение реального базового уровня Чемпиона {mode_tag}...")
            base_fit, base_met = self.evaluate_candidate(self.best_params, candidate_idx=0, is_champion=True)
            self.best_fitness = base_fit
            self.best_generation = 0
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
            print(f"Базовый фитнес Чемпиона установлен: {self.best_fitness:.1f}\n")

        try:
            for g in range(1, num_generations + 1):
                self.train_generation(g)
        finally:
            if self.overlay:
                self.overlay.stop()
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
    parser.add_argument("--max-frames", type=int, default=900, help="Max evaluation frames per episode")
    parser.add_argument("--sigma", type=float, default=config.EVOLUTION_MUTATION_SIGMA, help="Mutation sigma")
    parser.add_argument("--no-overlay", action="store_true", help="Disable in-game neural network overlay during training")

    args = parser.parse_args()

    trainer = EvolutionaryTrainer(
        population_size=args.population,
        mutation_sigma=args.sigma,
        max_eval_frames=args.max_frames,
        use_synthetic=not args.live,
        show_overlay=not args.no_overlay,
    )
    trainer.run(num_generations=args.generations)


if __name__ == "__main__":
    main()
