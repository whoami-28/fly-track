"""Closed-Loop Orchestrator for Drosophila Vision-Motor Polytrack Agent.

This module implements Module 5 of the Drosophila Vision-Motor Loop:
1. High-speed closed-loop inference loop:
   Screen capture (mss) -> Retina preprocessing (HRC) -> SNN LIF simulation (PyTorch) -> WASD actuation (pynput)
2. Live diagnostic dashboard & HUD overlay:
   - Driving view with Focus of Expansion
   - E-PG Ring Attractor Compass needle (0-360 deg)
   - Real-time WASD pedal/wheel key status
   - Descending Neurons (DN) motor activation gauges
   - Latency monitor and loop FPS counter
3. Game lifecycle management:
   - Automatic track recovery / reset upon fall detection
   - Pause / resume controls ('p' or SPACE)
   - Manual vs Autonomous pilot toggle ('m')
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch

import config
from fly_brain import BrainInferenceOutput, FlyBrain
from game_harness import PolytrackHarness, ScreenCapture
from retina import DrosophilaRetina, RetinaOutput

logger = logging.getLogger("PolytrackFlyMain")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


class PolytrackOrchestrator:
    """Coordinates screen capture, visual processing, SNN brain simulation, and keyboard dispatch."""

    def __init__(
        self,
        use_synthetic: bool = False,
        weights_path: Optional[Path] = None,
        device: str = config.DEVICE,
    ) -> None:
        self.use_synthetic = use_synthetic
        self.device = device

        # Initialize core components
        logger.info("Initializing Polytrack Game Harness...")
        self.harness = PolytrackHarness(fallback_synthetic=use_synthetic)

        logger.info("Initializing Drosophila Retina & Motion Preprocessor...")
        self.retina = DrosophilaRetina()

        logger.info("Initializing Spiking Neural Network (FlyBrain)...")
        self.brain = FlyBrain(k_substeps=config.SNN_SUBSTEPS_PER_FRAME, device=device)

        # Load weights if provided
        if weights_path and Path(weights_path).exists():
            logger.info(f"Loading custom plastic weights from {weights_path}...")
            self.brain.load_weights(Path(weights_path))

        # Runtime control states
        self.is_paused: bool = False
        self.autonomous_mode: bool = True
        self.step_counter: int = 0
        self.episode_start_time: float = time.time()
        self.survival_time_sec: float = 0.0

    def step(self) -> Tuple[np.ndarray, RetinaOutput, BrainInferenceOutput, Dict[str, Any]]:
        """Execute one complete synchronized cycle of the sensory-motor loop.

        Returns: (frame_bgr, retina_output, brain_output, telemetry_dict)
        """
        t0 = time.perf_counter()

        # -------------------------------------------------------------
        # Phase 1: Screen Capture & Game State
        # -------------------------------------------------------------
        if self.use_synthetic:
            self.step_counter += 1
            phase = self.step_counter * 0.04
            is_falling = (self.step_counter // 500) % 2 == 1
            is_respawn = (self.step_counter > 400 and (self.step_counter // 200) % 3 == 0)
            frame_bgr = ScreenCapture.generate_synthetic_polytrack_frame(
                800, 600, is_falling=is_falling, has_respawn_banner=is_respawn
            )
            # Add synthetic steering displacement
            shift = int(np.sin(phase) * 35.0)
            M = np.float32([[1, 0, shift], [0, 1, 0]])
            frame_bgr = cv2.warpAffine(frame_bgr, M, (800, 600), borderMode=cv2.BORDER_REFLECT)
            cap_ms = 1.0
            game_telem = {
                "is_terminal": is_falling or is_respawn,
                "is_falling": is_falling,
                "is_respawn_prompt": is_respawn,
                "prompt_score": 1.0 if is_respawn else 0.1,
                "laplacian_var": 120.0,
                "hud_delta": 0.5,
            }
        else:
            frame_bgr, cap_ms = self.harness.capture.grab_bgr()
            is_terminal, is_falling, is_respawn, lap_var, prompt_score = self.harness.state_detector.detect_fall_or_crash(frame_bgr)
            hud_delta = self.harness.state_detector.check_hud_activity(frame_bgr)
            game_telem = {
                "is_terminal": is_terminal,
                "is_falling": is_falling,
                "is_respawn_prompt": is_respawn,
                "prompt_score": prompt_score,
                "laplacian_var": lap_var,
                "hud_delta": hud_delta,
            }

        # -------------------------------------------------------------
        # Phase 2: Drosophila Retina & Hassenstein-Reichardt Correlator
        # -------------------------------------------------------------
        retina_out = self.retina.process(frame_bgr)

        # -------------------------------------------------------------
        # Phase 3: Spiking Neural Network (LIF Connectome Simulation)
        # -------------------------------------------------------------
        if self.is_paused:
            brain_out = self.brain.forward_frame(np.zeros_like(retina_out.stimulation_currents))
            actions = {"w": False, "a": False, "s": False, "d": False}
        else:
            brain_out = self.brain.forward_frame(retina_out.stimulation_currents, vs_forward=retina_out.lptc_vs_forward)
            actions = brain_out.actions if self.autonomous_mode else {"w": False, "a": False, "s": False, "d": False}

        # -------------------------------------------------------------
        # Phase 4: Actuation / Keyboard Dispatch
        # -------------------------------------------------------------
        if not self.is_paused and self.autonomous_mode:
            input_ms = self.harness.inputs.set_actions(
                w=actions["w"],
                a=actions["a"],
                s=actions["s"],
                d=actions["d"],
            )
        else:
            input_ms = 0.0
            self.harness.inputs.release_all()

        # Handle terminal conditions: respawn prompt or fall off track -> auto-reset game
        if game_telem["is_terminal"]:
            reason = "RESPAWN BANNER PROMPT" if game_telem["is_respawn_prompt"] else "FALL OFF TRACK"
            logger.warning(f"Terminal condition detected ({reason})! Terminating attempt and resetting ('r')...")
            if not self.use_synthetic:
                self.harness.inputs.reset_game()
            self.brain.reset()
            self.retina.reset()
            self.episode_start_time = time.time()

        total_loop_ms = (time.perf_counter() - t0) * 1000.0
        self.survival_time_sec = time.time() - self.episode_start_time

        telemetry = {
            "latency_total_ms": total_loop_ms,
            "latency_cap_ms": cap_ms,
            "latency_retina_ms": retina_out.processing_time_ms,
            "latency_brain_ms": brain_out.inference_time_ms,
            "latency_input_ms": input_ms,
            "survival_time_sec": self.survival_time_sec,
            "is_paused": self.is_paused,
            "autonomous_mode": self.autonomous_mode,
            "is_terminal": game_telem["is_terminal"],
            "is_falling": game_telem["is_falling"],
            "is_respawn_prompt": game_telem["is_respawn_prompt"],
            "prompt_score": game_telem["prompt_score"],
        }

        return frame_bgr, retina_out, brain_out, telemetry

    def render_hud_overlay(
        self,
        frame_bgr: np.ndarray,
        retina_out: RetinaOutput,
        brain_out: BrainInferenceOutput,
        telem: Dict[str, Any],
    ) -> np.ndarray:
        """Compose live visual telemetry HUD window."""
        canvas_h, canvas_w = 640, 1024
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        canvas[:] = (22, 25, 32)

        # 1. Main Driving View (Left: 600 x 450)
        view_w, view_h = 600, 450
        resized_view = cv2.resize(frame_bgr, (view_w, view_h))

        # Draw FOE horizon & Center heading line
        cv2.line(resized_view, (view_w // 2, 0), (view_w // 2, view_h), (0, 165, 255), 1)
        cv2.circle(resized_view, (view_w // 2, int(view_h * 0.45)), 5, (0, 0, 255), -1)

        # Status overlay on view
        status_text = "AUTO-PILOT ACTIVE" if telem["autonomous_mode"] else "MANUAL / PASSIVE"
        status_color = (0, 255, 120) if telem["autonomous_mode"] else (0, 200, 255)
        if telem["is_paused"]:
            status_text = "PAUSED (SPACE to resume)"
            status_color = (0, 140, 255)
        elif telem.get("is_respawn_prompt"):
            status_text = f"RESPAWN BANNER DETECTED (Score: {telem.get('prompt_score', 0):.2f})"
            status_color = (0, 0, 255)
        elif telem.get("is_falling"):
            status_text = "FALL OFF TRACK DETECTED"
            status_color = (0, 0, 255)

        cv2.putText(resized_view, status_text, (20, 35), cv2.FONT_HERSHEY_DUPLEX, 0.7, status_color, 2)
        canvas[50 : 50 + view_h, 20 : 20 + view_w] = resized_view
        cv2.rectangle(canvas, (18, 48), (22 + view_w, 52 + view_h), (60, 68, 85), 1)

        # Title Header
        cv2.putText(
            canvas,
            "POLYTRACK DROSOPHILA VISION-MOTOR CONTROLLER (FlyWire SNN)",
            (20, 32),
            cv2.FONT_HERSHEY_DUPLEX,
            0.65,
            (230, 235, 245),
            1,
        )

        # 2. Key Action Pedals (Under view: x=20, y=515)
        self._draw_key_pedals(canvas, brain_out.actions, 20, 515)

        # 3. E-PG Compass Dial (Right panel top: x=645, y=50)
        self._draw_compass_dial(canvas, brain_out.compass_heading_deg, brain_out.compass_bump_strength, 730, 150, 75)

        # 4. Motor Descending Neuron Firing Rates (Right panel middle: x=645, y=250)
        self._draw_dn_bars(canvas, brain_out.dn_spike_rates, 645, 260)

        # 5. Loop Latency & Performance Breakdown (Right panel bottom: x=645, y=475)
        self._draw_latency_card(canvas, telem, 645, 475)

        # Bottom help bar
        help_str = "Controls: [SPACE] Pause/Resume | [M] Manual/Auto | [R] Reset Track | [Q/ESC] Quit"
        cv2.putText(canvas, help_str, (20, 625), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 160, 180), 1)

        return canvas

    def _draw_key_pedals(self, canvas: np.ndarray, actions: Dict[str, bool], x: int, y: int) -> None:
        """Render active WASD key status indicators."""
        keys = [
            ("W (Gas)", actions.get("w", False), (0, 220, 100)),
            ("A (Left)", actions.get("a", False), (0, 200, 255)),
            ("S (Brake)", actions.get("s", False), (0, 80, 255)),
            ("D (Right)", actions.get("d", False), (0, 200, 255)),
        ]
        kw = 90
        kh = 38
        gap = 12

        cv2.putText(canvas, "Motor WASD Actuation:", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 190, 210), 1)

        for i, (name, active, color) in enumerate(keys):
            kx = x + i * (kw + gap)
            ky = y
            # Key box
            bg = color if active else (40, 45, 58)
            border = (255, 255, 255) if active else (70, 78, 96)
            cv2.rectangle(canvas, (kx, ky), (kx + kw, ky + kh), bg, -1)
            cv2.rectangle(canvas, (kx, ky), (kx + kw, ky + kh), border, 1)

            text_color = (0, 0, 0) if active else (180, 185, 200)
            cv2.putText(canvas, name, (kx + 12, ky + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, text_color, 1)

    def _draw_compass_dial(
        self,
        canvas: np.ndarray,
        heading_deg: float,
        strength: float,
        cx: int,
        cy: int,
        radius: int,
    ) -> None:
        """Render circular E-PG Ring Attractor compass heading dial."""
        # Dial background
        cv2.circle(canvas, (cx, cy), radius, (30, 35, 46), -1)
        cv2.circle(canvas, (cx, cy), radius, (70, 80, 100), 2)

        # Cardinal ticks
        for angle, label in [(0, "N"), (90, "E"), (180, "S"), (270, "W")]:
            rad = math.radians(angle - 90)
            tx = int(cx + (radius - 14) * math.cos(rad))
            ty = int(cy + (radius - 14) * math.sin(rad))
            cv2.putText(canvas, label, (tx - 5, ty + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 170, 190), 1)

        # Compass needle (pointing along heading)
        needle_rad = math.radians(heading_deg - 90)
        nx = int(cx + (radius - 18) * math.cos(needle_rad))
        ny = int(cy + (radius - 18) * math.sin(needle_rad))
        cv2.line(canvas, (cx, cy), (nx, ny), (0, 140, 255), 3)
        cv2.circle(canvas, (cx, cy), 4, (255, 255, 255), -1)

        # Title & values
        cv2.putText(canvas, "E-PG Compass Heading", (cx - 75, cy - radius - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 210, 230), 1)
        val_text = f"{heading_deg:05.1f} deg (Bump: {strength:.2f})"
        cv2.putText(canvas, val_text, (cx - 65, cy + radius + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 200, 255), 1)

    def _draw_dn_bars(self, canvas: np.ndarray, dn_rates: Dict[str, float], x: int, y: int) -> None:
        """Render descending neuron motor firing rates."""
        cv2.putText(canvas, "Descending Neurons (DN Motor Outputs):", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 210, 230), 1)
        bar_w = 180
        bar_h = 12
        gy = y + 15

        for name, rate in dn_rates.items():
            cv2.putText(canvas, f"{name:8s}", (x, gy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (170, 180, 200), 1)
            # Background
            cv2.rectangle(canvas, (x + 85, gy), (x + 85 + bar_w, gy + bar_h), (40, 45, 58), -1)
            # Fill
            fill_len = int(np.clip(rate, 0.0, 1.0) * bar_w)
            # Color code
            color = (0, 220, 100) if "p" in name else ((0, 80, 255) if "b" in name else (0, 200, 255))
            if fill_len > 0:
                cv2.rectangle(canvas, (x + 85, gy), (x + 85 + fill_len, gy + bar_h), color, -1)
            cv2.putText(canvas, f"{rate:.2f}", (x + 85 + bar_w + 8, gy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.34, color, 1)
            gy += 18

    def _draw_latency_card(self, canvas: np.ndarray, telem: Dict[str, Any], x: int, y: int) -> None:
        """Render precise latency and timing telemetry."""
        cv2.rectangle(canvas, (x, y), (x + 350, y + 120), (32, 36, 48), -1)
        cv2.rectangle(canvas, (x, y), (x + 350, y + 120), (55, 62, 80), 1)

        total_ms = telem["latency_total_ms"]
        fps = 1000.0 / max(0.1, total_ms)
        color = (100, 255, 100) if total_ms <= config.MAX_ALLOWED_LATENCY_MS else (0, 0, 255)

        cv2.putText(canvas, f"Loop Turnaround: {total_ms:5.1f} ms ({fps:4.1f} FPS)", (x + 15, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)
        cv2.line(canvas, (x + 15, y + 35), (x + 335, y + 35), (60, 68, 85), 1)

        cv2.putText(canvas, f"* Screen Grab (mss):    {telem['latency_cap_ms']:5.2f} ms", (x + 15, y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 190, 210), 1)
        cv2.putText(canvas, f"* Retina & HRC:         {telem['latency_retina_ms']:5.2f} ms", (x + 15, y + 72), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 190, 210), 1)
        cv2.putText(canvas, f"* FlyBrain SNN (8 LIF): {telem['latency_brain_ms']:5.2f} ms", (x + 15, y + 89), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 190, 210), 1)
        cv2.putText(canvas, f"* Survival Time:        {telem['survival_time_sec']:5.1f} s", (x + 15, y + 106), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 200, 255), 1)

    def run(self) -> None:
        """Main game loop."""
        print("\n" + "=" * 65)
        print("STARTING DROSOPHILA CLOSED-LOOP POLYTRACK AGENT")
        print(f"Mode: {'SYNTHETIC SIMULATION' if self.use_synthetic else 'LIVE DESKTOP POLYTRACK'}")
        print("Press 'q' or ESC in HUD window to quit safely.")
        print("Press SPACE to toggle Pause / Resume.")
        print("Press 'm' to toggle Manual / Autonomous mode.")
        print("Press 'r' to trigger game restart.")
        print("=" * 65 + "\n")

        win_name = "Polytrack Drosophila Pilot Dashboard"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 1024, 640)

        try:
            while True:
                frame_bgr, retina_out, brain_out, telem = self.step()
                hud = self.render_hud_overlay(frame_bgr, retina_out, brain_out, telem)
                cv2.imshow(win_name, hud)

                key = cv2.waitKey(1) & 0xFF
                if key in [ord("q"), 27]:  # 'q' or ESC
                    break
                elif key == ord(" "):  # Space: Pause
                    self.is_paused = not self.is_paused
                    logger.info(f"Pause state toggled: {self.is_paused}")
                elif key == ord("m"):  # 'm': Manual toggle
                    self.autonomous_mode = not self.autonomous_mode
                    logger.info(f"Autonomous pilot toggled: {self.autonomous_mode}")
                elif key == ord("r"):  # 'r': Reset track
                    logger.info("Manual track reset requested ('r').")
                    self.harness.inputs.reset_game()
                    self.brain.reset()
                    self.retina.reset()
                    self.episode_start_time = time.time()

        finally:
            self.harness.inputs.release_all()
            cv2.destroyAllWindows()
            print("\nClosed-loop agent stopped. All keyboard inputs released safely.\n")


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Drosophila Polytrack Autonomous Orchestrator (Module 5)")
    parser.add_argument("--synthetic", action="store_true", help="Run with synthetic Polytrack simulation")
    parser.add_argument("--weights", type=str, default=None, help="Path to trained plastic brain weights (.pt)")
    parser.add_argument("--calibrate", action="store_true", help="Calibrate Polytrack desktop window before running")

    args = parser.parse_args()

    if args.calibrate:
        from game_harness import PolytrackHarness
        harness = PolytrackHarness()
        harness.interactive_calibrate()

    orchestrator = PolytrackOrchestrator(
        use_synthetic=args.synthetic,
        weights_path=Path(args.weights) if args.weights else None,
    )
    orchestrator.run()


if __name__ == "__main__":
    main()
