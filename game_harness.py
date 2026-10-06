"""Game Harness for Polytrack: High-speed screen capture, input emulation, and latency tracking.

This module satisfies Module 1 of the Drosophila Vision-Motor Loop:
1. Fast screen capture via mss
2. WASD emulation via pynput
3. Game state detection (fall off track & HUD progress)
4. Sub-40ms latency benchmarking
"""

from __future__ import annotations

import atexit
import ctypes
import ctypes.wintypes
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mss
import numpy as np
from pynput.keyboard import Controller as KeyboardController, KeyCode

import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("GameHarness")

# Set Windows DPI Awareness so screen capture coordinates map 1:1 to physical pixels
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


@dataclass
class WindowRect:
    left: int
    top: int
    width: int
    height: int

    def to_mss_dict(self) -> Dict[str, int]:
        return {
            "left": int(self.left),
            "top": int(self.top),
            "width": int(self.width),
            "height": int(self.height),
        }


def find_windows_by_keywords(keywords: List[str]) -> List[Tuple[int, str, WindowRect]]:
    """Enumerate visible top-level windows matching any of the keyword substrings."""
    matches: List[Tuple[int, str, WindowRect]] = []
    user32 = ctypes.windll.user32

    def enum_windows_proc(hwnd: int, lparam: int) -> bool:
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length == 0:
            return True
        buff = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buff, length + 1)
        title = buff.value

        for kw in keywords:
            if kw.lower() in title.lower():
                # Prefer client rect (canvas only, excluding Windows title bar & borders)
                client_rect = ctypes.wintypes.RECT()
                user32.GetClientRect(hwnd, ctypes.byref(client_rect))
                pt = ctypes.wintypes.POINT(0, 0)
                user32.ClientToScreen(hwnd, ctypes.byref(pt))
                w = client_rect.right - client_rect.left
                h = client_rect.bottom - client_rect.top

                if w > 100 and h > 100:
                    matches.append((hwnd, title, WindowRect(pt.x, pt.y, w, h)))
                else:
                    # Fallback to window rect if client rect is unavailable
                    rect = ctypes.wintypes.RECT()
                    user32.GetWindowRect(hwnd, ctypes.byref(rect))
                    w_win = rect.right - rect.left
                    h_win = rect.bottom - rect.top
                    if w_win > 100 and h_win > 100:
                        matches.append((hwnd, title, WindowRect(rect.left, rect.top, w_win, h_win)))
                break
        return True

    EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_int, ctypes.c_int)
    user32.EnumWindows(EnumWindowsProc(enum_windows_proc), 0)
    return matches


def focus_window(hwnd: int) -> None:
    """Bring the target window to foreground."""
    try:
        user32 = ctypes.windll.user32
        user32.ShowWindow(hwnd, 9)  # SW_RESTORE
        user32.SetForegroundWindow(hwnd)
    except Exception as e:
        logger.warning(f"Could not focus window {hwnd}: {e}")


class LatencyTracker:
    """High-precision latency and frame-rate tracker."""

    def __init__(self, max_samples: int = 500) -> None:
        self.max_samples = max_samples
        self.capture_times_ms: List[float] = []
        self.processing_times_ms: List[float] = []
        self.input_times_ms: List[float] = []
        self.total_times_ms: List[float] = []

    def record(self, capture_ms: float, proc_ms: float, input_ms: float) -> float:
        total = capture_ms + proc_ms + input_ms
        self.capture_times_ms.append(capture_ms)
        self.processing_times_ms.append(proc_ms)
        self.input_times_ms.append(input_ms)
        self.total_times_ms.append(total)

        if len(self.total_times_ms) > self.max_samples:
            self.capture_times_ms.pop(0)
            self.processing_times_ms.pop(0)
            self.input_times_ms.pop(0)
            self.total_times_ms.pop(0)

        if total > config.MAX_ALLOWED_LATENCY_MS:
            logger.warning(
                f"Latency spike detected! Total: {total:.2f}ms "
                f"(Cap: {capture_ms:.2f}ms, Proc: {proc_ms:.2f}ms, In: {input_ms:.2f}ms) "
                f"> limit {config.MAX_ALLOWED_LATENCY_MS}ms"
            )
        return total

    def get_summary(self) -> Dict[str, Any]:
        if not self.total_times_ms:
            return {}
        arr = np.array(self.total_times_ms)
        cap = np.array(self.capture_times_ms)
        proc = np.array(self.processing_times_ms)
        inp = np.array(self.input_times_ms)

        mean_total = float(np.mean(arr))
        return {
            "samples": len(arr),
            "mean_total_ms": mean_total,
            "median_total_ms": float(np.median(arr)),
            "p95_total_ms": float(np.percentile(arr, 95)),
            "p99_total_ms": float(np.percentile(arr, 99)),
            "min_total_ms": float(np.min(arr)),
            "max_total_ms": float(np.max(arr)),
            "mean_capture_ms": float(np.mean(cap)),
            "mean_processing_ms": float(np.mean(proc)),
            "mean_input_ms": float(np.mean(inp)),
            "estimated_fps": float(1000.0 / mean_total) if mean_total > 0 else 0.0,
            "within_spec": bool(np.percentile(arr, 95) <= config.MAX_ALLOWED_LATENCY_MS),
        }


class ScreenCapture:
    """Captures desktop screen areas at high framerates using mss.

    Supports synthetic frame fallback when desktop is locked or in headless test environments.
    """

    def __init__(self, bbox: Optional[Dict[str, int]] = None, fallback_synthetic: bool = False) -> None:
        self.sct = mss.MSS() if hasattr(mss, "MSS") else mss.mss()
        self.bbox = bbox or config.DEFAULT_BBOX.copy()
        self.fallback_synthetic = fallback_synthetic

    def set_bbox(self, bbox: Dict[str, int]) -> None:
        self.bbox = bbox.copy()

    @staticmethod
    def generate_synthetic_polytrack_frame(width: int, height: int, is_falling: bool = False) -> np.ndarray:
        """Generate a realistic synthetic Polytrack frame for testing and calibration validation."""
        # Sky background (light cyan/sky tone)
        frame = np.full((height, width, 3), (235, 206, 135), dtype=np.uint8)

        if not is_falling:
            # Draw perspective road polygon
            pts = np.array(
                [
                    [int(width * 0.44), int(height * 0.42)],
                    [int(width * 0.56), int(height * 0.42)],
                    [int(width * 0.82), height],
                    [int(width * 0.18), height],
                ],
                np.int32,
            )
            cv2.fillPoly(frame, [pts], (75, 75, 80))  # Asphalt road
            cv2.polylines(frame, [pts], False, (0, 140, 255), 3)  # Orange curb boundaries

            # Center dashed line
            cv2.line(
                frame,
                (int(width * 0.50), int(height * 0.42)),
                (int(width * 0.50), height),
                (255, 255, 255),
                2,
            )

            # Polytrack vehicle (red low-poly sports car)
            car_pts = np.array(
                [
                    [int(width * 0.48), int(height * 0.74)],
                    [int(width * 0.52), int(height * 0.74)],
                    [int(width * 0.54), int(height * 0.88)],
                    [int(width * 0.46), int(height * 0.88)],
                ],
                np.int32,
            )
            cv2.fillPoly(frame, [car_pts], (30, 30, 210))

            # HUD race timer (top-middle)
            timer_text = "00:12.45"
            cv2.putText(
                frame,
                timer_text,
                (int(width * 0.44), int(height * 0.08)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        return frame

    def grab_bgr(self) -> Tuple[np.ndarray, float]:
        """Grab frame in BGR format. Returns (frame, capture_duration_ms)."""
        t0 = time.perf_counter_ns()
        try:
            sct_img = self.sct.grab(self.bbox)
            # Fast conversion from raw BGRA buffer to BGR numpy array
            frame = np.frombuffer(sct_img.raw, dtype=np.uint8).reshape((sct_img.height, sct_img.width, 4))
            bgr = frame[:, :, :3].copy()
        except Exception as e:
            if self.fallback_synthetic:
                bgr = self.generate_synthetic_polytrack_frame(
                    self.bbox.get("width", 800), self.bbox.get("height", 600)
                )
            else:
                raise RuntimeError(
                    f"Screen capture failed ({e}). Note: On Windows, screen capture requires "
                    "an active, unlocked desktop session. If Windows is locked (Win+L), "
                    "please unlock the desktop or pass fallback_synthetic=True for offline testing."
                ) from e
        dt_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
        return bgr, dt_ms

    def grab_rgb(self) -> Tuple[np.ndarray, float]:
        """Grab frame in RGB format. Returns (frame, capture_duration_ms)."""
        bgr, dt_ms = self.grab_bgr()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb, dt_ms

    def close(self) -> None:
        self.sct.close()


class InputEmulator:
    """Keyboard input emulator for WASD controls with safety cleanup."""

    def __init__(self) -> None:
        self.keyboard = KeyboardController()
        self.active_keys: Dict[str, bool] = {
            config.KEY_FORWARD: False,
            config.KEY_BRAKE: False,
            config.KEY_LEFT: False,
            config.KEY_RIGHT: False,
        }
        # Pre-resolve key codes
        self._key_codes = {k: KeyCode.from_char(k) for k in self.active_keys}
        self._reset_key = KeyCode.from_char(config.KEY_RESET)
        # Register atexit to prevent keys being stuck down if terminated unexpectedly
        atexit.register(self.release_all)

    def set_actions(self, w: bool = False, a: bool = False, s: bool = False, d: bool = False) -> float:
        """Apply desired WASD state.

        Only dispatches OS events when state changes to minimize event queue overhead.
        Returns dispatch duration in ms.
        """
        t0 = time.perf_counter_ns()
        target = {
            config.KEY_FORWARD: bool(w),
            config.KEY_LEFT: bool(a),
            config.KEY_BRAKE: bool(s),
            config.KEY_RIGHT: bool(d),
        }

        for key_char, should_press in target.items():
            is_pressed = self.active_keys[key_char]
            if should_press and not is_pressed:
                self.keyboard.press(self._key_codes[key_char])
                self.active_keys[key_char] = True
            elif not should_press and is_pressed:
                self.keyboard.release(self._key_codes[key_char])
                self.active_keys[key_char] = False

        dt_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
        return dt_ms

    def reset_game(self) -> None:
        """Trigger track restart via the reset key ('r')."""
        self.release_all()
        self.keyboard.press(self._reset_key)
        time.sleep(0.05)
        self.keyboard.release(self._reset_key)

    def release_all(self) -> None:
        """Emergency release for all controlled keys."""
        for key_char, is_pressed in list(self.active_keys.items()):
            if is_pressed:
                try:
                    self.keyboard.release(self._key_codes[key_char])
                except Exception:
                    pass
                self.active_keys[key_char] = False


class GameStateDetector:
    """Detects game state: fall off track, void detection, and timer/progress tracking."""

    def __init__(self, variance_threshold: float = config.FALL_CONTRAST_VAR_THRESHOLD) -> None:
        self.variance_threshold = variance_threshold
        self.fall_streak: int = 0
        self.last_hud_snapshot: Optional[np.ndarray] = None

    def detect_fall(self, frame_bgr: np.ndarray) -> Tuple[bool, float]:
        """Detect if the car has fallen off the track.

        Method: Polytrack tracks contain high-contrast geometric road boundaries and texture.
        When falling into the sky or void, spatial edge variance (Laplacian variance) collapses.
        Returns: (is_falling, laplacian_variance)
        """
        # Downsample frame for fast variance calculation
        h, w = frame_bgr.shape[:2]
        small = cv2.resize(frame_bgr, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        if lap_var < self.variance_threshold:
            self.fall_streak += 1
        else:
            self.fall_streak = max(0, self.fall_streak - 1)

        is_falling = self.fall_streak >= config.FALL_CONSECUTIVE_FRAMES
        return is_falling, lap_var

    def get_hud_timer_roi(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Extract the HUD region where the race timer is positioned."""
        h, w = frame_bgr.shape[:2]
        y1, y2, x1, x2 = config.HUD_TIMER_REL_ROI
        roi = frame_bgr[int(h * y1) : int(h * y2), int(w * x1) : int(w * x2)]
        return roi

    def check_hud_activity(self, frame_bgr: np.ndarray) -> float:
        """Calculate pixel delta in the HUD timer region to verify race is actively progressing."""
        current_hud = cv2.cvtColor(self.get_hud_timer_roi(frame_bgr), cv2.COLOR_BGR2GRAY)
        if self.last_hud_snapshot is None:
            self.last_hud_snapshot = current_hud
            return 0.0

        if current_hud.shape != self.last_hud_snapshot.shape:
            self.last_hud_snapshot = current_hud
            return 0.0

        diff = cv2.absdiff(current_hud, self.last_hud_snapshot)
        delta_score = float(np.mean(diff))
        self.last_hud_snapshot = current_hud
        return delta_score


class PolytrackHarness:
    """Main facade coordinating screen capture, inputs, state detection, and calibration."""

    def __init__(self, config_path: Path = config.CALIBRATION_FILE, fallback_synthetic: bool = False) -> None:
        self.config_path = config_path
        self.bbox = self.load_calibration() or config.DEFAULT_BBOX.copy()
        self.capture = ScreenCapture(self.bbox, fallback_synthetic=fallback_synthetic)
        self.inputs = InputEmulator()
        self.state_detector = GameStateDetector()
        self.latency_tracker = LatencyTracker()

    def load_calibration(self) -> Optional[Dict[str, int]]:
        """Load bounding box calibration from json file if available."""
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    bbox = data.get("bbox")
                    if bbox and all(k in bbox for k in ["top", "left", "width", "height"]):
                        logger.info(f"Loaded calibration from {self.config_path}: {bbox}")
                        return bbox
            except Exception as e:
                logger.error(f"Failed to read calibration file: {e}")
        return None

    def save_calibration(self, bbox: Dict[str, int], metadata: Optional[Dict[str, Any]] = None) -> None:
        """Save bounding box calibration to json file."""
        self.bbox = bbox
        self.capture.set_bbox(bbox)
        data = {
            "bbox": bbox,
            "saved_at": time.time(),
            "metadata": metadata or {},
        }
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved calibration to {self.config_path}: {bbox}")

    def auto_calibrate_from_window(self) -> bool:
        """Search open windows for Polytrack and set bounding box automatically."""
        windows = find_windows_by_keywords(config.WINDOW_TITLE_KEYWORDS)
        if not windows:
            logger.warning("No matching window found for Polytrack.")
            return False

        # Prefer windows explicitly containing Polytrack
        polytrack_matches = [w for w in windows if "polytrack" in w[1].lower()]
        target = polytrack_matches[0] if polytrack_matches else windows[0]
        hwnd, title, rect = target

        logger.info(f"Target window detected: '{title}' (hwnd: {hwnd}, rect: {rect})")
        bbox = rect.to_mss_dict()
        self.save_calibration(bbox, metadata={"window_title": title, "hwnd": hwnd})
        return True

    def interactive_calibrate(self) -> bool:
        """Interactively select the Polytrack game window area using mouse drag."""
        mss_cls = getattr(mss, "MSS", mss.mss)
        with mss_cls() as sct:
            # Grab primary monitor
            mon = sct.monitors[1]
            raw = sct.grab(mon)
            screen = np.frombuffer(raw.raw, dtype=np.uint8).reshape((raw.height, raw.width, 4))[:, :, :3]

        window_name = "Select Polytrack Game Area (Drag ROI and press SPACE/ENTER, or 'c' to cancel)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, min(1280, mon["width"]), min(720, mon["height"]))

        # OpenCV selectROI: returns (x, y, w, h)
        roi = cv2.selectROI(window_name, screen, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow(window_name)

        x, y, w, h = roi
        if w > 50 and h > 50:
            bbox = {
                "left": int(mon["left"] + x),
                "top": int(mon["top"] + y),
                "width": int(w),
                "height": int(h),
            }
            self.save_calibration(bbox, metadata={"type": "interactive_roi"})
            print(f"\n[OK] Calibration successful! New game area: {bbox}\n")
            return True
        else:
            logger.warning("Calibration canceled or selected region too small.")
            return False

    def step(self, w: bool = False, a: bool = False, s: bool = False, d: bool = False) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
        """Run a single synchronized control step:

        1. Capture frame
        2. Detect game state (fall off track, HUD delta)
        3. Dispatch keyboard actions
        4. Record component and total latencies

        Returns: (frame_rgb, is_falling, telemetry_dict)
        """
        # Step 1: Screen grab
        frame_bgr, cap_ms = self.capture.grab_bgr()

        # Step 2: Game state analysis
        t_proc_0 = time.perf_counter_ns()
        is_falling, lap_var = self.state_detector.detect_fall(frame_bgr)
        hud_delta = self.state_detector.check_hud_activity(frame_bgr)
        proc_ms = (time.perf_counter_ns() - t_proc_0) / 1_000_000.0

        # Step 3: Input dispatch
        input_ms = self.inputs.set_actions(w=w, a=a, s=s, d=d)

        # Step 4: Latency record
        total_ms = self.latency_tracker.record(cap_ms, proc_ms, input_ms)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        telemetry = {
            "latency_total_ms": total_ms,
            "latency_capture_ms": cap_ms,
            "latency_proc_ms": proc_ms,
            "latency_input_ms": input_ms,
            "is_falling": is_falling,
            "laplacian_var": lap_var,
            "hud_delta": hud_delta,
            "actions": {"w": w, "a": a, "s": s, "d": d},
        }
        return frame_rgb, is_falling, telemetry

    def run_benchmark(self, num_cycles: int = 150) -> Dict[str, Any]:
        """Run latency and throughput benchmark over N cycles and display statistics."""
        print("=" * 60)
        print(f"BENCHMARKING POLYTRACK HARNESS: {num_cycles} CYCLES")
        print(f"Capture Area: {self.bbox}")
        print(f"Max Allowed Latency Budget: {config.MAX_ALLOWED_LATENCY_MS} ms")
        print("=" * 60)

        # Warmup
        for _ in range(10):
            self.step(w=False, a=False, s=False, d=False)

        # Clean tracker for benchmark
        self.latency_tracker = LatencyTracker(max_samples=num_cycles + 10)

        # Alternating mock inputs to benchmark dispatch
        actions_seq = [
            (True, False, False, False),  # W
            (True, True, False, False),   # W+A
            (True, False, False, True),   # W+D
            (False, False, True, False),  # S
            (False, False, False, False), # Neutral
        ]

        t_start = time.perf_counter()
        for i in range(num_cycles):
            w, a, s, d = actions_seq[i % len(actions_seq)]
            _, _, telem = self.step(w=w, a=a, s=s, d=d)
            if (i + 1) % 50 == 0:
                print(
                    f" Cycle {i + 1:3d}/{num_cycles}: "
                    f"Latency = {telem['latency_total_ms']:.2f} ms "
                    f"(Cap: {telem['latency_capture_ms']:.2f}ms, "
                    f"Proc: {telem['latency_proc_ms']:.2f}ms, "
                    f"Input: {telem['latency_input_ms']:.2f}ms)"
                )
        t_total_wall = time.perf_counter() - t_start

        # Release keys safely
        self.inputs.release_all()

        summary = self.latency_tracker.get_summary()
        summary["wall_time_sec"] = t_total_wall
        summary["throughput_fps"] = float(num_cycles / t_total_wall)

        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS REPORT")
        print("=" * 60)
        print(f" Samples Evaluated:      {summary['samples']}")
        print(f" Throughput:             {summary['throughput_fps']:.1f} FPS")
        print(f" Mean Latency:           {summary['mean_total_ms']:.2f} ms")
        print(f" Median Latency:         {summary['median_total_ms']:.2f} ms")
        print(f" 95th Percentile (P95):  {summary['p95_total_ms']:.2f} ms")
        print(f" 99th Percentile (P99):  {summary['p99_total_ms']:.2f} ms")
        print(f" Min / Max Latency:      {summary['min_total_ms']:.2f} ms / {summary['max_total_ms']:.2f} ms")
        print("-" * 60)
        print(f" Component Breakdown:")
        print(f"   Capture (mss):        {summary['mean_capture_ms']:.2f} ms")
        print(f"   State Proc (OpenCV):  {summary['mean_processing_ms']:.2f} ms")
        print(f"   Input Dispatch:       {summary['mean_input_ms']:.3f} ms")
        print("-" * 60)
        if summary["within_spec"]:
            print(f" [PASS] Latency is well within required 30-40 ms budget!")
        else:
            print(f" [FAIL] P95 Latency ({summary['p95_total_ms']:.2f} ms) exceeded budget {config.MAX_ALLOWED_LATENCY_MS} ms!")
        print("=" * 60 + "\n")
        return summary

    def test_input_sequence(self) -> None:
        """Test sending WASD inputs with safety countdown to observe in-game response."""
        print("\n[INPUT TEST]")
        print("Prepare to focus your Polytrack window!")
        print("Starting 3-second countdown...")
        for i in range(3, 0, -1):
            print(f"  {i}...")
            time.sleep(1)
        print("Testing WASD inputs now:")

        sequence = [
            ("Forward (W)", True, False, False, False, 1.0),
            ("Turn Left (W+A)", True, True, False, False, 0.8),
            ("Turn Right (W+D)", True, False, False, True, 0.8),
            ("Brake / Reverse (S)", False, False, True, False, 0.8),
            ("Neutral (Release All)", False, False, False, False, 0.5),
        ]

        for name, w, a, s, d, duration in sequence:
            print(f" -> Dispatching: {name} for {duration:.1f}s")
            dt = self.inputs.set_actions(w=w, a=a, s=s, d=d)
            print(f"    Dispatch latency: {dt * 1000.0:.1f} us")
            time.sleep(duration)

        self.inputs.release_all()
        print("[INPUT TEST COMPLETED] All keys successfully released.\n")

    def run_live_preview(self) -> None:
        """Open a live visual preview window with real-time HUD and latency statistics overlay."""
        print("\nStarting Live Preview. Press 'q' or ESC in preview window to exit.")
        print("Press 'c' to recalibrate ROI, 'r' to trigger game restart.\n")
        win_name = "Polytrack Drosophila Harness Preview"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

        try:
            while True:
                t0 = time.perf_counter()
                frame_bgr, cap_ms = self.capture.grab_bgr()
                is_falling, lap_var = self.state_detector.detect_fall(frame_bgr)
                hud_delta = self.state_detector.check_hud_activity(frame_bgr)
                dt_total = (time.perf_counter() - t0) * 1000.0

                # Draw HUD timer ROI rectangle
                h, w = frame_bgr.shape[:2]
                y1, y2, x1, x2 = config.HUD_TIMER_REL_ROI
                cv2.rectangle(
                    frame_bgr,
                    (int(w * x1), int(h * y1)),
                    (int(w * x2), int(h * y2)),
                    (0, 255, 255),
                    2,
                )

                # Status banner
                state_text = "FALLING OFF TRACK!" if is_falling else "ALIVE ON TRACK"
                color = (0, 0, 255) if is_falling else (0, 255, 0)
                cv2.putText(frame_bgr, state_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                stats_text = f"Latency: {dt_total:.1f}ms | LapVar: {lap_var:.1f} | HUD Delta: {hud_delta:.1f}"
                cv2.putText(frame_bgr, stats_text, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                cv2.imshow(win_name, frame_bgr)
                key = cv2.waitKey(1) & 0xFF
                if key in [ord("q"), 27]:  # 'q' or ESC
                    break
                elif key == ord("c"):
                    cv2.destroyAllWindows()
                    self.interactive_calibrate()
                    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
                elif key == ord("r"):
                    logger.info("Restart trigger received ('r')")
                    self.inputs.reset_game()
        finally:
            self.inputs.release_all()
            cv2.destroyAllWindows()


def main() -> None:
    """CLI entrypoint for calibration and benchmarking."""
    import argparse

    parser = argparse.ArgumentParser(description="Polytrack Drosophila Harness (Module 1)")
    parser.add_argument("--calibrate", action="store_true", help="Interactive screen ROI calibration")
    parser.add_argument("--autofind", action="store_true", help="Auto-detect Polytrack window position")
    parser.add_argument("--benchmark", action="store_true", help="Run latency and throughput benchmark")
    parser.add_argument("--test-input", action="store_true", help="Run WASD input test sequence")
    parser.add_argument("--preview", action="store_true", help="Live preview with overlay")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic frames (for offline test/benchmark)")
    parser.add_argument("--cycles", type=int, default=150, help="Cycles for benchmark")

    args = parser.parse_args()
    harness = PolytrackHarness(fallback_synthetic=args.synthetic)

    # If no flags passed, list options or run benchmark
    if not any([args.calibrate, args.autofind, args.benchmark, args.test_input, args.preview]):
        print("\nPolytrack Drosophila Game Harness:")
        print("  --autofind   : Auto-detect Polytrack window")
        print("  --calibrate  : Interactively select game screen area")
        print("  --benchmark  : Benchmark capture and input latency (< 30-40ms requirement)")
        print("  --test-input : Test sending WASD keys to game")
        print("  --preview    : Open real-time capture window with diagnostics")
        print("  --synthetic  : Use synthetic Polytrack frame generator (offline / locked screen test)\n")
        print("Running window detection & benchmark...\n")
        if not harness.load_calibration():
            harness.auto_calibrate_from_window()
        harness.run_benchmark(num_cycles=args.cycles)
        return

    if args.autofind:
        harness.auto_calibrate_from_window()
    if args.calibrate:
        harness.interactive_calibrate()
    if args.test_input:
        harness.test_input_sequence()
    if args.benchmark:
        harness.run_benchmark(num_cycles=args.cycles)
    if args.preview:
        harness.run_live_preview()


if __name__ == "__main__":
    main()
