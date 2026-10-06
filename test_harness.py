"""Unit tests for the Polytrack Game Harness (Module 1)."""

import numpy as np

from game_harness import (
    GameStateDetector,
    InputEmulator,
    LatencyTracker,
    PolytrackHarness,
    ScreenCapture,
)


def test_screen_capture_dimensions() -> None:
    test_bbox = {"top": 100, "left": 100, "width": 400, "height": 300}
    cap = ScreenCapture(test_bbox, fallback_synthetic=True)
    cap.grab_bgr()  # Warmup cold start
    try:
        frame_bgr, dt_ms = cap.grab_bgr()
        assert frame_bgr is not None
        assert frame_bgr.shape == (300, 400, 3)
        assert frame_bgr.dtype == np.uint8
        assert dt_ms >= 0
        assert dt_ms < 50.0

        frame_rgb, _ = cap.grab_rgb()
        assert frame_rgb.shape == (300, 400, 3)
    finally:
        cap.close()


def test_input_emulator_state() -> None:
    inp = InputEmulator()
    try:
        # Initial state all released
        assert not any(inp.active_keys.values())

        # Press W and A
        inp.set_actions(w=True, a=True, s=False, d=False)
        assert inp.active_keys["w"] is True
        assert inp.active_keys["a"] is True
        assert inp.active_keys["s"] is False
        assert inp.active_keys["d"] is False

        # Switch to S and D
        inp.set_actions(w=False, a=False, s=True, d=True)
        assert inp.active_keys["w"] is False
        assert inp.active_keys["a"] is False
        assert inp.active_keys["s"] is True
        assert inp.active_keys["d"] is True

        # Release all
        inp.release_all()
        assert not any(inp.active_keys.values())
    finally:
        inp.release_all()


def test_game_state_detector_fall() -> None:
    detector = GameStateDetector(variance_threshold=50.0)

    # 1. Textured track image (high variance)
    np.random.seed(42)
    textured_track = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    # Give it structured edges
    textured_track[50:150, 50:150] = 0

    for _ in range(6):
        is_falling, var = detector.detect_fall(textured_track)
    assert not is_falling
    assert var > 50.0

    # 2. Blank void / sky image (uniform color = zero variance)
    blank_sky = np.ones((200, 200, 3), dtype=np.uint8) * 200
    for _ in range(6):
        is_falling, var = detector.detect_fall(blank_sky)
    assert is_falling
    assert var < 1.0


def test_game_state_detector_respawn_prompt() -> None:
    detector = GameStateDetector()
    # 1. Normal synthetic frame
    normal = ScreenCapture.generate_synthetic_polytrack_frame(800, 600, is_falling=False, has_respawn_banner=False)
    is_term, is_fall, is_respawn, _, score = detector.detect_fall_or_crash(normal)
    assert not is_respawn, "False positive on normal frame"

    # 2. Frame with respawn banner
    with_banner = ScreenCapture.generate_synthetic_polytrack_frame(800, 600, is_falling=False, has_respawn_banner=True)
    is_term_r, is_fall_r, is_respawn_r, _, score_r = detector.detect_fall_or_crash(with_banner)
    assert is_respawn_r, "Failed to detect respawn banner"
    assert is_term_r, "Terminal state not triggered by banner"
    assert score_r >= 0.70, f"Score too low: {score_r}"


def test_latency_tracker_statistics() -> None:
    tracker = LatencyTracker()
    for _ in range(50):
        tracker.record(capture_ms=5.0, proc_ms=2.0, input_ms=0.5)

    summary = tracker.get_summary()
    assert summary["samples"] == 50
    assert abs(summary["mean_total_ms"] - 7.5) < 0.1
    assert summary["within_spec"] is True
    assert summary["estimated_fps"] > 100.0


def test_harness_step_cycle() -> None:
    harness = PolytrackHarness(fallback_synthetic=True)
    try:
        frame, is_falling, telemetry = harness.step(w=True, a=False, s=False, d=False)
        assert frame is not None
        assert frame.ndim == 3
        assert "latency_total_ms" in telemetry
        assert telemetry["latency_total_ms"] < 40.0
        assert telemetry["actions"]["w"] is True
        assert "is_terminal" in telemetry
        assert "is_respawn_prompt" in telemetry
    finally:
        harness.inputs.release_all()


if __name__ == "__main__":
    print("Running tests manually...")
    test_screen_capture_dimensions()
    print("test_screen_capture_dimensions passed.")
    test_input_emulator_state()
    print("test_input_emulator_state passed.")
    test_game_state_detector_fall()
    print("test_game_state_detector_fall passed.")
    test_game_state_detector_respawn_prompt()
    print("test_game_state_detector_respawn_prompt passed.")
    test_latency_tracker_statistics()
    print("test_latency_tracker_statistics passed.")
    test_harness_step_cycle()
    print("test_harness_step_cycle passed.")
    print("ALL TESTS PASSED SUCCESSFULLY!")
