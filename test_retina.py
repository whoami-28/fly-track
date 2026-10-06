"""Unit test suite for the Drosophila Retina and Motion Correlator (Module 2)."""

import cv2
import numpy as np

import config
from retina import DrosophilaRetina, RetinaOutput, RetinaVisualizer


def test_retina_output_shapes() -> None:
    retina = DrosophilaRetina(grid_height=16, grid_width=32)
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    retina.process(dummy_frame)  # Warmup
    out = retina.process(dummy_frame)
    assert isinstance(out, RetinaOutput)
    assert out.ommatidia_luminance.shape == (16, 32)
    assert out.delta_luminance.shape == (16, 32)
    assert out.on_channel.shape == (16, 32)
    assert out.off_channel.shape == (16, 32)
    assert out.emd_x.shape == (16, 31)
    assert out.emd_y.shape == (15, 32)
    assert out.stimulation_currents.ndim == 1
    assert len(out.stimulation_currents) == (16 * 32) + (16 * 32) + (16 * 31) + (15 * 32) + 4
    assert out.spikes.shape == out.stimulation_currents.shape
    assert set(np.unique(out.spikes)).issubset({0, 1})
    assert out.processing_time_ms < 10.0


def test_hrc_directional_sensitivity() -> None:
    """Test that Hassenstein-Reichardt correlator detects rightward vs leftward motion."""
    retina = DrosophilaRetina()

    # Create a vertical bar pattern moving to the right
    base_frame = np.zeros((200, 200, 3), dtype=np.uint8)
    base_frame[:, 70:110] = 255  # Bright vertical stripe

    # Frame 1: baseline
    retina.process(base_frame)

    # Frame 2: Shift bar to the RIGHT by 20 pixels
    right_shifted = np.zeros((200, 200, 3), dtype=np.uint8)
    right_shifted[:, 90:130] = 255
    out_right = retina.process(right_shifted)

    # Rightward motion must yield positive mean EMD_x
    assert np.mean(out_right.emd_x) > 0.05, f"Expected positive EMD_x for rightward motion, got {np.mean(out_right.emd_x)}"

    # Reset and test leftward motion
    retina.reset()
    retina.process(base_frame)
    left_shifted = np.zeros((200, 200, 3), dtype=np.uint8)
    left_shifted[:, 50:90] = 255
    out_left = retina.process(left_shifted)

    # Leftward motion must yield negative mean EMD_x
    assert np.mean(out_left.emd_x) < -0.05, f"Expected negative EMD_x for leftward motion, got {np.mean(out_left.emd_x)}"


def test_on_off_channel_separation() -> None:
    retina = DrosophilaRetina()
    dark_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    bright_frame = np.full((100, 100, 3), 255, dtype=np.uint8)

    # Transition 1: Dark -> Bright (Expect high ON, zero OFF)
    retina.process(dark_frame)
    out_on = retina.process(bright_frame)
    assert np.mean(out_on.on_channel) > 0.1
    assert np.max(out_on.off_channel) == 0.0

    # Transition 2: Bright -> Dark (Expect zero ON, high OFF)
    retina.process(bright_frame)
    out_off = retina.process(dark_frame)
    assert np.mean(out_off.off_channel) > 0.1
    assert np.max(out_off.on_channel) == 0.0


def test_retina_dashboard_render() -> None:
    retina = DrosophilaRetina()
    visualizer = RetinaVisualizer()
    frame = np.zeros((600, 800, 3), dtype=np.uint8)
    # Add some road texture
    cv2.circle(frame, (400, 300), 80, (200, 100, 50), -1)

    out = retina.process(frame)
    dashboard = visualizer.render_dashboard(frame, out)

    assert dashboard is not None
    assert dashboard.shape == (760, 1280, 3)
    assert dashboard.dtype == np.uint8


if __name__ == "__main__":
    print("Running Retina unit tests...")
    test_retina_output_shapes()
    print("test_retina_output_shapes passed.")
    test_hrc_directional_sensitivity()
    print("test_hrc_directional_sensitivity passed.")
    test_on_off_channel_separation()
    print("test_on_off_channel_separation passed.")
    test_retina_dashboard_render()
    print("test_retina_dashboard_render passed.")
    print("ALL RETINA TESTS PASSED SUCCESSFULLY!")
