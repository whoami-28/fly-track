"""Unit tests for NeuralOverlayController and telemetry packet formatting."""

from __future__ import annotations

import time

from neural_overlay import NeuralOverlayController, OverlayTelemetryPacket


def test_overlay_packet_dataclass() -> None:
    pkt = OverlayTelemetryPacket(
        actions={"w": True, "a": False, "s": False, "d": False},
        action_probs={"w": 0.9, "a": 0.1, "s": 0.05, "d": 0.1},
        dn_rates={"DNp01_L": 0.8},
        heading_deg=45.0,
        vs_forward=0.05,
        hs_left=0.01,
        hs_right=0.01,
        lateral_offset=0.0,
        fitness=120.0,
        generation=5,
        fps=30.0,
        curb_status="ok",
    )
    assert pkt.actions["w"] is True
    assert pkt.heading_deg == 45.0
    assert pkt.curb_status == "ok"


def test_overlay_controller_lifecycle() -> None:
    controller = NeuralOverlayController(target_hwnd=0)
    # Start process
    started = controller.start()
    assert started is True
    assert controller.process is not None
    assert controller.process.is_alive()

    # Send a few updates
    for i in range(5):
        controller.update(
            actions={"w": True, "a": False, "s": False, "d": False},
            action_probs={"w": 0.9, "a": 0.1, "s": 0.0, "d": 0.1},
            dn_rates={"DNp01_L": 0.8},
            heading_deg=float(i * 10),
            vs_forward=0.05,
            hs_left=0.0,
            hs_right=0.0,
            lateral_offset=0.0,
            fitness=10.0 * i,
            generation=1,
            fps=30.0,
            curb_status="ok",
        )
        time.sleep(0.02)

    # Stop process cleanly
    controller.stop()
    assert controller.process is None
