"""Unit tests for Drosophila Flight Recorder Telemetry System.

Tests cover:
- FlightRecorder lifecycle (start_episode, record_frame, end_episode)
- Telemetry CSV schema, values, and disk persistence
- Summary log updating (episodes_summary.csv)
- Offline incident analysis and telemetry CLI report generation
"""

import csv
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from boundary_detector import BoundaryPerceptionOutput
from retina import DrosophilaRetina
from telemetry_logger import (
    FlightRecorder,
    TelemetryFrame,
    analyze_telemetry_file,
    format_incident_report,
)


@pytest.fixture
def temp_log_dir():
    """Create a temporary directory for telemetry logs and clean up afterwards."""
    temp_dir = Path(tempfile.mkdtemp(prefix="telemetry_test_"))
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_flight_recorder_lifecycle(temp_log_dir):
    """Test start, frame recording, and clean completion of an episode."""
    recorder = FlightRecorder(log_dir=temp_log_dir, mode="autopilot")
    assert not recorder.is_active

    recorder.start_episode(episode_id=1, generation=0, candidate_id=0, niche="speedrunner")
    assert recorder.is_active
    assert recorder.episode_id == "1"

    # Mock boundary and mock retina output
    boundary = BoundaryPerceptionOutput(
        corridor_detected=True,
        lateral_offset=0.15,
        track_heading_deg=12.0,
        immediate_heading_deg=2.5,
        norm_pos=0.08,
        proximity_left=0.45,
        proximity_right=0.55,
        is_curb_breached=False,
        is_curb_beached=False,
        departure_side="none",
        bottleneck_ahead=False,
        constriction_risk=0.05,
    )

    class MockBrain:
        compass_heading_deg = 3.5
        compass_bump_strength = 0.88
        dn_spike_rates = {
            "DNa01_L": 0.1,
            "DNa02_L": 0.05,
            "DNa01_R": 0.4,
            "DNa02_R": 0.35,
            "DNp01_L": 0.9,
            "DNp01_R": 0.9,
            "DNb01_L": 0.0,
            "DNb01_R": 0.0,
        }
        actions = {"w": True, "a": False, "s": False, "d": True}
        action_probabilities = {"w": 0.92, "a": 0.08, "s": 0.02, "d": 0.76}

    class MockRetina:
        lptc_vs_forward = 0.045
        lptc_yaw_signal = 0.012

    dec_state = {
        "raw_steer_demand": 0.35,
        "smoothed_steer_demand": 0.28,
        "drift_rate": 0.05,
        "active_steer": "d",
        "lockout_counter": 3,
    }

    # Record 10 frames
    for f in range(1, 11):
        recorder.record_frame(
            frame_idx=f,
            elapsed_sec=f / 30.0,
            boundary=boundary,
            retina=MockRetina(),
            brain=MockBrain(),
            actions=MockBrain.actions,
            action_probs=MockBrain.action_probabilities,
            decoder_state=dec_state,
        )

    assert len(recorder.frames) == 10

    # End episode
    csv_file = recorder.end_episode(
        death_reason="survived",
        fitness=150.0,
        distance=45.0,
        avg_speed=4.5,
    )

    assert not recorder.is_active
    assert csv_file is not None
    assert csv_file.exists()

    # Verify latest file exists
    latest_file = recorder.telemetry_dir / "autopilot_latest.csv"
    assert latest_file.exists()

    # Verify master summary exists
    summary_file = recorder.master_summary_file
    assert summary_file.exists()

    # Read CSV and verify schema
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 10
        first_row = rows[0]
        assert first_row["frame_idx"] == "1"
        assert float(first_row["lateral_offset"]) == pytest.approx(0.15, abs=1e-3)
        assert float(first_row["immediate_heading_deg"]) == pytest.approx(2.5, abs=1e-3)
        assert float(first_row["track_heading_deg"]) == pytest.approx(12.0, abs=1e-3)
        assert first_row["action_w"].strip().lower() in ("true", "1")
        assert first_row["action_d"].strip().lower() in ("true", "1")
        assert first_row["action_a"].strip().lower() in ("false", "0")
        assert float(first_row["raw_steer_demand"]) == pytest.approx(0.35, abs=1e-3)


def test_empty_episode_flush_noop(temp_log_dir):
    """Test that ending an episode with 0 frames recorded does not create spurious CSV files."""
    recorder = FlightRecorder(log_dir=temp_log_dir, mode="trainer")
    recorder.start_episode(episode_id="empty_run")
    csv_file = recorder.end_episode(death_reason="none")

    assert csv_file is None
    assert not (recorder.telemetry_dir / "trainer_latest.csv").exists()


def test_analyzer_parses_crash_incident(temp_log_dir):
    """Test offline incident parser and diagnostic reporter."""
    recorder = FlightRecorder(log_dir=temp_log_dir, mode="autopilot")
    recorder.start_episode(episode_id=42)

    # Simulate 30 frames culminating in a curb breach on the right side
    for f in range(1, 31):
        offset = 0.05 * f / 30.0 if f < 20 else 0.85 + (0.2 * (f - 20) / 10.0)
        breached = f >= 28

        class DynamicBoundary:
            corridor_detected = True
            lateral_offset = offset
            track_heading_deg = 25.0
            immediate_heading_deg = 5.0
            norm_pos = offset * 0.5
            proximity_left = max(0.0, 1.0 - offset)
            proximity_right = min(1.0, offset)
            is_curb_breached = breached
            is_curb_beached = False
            departure_side = "right" if breached else "none"
            bottleneck_ahead = False
            constriction_risk = 0.1

        class DynamicBrain:
            compass_heading_deg = 10.0
            compass_bump_strength = 0.5
            dn_spike_rates = {"DNa01_L": 0.0, "DNa02_L": 0.0, "DNa01_R": 0.5, "DNa02_R": 0.5, "DNp01_L": 0.8, "DNp01_R": 0.8, "DNb01_L": 0.0, "DNb01_R": 0.0}
            actions = {"w": True, "a": False, "s": False, "d": True}
            action_probabilities = {"w": 0.8, "a": 0.1, "s": 0.0, "d": 0.7}

        recorder.record_frame(
            frame_idx=f,
            elapsed_sec=f / 30.0,
            boundary=DynamicBoundary(),
            retina=None,
            brain=DynamicBrain(),
            actions={"w": True, "a": False, "s": False, "d": True},
            action_probs={"w": 0.8, "a": 0.1, "s": 0.0, "d": 0.7},
            decoder_state={"raw_steer_demand": 0.6, "smoothed_steer_demand": 0.5, "drift_rate": 0.0, "active_steer": "d", "lockout_counter": 0},
        )

    csv_path = recorder.end_episode(death_reason="curb_breached_right", distance=120.0, avg_speed=4.0)
    assert csv_path is not None

    stats = analyze_telemetry_file(csv_path, tail=20)
    assert stats["total_frames"] == 30
    assert stats["curb_breach_count"] == 3  # frames 28, 29, 30
    assert stats["duty_w"] == pytest.approx(100.0)
    assert stats["duty_d"] == pytest.approx(100.0)
    assert stats["duty_a"] == pytest.approx(0.0)
    assert len(stats["tail_frames"]) == 20

    report = format_incident_report(stats)
    assert "DROSOPHILA FLIGHT RECORDER: INCIDENT DIAGNOSTIC REPORT" in report
    assert "Total Recorded Frames: 30" in report
    assert "Curb Breaches:        3" in report
    assert "PRE-INCIDENT TRAJECTORY TRACE" in report
