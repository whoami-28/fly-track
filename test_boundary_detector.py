"""Unit tests for TrackBoundaryDetector (PolyTrack Red/White Curb Perception)."""

from __future__ import annotations

import cv2
import numpy as np

from boundary_detector import BoundaryPerceptionOutput, TrackBoundaryDetector


def _generate_synthetic_track(
    width: int = 480,
    height: int = 360,
    curve: str = "straight",
    car_offset_x: int = 0,
) -> np.ndarray:
    """Generate synthetic test frame with red-and-white alternating curbs."""
    frame = np.full((height, width, 3), (235, 206, 135), dtype=np.uint8)  # Sky
    frame[int(height * 0.38) :, :] = (55, 155, 60)  # Terrain

    t = np.linspace(0.40, 1.0, 25)
    y = (t * height).astype(int)

    if curve == "straight":
        xl = (width * 0.42 - (t - 0.40) * width * 0.30 - car_offset_x).astype(int)
        xr = (width * 0.58 + (t - 0.40) * width * 0.30 - car_offset_x).astype(int)
    elif curve == "right":
        xl = (width * 0.60 + (1.0 - t) ** 1.5 * width * 0.30 - car_offset_x).astype(int)
        xr = xl + (width * 0.40 * t).astype(int)
    elif curve == "left":
        xr = (width * 0.40 - (1.0 - t) ** 1.5 * width * 0.30 - car_offset_x).astype(int)
        xl = xr - (width * 0.40 * t).astype(int)
    else:
        xl = (width * 0.42 - car_offset_x).astype(int)
        xr = (width * 0.58 - car_offset_x).astype(int)

    pts_l = np.column_stack([xl, y])
    pts_r = np.column_stack([xr, y])
    poly = np.vstack([pts_l, pts_r[::-1]])
    cv2.fillPoly(frame, [poly], (75, 75, 80))  # Road

    # Alternating red & white curbs
    for i, (pt_l, pt_r) in enumerate(zip(pts_l, pts_r)):
        c_l = (30, 30, 220) if i % 2 == 0 else (240, 240, 240)
        c_r = (30, 30, 220) if (i + 1) % 2 == 0 else (240, 240, 240)
        radius = max(2, int(pt_l[1] / height * 8))
        cv2.circle(frame, tuple(pt_l), radius, c_l, -1)
        cv2.circle(frame, tuple(pt_r), radius, c_r, -1)

    return frame


def test_straight_track_detection() -> None:
    detector = TrackBoundaryDetector()
    frame = _generate_synthetic_track(curve="straight", car_offset_x=0)
    # Warmup call
    detector.detect(frame)
    out = detector.detect(frame)

    assert isinstance(out, BoundaryPerceptionOutput)
    assert out.corridor_detected is True
    assert len(out.left_curb_pts) > 0
    assert len(out.right_curb_pts) > 0
    assert len(out.center_pts) > 0

    # For a centered car on a straight track, offset and heading are near zero
    assert abs(out.lateral_offset) < 0.15, f"Expected near zero offset, got {out.lateral_offset}"
    assert abs(out.track_heading_deg) < 10.0, f"Expected near zero heading, got {out.track_heading_deg}"
    assert out.proximity_left < 0.5
    assert out.proximity_right < 0.5
    assert out.processing_time_ms < 10.0  # Must be fast (< 10 ms)


def test_right_curve_detection() -> None:
    detector = TrackBoundaryDetector()
    frame = _generate_synthetic_track(curve="right", car_offset_x=0)
    out = detector.detect(frame)

    assert out.corridor_detected is True
    # Heading angle should be strongly positive (rightward turn)
    assert out.track_heading_deg > 2.5, f"Expected positive heading for right curve, got {out.track_heading_deg}"


def test_left_curve_detection() -> None:
    detector = TrackBoundaryDetector()
    frame = _generate_synthetic_track(curve="left", car_offset_x=0)
    out = detector.detect(frame)

    assert out.corridor_detected is True
    # Heading angle should be strongly negative (leftward turn)
    assert out.track_heading_deg < -2.5, f"Expected negative heading for left curve, got {out.track_heading_deg}"


def test_lateral_drift_detection() -> None:
    detector = TrackBoundaryDetector()

    # Car shifted right relative to track (track shifted left in image)
    frame_drift_right = _generate_synthetic_track(curve="straight", car_offset_x=50)
    out_right = detector.detect(frame_drift_right)
    assert out_right.corridor_detected is True
    assert out_right.lateral_offset > 0.15, f"Expected positive offset, got {out_right.lateral_offset}"
    assert out_right.proximity_right > out_right.proximity_left

    # Car shifted left relative to track (track shifted right in image)
    frame_drift_left = _generate_synthetic_track(curve="straight", car_offset_x=-50)
    out_left = detector.detect(frame_drift_left)
    assert out_left.corridor_detected is True
    assert out_left.lateral_offset < -0.15, f"Expected negative offset, got {out_left.lateral_offset}"
    assert out_left.proximity_left > out_left.proximity_right


def test_blank_frame_no_corridor() -> None:
    detector = TrackBoundaryDetector()
    blank = np.full((360, 480, 3), (200, 200, 200), dtype=np.uint8)
    out = detector.detect(blank)

    assert out.corridor_detected is False
    assert out.lateral_offset == 0.0
    assert out.track_heading_deg == 0.0
