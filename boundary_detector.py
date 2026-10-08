"""Track Boundary and Curbstone Detector for PolyTrack (Drosophila Centering Reflex).

This module implements the edge/curb perception pathway inspired by the Drosophila visual system:
In fruit flies, visual flight down corridors is mediated by the Optomotor Centering Response
(Srinivasan et al. 1991, 1996; Maimon et al. 2010). Flies balance the optical contrast and
velocity between their left and right eyes to steer safely away from boundaries and walls.

In PolyTrack, the road corridor is marked by high-contrast alternating red-and-white rumble strips
(curbstones / поребрики). This detector:
1. Segments red and white curb markings using vectorised HSV thresholding.
2. Performs adaptive scanline sampling to extract left and right track boundary coordinates.
3. Computes the corridor centerline, normalized lateral offset (car position vs track center),
   lookahead curvature/heading angle, and proximity collision risk to left/right curbs.
4. Executes in < 2.0 ms to preserve the strict sub-40ms real-time loop budget.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class BoundaryPerceptionOutput:
    """Encapsulates perceived track boundaries and centering metrics."""

    lateral_offset: float  # Normalized deviation: -1.0 (far left) .. 0.0 (center) .. +1.0 (far right)
    track_heading_deg: float  # Lookahead road direction in degrees (-90 left .. +90 right)
    proximity_left: float  # Collision danger to left curb [0.0 safe .. 1.0 collision]
    proximity_right: float  # Collision danger to right curb [0.0 safe .. 1.0 collision]
    corridor_detected: bool  # True if both or strong curbs are recognized
    left_curb_pts: List[Tuple[int, int]] = field(default_factory=list)
    right_curb_pts: List[Tuple[int, int]] = field(default_factory=list)
    center_pts: List[Tuple[int, int]] = field(default_factory=list)
    processing_time_ms: float = 0.0


class TrackBoundaryDetector:
    """High-speed real-time detector for PolyTrack red-and-white track curbs."""

    def __init__(
        self,
        horizon_rel_y: float = 0.38,
        bottom_rel_y: float = 0.96,
        scan_step_y: int = 4,
        min_corridor_width_px: int = 24,
    ) -> None:
        self.horizon_rel_y = horizon_rel_y
        self.bottom_rel_y = bottom_rel_y
        self.scan_step_y = max(2, scan_step_y)
        self.min_corridor_width_px = min_corridor_width_px

        # HSV Threshold ranges for PolyTrack rumble strips
        # Red wraps around 0 and 180 in OpenCV HSV (H: 0-180, S: 0-255, V: 0-255)
        self.red_lower1 = np.array([0, 75, 75], dtype=np.uint8)
        self.red_upper1 = np.array([14, 255, 255], dtype=np.uint8)
        self.red_lower2 = np.array([166, 75, 75], dtype=np.uint8)
        self.red_upper2 = np.array([180, 255, 255], dtype=np.uint8)

        # White curb stripes: low saturation, high luminance
        self.white_lower = np.array([0, 0, 180], dtype=np.uint8)
        self.white_upper = np.array([180, 45, 255], dtype=np.uint8)

    def detect(self, frame_bgr: np.ndarray) -> BoundaryPerceptionOutput:
        """Detect track boundaries, centerline, and lateral offset from a video frame.

        Args:
            frame_bgr: NumPy array of shape (H, W, 3) in BGR color format.
        Returns:
            BoundaryPerceptionOutput containing centering signals and visual overlay coordinates.
        """
        t0 = time.perf_counter_ns()
        h, w = frame_bgr.shape[:2]

        # 1. Restrict analysis to driving road ROI (below the sky/horizon)
        y_min = int(h * self.horizon_rel_y)
        y_max = int(h * self.bottom_rel_y)
        roi_bgr = frame_bgr[y_min:y_max, :]
        roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

        # 2. Vectorized color segmentation of red and white stripes
        m_r1 = cv2.inRange(roi_hsv, self.red_lower1, self.red_upper1)
        m_r2 = cv2.inRange(roi_hsv, self.red_lower2, self.red_upper2)
        mask_red = cv2.bitwise_or(m_r1, m_r2)
        mask_white = cv2.inRange(roi_hsv, self.white_lower, self.white_upper)
        curb_mask = cv2.bitwise_or(mask_red, mask_white)

        # 3. Adaptive horizontal scanline boundary extraction
        mid_x = w // 2
        left_curb_pts: List[Tuple[int, int]] = []
        right_curb_pts: List[Tuple[int, int]] = []
        center_pts: List[Tuple[int, int]] = []

        roi_height = y_max - y_min
        for y_rel in range(0, roi_height, self.scan_step_y):
            row_y = y_min + y_rel
            row = curb_mask[y_rel, :]

            curb_indices = np.where(row > 0)[0]
            curb_count = len(curb_indices)
            # A genuine curb strip is bounded; if the whole screen is white (e.g. blank screen), ignore
            if curb_count == 0 or curb_count > int(w * 0.55):
                continue

            min_x = int(curb_indices[0])
            max_x = int(curb_indices[-1])

            if (max_x - min_x) >= self.min_corridor_width_px:
                # Both curbs are present on this scanline!
                left_curb_pts.append((min_x, row_y))
                right_curb_pts.append((max_x, row_y))
                c_x = (min_x + max_x) // 2
                center_pts.append((c_x, row_y))
            else:
                # Only one curb edge visible on this row
                if min_x < mid_x:
                    left_curb_pts.append((min_x, row_y))
                else:
                    right_curb_pts.append((max_x, row_y))

        # 4. Synthesize centering and heading cues
        corridor_detected = len(center_pts) >= 6 or (len(left_curb_pts) >= 6 and len(right_curb_pts) >= 6)
        lateral_offset = 0.0
        track_heading_deg = 0.0
        proximity_left = 0.0
        proximity_right = 0.0

        if len(center_pts) >= 4:
            # Near center (closest to car in front of wheels)
            near_c = center_pts[-1]
            # Far center (lookahead point towards horizon/curve apex)
            far_c = center_pts[0]

            # Lateral offset: relative distance between car center (mid_x) and track center
            lateral_offset = float(np.clip((mid_x - near_c[0]) / float(w * 0.40), -1.0, 1.0))

            # Lookahead road heading angle
            dx = far_c[0] - near_c[0]
            dy = near_c[1] - far_c[1]
            if dy > 0:
                track_heading_deg = float(np.degrees(np.arctan2(dx, dy)))
                track_heading_deg = float(np.clip(track_heading_deg, -90.0, 90.0))

            # Proximity risk: measure distance from car front center to nearest left/right curbs
            if len(left_curb_pts) > 0 and len(right_curb_pts) > 0:
                near_l = left_curb_pts[-1]
                near_r = right_curb_pts[-1]
                dist_left = max(1.0, float(mid_x - near_l[0]))
                dist_right = max(1.0, float(near_r[0] - mid_x))
                corridor_width = max(self.min_corridor_width_px, float(near_r[0] - near_l[0]))

                proximity_left = float(np.clip(1.0 - (dist_left / (corridor_width * 0.45)), 0.0, 1.0))
                proximity_right = float(np.clip(1.0 - (dist_right / (corridor_width * 0.45)), 0.0, 1.0))

        elif len(left_curb_pts) >= 4 and len(right_curb_pts) < 4:
            # Only left curb visible
            near_l = left_curb_pts[-1]
            far_l = left_curb_pts[0]
            dx = far_l[0] - near_l[0]
            dy = near_l[1] - far_l[1]
            if dy > 0:
                track_heading_deg = float(np.degrees(np.arctan2(dx, dy)))
                track_heading_deg = float(np.clip(track_heading_deg, -90.0, 90.0))

            dist_left = float(mid_x - near_l[0])
            if dist_left < w * 0.15:
                proximity_left = 0.85
                lateral_offset = -0.75
            else:
                lateral_offset = 0.45

        elif len(right_curb_pts) >= 4 and len(left_curb_pts) < 4:
            # Only right curb visible
            near_r = right_curb_pts[-1]
            far_r = right_curb_pts[0]
            dx = far_r[0] - near_r[0]
            dy = near_r[1] - far_r[1]
            if dy > 0:
                track_heading_deg = float(np.degrees(np.arctan2(dx, dy)))
                track_heading_deg = float(np.clip(track_heading_deg, -90.0, 90.0))

            dist_right = float(near_r[0] - mid_x)
            if dist_right < w * 0.15:
                proximity_right = 0.85
                lateral_offset = 0.75
            else:
                lateral_offset = -0.45

        proc_ms = (time.perf_counter_ns() - t0) / 1_000_000.0

        return BoundaryPerceptionOutput(
            lateral_offset=lateral_offset,
            track_heading_deg=track_heading_deg,
            proximity_left=proximity_left,
            proximity_right=proximity_right,
            corridor_detected=corridor_detected,
            left_curb_pts=left_curb_pts,
            right_curb_pts=right_curb_pts,
            center_pts=center_pts,
            processing_time_ms=proc_ms,
        )
