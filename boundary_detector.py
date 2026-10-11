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
    norm_pos: float = 0.0  # Unclipped normalized position (-1.0 = left curb, 0.0 = center, +1.0 = right curb)
    is_curb_breached: bool = False  # True if car center has crossed beyond outer curb (|norm_pos| > 1.15)
    is_curb_beached: bool = False  # True if car is straddling curb (|norm_pos| in [0.85, 1.25])
    departure_side: str = "none"  # "left", "right" or "none"
    immediate_offset: float = 0.0  # Immediate deviation in near zone (0-15m)
    immediate_heading_deg: float = 0.0  # Immediate trajectory in near zone
    bottleneck_ahead: bool = False  # True if road narrows sharply or chicane looms ahead
    constriction_risk: float = 0.0  # Looming risk: 0.0 (open) .. 1.0 (severe constriction)
    aperture_ratio: float = 1.0  # Width ahead / Width near
    left_curb_pts: List[Tuple[int, int]] = field(default_factory=list)
    right_curb_pts: List[Tuple[int, int]] = field(default_factory=list)
    center_pts: List[Tuple[int, int]] = field(default_factory=list)
    processing_time_ms: float = 0.0


class TrackBoundaryDetector:
    """High-speed real-time detector for PolyTrack red-and-white track curbs."""

    def __init__(
        self,
        horizon_rel_y: float = 0.38,
        bottom_rel_y: float = 0.72,
        scan_step_y: int = 4,
        min_corridor_width_px: int = 24,
    ) -> None:
        self.horizon_rel_y = horizon_rel_y
        self.bottom_rel_y = bottom_rel_y
        self.scan_step_y = max(2, scan_step_y)
        self.min_corridor_width_px = min_corridor_width_px

        # HSV Threshold ranges for PolyTrack rumble strips
        # Red wraps around 0 and 180 in OpenCV HSV (H: 0-180, S: 0-255, V: 0-255)
        # Widened to catch real PolyTrack curb variations (darker/brighter, compressed by game renderer)
        self.red_lower1 = np.array([0, 50, 50], dtype=np.uint8)    # Red-orange low end (включая оранжевый)
        self.red_upper1 = np.array([18, 255, 255], dtype=np.uint8)  # Covers orange-red → pure red
        self.red_lower2 = np.array([158, 50, 50], dtype=np.uint8)   # Magenta-red high end (wrap-around)
        self.red_upper2 = np.array([180, 255, 255], dtype=np.uint8)

        # Orange-brown curb band (PolyTrack desert/brick curbs)
        self.orange_lower = np.array([8, 80, 80], dtype=np.uint8)
        self.orange_upper = np.array([28, 255, 255], dtype=np.uint8)

        # White curb stripes: low saturation, high luminance (also catches light grey)
        self.white_lower = np.array([0, 0, 160], dtype=np.uint8)    # Lowered V threshold to catch dim white
        self.white_upper = np.array([180, 60, 255], dtype=np.uint8)  # Slightly raised S to catch grey-white

        # Temporal filter buffers to eliminate scanline flicker from red/white texture blocks
        self._smooth_lateral_offset: Optional[float] = None
        self._smooth_heading_deg: Optional[float] = None

    def reset(self) -> None:
        """Reset temporal smoothing filters."""
        self._smooth_lateral_offset = None
        self._smooth_heading_deg = None

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
        mask_orange = cv2.inRange(roi_hsv, self.orange_lower, self.orange_upper)  # Оранжевые поребрики
        mask_white = cv2.inRange(roi_hsv, self.white_lower, self.white_upper)
        curb_mask = cv2.bitwise_or(cv2.bitwise_or(mask_red, mask_orange), mask_white)

        # 3. Mask out on-screen UI overlays and player car body:
        # Bottom-left touch d-pad (white arrows on mobile/browser viewport)
        ui_bl_y = int(h * 0.65) - y_min
        if ui_bl_y < (y_max - y_min):
            ui_bl_x = int(w * 0.22)
            curb_mask[max(0, ui_bl_y):, :ui_bl_x] = 0

        # Bottom-right speedometer / timer UI box
        ui_br_y = int(h * 0.70) - y_min
        if ui_br_y < (y_max - y_min):
            ui_br_x = int(w * 0.85)
            curb_mask[max(0, ui_br_y):, ui_br_x:] = 0

        # Central player car body zone (roof / cockpit / spoiler / antenna)
        car_y = max(0, int(h * 0.58) - y_min)
        if car_y < (y_max - y_min):
            car_x1 = int(w * 0.43)
            car_x2 = int(w * 0.57)
            curb_mask[car_y:, car_x1:car_x2] = 0

        # 4. Adaptive horizontal scanline boundary extraction
        mid_x = w // 2
        left_curb_pts: List[Tuple[int, int]] = []
        right_curb_pts: List[Tuple[int, int]] = []
        center_pts: List[Tuple[int, int]] = []

        roi_height = y_max - y_min
        prev_cx: Optional[int] = None

        corridor_rows: List[Tuple[int, int, int, int]] = []  # (lx, rx, cx, row_y)

        for y_rel in range(0, roi_height, self.scan_step_y):
            row_y = y_min + y_rel
            row = curb_mask[y_rel, :]

            curb_indices = np.where(row > 0)[0]
            curb_count = len(curb_indices)
            if curb_count == 0 or curb_count > int(w * 0.55):
                continue

            ref_x = prev_cx if prev_cx is not None else mid_x

            # Detect asphalt gap between left and right curbs (gap > 15 pixels)
            if len(curb_indices) >= 2:
                diffs = np.where(np.diff(curb_indices) > 15)[0]
                if len(diffs) > 0:
                    lx = int(curb_indices[diffs[0]])
                    rx = int(curb_indices[diffs[-1] + 1])
                    if (rx - lx) >= self.min_corridor_width_px:
                        cx = (lx + rx) // 2
                        if prev_cx is None or abs(cx - prev_cx) <= int(w * 0.15):
                            left_curb_pts.append((lx, row_y))
                            right_curb_pts.append((rx, row_y))
                            center_pts.append((cx, row_y))
                            corridor_rows.append((lx, rx, cx, row_y))
                            prev_cx = cx
                            continue

            # Fallback for single-curb rows
            if curb_indices[0] < ref_x:
                left_curb_pts.append((int(curb_indices[-1]), row_y))
            else:
                right_curb_pts.append((int(curb_indices[0]), row_y))

        # 5. Synthesize centering, curvature, and proximity cues
        corridor_detected = len(corridor_rows) >= 3 or (len(left_curb_pts) >= 4 and len(right_curb_pts) >= 4)
        lateral_offset = 0.0
        norm_pos = 0.0
        is_curb_breached = False
        is_curb_beached = False
        departure_side = "none"
        track_heading_deg = 0.0
        proximity_left = 0.0
        proximity_right = 0.0
        immediate_offset = 0.0
        immediate_heading_deg = 0.0
        bottleneck_ahead = False
        constriction_risk = 0.0
        aperture_ratio = 1.0

        if len(corridor_rows) >= 4:
            near_l, near_r, near_c_x, near_y = corridor_rows[-1]
            far_l, far_r, far_c_x, far_y = corridor_rows[0]

            # Lookahead road heading angle (horizon zone)
            dx = far_c_x - near_c_x
            dy = near_y - far_y
            if dy > 0:
                track_heading_deg = float(np.degrees(np.arctan2(dx, dy)))
                track_heading_deg = float(np.clip(track_heading_deg, -90.0, 90.0))

            # Immediate road heading angle (near zone: 5-15m ahead)
            near_step = max(2, len(corridor_rows) // 3)
            imm_row = corridor_rows[-near_step]
            dx_imm = imm_row[2] - near_c_x
            dy_imm = near_y - imm_row[3]
            if dy_imm > 0:
                immediate_heading_deg = float(np.degrees(np.arctan2(dx_imm, dy_imm)))
                immediate_heading_deg = float(np.clip(immediate_heading_deg, -90.0, 90.0))
            else:
                immediate_heading_deg = track_heading_deg

            # Continuous, normalized proximity & centering metrics from synchronized near row
            half_w = max(20.0, (near_r - near_l) / 2.0)
            norm_pos = float((mid_x - near_c_x) / half_w)
            lateral_offset = float(np.clip(norm_pos, -1.0, 1.0))
            immediate_offset = lateral_offset

            # Curb breach and beached status
            is_curb_breached = abs(norm_pos) > 1.15
            is_curb_beached = 0.85 <= abs(norm_pos) <= 1.25
            departure_side = "left" if norm_pos < -0.85 else ("right" if norm_pos > 0.85 else "none")

            # Proximity is 0.0 within safe 40% central corridor, ramps smoothly to 1.0 at curb edge
            if norm_pos > 0.40:
                proximity_right = float(np.clip((norm_pos - 0.40) / 0.60, 0.0, 1.0))
                proximity_left = 0.0
            elif norm_pos < -0.40:
                proximity_left = float(np.clip((-norm_pos - 0.40) / 0.60, 0.0, 1.0))
                proximity_right = 0.0

            # Mid-zone corridor width & Constriction detection (Looming aperture)
            mid_l_idx = len(left_curb_pts) // 2
            mid_r_idx = len(right_curb_pts) // 2
            corridor_width_near = max(self.min_corridor_width_px, float(near_r - near_l))
            corridor_width_mid = max(1.0, float(right_curb_pts[mid_r_idx][0] - left_curb_pts[mid_l_idx][0]))
            expected_mid_width = max(self.min_corridor_width_px, corridor_width_near * 0.55)
            aperture_ratio = float(np.clip(corridor_width_mid / expected_mid_width, 0.1, 2.0))

            mid_dist_l = float(mid_x - left_curb_pts[mid_l_idx][0])
            mid_dist_r = float(right_curb_pts[mid_r_idx][0] - mid_x)
            is_pinched = (mid_dist_l < corridor_width_near * 0.18) or (mid_dist_r < corridor_width_near * 0.18)

            if aperture_ratio < 0.70 or is_pinched:
                bottleneck_ahead = True
                constriction_risk = float(np.clip(max(1.0 - aperture_ratio, 0.6 if is_pinched else 0.0), 0.0, 1.0))

        elif len(left_curb_pts) >= 4 and len(right_curb_pts) < 4:
            # Only left curb visible
            near_l = left_curb_pts[-1]
            far_l = left_curb_pts[0]
            dx = far_l[0] - near_l[0]
            dy = near_l[1] - far_l[1]
            if dy > 0:
                track_heading_deg = float(np.degrees(np.arctan2(dx, dy)))
                track_heading_deg = float(np.clip(track_heading_deg, -90.0, 90.0))
            immediate_heading_deg = track_heading_deg

            dist_left = float(mid_x - near_l[0])
            if dist_left < 0:
                # Car center is to the LEFT of the left curb -> breached off track left!
                norm_pos = -1.50
                is_curb_breached = True
                is_curb_beached = False
                departure_side = "left"
                proximity_left = 1.0
                lateral_offset = -1.0
            elif dist_left < w * 0.12:
                proximity_left = 0.85
                lateral_offset = -0.75
                norm_pos = -0.90
                is_curb_beached = True
                departure_side = "left"
            else:
                lateral_offset = 0.45
                norm_pos = 0.45
                departure_side = "none"
            immediate_offset = lateral_offset

        elif len(right_curb_pts) >= 4 and len(left_curb_pts) < 4:
            # Only right curb visible
            near_r = right_curb_pts[-1]
            far_r = right_curb_pts[0]
            dx = far_r[0] - near_r[0]
            dy = near_r[1] - far_r[1]
            if dy > 0:
                track_heading_deg = float(np.degrees(np.arctan2(dx, dy)))
                track_heading_deg = float(np.clip(track_heading_deg, -90.0, 90.0))
            immediate_heading_deg = track_heading_deg

            dist_right = float(near_r[0] - mid_x)
            if dist_right < 0:
                # Car center is to the RIGHT of the right curb -> breached off track right!
                norm_pos = 1.50
                is_curb_breached = True
                is_curb_beached = False
                departure_side = "right"
                proximity_right = 1.0
                lateral_offset = 1.0
            elif dist_right < w * 0.12:
                proximity_right = 0.85
                lateral_offset = 0.75
                norm_pos = 0.90
                is_curb_beached = True
                departure_side = "right"
            else:
                lateral_offset = -0.45
                norm_pos = -0.45
                departure_side = "none"
            immediate_offset = lateral_offset

        # Temporal filter to remove texture block flicker from alternating red/white rumble strips
        if corridor_detected:
            if self._smooth_lateral_offset is None or abs(lateral_offset - self._smooth_lateral_offset) > 0.35:
                self._smooth_lateral_offset = lateral_offset
            else:
                self._smooth_lateral_offset = 0.50 * lateral_offset + 0.50 * self._smooth_lateral_offset

            if self._smooth_heading_deg is None or abs(track_heading_deg - self._smooth_heading_deg) > 30.0:
                self._smooth_heading_deg = track_heading_deg
            else:
                self._smooth_heading_deg = 0.50 * track_heading_deg + 0.50 * self._smooth_heading_deg

            lateral_offset = float(self._smooth_lateral_offset)
            track_heading_deg = float(self._smooth_heading_deg)
        else:
            self._smooth_lateral_offset = None
            self._smooth_heading_deg = None

        proc_ms = (time.perf_counter_ns() - t0) / 1_000_000.0

        return BoundaryPerceptionOutput(
            lateral_offset=lateral_offset,
            track_heading_deg=track_heading_deg,
            proximity_left=proximity_left,
            proximity_right=proximity_right,
            corridor_detected=corridor_detected,
            norm_pos=norm_pos,
            is_curb_breached=is_curb_breached,
            is_curb_beached=is_curb_beached,
            departure_side=departure_side,
            immediate_offset=immediate_offset,
            immediate_heading_deg=immediate_heading_deg,
            bottleneck_ahead=bottleneck_ahead,
            constriction_risk=constriction_risk,
            aperture_ratio=aperture_ratio,
            left_curb_pts=left_curb_pts,
            right_curb_pts=right_curb_pts,
            center_pts=center_pts,
            processing_time_ms=proc_ms,
        )
