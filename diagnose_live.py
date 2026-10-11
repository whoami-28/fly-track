"""Live Diagnostic Script — checks keyboard input, boundary detector, and calibration.

Run this WHILE PolyTrack is open and you can see it on screen.
"""
import sys
import time
import ctypes
import json
from pathlib import Path

import cv2
import numpy as np

# ─── 1. Check calibration ───────────────────────────────────────────────────
cfg_path = Path(__file__).parent / "calibration_config.json"
print("=" * 60)
print("STEP 1: Calibration check")
print("=" * 60)
if cfg_path.exists():
    with open(cfg_path) as f:
        cfg = json.load(f)
    bbox = cfg.get("bbox", {})
    print(f"  bbox = {bbox}")
    if bbox.get("width", 0) < 300 or bbox.get("height", 0) < 200:
        print("  [WARN] bbox is too small! Window may not be captured correctly.")
        print("         Run 1_CALIBRATE_WINDOW.bat to recalibrate!")
    else:
        print("  [OK] bbox looks reasonable.")
else:
    print("  [WARN] No calibration_config.json found!")

# ─── 2. Capture a live frame ─────────────────────────────────────────────────
print()
print("=" * 60)
print("STEP 2: Live frame capture")
print("=" * 60)
try:
    from game_harness import PolytrackHarness, ScreenCapture
    h = PolytrackHarness(fallback_synthetic=False)
    print(f"  Using bbox: {h.bbox}")
    frame, cap_ms = h.capture.grab_bgr()
    print(f"  Frame shape: {frame.shape} | Capture time: {cap_ms:.2f}ms")
    out_path = str(Path(__file__).parent / "data" / "diag_live_frame.png")
    Path(out_path).parent.mkdir(exist_ok=True)
    cv2.imwrite(out_path, frame)
    print(f"  Saved live frame to: {out_path}")
    print(f"  Frame mean brightness: {frame.mean():.1f} (should be > 50 if game is visible)")
except Exception as e:
    print(f"  [ERROR] Could not capture live frame: {e}")
    frame = None

# ─── 3. Boundary detector on live frame ─────────────────────────────────────
print()
print("=" * 60)
print("STEP 3: Boundary detector (curb detection)")
print("=" * 60)
if frame is not None:
    from boundary_detector import TrackBoundaryDetector
    det = TrackBoundaryDetector()
    out = det.detect(frame)
    print(f"  corridor_detected: {out.corridor_detected}")
    print(f"  lateral_offset:    {out.lateral_offset:.3f}")
    print(f"  track_heading_deg: {out.track_heading_deg:.1f}")
    print(f"  proximity_left:    {out.proximity_left:.3f}")
    print(f"  proximity_right:   {out.proximity_right:.3f}")
    print(f"  left_curb_pts:     {len(out.left_curb_pts)}")
    print(f"  right_curb_pts:    {len(out.right_curb_pts)}")
    print(f"  center_pts:        {len(out.center_pts)}")
    print(f"  processing_ms:     {out.processing_time_ms:.2f}ms")

    # Sample actual HSV values in the lower part of the frame (where curbs should be)
    h_px, w_px = frame.shape[:2]
    roi_bgr = frame[int(h_px * 0.5):int(h_px * 0.9), :]
    roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

    # Sample random pixels in the curb area — print min/max of each channel
    print()
    print("  --- HSV color analysis in lower 40-90% of frame ---")
    print(f"  H range: [{roi_hsv[:,:,0].min()}, {roi_hsv[:,:,0].max()}]")
    print(f"  S range: [{roi_hsv[:,:,1].min()}, {roi_hsv[:,:,1].max()}]")
    print(f"  V range: [{roi_hsv[:,:,2].min()}, {roi_hsv[:,:,2].max()}]")

    # Check how many red pixels and white pixels the current thresholds catch
    m_r1 = cv2.inRange(roi_hsv, det.red_lower1, det.red_upper1)
    m_r2 = cv2.inRange(roi_hsv, det.red_lower2, det.red_upper2)
    m_red = cv2.bitwise_or(m_r1, m_r2)
    m_white = cv2.inRange(roi_hsv, det.white_lower, det.white_upper)
    total = roi_hsv.shape[0] * roi_hsv.shape[1]
    print(f"  Red pixels detected:   {cv2.countNonZero(m_red)} / {total} ({100*cv2.countNonZero(m_red)/total:.1f}%)")
    print(f"  White pixels detected: {cv2.countNonZero(m_white)} / {total} ({100*cv2.countNonZero(m_white)/total:.1f}%)")

    # Save annotated frame showing the curb mask
    curb_mask = cv2.bitwise_or(m_red, m_white)
    curb_viz = frame[int(h_px * 0.5):int(h_px * 0.9), :].copy()
    curb_viz[curb_mask > 0] = (0, 255, 0)  # Green overlay where curbs detected
    out_mask_path = str(Path(__file__).parent / "data" / "diag_curb_mask.png")
    cv2.imwrite(out_mask_path, curb_viz)
    print(f"  Curb mask overlay saved to: {out_mask_path}")

# ─── 4. Test keyboard input ───────────────────────────────────────────────────
print()
print("=" * 60)
print("STEP 4: Keyboard input test (W key for 2 seconds)")
print("         Watch PolyTrack — the car should start moving!")
print("=" * 60)
if frame is not None:
    try:
        from game_harness import PolytrackHarness
        h2 = PolytrackHarness(fallback_synthetic=False)
        print("  Ensuring window focus...")
        focused = h2.ensure_window_focus()
        print(f"  ensure_window_focus() returned: {focused}")
        time.sleep(0.5)

        # Grab frame before
        f0, _ = h2.capture.grab_bgr()
        print("  Pressing W for 2 seconds...")
        h2.inputs.set_actions(w=True)
        time.sleep(2.0)
        h2.inputs.set_actions(w=False)

        # Grab frame after
        f1, _ = h2.capture.grab_bgr()
        diff = float(np.abs(f0.astype(np.int32) - f1.astype(np.int32)).mean())
        print(f"  Frame diff before/after W: {diff:.2f}")
        if diff < 1.0:
            print("  [FAIL] Frame did NOT change! Keys are NOT reaching PolyTrack.")
            print("         Possible causes:")
            print("         1. PolyTrack window lost focus (click on it manually first)")
            print("         2. PolyTrack runs in browser — click inside the game first")
            print("         3. SendInput is blocked by UAC — run script as Administrator")
        elif diff < 5.0:
            print("  [WARN] Minimal frame change. Car may not be moving properly.")
        else:
            print(f"  [OK] Frame changed significantly (diff={diff:.1f}) — keys working!")

        cv2.imwrite(str(Path(__file__).parent / "data" / "diag_before_w.png"), f0)
        cv2.imwrite(str(Path(__file__).parent / "data" / "diag_after_w.png"), f1)
        print("  Saved before/after frames to data/")
    except Exception as e:
        print(f"  [ERROR] Keyboard test failed: {e}")

print()
print("=" * 60)
print("DONE. Check data/ folder for diagnostic images.")
print("=" * 60)
