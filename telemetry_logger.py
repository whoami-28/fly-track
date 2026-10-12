"""High-Performance Flight-Recorder Telemetry Logger for Polytrack Drosophila Autopilot & Trainer.

Records detailed frame-by-frame sensory, neural, and motor states to CSV/JSONL for
in-depth flight path analysis, incident diagnosis, and evolutionary telemetry tracking.
"""

import argparse
import csv
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


@dataclass
class TelemetryFrame:
    """Encapsulates telemetry state of a single camera/inference frame."""

    frame_idx: int
    elapsed_sec: float

    # Senses: Curbs & Road Geometry
    lateral_offset: float
    track_heading_deg: float
    immediate_heading_deg: float
    proximity_left: float
    proximity_right: float
    norm_pos: float
    corridor_detected: bool
    is_curb_breached: bool
    is_curb_beached: bool
    departure_side: str
    bottleneck_ahead: bool
    constriction_risk: float

    # Senses: Optic Flow
    vs_forward: float
    hs_yaw: float

    # Brain: E-PG Compass & Descending Neurons
    compass_heading_deg: float
    compass_strength: float
    dn_steer_left: float
    dn_steer_right: float
    dn_throttle: float
    dn_brake: float

    # Motor Decoding & Control State Machine
    raw_steer_demand: float
    steer_demand_smooth: float
    drift_rate: float
    active_steer: str
    lockout_counter: int

    # Probabilities & Output Actions
    prob_w: float
    prob_a: float
    prob_s: float
    prob_d: float
    action_w: bool
    action_a: bool
    action_s: bool
    action_d: bool


class FlightRecorder:
    """Non-blocking in-memory flight recorder with buffered disk flush at episode end."""

    def __init__(
        self,
        mode: str = "autopilot",
        base_dir: Optional[Path] = None,
        log_dir: Optional[Path] = None,
    ) -> None:
        self.mode = mode.lower()
        target_dir = log_dir or base_dir
        self.base_dir = Path(target_dir) if target_dir else Path(__file__).parent / "logs"
        self.telemetry_dir = self.base_dir / "telemetry"
        self.master_summary_file = self.base_dir / "episodes_summary.csv"

        self.telemetry_dir.mkdir(parents=True, exist_ok=True)

        self.episode_id: str = "0"
        self.generation: Optional[int] = None
        self.candidate_id: Optional[int] = None
        self.niche: Optional[str] = None

        self.start_time: float = time.time()
        self.frames: List[TelemetryFrame] = []
        self.is_recording: bool = False

    @property
    def is_active(self) -> bool:
        """True if recorder is currently tracking an episode."""
        return self.is_recording

    def start_episode(
        self,
        episode_id: Union[int, str],
        generation: Optional[int] = None,
        candidate_id: Optional[int] = None,
        niche: Optional[str] = None,
    ) -> None:
        """Start a new recording episode, resetting in-memory frame buffer."""
        self.episode_id = str(episode_id)
        self.generation = generation
        self.candidate_id = candidate_id
        self.niche = niche
        self.start_time = time.time()
        self.frames = []
        self.is_recording = True

    def record_frame(
        self,
        frame_idx: int,
        elapsed_sec: float,
        boundary: Optional[Any] = None,
        retina: Optional[Any] = None,
        brain: Optional[Any] = None,
        actions: Optional[Dict[str, bool]] = None,
        action_probs: Optional[Dict[str, float]] = None,
        decoder_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record telemetry snapshot of the current frame into fast in-memory buffer."""
        if not self.is_recording:
            return

        # 1. Curb & boundary perception
        b_offset = float(getattr(boundary, "lateral_offset", 0.0)) if boundary else 0.0
        b_far_hdg = float(getattr(boundary, "track_heading_deg", 0.0)) if boundary else 0.0
        b_imm_hdg = float(getattr(boundary, "immediate_heading_deg", b_far_hdg)) if boundary else 0.0
        b_prox_l = float(getattr(boundary, "proximity_left", 0.0)) if boundary else 0.0
        b_prox_r = float(getattr(boundary, "proximity_right", 0.0)) if boundary else 0.0
        b_norm_pos = float(getattr(boundary, "norm_pos", 0.0)) if boundary else 0.0
        b_corridor = bool(getattr(boundary, "corridor_detected", False)) if boundary else False
        b_breached = bool(getattr(boundary, "is_curb_breached", False)) if boundary else False
        b_beached = bool(getattr(boundary, "is_curb_beached", False)) if boundary else False
        b_dep_side = str(getattr(boundary, "departure_side", "none")) if boundary else "none"
        b_bottle = bool(getattr(boundary, "bottleneck_ahead", False)) if boundary else False
        b_constrict = float(getattr(boundary, "constriction_risk", 0.0)) if boundary else 0.0

        # 2. Optic flow
        vs_fwd = float(getattr(retina, "lptc_vs_forward", 0.0)) if retina else 0.0
        hs_yaw = float(getattr(retina, "lptc_yaw_signal", 0.0)) if retina else 0.0

        # 3. Brain SNN & Compass
        c_hdg = float(getattr(brain, "compass_heading_deg", 0.0)) if brain else 0.0
        c_bump = float(getattr(brain, "compass_bump_strength", 0.0)) if brain else 0.0

        dn_rates = getattr(brain, "dn_spike_rates", {}) if brain else {}
        dn_l = float(dn_rates.get("DNa01_L", 0.0) + dn_rates.get("DNa02_L", 0.0)) * 0.5
        dn_r = float(dn_rates.get("DNa01_R", 0.0) + dn_rates.get("DNa02_R", 0.0)) * 0.5
        dn_w = float(dn_rates.get("DNp01_L", 0.0) + dn_rates.get("DNp01_R", 0.0)) * 0.5
        dn_s = float(dn_rates.get("DNb01_L", 0.0) + dn_rates.get("DNb01_R", 0.0)) * 0.5

        # 4. Decoder internal state
        dec = decoder_state or {}
        raw_demand = float(dec.get("raw_steer_demand", 0.0))
        smooth_demand = float(dec.get("smoothed_steer_demand", 0.0))
        d_rate = float(dec.get("drift_rate", 0.0))
        act_steer = str(dec.get("active_steer", "none"))
        lockout = int(dec.get("lockout_counter", 0))

        # 5. Output actions and probabilities
        probs = action_probs or (getattr(brain, "action_probabilities", {}) if brain else {})
        pw = float(probs.get("w", 0.0))
        pa = float(probs.get("a", 0.0))
        ps = float(probs.get("s", 0.0))
        pd = float(probs.get("d", 0.0))

        acts = actions or (getattr(brain, "actions", {}) if brain else {})
        aw = bool(acts.get("w", False))
        aa = bool(acts.get("a", False))
        as_ = bool(acts.get("s", False))
        ad = bool(acts.get("d", False))

        tf = TelemetryFrame(
            frame_idx=frame_idx,
            elapsed_sec=elapsed_sec,
            lateral_offset=b_offset,
            track_heading_deg=b_far_hdg,
            immediate_heading_deg=b_imm_hdg,
            proximity_left=b_prox_l,
            proximity_right=b_prox_r,
            norm_pos=b_norm_pos,
            corridor_detected=b_corridor,
            is_curb_breached=b_breached,
            is_curb_beached=b_beached,
            departure_side=b_dep_side,
            bottleneck_ahead=b_bottle,
            constriction_risk=b_constrict,
            vs_forward=vs_fwd,
            hs_yaw=hs_yaw,
            compass_heading_deg=c_hdg,
            compass_strength=c_bump,
            dn_steer_left=dn_l,
            dn_steer_right=dn_r,
            dn_throttle=dn_w,
            dn_brake=dn_s,
            raw_steer_demand=raw_demand,
            steer_demand_smooth=smooth_demand,
            drift_rate=d_rate,
            active_steer=act_steer,
            lockout_counter=lockout,
            prob_w=pw,
            prob_a=pa,
            prob_s=ps,
            prob_d=pd,
            action_w=aw,
            action_a=aa,
            action_s=as_,
            action_d=ad,
        )
        self.frames.append(tf)

    def end_episode(
        self,
        death_reason: str = "survived",
        fitness: float = 0.0,
        distance: float = 0.0,
        avg_speed: float = 0.0,
    ) -> Optional[Path]:
        """Flush episode telemetry to CSV and record summary entry in master log."""
        if not self.is_recording or len(self.frames) == 0:
            self.is_recording = False
            return None

        total_frames = len(self.frames)
        duration_sec = time.time() - self.start_time
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        if self.mode == "trainer" and self.generation is not None:
            filename = f"trainer_gen{self.generation:02d}_cand{self.candidate_id or 0:02d}_{timestamp_str}.csv"
        else:
            filename = f"autopilot_ep{self.episode_id}_{timestamp_str}.csv"

        episode_filepath = self.telemetry_dir / filename
        latest_filepath = self.telemetry_dir / f"{self.mode}_latest.csv"

        # 1. Write detailed per-frame telemetry CSV
        fieldnames = list(asdict(self.frames[0]).keys())
        try:
            with open(episode_filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for frame in self.frames:
                    writer.writerow(asdict(frame))

            # Also write a copy to {mode}_latest.csv for instant inspection
            with open(latest_filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for frame in self.frames:
                    writer.writerow(asdict(frame))

        except Exception as e:
            print(f"[WARN] Failed to write telemetry CSV: {e}", file=sys.stderr)

        # 2. Append to master episodes summary CSV
        self._append_to_master_summary(
            timestamp=datetime.now().isoformat(),
            duration_sec=duration_sec,
            total_frames=total_frames,
            death_reason=death_reason,
            fitness=fitness,
            distance=distance,
            avg_speed=avg_speed,
            filepath=episode_filepath,
        )

        self.is_recording = False
        return episode_filepath

    def _append_to_master_summary(
        self,
        timestamp: str,
        duration_sec: float,
        total_frames: int,
        death_reason: str,
        fitness: float,
        distance: float,
        avg_speed: float,
        filepath: Path,
    ) -> None:
        """Append summary row to master episodes_summary.csv."""
        summary_headers = [
            "timestamp",
            "mode",
            "episode_id",
            "generation",
            "candidate_id",
            "niche",
            "duration_sec",
            "total_frames",
            "fitness",
            "distance",
            "avg_speed",
            "death_reason",
            "telemetry_file",
        ]

        file_exists = self.master_summary_file.exists()
        try:
            with open(self.master_summary_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(summary_headers)
                writer.writerow(
                    [
                        timestamp,
                        self.mode,
                        self.episode_id,
                        self.generation if self.generation is not None else "",
                        self.candidate_id if self.candidate_id is not None else "",
                        self.niche or "",
                        f"{duration_sec:.2f}",
                        total_frames,
                        f"{fitness:.2f}",
                        f"{distance:.2f}",
                        f"{avg_speed:.3f}",
                        death_reason,
                        filepath.name,
                    ]
                )
        except Exception as e:
            print(f"[WARN] Failed to append master summary: {e}", file=sys.stderr)


# =============================================================================
# Telemetry Analysis Utilities (CLI & API)
# =============================================================================


def analyze_telemetry_file(filepath: Path, tail: int = 25) -> Dict[str, Any]:
    """Parse a telemetry CSV file, extract incident statistics, and print a diagnostic report."""
    if not filepath.exists():
        raise FileNotFoundError(f"Telemetry file not found: {filepath}")

    rows: List[Dict[str, str]] = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    total_frames = len(rows)
    if total_frames == 0:
        return {"total_frames": 0, "tail_frames": []}

    # Helper for boolean string check
    def is_true(val: Any) -> bool:
        return str(val).strip().lower() in ("true", "1", "yes")

    dur_sec = float(rows[-1]["elapsed_sec"])
    w_count = sum(1 for r in rows if is_true(r["action_w"]))
    a_count = sum(1 for r in rows if is_true(r["action_a"]))
    s_count = sum(1 for r in rows if is_true(r["action_s"]))
    d_count = sum(1 for r in rows if is_true(r["action_d"]))
    neutral_count = sum(1 for r in rows if not is_true(r["action_a"]) and not is_true(r["action_d"]))

    offsets = [float(r["lateral_offset"]) for r in rows]
    speeds = [float(r["vs_forward"]) for r in rows]

    breach_count = sum(1 for r in rows if is_true(r.get("is_curb_breached", False)))
    beached_count = sum(1 for r in rows if is_true(r.get("is_curb_beached", False)))

    last_row = rows[-1]
    is_breached = is_true(last_row.get("is_curb_breached", False))
    is_beached = is_true(last_row.get("is_curb_beached", False))
    departure_side = last_row.get("departure_side", "none")

    tail_n = min(tail, total_frames)
    tail_rows = rows[-tail_n:]

    stats: Dict[str, Any] = {
        "filepath": str(filepath.name),
        "total_frames": total_frames,
        "duration_sec": dur_sec,
        "mean_speed": float(np.mean(speeds)),
        "max_drift": float(np.max(np.abs(offsets))),
        "w_pct": w_count / total_frames,
        "a_pct": a_count / total_frames,
        "s_pct": s_count / total_frames,
        "d_pct": d_count / total_frames,
        "duty_w": (w_count / total_frames) * 100.0,
        "duty_a": (a_count / total_frames) * 100.0,
        "duty_s": (s_count / total_frames) * 100.0,
        "duty_d": (d_count / total_frames) * 100.0,
        "neutral_pct": neutral_count / total_frames,
        "curb_breach_count": breach_count,
        "curb_beached_count": beached_count,
        "is_breached": is_breached,
        "is_beached": is_beached,
        "departure_side": departure_side,
        "tail_frames": tail_rows,
    }

    report_str = format_incident_report(stats)
    print(report_str)

    return stats


def format_incident_report(stats: Dict[str, Any]) -> str:
    """Format parsed telemetry incident statistics into a diagnostic ASCII report."""
    def is_true(val: Any) -> bool:
        return str(val).strip().lower() in ("true", "1", "yes")

    total_frames = stats.get("total_frames", 0)
    dur_sec = stats.get("duration_sec", 0.0)
    fps = total_frames / max(0.01, dur_sec)

    lines: List[str] = [
        "=" * 75,
        "DROSOPHILA FLIGHT RECORDER: INCIDENT DIAGNOSTIC REPORT",
        "=" * 75,
        f" Log File:             {stats.get('filepath', 'unknown')}",
        f" Total Recorded Frames: {total_frames} ({dur_sec:.2f}s @ {fps:.1f} FPS)",
        f" Mean Optical Speed:   {stats.get('mean_speed', 0.0):.4f}",
        f" Max Lateral Drift:    |offset| = {stats.get('max_drift', 0.0):.3f}",
        f" Curb Breaches:        {stats.get('curb_breach_count', 0)}",
        f" Curb Beached Frames:  {stats.get('curb_beached_count', 0)}",
        f" Actions Duty Cycle:   W: {stats.get('duty_w', 0.0):.1f}% | A: {stats.get('duty_a', 0.0):.1f}% | S: {stats.get('duty_s', 0.0):.1f}% | D: {stats.get('duty_d', 0.0):.1f}%",
        "-" * 75,
    ]

    tail_rows = stats.get("tail_frames", [])
    if tail_rows:
        last = tail_rows[-1]
        if stats.get("is_breached"):
            lines.append(f" [INCIDENT] CURB BREACHED ({stats.get('departure_side', '').upper()}) at Frame {last.get('frame_idx')}")
        elif stats.get("is_beached"):
            lines.append(f" [INCIDENT] BEACHED ON CURB ({stats.get('departure_side', '').upper()}) at Frame {last.get('frame_idx')}")
        else:
            lines.append(f" [STATUS] Final frame offset={last.get('lateral_offset')}, heading={last.get('track_heading_deg')} deg")

        lines.extend([
            "-" * 75,
            f"PRE-INCIDENT TRAJECTORY TRACE (Last {len(tail_rows)} frames):",
            "Frame | Time(s) | Offset | ImmHdg | FarHdg | ProxL | ProxR | Keys | Demand | State",
            "-" * 75,
        ])

        for r in tail_rows:
            keys_str = ""
            keys_str += "W" if is_true(r.get("action_w")) else "."
            keys_str += "A" if is_true(r.get("action_a")) else "."
            keys_str += "S" if is_true(r.get("action_s")) else "."
            keys_str += "D" if is_true(r.get("action_d")) else "."

            f_idx = int(r.get("frame_idx", 0))
            t_sec = float(r.get("elapsed_sec", 0.0))
            off = float(r.get("lateral_offset", 0.0))
            imm = float(r.get("immediate_heading_deg", 0.0))
            far = float(r.get("track_heading_deg", 0.0))
            pl = float(r.get("proximity_left", 0.0))
            pr = float(r.get("proximity_right", 0.0))
            dem = float(r.get("steer_demand_smooth", 0.0))
            st = str(r.get("active_steer", "none"))

            lines.append(f"{f_idx:5d} | {t_sec:6.2f} | {off:+6.2f} | {imm:+6.1f} | {far:+6.1f} | {pl:5.2f} | {pr:5.2f} | {keys_str} | {dem:+6.2f} | {st}")

    lines.append("=" * 75)
    return "\n".join(lines)


def show_summary_table(summary_file: Path, limit: int = 15) -> None:
    """Print master episode summary log table."""
    if not summary_file.exists():
        print(f"[INFO] No summary file found at {summary_file}. Run an autopilot or training session first!")
        return

    rows: List[Dict[str, str]] = []
    with open(summary_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if not rows:
        print("[INFO] Summary file is empty.")
        return

    print("=" * 85)
    print(f"RECENT FLIGHT SESSIONS (Last {min(limit, len(rows))} of {len(rows)} entries)")
    print("=" * 85)
    print("Timestamp           | Mode      | Gen/Cand | Frames | Dist(m) | Fitness | Death Reason")
    print("-" * 85)
    for r in rows[-limit:]:
        ts = r["timestamp"][:19].replace("T", " ")
        mode = r["mode"][:9].ljust(9)
        gen = r.get("generation", "")
        cand = r.get("candidate_id", "")
        gc_str = f"G{gen}:C{cand}".ljust(8) if gen else "AP".ljust(8)
        frames = int(r["total_frames"])
        dist = float(r.get("distance", 0.0))
        fit = float(r.get("fitness", 0.0))
        reason = r.get("death_reason", "")[:18]
        print(f"{ts} | {mode} | {gc_str} | {frames:6d} | {dist:7.1f} | {fit:7.1f} | {reason}")
    print("=" * 85)


def main() -> None:
    parser = argparse.ArgumentParser(description="PolyTrack Flight-Recorder Telemetry Analyzer")
    parser.add_argument("--latest", action="store_true", help="Analyze most recent telemetry run")
    parser.add_argument("--summary", action="store_true", help="Show master episodes summary table")
    parser.add_argument("--file", type=str, default=None, help="Path to specific telemetry CSV file")
    parser.add_argument("--tail", type=int, default=25, help="Number of trailing frames to display")
    args = parser.parse_args()

    base_dir = Path(__file__).parent / "logs"
    telemetry_dir = base_dir / "telemetry"

    if args.summary:
        show_summary_table(base_dir / "episodes_summary.csv")
        return

    target_file: Optional[Path] = None
    if args.file:
        target_file = Path(args.file)
    elif args.latest or len(sys.argv) == 1:
        # Check autopilot_latest.csv or trainer_latest.csv or most recent file in telemetry dir
        candidates = [
            telemetry_dir / "autopilot_latest.csv",
            telemetry_dir / "trainer_latest.csv",
        ]
        for c in candidates:
            if c.exists():
                target_file = c
                break

        if target_file is None and telemetry_dir.exists():
            files = sorted(telemetry_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
            if files:
                target_file = files[0]

    if target_file and target_file.exists():
        analyze_telemetry_file(target_file, tail=args.tail)
    else:
        print("[INFO] No telemetry file found. Run an autopilot session (main.py) or training session (trainer.py) to generate logs.")
        show_summary_table(base_dir / "episodes_summary.csv")


if __name__ == "__main__":
    main()
