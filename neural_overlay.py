"""In-Game Real-Time Neural Network Overlay for PolyTrack.

Visualizes the Drosophila biological connectome and spiking motor telemetry:
- 26 LPTC sensory neurons (Horizontal & Vertical Optic Flow)
- Central Complex 16 E-PG Compass Ring (Heading Bump Attractor)
- 8 Descending Motor Neurons (DN Steering, Throttle, Brake)
- Synaptic connection pathways
- Real-time WASD Actuator Pedals with firing probabilities

Runs in an isolated multiprocessing process positioned at the middle of the
right edge of the PolyTrack game window.
Uses Win32 SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAPTURE) to ensure the overlay
is visible to human eyes on screen, but 100% invisible to mss / ScreenCapture,
preventing any occlusion of the car's computer vision.
"""

from __future__ import annotations

import ctypes
import math
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass
class OverlayTelemetryPacket:
    """State packet transmitted from main/trainer loop to overlay process."""

    actions: Dict[str, bool]
    action_probs: Dict[str, float]
    dn_rates: Dict[str, float]
    heading_deg: float
    vs_forward: float
    hs_left: float
    hs_right: float
    lateral_offset: float
    fitness: float = 0.0
    generation: int = 0
    fps: float = 0.0
    curb_status: str = "ok"  # "ok", "breached", "beached"


def _apply_win32_overlay_styles(hwnd: int) -> None:
    """Apply Win32 styles: Click-through, Layered, and Exclude from desktop screen capture."""
    if sys.platform != "win32":
        return

    try:
        user32 = ctypes.windll.user32
        GWL_EXSTYLE = -20
        WS_EX_TRANSPARENT = 0x00000020
        WS_EX_LAYERED = 0x00080000
        WS_EX_TOOLWINDOW = 0x00000080

        # 1. Enable layered click-through
        cur_style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        user32.SetWindowLongW(
            hwnd, GWL_EXSTYLE, cur_style | WS_EX_TRANSPARENT | WS_EX_LAYERED | WS_EX_TOOLWINDOW
        )

        # 2. Hardware-level screen capture exclusion (WDA_EXCLUDEFROMCAPTURE = 0x11)
        # Keeps overlay visible to human player eyes, but BitBlt/mss grabs the game underneath cleanly!
        WDA_EXCLUDEFROMCAPTURE = 0x00000011
        user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)
    except Exception as e:
        # Fallback gracefully if graphics driver or API is unavailable
        pass


def _get_target_window_rect(target_hwnd: int) -> Optional[Tuple[int, int, int, int]]:
    """Get screen coordinates (left, top, width, height) of target window client area."""
    if sys.platform != "win32" or target_hwnd <= 0:
        return None

    try:
        user32 = ctypes.windll.user32
        if not user32.IsWindow(target_hwnd):
            return None

        client_rect = ctypes.wintypes.RECT()
        user32.GetClientRect(target_hwnd, ctypes.byref(client_rect))
        pt = ctypes.wintypes.POINT(0, 0)
        user32.ClientToScreen(target_hwnd, ctypes.byref(pt))

        w = client_rect.right - client_rect.left
        h = client_rect.bottom - client_rect.top
        if w > 100 and h > 100:
            return pt.x, pt.y, w, h
    except Exception:
        pass
    return None


class NeuralOverlayWindow:
    """Tkinter Canvas visualizer running inside the dedicated overlay process."""

    WIDTH = 210
    HEIGHT = 370

    def __init__(
        self,
        data_queue: mp.Queue,
        target_hwnd: int = 0,
        stop_event: Optional[mp.Event] = None,
    ) -> None:
        self.queue = data_queue
        self.target_hwnd = target_hwnd
        self.stop_event = stop_event

        import tkinter as tk

        self.root = tk.Tk()
        self.root.title("PolyTrack Drosophila Connectome")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 0.88)
        self.root.configure(bg="#0A0D14")

        self.canvas = tk.Canvas(
            self.root,
            width=self.WIDTH,
            height=self.HEIGHT,
            bg="#0A0D14",
            highlightthickness=1,
            highlightbackground="#1C2333",
        )
        self.canvas.pack(fill="both", expand=True)

        self.latest_packet: Optional[OverlayTelemetryPacket] = None
        self.last_pos_check = 0.0
        self.last_target_rect: Optional[Tuple[int, int, int, int]] = None

        # Position window immediately
        self._reposition_overlay()

        # Apply Win32 attributes after window realization
        self.root.update_idletasks()
        try:
            hwnd = int(self.root.frame(), 16)
            _apply_win32_overlay_styles(hwnd)
        except Exception:
            pass

        # Schedule animation loop
        self.root.after(30, self._render_tick)

    def _reposition_overlay(self) -> None:
        """Reposition overlay to the middle of the right edge of target PolyTrack window."""
        target_rect = _get_target_window_rect(self.target_hwnd)
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()

        if target_rect:
            tx, ty, tw, th = target_rect
            # Positioned at middle of right edge:
            # X = tx + tw - WIDTH - 8 (inset 8px from right border)
            # Y = ty + (th - HEIGHT) // 2 (centered vertically)
            x = tx + tw - self.WIDTH - 10
            y = ty + (th - self.HEIGHT) // 2
            # Clamp to screen bounds
            x = max(0, min(screen_w - self.WIDTH, x))
            y = max(0, min(screen_h - self.HEIGHT, y))
        else:
            # Fallback: right edge of primary display
            x = screen_w - self.WIDTH - 16
            y = (screen_h - self.HEIGHT) // 2

        self.root.geometry(f"{self.WIDTH}x{self.HEIGHT}+{x}+{y}")

    def _render_tick(self) -> None:
        """Periodic tick (~30-35 FPS): fetch latest packet and repaint canvas."""
        if self.stop_event and self.stop_event.is_set():
            self.root.destroy()
            return

        # 1. Drain queue to latest state
        while not self.queue.empty():
            try:
                self.latest_packet = self.queue.get_nowait()
            except Exception:
                break

        # 2. Recheck target window position once every 1.2 seconds
        now = time.time()
        if now - self.last_pos_check > 1.2:
            self.last_pos_check = now
            self._reposition_overlay()

        # 3. Draw visual layers
        self._draw_frame()

        # Schedule next tick
        self.root.after(30, self._render_tick)

    def _draw_frame(self) -> None:
        """Repaint connectome, E-PG compass, and WASD actuators."""
        self.canvas.delete("all")
        pkt = self.latest_packet

        # Default telemetry if no packet yet
        actions = pkt.actions if pkt else {"w": False, "a": False, "s": False, "d": False}
        probs = pkt.action_probs if pkt else {"w": 0.0, "a": 0.0, "s": 0.0, "d": 0.0}
        dn_rates = pkt.dn_rates if pkt else {}
        heading_deg = pkt.heading_deg if pkt else 0.0
        vs_forward = pkt.vs_forward if pkt else 0.0
        hs_l = pkt.hs_left if pkt else 0.0
        hs_r = pkt.hs_right if pkt else 0.0
        curb_status = pkt.curb_status if pkt else "ok"
        gen = pkt.generation if pkt else 0
        fitness = pkt.fitness if pkt else 0.0
        fps = pkt.fps if pkt else 0.0

        # -------------------------------------------------------------
        # 1. HEADER & STATUS BANNER
        # -------------------------------------------------------------
        self.canvas.create_text(
            10, 14, text="DROSOPHILA CONNECTOME", anchor="w", fill="#5AC8FA", font=("Segoe UI", 8, "bold")
        )
        status_color = "#30D158" if curb_status == "ok" else ("#FF453A" if curb_status == "breached" else "#FF9F0A")
        self.canvas.create_text(
            self.WIDTH - 10,
            14,
            text=curb_status.upper(),
            anchor="e",
            fill=status_color,
            font=("Segoe UI", 7, "bold"),
        )

        sub_info = f"GEN {gen} | FIT {fitness:.1f} | {fps:.0f} FPS"
        self.canvas.create_text(
            10, 27, text=sub_info, anchor="w", fill="#8E8E93", font=("Segoe UI", 7)
        )

        self.canvas.create_line(10, 34, self.WIDTH - 10, 34, fill="#1C2333")

        # -------------------------------------------------------------
        # 2. TIER 1: LPTC SENSORY NEURONS (OPTIC FLOW)
        # -------------------------------------------------------------
        self.canvas.create_text(
            10, 44, text="LPTC OPTIC FLOW", anchor="w", fill="#A2845E", font=("Segoe UI", 7, "bold")
        )

        # HS Left Eye (Optic Flow Yaw Left)
        hs_l_val = min(1.0, max(0.0, abs(hs_l) * 2.0))
        c_hsl = "#00D2FF" if hs_l_val > 0.2 else "#1E2A3A"
        self.canvas.create_rectangle(10, 52, 45, 66, fill=c_hsl, outline="#1C2333")
        self.canvas.create_text(27, 59, text="HS_L", fill="#FFFFFF", font=("Segoe UI", 6))

        # VS Strips (Forward Optic Flow Expansion)
        vs_val = min(1.0, max(0.0, vs_forward * 12.0))
        c_vs = "#30D158" if vs_val > 0.15 else "#16281E"
        self.canvas.create_rectangle(52, 52, self.WIDTH - 52, 66, fill=c_vs, outline="#1C2333")
        self.canvas.create_text(self.WIDTH // 2, 59, text=f"VS FW ({vs_forward*100:.1f})", fill="#FFFFFF", font=("Segoe UI", 6))

        # HS Right Eye (Optic Flow Yaw Right)
        hs_r_val = min(1.0, max(0.0, abs(hs_r) * 2.0))
        c_hsr = "#00D2FF" if hs_r_val > 0.2 else "#1E2A3A"
        self.canvas.create_rectangle(self.WIDTH - 45, 52, self.WIDTH - 10, 66, fill=c_hsr, outline="#1C2333")
        self.canvas.create_text(self.WIDTH - 27, 59, text="HS_R", fill="#FFFFFF", font=("Segoe UI", 6))

        # -------------------------------------------------------------
        # 3. TIER 2: CENTRAL COMPLEX E-PG COMPASS RING (16 NEURONS)
        # -------------------------------------------------------------
        ring_cx = self.WIDTH // 2
        ring_cy = 135
        ring_r = 42

        self.canvas.create_text(
            10, 80, text=f"E-PG COMPASS: {heading_deg:.0f}°", anchor="w", fill="#FF9F0A", font=("Segoe UI", 7, "bold")
        )

        # Background circular track
        self.canvas.create_oval(
            ring_cx - ring_r, ring_cy - ring_r, ring_cx + ring_r, ring_cy + ring_r, outline="#1C2333", width=1
        )

        # Draw 16 E-PG Compass Wedge Neurons
        head_rad = math.radians(heading_deg - 90)  # Align 0 deg to top
        for i in range(16):
            angle = (i / 16.0) * 2.0 * math.pi - math.pi / 2.0
            nx = ring_cx + ring_r * math.cos(angle)
            ny = ring_cy + ring_r * math.sin(angle)

            # Proximity to bump head direction
            angle_diff = (angle - head_rad + math.pi) % (2 * math.pi) - math.pi
            proximity = max(0.0, 1.0 - abs(angle_diff) / (math.pi / 3.0))

            if proximity > 0.5:
                node_col = "#FFD60A"  # Glowing golden bump
                node_r = 4.5
            elif proximity > 0.2:
                node_col = "#FF9F0A"
                node_r = 3.5
            else:
                node_col = "#1E222D"
                node_r = 2.5

            self.canvas.create_oval(
                nx - node_r, ny - node_r, nx + node_r, ny + node_r, fill=node_col, outline="#0A0D14"
            )

        # Center Heading Indicator Needle
        nx_head = ring_cx + (ring_r - 12) * math.cos(head_rad)
        ny_head = ring_cy + (ring_r - 12) * math.sin(head_rad)
        self.canvas.create_line(ring_cx, ring_cy, nx_head, ny_head, fill="#FFD60A", width=2)
        self.canvas.create_oval(ring_cx - 3, ring_cy - 3, ring_cx + 3, ring_cy + 3, fill="#FF9F0A", outline="")

        # -------------------------------------------------------------
        # 4. TIER 3: DESCENDING MOTOR NEURONS (DN SPIKE RATES)
        # -------------------------------------------------------------
        self.canvas.create_text(
            10, 196, text="DESCENDING NEURONS (DN)", anchor="w", fill="#BF5AF2", font=("Segoe UI", 7, "bold")
        )

        dn_labels = [
            ("DNa01_L", 25, 212),
            ("DNa02_L", 25, 226),
            ("DNp01_L", 70, 212),
            ("DNb01_L", 70, 226),
            ("DNp01_R", 140, 212),
            ("DNb01_R", 140, 226),
            ("DNa01_R", 185, 212),
            ("DNa02_R", 185, 226),
        ]

        for name, px, py in dn_labels:
            rate = dn_rates.get(name, 0.0)
            is_firing = rate > 0.15
            color = "#BF5AF2" if is_firing else "#1E222D"
            r = 4.5 if is_firing else 3.0
            self.canvas.create_oval(px - r, py - r, px + r, py + r, fill=color, outline="#0A0D14")
            # Connective synaptic lines from DN down to pedals
            self.canvas.create_line(px, py + r, px, 245, fill="#141A28", width=1)

        # -------------------------------------------------------------
        # 5. TIER 4: WASD ACTUATOR PEDALS (WITH PROBABILITIES)
        # -------------------------------------------------------------
        self.canvas.create_line(10, 244, self.WIDTH - 10, 244, fill="#1C2333")
        self.canvas.create_text(
            10, 254, text="WASD ACTUATOR PEDALS", anchor="w", fill="#64D2FF", font=("Segoe UI", 7, "bold")
        )

        # Pedals Layout:
        #           [ W ] (Throttle)
        # [ A ] (Left) [ S ] (Brake) [ D ] (Right)
        box_w, box_h = 36, 32
        pedal_defs = [
            ("w", "W", self.WIDTH // 2 - box_w // 2, 266, "#30D158", actions["w"], probs.get("w", 0.0)),
            ("a", "A", self.WIDTH // 2 - box_w // 2 - box_w - 6, 304, "#00D2FF", actions["a"], probs.get("a", 0.0)),
            ("s", "S", self.WIDTH // 2 - box_w // 2, 304, "#FF453A", actions["s"], probs.get("s", 0.0)),
            ("d", "D", self.WIDTH // 2 + box_w // 2 + 6, 304, "#00D2FF", actions["d"], probs.get("d", 0.0)),
        ]

        for _, label, bx, by, active_col, is_pressed, prob in pedal_defs:
            bg_col = active_col if is_pressed else "#141A24"
            text_col = "#000000" if is_pressed else "#8E8E93"
            prob_pct = int(prob * 100)

            # Key button box
            self.canvas.create_rectangle(
                bx, by, bx + box_w, by + box_h, fill=bg_col, outline="#2C3549" if not is_pressed else active_col, width=1
            )
            # Key letter
            self.canvas.create_text(
                bx + box_w // 2, by + 12, text=label, fill=text_col, font=("Segoe UI", 9, "bold")
            )
            # Probability sub-label
            self.canvas.create_text(
                bx + box_w // 2, by + 24, text=f"{prob_pct}%", fill=text_col, font=("Segoe UI", 6)
            )

        # Outer bounding glow border
        self.canvas.create_rectangle(1, 1, self.WIDTH - 1, self.HEIGHT - 1, outline="#1C2333", width=1)

    def start_loop(self) -> None:
        """Start Tkinter event loop."""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.root.destroy()


def _overlay_process_entry(queue: mp.Queue, target_hwnd: int, stop_event: mp.Event) -> None:
    """Entry point for the isolated background overlay process."""
    try:
        app = NeuralOverlayWindow(queue, target_hwnd=target_hwnd, stop_event=stop_event)
        app.start_loop()
    except Exception as e:
        print(f"[NeuralOverlay] Process exited: {e}")


class NeuralOverlayController:
    """Thread-safe controller for starting, updating, and stopping the overlay from Python."""

    def __init__(self, target_hwnd: int = 0) -> None:
        self.target_hwnd = target_hwnd
        self.queue: Optional[mp.Queue] = None
        self.stop_event: Optional[mp.Event] = None
        self.process: Optional[mp.Process] = None

    def start(self, target_hwnd: Optional[int] = None) -> bool:
        """Start the background neural overlay window process."""
        if target_hwnd is not None:
            self.target_hwnd = target_hwnd

        if self.process is not None and self.process.is_alive():
            return True

        try:
            self.queue = mp.Queue(maxsize=4)
            self.stop_event = mp.Event()
            self.process = mp.Process(
                target=_overlay_process_entry,
                args=(self.queue, self.target_hwnd, self.stop_event),
                daemon=True,
            )
            self.process.start()
            return True
        except Exception as e:
            print(f"[NeuralOverlay] Failed to start overlay process: {e}")
            return False

    def update(
        self,
        actions: Dict[str, bool],
        action_probs: Dict[str, float],
        dn_rates: Dict[str, float],
        heading_deg: float = 0.0,
        vs_forward: float = 0.0,
        hs_left: float = 0.0,
        hs_right: float = 0.0,
        lateral_offset: float = 0.0,
        fitness: float = 0.0,
        generation: int = 0,
        fps: float = 0.0,
        curb_status: str = "ok",
    ) -> None:
        """Post a new telemetry frame to the overlay (non-blocking)."""
        if self.queue is None or self.process is None or not self.process.is_alive():
            return

        pkt = OverlayTelemetryPacket(
            actions=actions,
            action_probs=action_probs,
            dn_rates=dn_rates,
            heading_deg=heading_deg,
            vs_forward=vs_forward,
            hs_left=hs_left,
            hs_right=hs_right,
            lateral_offset=lateral_offset,
            fitness=fitness,
            generation=generation,
            fps=fps,
            curb_status=curb_status,
        )

        try:
            # Drop older frames if buffer is full
            while self.queue.full():
                try:
                    self.queue.get_nowait()
                except Exception:
                    break
            self.queue.put_nowait(pkt)
        except Exception:
            pass

    def stop(self) -> None:
        """Gracefully terminate the overlay window process."""
        if self.stop_event:
            self.stop_event.set()
        if self.process and self.process.is_alive():
            self.process.join(timeout=1.0)
            if self.process.is_alive():
                self.process.terminate()
        self.process = None


def run_demo() -> None:
    """Run an interactive demonstration of the neural overlay with synthetic signals."""
    print("Starting Drosophila Neural Connectome Overlay Demo...")
    controller = NeuralOverlayController(target_hwnd=0)
    controller.start()

    t0 = time.time()
    try:
        for step in range(300):
            elapsed = time.time() - t0
            head_deg = (elapsed * 45.0) % 360.0
            vs_fw = 0.04 + 0.03 * math.sin(elapsed * 2.0)
            hs_l = 0.02 * math.cos(elapsed * 1.5)
            hs_r = -0.02 * math.cos(elapsed * 1.5)

            p_w = 0.85 + 0.10 * math.sin(elapsed)
            p_a = max(0.0, 0.40 * math.sin(elapsed * 3.0))
            p_d = max(0.0, -0.40 * math.sin(elapsed * 3.0))
            p_s = 0.15 if vs_fw < 0.03 else 0.02

            actions = {
                "w": p_w > 0.5,
                "a": p_a > 0.5,
                "s": p_s > 0.5,
                "d": p_d > 0.5,
            }
            probs = {"w": p_w, "a": p_a, "s": p_s, "d": p_d}
            dn_rates = {
                "DNa01_L": p_a,
                "DNa02_L": p_a * 0.8,
                "DNp01_L": p_w,
                "DNb01_L": p_s,
                "DNp01_R": p_w,
                "DNb01_R": p_s,
                "DNa01_R": p_d,
                "DNa02_R": p_d * 0.8,
            }

            curb_st = "breached" if p_a > 0.7 else ("beached" if vs_fw < 0.02 else "ok")

            controller.update(
                actions=actions,
                action_probs=probs,
                dn_rates=dn_rates,
                heading_deg=head_deg,
                vs_forward=vs_fw,
                hs_left=hs_l,
                hs_right=hs_r,
                lateral_offset=0.1 * math.sin(elapsed),
                fitness=142.5 + elapsed * 5.0,
                generation=12,
                fps=30.0,
                curb_status=curb_st,
            )
            time.sleep(0.033)
    finally:
        controller.stop()
        print("Demo completed.")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        run_demo()
