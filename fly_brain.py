"""Drosophila Spiking Neural Network (SNN) Brain Model.

This module implements Module 4 of the Drosophila Vision-Motor Loop:
1. PyTorch vectorized Leaky Integrate-and-Fire (LIF) neural dynamics with potential reset:
   V_i[t] = alpha * V_i[t-1] + sum_j W_{ji} * S_j[t-1] + I_ext
2. Biologically wired FlyWire connectome recurrent core (102 neurons, signed W_syn).
3. Plastic Input Adapter: [2004 retina features] -> [26 sensory LPTC neurons].
4. Plastic Motor Decoder: [8 Descending Neurons spike rates] -> [WASD keyboard controls].
5. Population vector decoding of the internal heading compass (E-PG bump attractor).
6. High-speed CPU/GPU inference step benchmarking (< 1 ms per frame).
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from connectome_loader import ConnectomeGraph, get_or_build_connectome

logger = logging.getLogger("FlyBrain")


@dataclass
class BrainInferenceOutput:
    """Encapsulates all outputs, neural states, and motor commands produced during one frame."""

    # Decoded motor actions for Polytrack
    actions: Dict[str, bool]             # {"w": bool, "a": bool, "s": bool, "d": bool}
    action_probabilities: Dict[str, float]

    # Motor descending neuron spike rates [0, 1]
    dn_spike_rates: Dict[str, float]

    # Decoded internal compass heading from E-PG bump attractor [0, 360) degrees
    compass_heading_deg: float
    compass_bump_strength: float

    # Raw neural simulation dynamics across the K sub-steps
    spike_history: np.ndarray            # Shape: (K, N_neurons) uint8
    voltage_history: np.ndarray          # Shape: (K, N_neurons) float32

    # Performance
    inference_time_ms: float             # Time spent in PyTorch inference


class PlasticInputAdapter(nn.Module):
    """Plastic linear projection mapping retina sensory currents to LPTC neurons.

    Input: 2004 channels (512 ommatidia + 512 delta + 496 EMD_x + 480 EMD_y + 4 LPTC).
    Output: 26 LPTC neurons (HS_L: 3, HS_R: 3, VS_L: 10, VS_R: 10).
    """

    def __init__(
        self,
        in_features: int = config.RETINA_FEATURE_DIM,
        out_features: int = config.LPTC_NEURON_COUNT,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self._init_biological_weights()

    def _init_biological_weights(self) -> None:
        """Initialize receptive field weights aligned with Drosophila visual neurobiology."""
        with torch.no_grad():
            # Small baseline noise
            self.linear.weight.data.normal_(0.0, 0.01)
            self.linear.bias.data.zero_()

            # The last 4 features of retina are pre-aggregated biological signals:
            # [-4: HS_Left, -3: HS_Right, -2: VS_Forward, -1: Net Yaw]
            # Connect these strongly to the respective LPTC sub-populations:
            # LPTC indices:
            # 0..2:   HS_L (North, Equatorial, South)
            # 3..5:   HS_R (North, Equatorial, South)
            # 6..15:  VS_L (VS1..VS10)
            # 16..25: VS_R (VS1..VS10)

            # Left HS tangentials receive left horizontal flow and progressive yaw
            self.linear.weight.data[0:3, -4] += 1.5   # HS_Left feature
            self.linear.weight.data[0:3, -1] += 1.0   # Yaw feature

            # Right HS tangentials receive right horizontal flow
            self.linear.weight.data[3:6, -3] += 1.5   # HS_Right feature
            self.linear.weight.data[3:6, -1] -= 1.0   # Negative yaw

            # Vertical VS tangentials receive forward optic flow expansion
            self.linear.weight.data[6:16, -2] += 1.2  # VS_Forward feature
            self.linear.weight.data[16:26, -2] += 1.2

    def forward(self, retina_tensor: torch.Tensor) -> torch.Tensor:
        """Project (Batch, 2004) -> (Batch, 26)."""
        return self.linear(retina_tensor)


class LIFConnectomeCore(nn.Module):
    """Vectorized Leaky Integrate-and-Fire (LIF) network wired by the FlyWire connectome.

    Membrane potential update:
        V[t] = alpha * V[t-1] + sum_j (W_ji * S_j[t-1]) * g_syn + I_ext[t]
        S[t] = (V[t] >= V_th)
        V[t] = V[t] * (1 - S[t]) + V_reset * S[t]
    """

    def __init__(
        self,
        connectome: ConnectomeGraph,
        alpha_decay: float = config.SNN_ALPHA_DECAY,
        v_threshold: float = config.SNN_V_THRESHOLD,
        v_reset: float = config.SNN_V_RESET,
        synaptic_gain: float = config.SNN_SYNAPTIC_GAIN,
    ) -> None:
        super().__init__()
        self.n_neurons = connectome.n_neurons
        self.alpha = float(alpha_decay)
        self.v_th = float(v_threshold)
        self.v_reset = float(v_reset)
        self.synaptic_gain = float(synaptic_gain)

        # Register connectome signed weights as a fixed buffer
        W_signed = connectome.get_signed_weights_dense()
        # W_signed[i, j] is presynaptic j -> postsynaptic i
        self.register_buffer("W_syn", torch.tensor(W_signed, dtype=torch.float32))

        # Layer indices
        self.lptc_indices = connectome.layer_indices["sensory_lptc"]
        self.cx_indices = connectome.layer_indices["central_complex"]
        self.dn_indices = connectome.layer_indices["motor_dn"]

        # Recurrent state buffers
        self.register_buffer("v_mem", torch.zeros(1, self.n_neurons, dtype=torch.float32))
        self.register_buffer("spikes_prev", torch.zeros(1, self.n_neurons, dtype=torch.float32))

    def reset_state(self, batch_size: int = 1) -> None:
        """Reset membrane potentials and spike history to resting state."""
        device = self.W_syn.device
        self.v_mem = torch.full((batch_size, self.n_neurons), self.v_reset, device=device)
        self.spikes_prev = torch.zeros((batch_size, self.n_neurons), device=device)

    def sub_step(self, i_ext: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Execute a single LIF integration sub-step.

        Args:
            i_ext: Tensor of shape (Batch, N_neurons) representing injected currents.
        Returns:
            (spikes, v_mem) of shape (Batch, N_neurons).
        """
        # Synaptic recurrent current: I_syn = spikes_prev @ W^T
        # Shape: (Batch, N) @ (N, N)^T -> (Batch, N)
        i_syn = torch.matmul(self.spikes_prev, self.W_syn.t()) * self.synaptic_gain

        # Membrane potential integration with exponential leak
        v_next = self.alpha * self.v_mem + i_syn + i_ext

        # Spike generation (Heaviside step)
        spikes = (v_next >= self.v_th).float()

        # Hard reset of membrane potential
        self.v_mem = v_next * (1.0 - spikes) + self.v_reset * spikes
        self.spikes_prev = spikes

        return spikes, self.v_mem


class PlasticMotorDecoder(nn.Module):
    """Decodes Descending Neurons (DN) spike rates into WASD game actions."""

    def __init__(
        self,
        connectome: ConnectomeGraph,
        steer_diff_th: float = config.MOTOR_STEER_DIFF_THRESHOLD,
        throttle_th: float = config.MOTOR_THROTTLE_RATE_THRESHOLD,
        brake_th: float = config.MOTOR_BRAKE_RATE_THRESHOLD,
    ) -> None:
        super().__init__()
        self.steer_diff_th = steer_diff_th
        self.throttle_th = throttle_th
        self.brake_th = brake_th

        # Map descending neuron names to indices
        self.dn_names = ["DNa01_L", "DNa01_R", "DNa02_L", "DNa02_R", "DNp01_L", "DNp01_R", "DNb01_L", "DNb01_R"]
        self.dn_indices = [connectome.name_to_idx[name] for name in self.dn_names]

        # Learnable linear readout layer for fine-tuning via Evolutionary Strategies
        # Maps 8 DN rates -> 4 action logits [W, A, S, D]
        self.readout = nn.Linear(len(self.dn_names), 4, bias=True)
        self._init_readout_weights()

    def _init_readout_weights(self) -> None:
        """Initialize motor readout weights matching Drosophila descending physiology."""
        with torch.no_grad():
            self.readout.weight.data.zero_()
            self.readout.bias.data.zero_()

            # Actions order: 0: W (Throttle), 1: A (Left), 2: S (Brake), 3: D (Right)
            # DN names order:
            # 0: DNa01_L, 1: DNa01_R, 2: DNa02_L, 3: DNa02_R
            # 4: DNp01_L, 5: DNp01_R, 6: DNb01_L, 7: DNb01_R

            # Throttle 'W': driven by DNp01
            self.readout.weight.data[0, 4] = +2.0  # DNp01_L
            self.readout.weight.data[0, 5] = +2.0  # DNp01_R

            # Turn Left 'A': driven by DNa01_L & DNa02_L, inhibited by Right
            self.readout.weight.data[1, 0] = +2.5  # DNa01_L
            self.readout.weight.data[1, 2] = +2.0  # DNa02_L
            self.readout.weight.data[1, 1] = -1.5  # DNa01_R (reciprocal inhibition)

            # Brake 'S': driven by DNb01
            self.readout.weight.data[2, 6] = +2.5  # DNb01_L
            self.readout.weight.data[2, 7] = +2.5  # DNb01_R

            # Turn Right 'D': driven by DNa01_R & DNa02_R, inhibited by Left
            self.readout.weight.data[3, 1] = +2.5  # DNa01_R
            self.readout.weight.data[3, 3] = +2.0  # DNa02_R
            self.readout.weight.data[3, 0] = -1.5  # DNa01_L

            # Threshold biases
            self.readout.bias.data[0] = -0.5  # W bias
            self.readout.bias.data[1] = -0.4  # A bias
            self.readout.bias.data[2] = -0.8  # S bias
            self.readout.bias.data[3] = -0.4  # D bias

    def decode(self, mean_dn_spikes: torch.Tensor) -> Tuple[Dict[str, bool], Dict[str, float]]:
        """Decode descending neuron firing rates into discrete WASD key actions and probabilities."""
        # Logits: (Batch, 4)
        logits = self.readout(mean_dn_spikes)
        probs = torch.sigmoid(logits)[0]  # Take first item in batch

        prob_w = float(probs[0].item())
        prob_a = float(probs[1].item())
        prob_s = float(probs[2].item())
        prob_d = float(probs[3].item())

        # Mutually exclusive steering: cannot press A and D at the same time
        steer_a = prob_a > 0.5 and (prob_a > prob_d)
        steer_d = prob_d > 0.5 and (prob_d > prob_a)

        actions = {
            "w": bool(prob_w > 0.5),
            "a": bool(steer_a),
            "s": bool(prob_s > 0.5),
            "d": bool(steer_d),
        }
        action_probs = {
            "w": prob_w,
            "a": prob_a,
            "s": prob_s,
            "d": prob_d,
        }
        return actions, action_probs


class FlyBrain(nn.Module):
    """Complete Drosophila Brain model orchestrating visual input, LIF connectome, and motor output."""

    def __init__(
        self,
        connectome: Optional[ConnectomeGraph] = None,
        k_substeps: int = config.SNN_SUBSTEPS_PER_FRAME,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.connectome = connectome or get_or_build_connectome()
        self.k_substeps = k_substeps

        # Components
        self.input_adapter = PlasticInputAdapter().to(self.device)
        self.connectome_core = LIFConnectomeCore(self.connectome).to(self.device)
        self.motor_decoder = PlasticMotorDecoder(self.connectome).to(self.device)

        # Compass E-PG indices for population vector decoding
        self.epg_indices = [self.connectome.name_to_idx[f"E-PG_{w:02d}"] for w in range(1, 17)]
        # Wedge angles: 16 wedges covering [0, 2*pi)
        self.epg_angles_rad = np.linspace(0, 2 * np.pi, 16, endpoint=False)

        # Descending neuron names for telemetry
        self.dn_names = self.motor_decoder.dn_names
        self.dn_indices = self.motor_decoder.dn_indices

        # Initialize internal state
        self.reset()

    def reset(self) -> None:
        """Reset internal membrane potentials and spike buffers."""
        self.connectome_core.reset_state(batch_size=1)

    @torch.no_grad()
    def forward_frame(
        self,
        retina_currents: np.ndarray,
        k_substeps: Optional[int] = None,
    ) -> BrainInferenceOutput:
        """Process a single camera frame through the SNN over K simulation sub-steps.

        Args:
            retina_currents: 1D NumPy array of shape (2004,) from DrosophilaRetina.
            k_substeps: Number of LIF integration steps (defaults to config.SNN_SUBSTEPS_PER_FRAME).
        Returns:
            BrainInferenceOutput dataclass.
        """
        t0 = time.perf_counter_ns()
        k = k_substeps or self.k_substeps

        # Step 1: Adapt input retina features -> 26 LPTC sensory currents
        retina_tensor = torch.from_numpy(retina_currents).float().unsqueeze(0).to(self.device)
        lptc_currents = self.input_adapter(retina_tensor)  # Shape: (1, 26)

        # Assemble full injection current vector I_ext for all 102 neurons
        i_ext_full = torch.zeros((1, self.connectome.n_neurons), device=self.device)
        i_ext_full[0, self.connectome_core.lptc_indices] = lptc_currents[0]

        # Step 2: Simulate K sub-steps of LIF dynamics
        spikes_list: List[torch.Tensor] = []
        voltages_list: List[torch.Tensor] = []

        for _ in range(k):
            s_t, v_t = self.connectome_core.sub_step(i_ext_full)
            spikes_list.append(s_t.clone())
            voltages_list.append(v_t.clone())

        # Stack over time: (K, N)
        all_spikes = torch.cat(spikes_list, dim=0)       # Shape: (K, N)
        all_voltages = torch.cat(voltages_list, dim=0)   # Shape: (K, N)

        # Step 3: Compute mean firing rates of Descending Neurons over the frame
        dn_spikes = all_spikes[:, self.dn_indices]        # Shape: (K, 8)
        mean_dn_rates = torch.mean(dn_spikes, dim=0, keepdim=True)  # Shape: (1, 8)

        # Step 4: Motor decoding -> WASD key actions
        actions, action_probs = self.motor_decoder.decode(mean_dn_rates)

        # Step 5: Decode internal heading compass from E-PG bump attractor
        epg_rates = torch.mean(all_spikes[:, self.epg_indices], dim=0).detach().cpu().numpy()
        heading_deg, bump_strength = self._decode_compass_heading(epg_rates)

        # Step 6: Package telemetry
        dn_rates_dict = {
            name: float(mean_dn_rates[0, i].item())
            for i, name in enumerate(self.dn_names)
        }

        dt_ms = (time.perf_counter_ns() - t0) / 1_000_000.0

        return BrainInferenceOutput(
            actions=actions,
            action_probabilities=action_probs,
            dn_spike_rates=dn_rates_dict,
            compass_heading_deg=heading_deg,
            compass_bump_strength=bump_strength,
            spike_history=all_spikes.detach().cpu().numpy().astype(np.uint8),
            voltage_history=all_voltages.detach().cpu().numpy().astype(np.float32),
            inference_time_ms=dt_ms,
        )

    def _decode_compass_heading(self, epg_rates: np.ndarray) -> Tuple[float, float]:
        """Decode head direction angle from the 16 E-PG compass wedges via population vector."""
        # Population vector: v = sum(r_k * [cos(theta_k), sin(theta_k)])
        cos_sum = np.sum(epg_rates * np.cos(self.epg_angles_rad))
        sin_sum = np.sum(epg_rates * np.sin(self.epg_angles_rad))
        bump_strength = float(np.sqrt(cos_sum**2 + sin_sum**2))

        if bump_strength < 1e-4:
            return 0.0, 0.0

        angle_rad = np.arctan2(sin_sum, cos_sum)
        angle_deg = float(np.degrees(angle_rad) % 360.0)
        return angle_deg, bump_strength

    def get_plastic_parameters(self) -> np.ndarray:
        """Extract all trainable plastic parameters (Input Adapter + Motor Decoder) as flat vector."""
        params = []
        for p in list(self.input_adapter.parameters()) + list(self.motor_decoder.parameters()):
            params.append(p.detach().cpu().numpy().ravel())
        return np.concatenate(params)

    def set_plastic_parameters(self, flat_params: np.ndarray) -> None:
        """Load flat parameter vector into plastic layers (for evolutionary optimization)."""
        idx = 0
        for p in list(self.input_adapter.parameters()) + list(self.motor_decoder.parameters()):
            n_el = p.numel()
            chunk = flat_params[idx : idx + n_el]
            p.data.copy_(torch.from_numpy(chunk.reshape(p.shape)).float().to(self.device))
            idx += n_el

    def save_weights(self, filepath: Path) -> None:
        """Save plastic adapter weights to .pt file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "input_adapter": self.input_adapter.state_dict(),
                "motor_decoder": self.motor_decoder.state_dict(),
            },
            filepath,
        )
        logger.info(f"Saved brain plastic weights to {filepath}")

    def load_weights(self, filepath: Path) -> None:
        """Load plastic adapter weights from .pt file."""
        if not filepath.exists():
            raise FileNotFoundError(f"Weight file not found: {filepath}")
        ckpt = torch.load(filepath, map_location=self.device)
        self.input_adapter.load_state_dict(ckpt["input_adapter"])
        self.motor_decoder.load_state_dict(ckpt["motor_decoder"])
        logger.info(f"Loaded brain plastic weights from {filepath}")


def benchmark_fly_brain(cycles: int = 500) -> Dict[str, Any]:
    """Benchmark inference latency of the FlyBrain SNN model on CPU/GPU."""
    print("=" * 60)
    print(f"BENCHMARKING FLYBRAIN SNN INFERENCE ({cycles} CYCLES)")
    print(f"Sub-steps per Frame (K): {config.SNN_SUBSTEPS_PER_FRAME}")
    print(f"Connectome Size: {config.LPTC_NEURON_COUNT} LPTC + 68 CX + 8 DN = 102 Neurons")
    print("=" * 60)

    brain = FlyBrain()
    dummy_retina = np.random.randn(config.RETINA_FEATURE_DIM).astype(np.float32)

    # Warmup
    for _ in range(20):
        brain.forward_frame(dummy_retina)

    times_ms = []
    t_start = time.perf_counter()
    for _ in range(cycles):
        out = brain.forward_frame(dummy_retina)
        times_ms.append(out.inference_time_ms)
    t_total = time.perf_counter() - t_start

    arr = np.array(times_ms)
    mean_ms = float(np.mean(arr))
    p95_ms = float(np.percentile(arr, 95))
    fps = cycles / t_total

    print("\n" + "=" * 60)
    print("FLYBRAIN INFERENCE BENCHMARK REPORT")
    print("=" * 60)
    print(f" Samples Evaluated:      {cycles}")
    print(f" Mean Inference Latency: {mean_ms:.3f} ms")
    print(f" Median Latency:         {np.median(arr):.3f} ms")
    print(f" 95th Percentile (P95):  {p95_ms:.3f} ms")
    print(f" 99th Percentile (P99):  {np.percentile(arr, 99):.3f} ms")
    print(f" Min / Max Latency:      {np.min(arr):.3f} ms / {np.max(arr):.3f} ms")
    print(f" Max SNN Throughput:     {fps:.1f} FPS")
    print("-" * 60)
    print(f" Sample Actions Decoded: W={out.actions['w']}, A={out.actions['a']}, S={out.actions['s']}, D={out.actions['d']}")
    print(f" Heading Compass:        {out.compass_heading_deg:.1f} deg (Strength: {out.compass_bump_strength:.2f})")
    print("-" * 60)
    if p95_ms < 5.0:
        print(f" [PASS] Ultra-fast inference ({mean_ms:.3f} ms) leaving maximum budget for game rendering!")
    else:
        print(f" [WARN] Inference took {mean_ms:.3f} ms")
    print("=" * 60 + "\n")

    return {
        "mean_ms": mean_ms,
        "p95_ms": p95_ms,
        "fps": fps,
    }


def main() -> None:
    """CLI entrypoint for testing and benchmarking the FlyBrain SNN."""
    parser = argparse.ArgumentParser(description="Drosophila SNN FlyBrain (Module 4)")
    parser.add_argument("--benchmark", action="store_true", help="Run SNN inference latency benchmark")
    parser.add_argument("--cycles", type=int, default=500, help="Benchmark cycles")

    args = parser.parse_args()
    benchmark_fly_brain(cycles=args.cycles)


if __name__ == "__main__":
    main()
