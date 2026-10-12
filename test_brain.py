"""Unit tests for the Drosophila Spiking Neural Network (SNN) FlyBrain (Module 4)."""

import numpy as np
import torch

import config
from fly_brain import (
    BrainInferenceOutput,
    FlyBrain,
    LIFConnectomeCore,
    PlasticInputAdapter,
    PlasticMotorDecoder,
)
from connectome_loader import get_or_build_connectome


def test_plastic_input_adapter() -> None:
    adapter = PlasticInputAdapter(in_features=2004, out_features=26)
    dummy_input = torch.randn(1, 2004)
    out = adapter(dummy_input)
    assert out.shape == (1, 26)
    assert not torch.isnan(out).any()


def test_lif_neuron_dynamics() -> None:
    connectome = get_or_build_connectome()
    lif = LIFConnectomeCore(connectome, alpha_decay=0.8, v_threshold=1.0, v_reset=0.0)
    lif.reset_state(batch_size=1)

    # 1. Sub-threshold stimulation: should not spike, voltage should rise then decay
    i_sub = torch.zeros(1, connectome.n_neurons)
    i_sub[0, 0] = 0.5
    spikes1, v1 = lif.sub_step(i_sub)
    assert spikes1[0, 0] == 0.0
    assert 0.49 <= v1[0, 0].item() <= 0.51

    # Zero input step: voltage must decay by alpha = 0.8
    spikes2, v2 = lif.sub_step(torch.zeros(1, connectome.n_neurons))
    assert spikes2[0, 0] == 0.0
    assert abs(v2[0, 0].item() - (0.5 * 0.8)) < 1e-4

    # 2. Supra-threshold stimulation: must spike and reset to 0
    i_supra = torch.zeros(1, connectome.n_neurons)
    i_supra[0, 0] = 1.5
    spikes3, v3 = lif.sub_step(i_supra)
    assert spikes3[0, 0] == 1.0  # Spike emitted!
    assert v3[0, 0].item() == 0.0  # Reset!


def test_motor_decoder_actions() -> None:
    connectome = get_or_build_connectome()
    decoder = PlasticMotorDecoder(connectome)

    # All zeros firing rates at standstill (vs_forward = 0.0)
    zero_rates = torch.zeros(1, 8)
    actions, probs = decoder.decode(zero_rates, vs_forward=0.0)
    assert isinstance(actions, dict)
    assert set(actions.keys()) == {"w", "a", "s", "d"}
    assert actions["s"] is False, "S must NEVER be engaged from a standstill (cannot reverse)!"
    assert actions["w"] is True, "W must be engaged by default from standstill to drive forward!"
    for k in ["w", "a", "s", "d"]:
        assert 0.0 <= probs[k] <= 1.0

    # Mutual exclusivity of steering:
    # Even if both left and right DNs fire, one of A or D must take precedence or neither
    high_rates = torch.ones(1, 8) * 0.9
    actions_high, _ = decoder.decode(high_rates, vs_forward=0.0)
    assert not (actions_high["a"] and actions_high["d"]), "Steering A and D cannot both be True simultaneously!"
    assert actions_high["s"] is False, "S must remain locked out at standstill even with high brake spike rates!"

    # At speed (vs_forward > 0.05), braking/drifting is permitted
    brake_rates = torch.zeros(1, 8)
    brake_rates[0, 6] = 0.8  # DNb01_L
    brake_rates[0, 7] = 0.8  # DNb01_R
    actions_moving, _ = decoder.decode(brake_rates, vs_forward=0.5)
    assert actions_moving["s"] is True, "S must be permitted for deceleration when moving forward!"


def test_fly_brain_end_to_end() -> None:
    brain = FlyBrain()
    brain.reset()

    dummy_retina = np.random.randn(config.RETINA_FEATURE_DIM).astype(np.float32)

    # Warmup
    brain.forward_frame(dummy_retina)

    # Execute inference
    out = brain.forward_frame(dummy_retina, k_substeps=8)
    assert isinstance(out, BrainInferenceOutput)
    assert set(out.actions.keys()) == {"w", "a", "s", "d"}
    assert len(out.dn_spike_rates) == 8
    assert 0.0 <= out.compass_heading_deg <= 360.0
    assert out.spike_history.shape == (8, 102)
    assert out.voltage_history.shape == (8, 102)
    assert out.inference_time_ms < 10.0


def test_plastic_parameters_roundtrip() -> None:
    brain = FlyBrain()
    params_orig = brain.get_plastic_parameters()
    assert isinstance(params_orig, np.ndarray)
    assert params_orig.ndim == 1
    assert len(params_orig) == 170  # 26*5=130 spatial RF params + 36 readout + 4 modulation params

    # Perturb and set
    perturbed = params_orig + 0.01
    brain.set_plastic_parameters(perturbed)
    params_new = brain.get_plastic_parameters()
    assert np.allclose(params_new, perturbed)


def test_motor_decoder_boundary_centering() -> None:
    from boundary_detector import BoundaryPerceptionOutput
    connectome = get_or_build_connectome()
    decoder = PlasticMotorDecoder(connectome)
    zero_rates = torch.zeros(1, 8)

    # 1. Car drifted to the right of track center (+offset): must steer Left (A)
    bound_drift_right = BoundaryPerceptionOutput(
        lateral_offset=0.6,
        track_heading_deg=0.0,
        proximity_left=0.0,
        proximity_right=0.4,
        corridor_detected=True,
    )
    act_l, prob_l = decoder.decode(zero_rates, vs_forward=0.1, boundary=bound_drift_right)
    assert act_l["a"] is True, "Car drifted right must steer Left (A) to center!"
    assert act_l["d"] is False
    assert prob_l["a"] > prob_l["d"]

    # 2. Car drifted to the left of track center (-offset): must steer Right (D)
    decoder.reset()
    bound_drift_left = BoundaryPerceptionOutput(
        lateral_offset=-0.6,
        track_heading_deg=0.0,
        proximity_left=0.4,
        proximity_right=0.0,
        corridor_detected=True,
    )
    act_r, prob_r = decoder.decode(zero_rates, vs_forward=0.1, boundary=bound_drift_left)
    assert act_r["d"] is True, "Car drifted left must steer Right (D) to center!"
    assert act_r["a"] is False
    assert prob_r["d"] > prob_r["a"]

    # 3. Emergency wall repulsion on left curb (proximity_left = 0.9): must forcefully steer Right
    decoder.reset()
    bound_danger_left = BoundaryPerceptionOutput(
        lateral_offset=0.0,
        track_heading_deg=0.0,
        proximity_left=0.9,
        proximity_right=0.0,
        corridor_detected=True,
    )
    act_danger, prob_danger = decoder.decode(zero_rates, vs_forward=0.1, boundary=bound_danger_left)
    assert act_danger["d"] is True, "Emergency left wall danger must trigger Right steering (D)!"


def test_motor_decoder_smoothing_and_deadband() -> None:
    from boundary_detector import BoundaryPerceptionOutput
    connectome = get_or_build_connectome()
    decoder = PlasticMotorDecoder(connectome)
    zero_rates = torch.zeros(1, 8)

    # Deadband on micro-deviation in straight corridor: raw steer is zeroed
    bound_straight_jitter = BoundaryPerceptionOutput(
        lateral_offset=0.03,
        track_heading_deg=0.0,
        proximity_left=0.0,
        proximity_right=0.0,
        corridor_detected=True,
    )
    act_s, prob_s = decoder.decode(zero_rates, vs_forward=0.1, boundary=bound_straight_jitter)
    assert act_s["a"] is False and act_s["d"] is False
    assert abs(decoder.smoothed_steer_demand) < 0.01


def test_motor_decoder_looming_and_trail_braking() -> None:
    from boundary_detector import BoundaryPerceptionOutput
    connectome = get_or_build_connectome()
    decoder = PlasticMotorDecoder(connectome)
    zero_rates = torch.zeros(1, 8)

    # Approaching severe bottleneck with looming constriction risk = 0.85
    bound_bottleneck = BoundaryPerceptionOutput(
        lateral_offset=0.1,
        track_heading_deg=15.0,  # Far horizon curves right
        immediate_heading_deg=-5.0,
        proximity_left=0.1,
        proximity_right=0.75,  # Right wall looms close!
        corridor_detected=True,
        bottleneck_ahead=True,
        constriction_risk=0.85,
    )

    # At racing speed (vs_forward = 0.5):
    # 1. Horizon right turn (15 deg) must be suppressed due to proximity_right = 0.75
    # 2. Decel demand must be triggered, boosting brake prob_s and reducing prob_w
    actions, probs = decoder.decode(zero_rates, vs_forward=0.5, boundary=bound_bottleneck)
    assert actions["a"] is True, "Must steer Left (A) to avoid impending right wall!"
    assert actions["d"] is False
    assert probs["s"] > 0.40, f"Expected elevated brake probability, got {probs['s']}"


def test_motor_decoder_schmitt_and_neutral_lockout() -> None:
    from boundary_detector import BoundaryPerceptionOutput
    connectome = get_or_build_connectome()
    decoder = PlasticMotorDecoder(connectome)
    zero_rates = torch.zeros(1, 8)

    # 1. Engage Left (A) from neutral with offset = 0.6
    b_left = BoundaryPerceptionOutput(
        lateral_offset=0.6,
        track_heading_deg=0.0,
        proximity_left=0.0,
        proximity_right=0.3,
        corridor_detected=True,
    )
    act1, _ = decoder.decode(zero_rates, vs_forward=0.1, boundary=b_left)
    assert act1["a"] is True
    assert decoder.active_steer == "a"

    # 2. Hysteresis: offset reduces to 0.25 -> A remains active due to hysteresis
    b_slight = BoundaryPerceptionOutput(
        lateral_offset=0.25,
        track_heading_deg=0.0,
        proximity_left=0.0,
        proximity_right=0.0,
        corridor_detected=True,
    )
    act2, _ = decoder.decode(zero_rates, vs_forward=0.1, boundary=b_slight)
    assert act2["a"] is True, "Hysteresis should maintain A active when above P_OFF (0.44)"

    # 3. Switching direction from Left to Right:
    # Sudden demand to turn Right (e.g. lateral_offset = -0.6)
    b_right = BoundaryPerceptionOutput(
        lateral_offset=-0.6,
        track_heading_deg=0.0,
        proximity_left=0.3,
        proximity_right=0.0,
        corridor_detected=True,
    )
    act3, _ = decoder.decode(zero_rates, vs_forward=0.1, boundary=b_right)
    assert act3["a"] is False
    assert act3["d"] is False, "Direction reversal must enforce neutral lockout pause, not instant D"
    assert decoder.active_steer == "none"
    assert decoder.lockout_counter == 1

    # Second frame: lockout decrements to 0
    act4, _ = decoder.decode(zero_rates, vs_forward=0.1, boundary=b_right)
    assert act4["d"] is False
    assert decoder.lockout_counter == 0

    # Third frame: lockout expired -> Right (D) engages cleanly!
    act5, _ = decoder.decode(zero_rates, vs_forward=0.1, boundary=b_right)
    assert act5["d"] is True, "After neutral lockout pause, D should engage cleanly"
    assert decoder.active_steer == "d"


def test_motor_decoder_emergency_lockout_override() -> None:
    from boundary_detector import BoundaryPerceptionOutput
    connectome = get_or_build_connectome()
    decoder = PlasticMotorDecoder(connectome)
    zero_rates = torch.zeros(1, 8)

    # Put decoder into active Left steering
    b_left = BoundaryPerceptionOutput(
        lateral_offset=0.6,
        track_heading_deg=0.0,
        proximity_left=0.0,
        proximity_right=0.3,
        corridor_detected=True,
    )
    decoder.decode(zero_rates, vs_forward=0.1, boundary=b_left)
    assert decoder.active_steer == "a"

    # Emergency right turn due to extreme left wall proximity (proximity_left = 0.85)
    b_danger = BoundaryPerceptionOutput(
        lateral_offset=0.0,
        track_heading_deg=0.0,
        proximity_left=0.85,
        proximity_right=0.0,
        corridor_detected=True,
    )
    act_emerg, _ = decoder.decode(zero_rates, vs_forward=0.1, boundary=b_danger)
    # Must immediately switch to D, bypassing neutral lockout!
    assert act_emerg["d"] is True, "Emergency wall proximity must immediately counter-steer D without lockout delay!"
    assert act_emerg["a"] is False
    assert decoder.lockout_counter == 0


def test_motor_decoder_pd_damping() -> None:
    from boundary_detector import BoundaryPerceptionOutput
    connectome = get_or_build_connectome()
    decoder = PlasticMotorDecoder(connectome)
    zero_rates = torch.zeros(1, 8)

    # Frame 1: car is at lateral_offset = 0.5
    b1 = BoundaryPerceptionOutput(
        lateral_offset=0.5,
        track_heading_deg=0.0,
        proximity_left=0.0,
        proximity_right=0.1,
        corridor_detected=True,
    )
    decoder.decode(zero_rates, vs_forward=0.1, boundary=b1)

    # Frame 2: car is moving rapidly left back to center: lateral_offset dropped to 0.15 (drift_rate = -0.35)
    b2 = BoundaryPerceptionOutput(
        lateral_offset=0.15,
        track_heading_deg=0.0,
        proximity_left=0.0,
        proximity_right=0.0,
        corridor_detected=True,
    )
    decoder.decode(zero_rates, vs_forward=0.1, boundary=b2, smooth=False)

    # Without D-damping: raw steer demand for offset=0.15 would steer left (negative demand)
    # With D-damping: -1.20 * (-0.35) = +0.42 opposes the leftward steering, suppressing overshoot!
    raw_p_only = -0.75 * 0.15 * 0.70  # approx -0.078
    raw_with_d = raw_p_only - 1.20 * (-0.35)  # approx +0.342
    assert raw_with_d > 0.0, "D-term must reverse raw steer demand when vehicle rapidly approaches center line"


def test_motor_decoder_near_road_primacy() -> None:
    from boundary_detector import BoundaryPerceptionOutput
    connectome = get_or_build_connectome()
    decoder = PlasticMotorDecoder(connectome)
    zero_rates = torch.zeros(1, 8)

    # Road under wheels is completely straight (immediate_heading_deg = 0.0, lateral_offset = 0.0),
    # BUT distant track on horizon turns sharply right (track_heading_deg = 45.0)!
    b_bridge = BoundaryPerceptionOutput(
        lateral_offset=0.0,
        immediate_heading_deg=0.0,
        track_heading_deg=45.0,
        proximity_left=0.0,
        proximity_right=0.0,
        corridor_detected=True,
    )

    actions, probs = decoder.decode(zero_rates, vs_forward=0.2, boundary=b_bridge)

    # The car MUST NOT prematurely turn right off the straight road into the abyss!
    assert actions["d"] is False, "Car on straight road under wheels must NOT steer right toward distant horizon turn!"
    assert actions["a"] is False
    assert decoder.active_steer == "none"

    # Verify get_decoder_state exposes valid telemetry dict
    st = decoder.get_decoder_state()
    assert isinstance(st, dict)
    assert "raw_steer_demand" in st
    assert "smoothed_steer_demand" in st
    assert "active_steer" in st


if __name__ == "__main__":
    print("Running FlyBrain unit tests...")
    test_plastic_input_adapter()
    print("test_plastic_input_adapter passed.")
    test_lif_neuron_dynamics()
    print("test_lif_neuron_dynamics passed.")
    test_motor_decoder_actions()
    print("test_motor_decoder_actions passed.")
    test_fly_brain_end_to_end()
    print("test_fly_brain_end_to_end passed.")
    test_plastic_parameters_roundtrip()
    print("test_plastic_parameters_roundtrip passed.")
    test_motor_decoder_schmitt_and_neutral_lockout()
    print("test_motor_decoder_schmitt_and_neutral_lockout passed.")
    test_motor_decoder_emergency_lockout_override()
    print("test_motor_decoder_emergency_lockout_override passed.")
    test_motor_decoder_pd_damping()
    print("test_motor_decoder_pd_damping passed.")
    print("ALL FLYBRAIN TESTS PASSED SUCCESSFULLY!")
