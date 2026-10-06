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
    assert len(params_orig) > 50000  # 2004*26 + biases + motor weights

    # Perturb and set
    perturbed = params_orig + 0.01
    brain.set_plastic_parameters(perturbed)
    params_new = brain.get_plastic_parameters()
    assert np.allclose(params_new, perturbed)


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
    print("ALL FLYBRAIN TESTS PASSED SUCCESSFULLY!")
