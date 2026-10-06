"""Unit tests for the FlyWire Connectome Loader (Module 3)."""

from pathlib import Path
import networkx as nx
import numpy as np
import scipy.sparse as sp

import config
from connectome_loader import (
    ConnectomeGraph,
    build_canonical_flywire_subgraph,
    get_or_build_connectome,
)


def test_build_canonical_subgraph() -> None:
    graph = build_canonical_flywire_subgraph()
    assert graph.n_neurons == 102
    assert isinstance(graph.adjacency_matrix, sp.csr_matrix)
    assert graph.adjacency_matrix.shape == (102, 102)
    assert graph.adjacency_matrix.nnz > 100

    # Test layer counts
    assert len(graph.layer_indices["sensory_lptc"]) == 26
    assert len(graph.layer_indices["central_complex"]) == 68
    assert len(graph.layer_indices["motor_dn"]) == 8

    # Verify key landmark cells exist
    for cell in ["HS_L_N", "HS_R_E", "VS_L_01", "E-PG_01", "P-EN_L_01", "Delta7_01", "PFL3_L_01", "DNa01_L", "DNa01_R", "DNp01_L", "DNb01_L"]:
        assert cell in graph.name_to_idx, f"Cell {cell} missing from connectome graph!"


def test_signed_neurotransmitter_polarities() -> None:
    graph = build_canonical_flywire_subgraph()
    W_signed = graph.get_signed_weights_dense()
    assert W_signed.shape == (102, 102)

    # 1. Delta7 interneurons are GABAergic: all outgoing synapses MUST be <= 0 (inhibitory)
    delta7_indices = [graph.name_to_idx[f"Delta7_{d:02d}"] for d in range(1, 9)]
    for d_idx in delta7_indices:
        out_weights = W_signed[:, d_idx]  # presynaptic column
        assert np.all(out_weights <= 0.0), f"Delta7 presynaptic weights must be non-positive, got {out_weights.max()}"
        assert np.any(out_weights < 0.0), "Delta7 must have at least one active inhibitory synapse"

    # 2. E-PG neurons are Cholinergic: all outgoing synapses MUST be >= 0 (excitatory)
    epg_indices = [graph.name_to_idx[f"E-PG_{w:02d}"] for w in range(1, 17)]
    for e_idx in epg_indices:
        out_weights = W_signed[:, e_idx]
        assert np.all(out_weights >= 0.0), f"E-PG presynaptic weights must be non-negative, got {out_weights.min()}"
        assert np.any(out_weights > 0.0), "E-PG must have active excitatory synapses"


def test_steering_and_motor_pathway() -> None:
    graph = build_canonical_flywire_subgraph()
    W_signed = graph.get_signed_weights_dense()

    # PFL3_L must drive DNa01_L (Left turn 'A')
    pfl3_l1 = graph.name_to_idx["PFL3_L_01"]
    dna01_l = graph.name_to_idx["DNa01_L"]
    assert W_signed[dna01_l, pfl3_l1] > 0.0, "Expected excitatory connection from PFL3_L to DNa01_L"

    # PFL3_R must drive DNa01_R (Right turn 'D')
    pfl3_r1 = graph.name_to_idx["PFL3_R_01"]
    dna01_r = graph.name_to_idx["DNa01_R"]
    assert W_signed[dna01_r, pfl3_r1] > 0.0, "Expected excitatory connection from PFL3_R to DNa01_R"


def test_cache_save_and_load(tmp_path: Path) -> None:
    test_cache = tmp_path / "test_connectome.npz"
    graph = build_canonical_flywire_subgraph()
    graph.save(test_cache)

    assert test_cache.exists()
    loaded = ConnectomeGraph.load(test_cache)

    assert loaded.n_neurons == graph.n_neurons
    assert loaded.adjacency_matrix.nnz == graph.adjacency_matrix.nnz
    assert np.allclose(loaded.adjacency_matrix.toarray(), graph.adjacency_matrix.toarray())
    assert loaded.name_to_idx == graph.name_to_idx


def test_networkx_conversion() -> None:
    graph = build_canonical_flywire_subgraph()
    G = graph.to_networkx()
    assert G.number_of_nodes() == 102
    assert G.number_of_edges() == graph.adjacency_matrix.nnz
    # Graph should be connected from sensory to motor
    assert nx.is_weakly_connected(G)


if __name__ == "__main__":
    print("Running Connectome unit tests...")
    test_build_canonical_subgraph()
    print("test_build_canonical_subgraph passed.")
    test_signed_neurotransmitter_polarities()
    print("test_signed_neurotransmitter_polarities passed.")
    test_steering_and_motor_pathway()
    print("test_steering_and_motor_pathway passed.")
    import tempfile
    test_cache_save_and_load(Path(tempfile.gettempdir()))
    print("test_cache_save_and_load passed.")
    test_networkx_conversion()
    print("test_networkx_conversion passed.")
    print("ALL CONNECTOME TESTS PASSED SUCCESSFULLY!")
