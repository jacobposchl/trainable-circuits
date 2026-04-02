from __future__ import annotations

import numpy as np

from flow_circuits.discovery import CandidateCircuitDiscoverer


def test_candidate_discovery_finds_connected_circuit():
    n_images = 6
    n_layers = 3
    n_cells = 4
    traj_dim = 8
    flow_dim = 6

    future = np.zeros((n_images, n_layers, n_cells, traj_dim), dtype=np.float32)
    future[:3, :, 0, 0] = 1.0
    future[3:, :, 0, 1] = 1.0
    future[:3, :, 1, 0] = 1.0
    future[3:, :, 1, 1] = 1.0
    future = future / np.clip(np.linalg.norm(future, axis=-1, keepdims=True), 1.0e-8, None)

    flow = np.zeros((n_images, n_layers, n_cells, flow_dim), dtype=np.float32)
    flow[:, :, :, 0] = 1.0
    predicted_next = flow[:, :-1].copy()

    discoverer = CandidateCircuitDiscoverer(
        grid_size=2,
        min_cluster_fraction=0.2,
        max_cluster_fraction=0.8,
        min_cluster_size=2,
        bootstrap_iterations=1,
        stability_threshold=0.0,
        merge_threshold=0.5,
        node_threshold=0.5,
        random_seed=0,
    )
    artifact = discoverer.discover(
        future_descriptors=future,
        predicted_next=predicted_next,
        flow_targets=flow,
        dataset_indices=np.arange(n_images),
        labels=np.array([0, 0, 0, 1, 1, 1]),
    )

    assert artifact["circuits"]
    circuit = artifact["circuits"][0]
    assert circuit["active_nodes"]
    assert circuit["thresholds"]
