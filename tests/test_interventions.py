from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch

from flow_circuits.interventions import ResidualPatchAblator, assign_circuit_members, run_circuit_interventions
from flow_circuits.training import FlowCircuitTrainer, collect_intervention_outputs, collect_model_outputs


def test_residual_patch_ablator_changes_logits(monkeypatch, minimal_config, tiny_loader):
    monkeypatch.setattr(
        "flow_circuits.training.trainer.build_cifar10_splits",
        lambda **kwargs: {"fit": tiny_loader, "val": tiny_loader, "discovery": tiny_loader, "test": tiny_loader},
    )
    trainer = FlowCircuitTrainer(copy.deepcopy(minimal_config))
    batch = next(iter(tiny_loader))[0]
    ablator = ResidualPatchAblator(trainer.components, grid_size=2)
    with torch.no_grad():
        before = trainer.components.observer.model(batch)
    after = ablator.ablate(batch, [(0, 0), (1, 1)])

    assert before.shape == after.shape
    assert not torch.allclose(before, after)


def test_run_circuit_interventions_returns_summary(monkeypatch, minimal_config, tiny_loader, tmp_path):
    monkeypatch.setattr(
        "flow_circuits.training.trainer.build_cifar10_splits",
        lambda **kwargs: {"fit": tiny_loader, "val": tiny_loader, "discovery": tiny_loader, "test": tiny_loader},
    )
    trainer = FlowCircuitTrainer(copy.deepcopy(minimal_config))
    trainer.components.observer.classifier_is_trained = True
    outputs = collect_model_outputs(trainer.components, tiny_loader, device=torch.device("cpu"), max_images=6)
    centroid = outputs["future_descriptors"][0, 0, 0].tolist()
    # Set representative-node threshold to 0.999 so only image 0 (whose
    # descriptor is exactly the centroid) is assigned as a member, leaving
    # the remaining images available as matched non-member controls.
    artifact = {
        "metadata": {"grid_size": 2, "n_layers": outputs["future_descriptors"].shape[1], "n_cells": 4},
        "circuits": [
            {
                "id": 0,
                "representative_node": [0, 0],
                "active_nodes": [[0, 0], [1, 0], [2, 0]],
                "centroids": {"0:0": centroid, "1:0": outputs["future_descriptors"][0, 1, 0].tolist(), "2:0": outputs["future_descriptors"][0, 2, 0].tolist()},
                "thresholds": {"0:0": 0.999, "1:0": -1.0, "2:0": -1.0},
            }
        ],
    }
    results = run_circuit_interventions(
        trainer.components,
        artifact,
        outputs,
        alpha=0.5,
        output_path=tmp_path / "interventions.json",
    )

    assert results
    assert Path(tmp_path / "interventions.json").exists()


def test_assign_circuit_members_matches_reference_loop():
    torch.manual_seed(0)
    future_descriptors = torch.randn(7, 4, 4, 8)
    future_descriptors = future_descriptors / future_descriptors.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
    circuit = {
        "representative_node": [1, 2],
        "active_nodes": [[0, 0], [1, 2], [2, 2], [3, 1]],
        "centroids": {},
        "thresholds": {},
    }
    for layer_idx, cell_idx in circuit["active_nodes"]:
        key = f"{layer_idx}:{cell_idx}"
        centroid = torch.randn(8)
        centroid = centroid / centroid.norm().clamp_min(1.0e-8)
        circuit["centroids"][key] = centroid.tolist()
        circuit["thresholds"][key] = -0.2

    actual = assign_circuit_members(circuit, future_descriptors, torch.arange(future_descriptors.shape[0]))

    active_nodes = [tuple(node) for node in circuit["active_nodes"]]
    representative_node = tuple(circuit["representative_node"])
    centroids = {
        tuple(int(value) for value in key.split(":")): torch.tensor(vec, dtype=future_descriptors.dtype)
        for key, vec in circuit["centroids"].items()
    }
    thresholds = {
        tuple(int(value) for value in key.split(":")): float(value)
        for key, value in circuit["thresholds"].items()
    }
    expected = torch.zeros(future_descriptors.shape[0], dtype=torch.bool)
    for row_idx in range(future_descriptors.shape[0]):
        rep_score = torch.dot(
            future_descriptors[row_idx, representative_node[0], representative_node[1]],
            centroids[representative_node],
        ).item()
        if rep_score < thresholds[representative_node]:
            continue
        satisfied = 0
        for node in active_nodes:
            score = torch.dot(future_descriptors[row_idx, node[0], node[1]], centroids[node]).item()
            if score >= thresholds[node]:
                satisfied += 1
        if satisfied >= max(1, int(np.ceil(0.5 * len(active_nodes)))):
            expected[row_idx] = True

    assert torch.equal(actual, expected)


def test_run_circuit_interventions_matches_for_parallel_stats(monkeypatch, minimal_config, tiny_loader, tmp_path):
    monkeypatch.setattr(
        "flow_circuits.training.trainer.build_cifar10_splits",
        lambda **kwargs: {"fit": tiny_loader, "val": tiny_loader, "discovery": tiny_loader, "test": tiny_loader},
    )
    trainer = FlowCircuitTrainer(copy.deepcopy(minimal_config))
    trainer.components.observer.classifier_is_trained = True
    outputs = collect_intervention_outputs(trainer.components, tiny_loader, device=torch.device("cpu"), max_images=6)
    centroid = outputs["future_descriptors"][0, 0, 0].tolist()
    artifact = {
        "metadata": {"grid_size": 2, "n_layers": outputs["future_descriptors"].shape[1], "n_cells": 4},
        "circuits": [
            {
                "id": 0,
                "representative_node": [0, 0],
                "active_nodes": [[0, 0], [1, 0], [2, 0]],
                "centroids": {"0:0": centroid, "1:0": outputs["future_descriptors"][0, 1, 0].tolist(), "2:0": outputs["future_descriptors"][0, 2, 0].tolist()},
                "thresholds": {"0:0": 0.999, "1:0": -1.0, "2:0": -1.0},
            }
        ],
    }

    serial = run_circuit_interventions(
        trainer.components,
        artifact,
        outputs,
        alpha=0.5,
        output_path=tmp_path / "serial.json",
        n_jobs=1,
    )
    parallel = run_circuit_interventions(
        trainer.components,
        artifact,
        outputs,
        alpha=0.5,
        output_path=tmp_path / "parallel.json",
        n_jobs=2,
    )

    assert [result.to_dict() for result in serial] == [result.to_dict() for result in parallel]
