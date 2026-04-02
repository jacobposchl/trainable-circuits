from __future__ import annotations

import copy
from pathlib import Path

import torch

from flow_circuits.interventions import ResidualPatchAblator, run_circuit_interventions
from flow_circuits.training import FlowCircuitTrainer, collect_model_outputs


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
    outputs = collect_model_outputs(trainer.components, tiny_loader, device=torch.device("cpu"), max_images=6)
    centroid = outputs["future_descriptors"][0, 0, 0].tolist()
    artifact = {
        "metadata": {"grid_size": 2, "n_layers": outputs["future_descriptors"].shape[1], "n_cells": 4},
        "circuits": [
            {
                "id": 0,
                "representative_node": [0, 0],
                "active_nodes": [[0, 0], [1, 0], [2, 0]],
                "centroids": {"0:0": centroid, "1:0": outputs["future_descriptors"][0, 1, 0].tolist(), "2:0": outputs["future_descriptors"][0, 2, 0].tolist()},
                "thresholds": {"0:0": -1.0, "1:0": -1.0, "2:0": -1.0},
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
