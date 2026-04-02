from __future__ import annotations

import copy
import json
from pathlib import Path

import torch

from flow_circuits.discovery import CandidateCircuitDiscoverer
from flow_circuits.interventions import run_circuit_interventions
from flow_circuits.training import FlowCircuitTrainer, collect_model_outputs, load_components_from_checkpoint


def test_end_to_end_tiny_pipeline(monkeypatch, minimal_config, tiny_loader, tmp_path):
    config = copy.deepcopy(minimal_config)
    config["logging"]["checkpoint_dir"] = str(tmp_path / "checkpoints")
    monkeypatch.setattr(
        "flow_circuits.training.trainer.build_cifar10_splits",
        lambda **kwargs: {"fit": tiny_loader, "val": tiny_loader, "discovery": tiny_loader, "test": tiny_loader},
    )
    trainer = FlowCircuitTrainer(config)
    trainer.train()

    checkpoint_path = Path(config["logging"]["checkpoint_dir"], "final.pt")
    loaded = load_components_from_checkpoint(checkpoint_path, device=torch.device("cpu"))
    discovery_outputs = collect_model_outputs(loaded, tiny_loader, device=torch.device("cpu"), max_images=6)

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
        future_descriptors=discovery_outputs["future_descriptors"].numpy(),
        predicted_next=discovery_outputs["predicted_next"].numpy(),
        flow_targets=discovery_outputs["flow_targets"].numpy(),
        dataset_indices=discovery_outputs["indices"].numpy(),
        labels=discovery_outputs["labels"].numpy(),
    )
    artifact_path = tmp_path / "candidate_circuits.json"
    discoverer.save(artifact, artifact_path)

    if artifact["circuits"]:
        results = run_circuit_interventions(
            loaded,
            artifact,
            discovery_outputs,
            alpha=0.5,
            output_path=tmp_path / "interventions.json",
        )
        assert isinstance(results, list)

    assert checkpoint_path.exists()
    assert artifact_path.exists()
    assert json.loads(artifact_path.read_text(encoding="utf-8"))["metadata"]["grid_size"] == 2
