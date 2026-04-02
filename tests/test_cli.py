from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

from flow_circuits.cli import discover as discover_cli
from flow_circuits.cli import evaluate as evaluate_cli
from flow_circuits.cli import intervene as intervene_cli
from flow_circuits.cli import train as train_cli


def test_train_cli_invokes_trainer(monkeypatch, minimal_config, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(minimal_config), encoding="utf-8")

    class DummyTrainer:
        def __init__(self, config):
            self.config = config

        def train(self):
            return {"ok": True}

    monkeypatch.setattr(train_cli, "FlowCircuitTrainer", DummyTrainer)
    monkeypatch.setattr(sys, "argv", ["flow-train", "--config", str(config_path)])
    train_cli.main()


def test_evaluate_cli_writes_json(monkeypatch, tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_text("stub", encoding="utf-8")

    class DummyMetrics:
        def to_dict(self):
            return {"prediction_cosine_mean": 0.5}

    class DummyBaseline:
        def to_dict(self):
            return {"best_baseline": 0.2}

    class DummyTrainer:
        def __init__(self, config):
            self.config = config

        def _fit_baselines(self):
            return object()

        def _evaluate_baselines(self, regressors):
            return DummyBaseline()

    monkeypatch.setattr(
        evaluate_cli,
        "load_components_from_checkpoint",
        lambda checkpoint, device: type("Loaded", (), {"config": {"data": {"data_dir": "x", "batch_size": 2, "num_workers": 0, "seed": 0, "augment_fit": False, "download": False}, "training": {"validation_images": 2}}})(),
    )
    monkeypatch.setattr(
        evaluate_cli,
        "build_cifar10_splits",
        lambda **kwargs: {"test": object()},
    )
    monkeypatch.setattr(
        evaluate_cli,
        "collect_model_outputs",
        lambda *args, **kwargs: {"z": None, "flow_targets": None, "future_descriptors": None, "predicted_next": None, "reconstructed_current": None},
    )
    monkeypatch.setattr(evaluate_cli, "evaluate_representation_metrics", lambda *args, **kwargs: DummyMetrics())
    monkeypatch.setattr(evaluate_cli, "FlowCircuitTrainer", DummyTrainer)
    monkeypatch.setattr(sys, "argv", ["flow-evaluate", "--checkpoint", str(checkpoint), "--output", str(tmp_path / "eval.json")])
    evaluate_cli.main()

    assert json.loads((tmp_path / "eval.json").read_text(encoding="utf-8"))["representation_metrics"]["prediction_cosine_mean"] == 0.5


def test_discover_and_intervene_clis_write_outputs(monkeypatch, tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_text("stub", encoding="utf-8")
    artifact_path = tmp_path / "circuits.json"

    loaded = type(
        "Loaded",
        (),
        {
            "config": {
                "data": {"data_dir": "x", "batch_size": 2, "num_workers": 0, "seed": 0, "augment_fit": False, "download": False},
                "discovery": {"min_cluster_fraction": 0.2, "max_cluster_fraction": 0.8, "min_cluster_size": 2, "bootstrap_iterations": 1, "stability_threshold": 0.0, "merge_threshold": 0.5, "node_threshold": 0.5, "seed": 0},
                "tokenization": {"grid_size": 2},
                "interventions": {"alpha": 0.5},
            },
        },
    )()
    monkeypatch.setattr(discover_cli, "load_components_from_checkpoint", lambda *args, **kwargs: loaded)
    monkeypatch.setattr(discover_cli, "build_cifar10_splits", lambda **kwargs: {"discovery": object()})
    monkeypatch.setattr(
        discover_cli,
        "collect_model_outputs",
        lambda *args, **kwargs: {
            "future_descriptors": __import__("torch").zeros(2, 3, 4, 8),
            "predicted_next": __import__("torch").zeros(2, 2, 4, 6),
            "flow_targets": __import__("torch").zeros(2, 3, 4, 6),
            "indices": __import__("torch").arange(2),
            "labels": __import__("torch").tensor([0, 1]),
        },
    )

    class DummyDiscoverer:
        def __init__(self, **kwargs):
            pass

        def discover(self, **kwargs):
            return {"metadata": {"grid_size": 2, "n_layers": 3, "n_cells": 4}, "circuits": []}

        def save(self, artifact, path):
            Path(path).write_text(json.dumps(artifact), encoding="utf-8")

    monkeypatch.setattr(discover_cli, "CandidateCircuitDiscoverer", DummyDiscoverer)
    monkeypatch.setattr(sys, "argv", ["flow-discover", "--checkpoint", str(checkpoint), "--output", str(artifact_path)])
    discover_cli.main()
    assert artifact_path.exists()

    monkeypatch.setattr(intervene_cli, "load_components_from_checkpoint", lambda *args, **kwargs: loaded)
    monkeypatch.setattr(intervene_cli, "build_cifar10_splits", lambda **kwargs: {"test": object()})
    monkeypatch.setattr(
        intervene_cli,
        "collect_model_outputs",
        lambda *args, **kwargs: {"images": None, "future_descriptors": None, "indices": None, "labels": None, "logits": None},
    )
    monkeypatch.setattr(
        intervene_cli,
        "run_circuit_interventions",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(sys, "argv", ["flow-intervene", "--checkpoint", str(checkpoint), "--circuits", str(artifact_path), "--output", str(tmp_path / "summary.json")])
    intervene_cli.main()
    assert (tmp_path / "summary.csv").exists()
