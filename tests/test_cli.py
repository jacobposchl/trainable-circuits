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
    seen = {}

    class DummyTrainer:
        def __init__(self, config, *, resume_from=None):
            self.config = config
            seen["resume_from"] = resume_from

        def train(self):
            return {"ok": True}

    monkeypatch.setattr(train_cli, "FlowCircuitTrainer", DummyTrainer)
    monkeypatch.setattr(sys, "argv", ["flow-train", "--config", str(config_path), "--resume", "phase_b.pt"])
    train_cli.main()
    assert seen["resume_from"] == "phase_b.pt"


def test_evaluate_cli_writes_json(monkeypatch, tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_text("stub", encoding="utf-8")

    class DummyMetrics:
        def to_dict(self):
            return {"prediction_cosine_mean": 0.5}

    class DummyBaseline:
        def to_dict(self):
            return {"best_baseline": 0.2}

    class DummyCheck:
        def to_dict(self):
            return {"passes": True}

    monkeypatch.setattr(
        evaluate_cli,
        "load_components_from_checkpoint",
        lambda checkpoint, device: type(
            "Loaded",
            (),
            {
                "config": {
                    "data": {"data_dir": "x", "batch_size": 2, "num_workers": 0, "seed": 0, "augment_fit": False, "download": False},
                    "training": {"validation_images": 2, "baseline_fit_images": 2},
                },
                "tokenizer": type("Tokenizer", (), {"build_future_descriptors": staticmethod(lambda flow_targets, depth_permutations=None: flow_targets)})(),
                "encoder": type("Encoder", (), {"final_norm": type("FinalNorm", (), {"weight": __import__("torch").zeros(1)})()})(),
            },
        )(),
    )
    monkeypatch.setattr(
        evaluate_cli,
        "build_cifar10_splits",
        lambda **kwargs: {"fit": object(), "test": object()},
    )
    monkeypatch.setattr(
        evaluate_cli,
        "collect_model_outputs",
        lambda *args, **kwargs: {
            "z": __import__("torch").zeros(2, 3, 4, 8),
            "local_features": [__import__("torch").zeros(2, 4, 5) for _ in range(3)],
            "flow_targets": __import__("torch").zeros(2, 3, 4, 6),
            "future_descriptors": __import__("torch").zeros(2, 3, 4, 6),
            "predicted_next": __import__("torch").zeros(2, 2, 4, 6),
            "reconstructed_current": __import__("torch").zeros(2, 3, 4, 6),
        },
    )
    monkeypatch.setattr(
        evaluate_cli,
        "collect_baseline_features",
        lambda *args, **kwargs: ([__import__("numpy").zeros((2, 4, 5)) for _ in range(2)], [__import__("numpy").zeros((2, 4, 6)) for _ in range(2)], [__import__("numpy").zeros((2, 4, 6)) for _ in range(2)]),
    )
    monkeypatch.setattr(
        evaluate_cli,
        "BaselineRegressors",
        type(
            "DummyRegressors",
            (),
            {
                "fit": staticmethod(lambda **kwargs: type("Reg", (), {"evaluate": lambda self, **kw: DummyBaseline(), "score_predictions": lambda self, **kw: {"mean_baseline": __import__("numpy").zeros(2), "local_baseline": __import__("numpy").zeros(2), "flow_baseline": __import__("numpy").zeros(2)}})()),
            },
        ),
    )
    monkeypatch.setattr(evaluate_cli, "evaluate_representation_metrics", lambda *args, **kwargs: DummyMetrics())
    monkeypatch.setattr(evaluate_cli, "compute_prediction_scores_by_image", lambda *args, **kwargs: __import__("numpy").zeros(2))
    monkeypatch.setattr(evaluate_cli, "compute_alignment_scores", lambda *args, **kwargs: {"model_node_scores": __import__("numpy").zeros(2), "local_node_scores": __import__("numpy").zeros(2), "flow_node_scores": __import__("numpy").zeros(2)})
    monkeypatch.setattr(evaluate_cli, "evaluate_prediction_check", lambda *args, **kwargs: DummyCheck())
    monkeypatch.setattr(evaluate_cli, "evaluate_alignment_check", lambda *args, **kwargs: DummyCheck())
    monkeypatch.setattr(evaluate_cli, "_future_shuffle_prediction_null", lambda *args, **kwargs: {"drop": 0.0})
    monkeypatch.setattr(evaluate_cli, "_depth_order_alignment_null", lambda *args, **kwargs: {"drop": 0.0})
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
                "discovery": {"min_cluster_fraction": 0.2, "max_cluster_fraction": 0.8, "min_cluster_size": 2, "bootstrap_iterations": 1, "stability_threshold": 0.0, "merge_threshold": 0.5, "node_threshold": 0.5, "seed": 0, "batch_size": 5},
                "tokenization": {"grid_size": 2},
                "interventions": {"alpha": 0.5, "batch_size": 7},
            },
        },
    )()
    monkeypatch.setattr(discover_cli, "load_components_from_checkpoint", lambda *args, **kwargs: loaded)
    discover_loader_kwargs = {}
    monkeypatch.setattr(discover_cli, "build_cifar10_splits", lambda **kwargs: discover_loader_kwargs.update(kwargs) or {"discovery": object()})
    monkeypatch.setattr(
        discover_cli,
        "collect_discovery_outputs",
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
    assert discover_loader_kwargs["batch_size"] == 5

    monkeypatch.setattr(intervene_cli, "load_components_from_checkpoint", lambda *args, **kwargs: loaded)
    intervene_loader_kwargs = {}
    monkeypatch.setattr(intervene_cli, "build_cifar10_splits", lambda **kwargs: intervene_loader_kwargs.update(kwargs) or {"test": object()})
    monkeypatch.setattr(
        intervene_cli,
        "collect_intervention_outputs",
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
    assert intervene_loader_kwargs["batch_size"] == 7


def test_discover_cli_parallel_seed_runs_preserve_seed_order(monkeypatch, tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_text("stub", encoding="utf-8")
    artifact_path = tmp_path / "circuits.json"
    loaded = type(
        "Loaded",
        (),
        {
            "config": {
                "data": {"data_dir": "x", "batch_size": 2, "num_workers": 0, "seed": 0, "augment_fit": False, "download": False},
                "discovery": {
                    "min_cluster_fraction": 0.2,
                    "max_cluster_fraction": 0.8,
                    "min_cluster_size": 2,
                    "bootstrap_iterations": 1,
                    "stability_threshold": 0.0,
                    "merge_threshold": 0.5,
                    "node_threshold": 0.5,
                    "seed": 0,
                    "seeds": [0, 1, 2],
                    "n_jobs": 2,
                    "compute_seed_stability": False,
                    "compute_node_shuffle_null": False,
                },
                "tokenization": {"grid_size": 2},
            },
        },
    )()
    monkeypatch.setattr(discover_cli, "load_components_from_checkpoint", lambda *args, **kwargs: loaded)
    monkeypatch.setattr(discover_cli, "build_cifar10_splits", lambda **kwargs: {"discovery": object()})
    monkeypatch.setattr(
        discover_cli,
        "collect_discovery_outputs",
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
            self.random_seed = kwargs["random_seed"]

        def discover(self, **kwargs):
            import time
            time.sleep(0.01 * (2 - self.random_seed))
            return {
                "metadata": {"grid_size": 2, "n_layers": 3, "n_cells": 4},
                "node_clusters": [{"seed": self.random_seed}],
                "circuits": [{"id": self.random_seed, "active_nodes": [], "image_set": [], "representative_node": [0, 0], "centroids": {"0:0": [1.0]}, "thresholds": {"0:0": 0.0}}],
            }

        def save(self, artifact, path):
            Path(path).write_text(json.dumps(artifact), encoding="utf-8")

    monkeypatch.setattr(discover_cli, "CandidateCircuitDiscoverer", DummyDiscoverer)
    monkeypatch.setattr(sys, "argv", ["flow-discover", "--checkpoint", str(checkpoint), "--output", str(artifact_path)])
    discover_cli.main()
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert [run["seed"] for run in artifact["seed_runs"]] == [0, 1, 2]
    assert artifact["stability_summary"]["skipped"] is True
    assert artifact["null_checks"]["node_shuffle"]["skipped"] is True
