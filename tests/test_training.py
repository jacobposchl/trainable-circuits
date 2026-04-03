from __future__ import annotations

import copy
from pathlib import Path

import torch

from flow_circuits.training import FlowCircuitTrainer, load_components_from_checkpoint
from flow_circuits.evaluation import BaselineComparison, RepresentationMetrics


def test_trainer_runs_base_phases_with_tiny_loaders(monkeypatch, minimal_config, tiny_loader):
    monkeypatch.setattr(
        "flow_circuits.training.trainer.build_cifar10_splits",
        lambda **kwargs: {"fit": tiny_loader, "val": tiny_loader, "discovery": tiny_loader, "test": tiny_loader},
    )
    trainer = FlowCircuitTrainer(copy.deepcopy(minimal_config))
    summary = trainer.train()

    assert summary["final_phase"] == "phase_b"
    assert Path(minimal_config["logging"]["checkpoint_dir"], "final.pt").exists()


def test_checkpoint_round_trip_loads_new_components(monkeypatch, minimal_config, tiny_loader):
    monkeypatch.setattr(
        "flow_circuits.training.trainer.build_cifar10_splits",
        lambda **kwargs: {"fit": tiny_loader, "val": tiny_loader, "discovery": tiny_loader, "test": tiny_loader},
    )
    trainer = FlowCircuitTrainer(copy.deepcopy(minimal_config))
    trainer.train()
    checkpoint_path = Path(minimal_config["logging"]["checkpoint_dir"], "final.pt")
    loaded = load_components_from_checkpoint(checkpoint_path, device=torch.device("cpu"))

    assert loaded.config["experiment"]["name"] == "pytest"
    assert len(loaded.observer.layer_channels) == 8


def test_aligned_mode_skips_phase_c_when_baseline_gate_fails(monkeypatch, aligned_config, tiny_loader):
    monkeypatch.setattr(
        "flow_circuits.training.trainer.build_cifar10_splits",
        lambda **kwargs: {"fit": tiny_loader, "val": tiny_loader, "discovery": tiny_loader, "test": tiny_loader},
    )
    trainer = FlowCircuitTrainer(copy.deepcopy(aligned_config))
    monkeypatch.setattr(
        trainer,
        "_full_validation_metrics",
        lambda loader: RepresentationMetrics(
            prediction_cosine_mean=0.1,
            prediction_cosine_sem=0.01,
            reconstruction_cosine_mean=0.1,
            reconstruction_cosine_sem=0.01,
            trajectory_alignment_mean=0.1,
            trajectory_alignment_std=0.0,
        ),
    )
    monkeypatch.setattr(
        trainer,
        "_evaluate_baselines",
        lambda regressors: BaselineComparison(0.2, 0.2, 0.2, 0.2),
    )
    summary = trainer.train()

    assert summary["final_phase"] == "phase_b"
    assert summary["phase_c"]["accepted"] is False


def test_phase_c_rejection_restores_phase_b_checkpoint_config(monkeypatch, aligned_config, tiny_loader):
    monkeypatch.setattr(
        "flow_circuits.training.trainer.build_cifar10_splits",
        lambda **kwargs: {"fit": tiny_loader, "val": tiny_loader, "discovery": tiny_loader, "test": tiny_loader},
    )
    trainer = FlowCircuitTrainer(copy.deepcopy(aligned_config))

    metrics_sequence = iter(
        [
            RepresentationMetrics(
                prediction_cosine_mean=0.30,
                prediction_cosine_sem=0.01,
                reconstruction_cosine_mean=0.10,
                reconstruction_cosine_sem=0.01,
                trajectory_alignment_mean=0.20,
                trajectory_alignment_std=0.0,
            ),
            RepresentationMetrics(
                prediction_cosine_mean=0.27,
                prediction_cosine_sem=0.01,
                reconstruction_cosine_mean=0.10,
                reconstruction_cosine_sem=0.01,
                trajectory_alignment_mean=0.19,
                trajectory_alignment_std=0.0,
            ),
        ]
    )
    monkeypatch.setattr(trainer, "_full_validation_metrics", lambda loader: next(metrics_sequence))
    monkeypatch.setattr(
        trainer,
        "_evaluate_baselines",
        lambda regressors: BaselineComparison(0.20, 0.20, 0.20, 0.20),
    )

    summary = trainer.train()
    checkpoint_path = Path(aligned_config["logging"]["checkpoint_dir"], "final.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    assert summary["final_phase"] == "phase_b"
    assert summary["phase_c"]["accepted"] is False
    assert checkpoint["phase"] == "phase_b"
    assert checkpoint["config"]["objectives"].get("lambda_traj") is None
    assert checkpoint["config"]["objectives"]["traj_topk"] == aligned_config["objectives"]["traj_topk"]
    assert checkpoint["config"]["objectives"]["traj_gamma"] == aligned_config["objectives"]["traj_gamma"]
    assert checkpoint["config"]["objectives"]["traj_tau"] == aligned_config["objectives"]["traj_tau"]


def test_phase_c_acceptance_persists_winning_checkpoint_config(monkeypatch, aligned_config, tiny_loader):
    monkeypatch.setattr(
        "flow_circuits.training.trainer.build_cifar10_splits",
        lambda **kwargs: {"fit": tiny_loader, "val": tiny_loader, "discovery": tiny_loader, "test": tiny_loader},
    )
    trainer = FlowCircuitTrainer(copy.deepcopy(aligned_config))

    metrics_sequence = iter(
        [
            RepresentationMetrics(
                prediction_cosine_mean=0.30,
                prediction_cosine_sem=0.01,
                reconstruction_cosine_mean=0.10,
                reconstruction_cosine_sem=0.01,
                trajectory_alignment_mean=0.20,
                trajectory_alignment_std=0.0,
            ),
            RepresentationMetrics(
                prediction_cosine_mean=0.295,
                prediction_cosine_sem=0.01,
                reconstruction_cosine_mean=0.10,
                reconstruction_cosine_sem=0.01,
                trajectory_alignment_mean=0.24,
                trajectory_alignment_std=0.0,
            ),
        ]
    )
    monkeypatch.setattr(trainer, "_full_validation_metrics", lambda loader: next(metrics_sequence))
    monkeypatch.setattr(
        trainer,
        "_evaluate_baselines",
        lambda regressors: BaselineComparison(0.20, 0.20, 0.20, 0.20),
    )

    summary = trainer.train()
    checkpoint_path = Path(aligned_config["logging"]["checkpoint_dir"], "final.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    loaded = load_components_from_checkpoint(checkpoint_path, device=torch.device("cpu"))

    assert summary["final_phase"] == "phase_c"
    assert summary["phase_c"]["accepted"] is True
    assert checkpoint["phase"] == "phase_c"
    assert checkpoint["config"]["objectives"]["lambda_traj"] == 0.1
    assert checkpoint["config"]["objectives"]["traj_topk"] == 2
    assert checkpoint["config"]["objectives"]["traj_gamma"] == 0.0
    assert checkpoint["config"]["objectives"]["traj_tau"] == 0.2
    assert loaded.checkpoint["phase"] == "phase_c"
    assert loaded.config["objectives"]["lambda_traj"] == 0.1
