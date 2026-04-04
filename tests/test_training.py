from __future__ import annotations

import copy
from pathlib import Path

import torch

from flow_circuits.training import (
    FlowCircuitTrainer,
    collect_discovery_outputs,
    collect_interpretability_outputs,
    collect_intervention_outputs,
    collect_probe_outputs,
    load_components_from_checkpoint,
)
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


def test_task_specific_collectors_return_only_needed_outputs(monkeypatch, minimal_config, tiny_loader):
    monkeypatch.setattr(
        "flow_circuits.training.trainer.build_cifar10_splits",
        lambda **kwargs: {"fit": tiny_loader, "val": tiny_loader, "discovery": tiny_loader, "test": tiny_loader},
    )
    trainer = FlowCircuitTrainer(copy.deepcopy(minimal_config))

    discovery = collect_discovery_outputs(trainer.components, tiny_loader, device=torch.device("cpu"), max_images=4)
    interpretability = collect_interpretability_outputs(
        trainer.components,
        tiny_loader,
        device=torch.device("cpu"),
        max_images=4,
    )
    intervention = collect_intervention_outputs(trainer.components, tiny_loader, device=torch.device("cpu"), max_images=4)
    probe = collect_probe_outputs(trainer.components, tiny_loader, device=torch.device("cpu"), max_images=4)

    assert set(discovery) == {"flow_targets", "future_descriptors", "predicted_next", "labels", "indices"}
    assert set(interpretability) == {"z", "future_descriptors", "images", "logits", "labels", "indices"}
    assert set(intervention) == {"future_descriptors", "images", "logits", "labels", "indices"}
    assert set(probe) == {"z", "local_features", "future_descriptors", "labels", "indices"}
    assert discovery["future_descriptors"].shape[0] == 4
    assert interpretability["images"].shape[0] == 4
    assert intervention["images"].shape[0] == 4
    assert probe["z"].shape[0] == 4


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
    phase_c_path = Path(aligned_config["logging"]["checkpoint_dir"], "phase_c.pt")
    phase_c_checkpoint = torch.load(phase_c_path, map_location="cpu", weights_only=False)

    assert summary["final_phase"] == "phase_b"
    assert summary["phase_c"]["accepted"] is False
    assert checkpoint["phase"] == "phase_b"
    assert phase_c_path.exists()
    assert phase_c_checkpoint["phase"] == "phase_c"
    assert summary["phase_c"]["saved_checkpoint"] == str(phase_c_path)
    assert summary["phase_c"]["metrics"].prediction_cosine_mean == 0.27
    assert phase_c_checkpoint["config"]["objectives"]["lambda_traj"] == 0.1
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
    phase_c_path = Path(aligned_config["logging"]["checkpoint_dir"], "phase_c.pt")
    phase_c_checkpoint = torch.load(phase_c_path, map_location="cpu", weights_only=False)
    loaded = load_components_from_checkpoint(checkpoint_path, device=torch.device("cpu"))

    assert summary["final_phase"] == "phase_c"
    assert summary["phase_c"]["accepted"] is True
    assert checkpoint["phase"] == "phase_c"
    assert phase_c_path.exists()
    assert phase_c_checkpoint["phase"] == "phase_c"
    assert summary["phase_c"]["saved_checkpoint"] == str(phase_c_path)
    assert checkpoint["config"]["objectives"]["lambda_traj"] == 0.1
    assert checkpoint["config"]["objectives"]["traj_topk"] == 2
    assert checkpoint["config"]["objectives"]["traj_gamma"] == 0.0
    assert checkpoint["config"]["objectives"]["traj_tau"] == 0.2
    assert loaded.checkpoint["phase"] == "phase_c"
    assert loaded.config["objectives"]["lambda_traj"] == 0.1


def test_phase_c_resets_learning_rate_before_candidate_sweep(monkeypatch, aligned_config, tiny_loader):
    monkeypatch.setattr(
        "flow_circuits.training.trainer.build_cifar10_splits",
        lambda **kwargs: {"fit": tiny_loader, "val": tiny_loader, "discovery": tiny_loader, "test": tiny_loader},
    )
    trainer = FlowCircuitTrainer(copy.deepcopy(aligned_config))
    phase_c_lrs = []
    original_run_phase = trainer._run_phase

    def wrapped_run_phase(phase, epochs, *, lambda_rec, lambda_traj):
        if phase == "phase_c":
            phase_c_lrs.append(trainer.optimizer.param_groups[0]["lr"])
            return []
        return original_run_phase(phase, epochs, lambda_rec=lambda_rec, lambda_traj=lambda_traj)

    monkeypatch.setattr(trainer, "_run_phase", wrapped_run_phase)
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

    trainer.train()

    assert phase_c_lrs == [aligned_config["training"]["lr"]]


def test_resume_from_phase_b_skips_retraining_ab(monkeypatch, aligned_config, tiny_loader):
    monkeypatch.setattr(
        "flow_circuits.training.trainer.build_cifar10_splits",
        lambda **kwargs: {"fit": tiny_loader, "val": tiny_loader, "discovery": tiny_loader, "test": tiny_loader},
    )
    initial_trainer = FlowCircuitTrainer(copy.deepcopy(aligned_config))
    initial_trainer.train()
    phase_b_path = Path(aligned_config["logging"]["checkpoint_dir"], "phase_b.pt")

    resumed_trainer = FlowCircuitTrainer(copy.deepcopy(aligned_config), resume_from=phase_b_path)
    seen_phases = []

    def wrapped_run_phase(phase, epochs, *, lambda_rec, lambda_traj):
        seen_phases.append(phase)
        return []

    metrics_sequence = iter(
        [
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
    monkeypatch.setattr(resumed_trainer, "_run_phase", wrapped_run_phase)
    monkeypatch.setattr(resumed_trainer, "_full_validation_metrics", lambda loader: next(metrics_sequence))
    monkeypatch.setattr(
        resumed_trainer,
        "_evaluate_baselines",
        lambda regressors: BaselineComparison(0.20, 0.20, 0.20, 0.20),
    )

    summary = resumed_trainer.train()

    assert seen_phases == ["phase_c"]
    assert summary["final_phase"] == "phase_b"
    assert summary["phase_c"]["accepted"] is False
