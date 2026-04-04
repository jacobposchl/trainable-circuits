from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from flow_circuits.training.branch_workflows import (
    _reconstruct_phase_c_candidates,
    run_backbone_and_z_training_workflow,
)


def test_branch_training_workflow_writes_candidate_manifest(tmp_path, monkeypatch):
    def fake_backbone(base_config, *, backbone_epochs, output_path):
        Path(output_path).write_bytes(b"backbone")
        return {"output_path": str(output_path), "best_epoch": 1}

    def fake_phase_ab(base_config, *, backbone_checkpoint, checkpoint_dir, phase_a_epochs, phase_b_epochs):
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "phase_b.pt").write_bytes(b"phaseb")
        (checkpoint_dir / "phase_ab_summary.json").write_text("{}", encoding="utf-8")
        return {"final_phase": "phase_b"}

    def fake_sweep(**kwargs):
        output_dir = Path(kwargs["output_dir"])
        return [
            {
                "branch_tag": kwargs["branch_tag"],
                "lambda_traj": 0.1,
                "epoch": 5,
                "checkpoint_path": str(output_dir / f"phase_c_{kwargs['branch_tag']}_lambda_0p1_epoch_5.pt"),
            }
        ]

    monkeypatch.setattr("flow_circuits.training.branch_workflows._train_supervised_backbone", fake_backbone)
    monkeypatch.setattr("flow_circuits.training.branch_workflows._train_frozen_phase_ab", fake_phase_ab)
    monkeypatch.setattr("flow_circuits.training.branch_workflows._run_phase_c_milestone_sweep", fake_sweep)

    result = run_backbone_and_z_training_workflow(
        {
            "experiment": {"name": "pytest", "mode": "aligned"},
            "data": {"data_dir": str(tmp_path / "data"), "batch_size": 2, "num_workers": 0, "seed": 0, "augment_fit": False, "download": False},
            "backbone": {"arch": "resnet18", "pretrained": False},
            "training": {"lr": 1.0e-3, "weight_decay": 1.0e-4, "grad_clip": 1.0, "validation_images": 4, "alignment_max_pairs": 16},
            "logging": {"checkpoint_dir": str(tmp_path / "ckpts")},
            "objectives": {"lambda_rec": 0.2},
        },
        backbone_epochs=1,
        phase_a_epochs=1,
        phase_b_epochs=1,
        phase_c_max_epochs=5,
        phase_c_milestones=[5],
        lambda_traj_candidates=[0.1],
        output_dir=tmp_path / "workflow",
        joint_branch_enabled=True,
        force_rerun=True,
    )

    manifest_path = tmp_path / "workflow" / "training_candidates.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["frozen_phase_c_candidates"][0]["branch_tag"] == "frozen"
    assert manifest["joint_phase_c_candidates"][0]["branch_tag"] == "joint"
    assert Path(result["frozen_phase_b_checkpoint"]).exists()


def test_reconstruct_phase_c_candidates_from_existing_checkpoints(tmp_path):
    output_dir = tmp_path / "frozen_branch"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "phase_c_frozen_lambda_0p1_epoch_10.pt"
    torch.save(
        {
            "validation": {
                "prediction_cosine_mean": 0.91,
                "reconstruction_cosine_mean": 0.97,
                "trajectory_alignment_mean": 0.62,
            },
            "summary": {
                "branch_tag": "frozen",
                "lambda_traj": 0.1,
                "epoch": 10,
                "train_backbone": False,
                "ce_weight": 0.0,
            },
        },
        checkpoint_path,
    )

    rows = _reconstruct_phase_c_candidates(
        output_dir=output_dir,
        branch_tag="frozen",
        lambda_traj_candidates=[0.1, 0.2],
        milestones=[5, 10],
    )

    assert len(rows) == 1
    assert rows[0]["checkpoint_path"] == str(checkpoint_path)
    assert rows[0]["lambda_traj"] == 0.1
    assert rows[0]["epoch"] == 10
    assert rows[0]["trajectory_alignment_mean"] == 0.62


def test_phase_c_sweep_resumes_from_latest_existing_milestone(tmp_path, monkeypatch):
    from flow_circuits.training import branch_workflows as workflows

    output_dir = tmp_path / "joint_branch"
    output_dir.mkdir(parents=True, exist_ok=True)
    phase_b_checkpoint = tmp_path / "phase_b_frozen.pt"
    phase_b_checkpoint.write_bytes(b"phaseb")

    existing_complete = output_dir / "phase_c_joint_lambda_0p05_epoch_20.pt"
    existing_partial = output_dir / "phase_c_joint_lambda_0p1_epoch_10.pt"
    for path, lambda_traj, epoch in [
        (existing_complete, 0.05, 20),
        (existing_partial, 0.1, 10),
    ]:
        torch.save(
            {
                "optimizer_state": {"state": {}, "param_groups": [{"lr": 1.0e-3, "initial_lr": 1.0e-3}]},
                "scheduler_state": {"T_max": 20, "eta_min": 0.0, "base_lrs": [1.0e-3], "last_epoch": epoch, "verbose": False, "_step_count": epoch + 1, "_get_lr_called_within_step": False, "_last_lr": [1.0e-3]},
                "validation": {
                    "prediction_cosine_mean": 0.9 + lambda_traj,
                    "reconstruction_cosine_mean": 0.97,
                    "trajectory_alignment_mean": 0.6 + lambda_traj,
                },
                "summary": {
                    "branch_tag": "joint",
                    "lambda_traj": lambda_traj,
                    "epoch": epoch,
                    "train_backbone": True,
                    "ce_weight": 1.0,
                },
            },
            path,
        )

    load_calls = []
    epoch_calls = []

    def fake_build_cifar10_splits(**kwargs):
        return {"fit": object(), "val": object()}

    def fake_load_components_from_checkpoint(checkpoint_path, device, config_overrides=None):
        checkpoint_path = Path(checkpoint_path)
        load_calls.append(checkpoint_path.name)
        checkpoint = {}
        if checkpoint_path.exists() and checkpoint_path.suffix == ".pt" and checkpoint_path != phase_b_checkpoint:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        return SimpleNamespace(checkpoint=checkpoint)

    class DummyOptimizer:
        def __init__(self):
            self.param_groups = [{"lr": 1.0e-3, "initial_lr": 1.0e-3}]
            self.loaded = None

        def state_dict(self):
            return {"state": {}, "param_groups": self.param_groups}

        def load_state_dict(self, state):
            self.loaded = state
            self.param_groups = state.get("param_groups", self.param_groups)

    class DummyScheduler:
        def __init__(self, optimizer, T_max):
            self.optimizer = optimizer
            self.T_max = T_max
            self.loaded = None

        def step(self):
            return None

        def state_dict(self):
            return {"T_max": self.T_max}

        def load_state_dict(self, state):
            self.loaded = state

    def fake_build_optimizer(*args, **kwargs):
        return DummyOptimizer()

    def fake_run_branch_epoch(*args, **kwargs):
        epoch_calls.append(kwargs["train"])
        return {
            "loss": 0.1,
            "pred_loss": 0.05,
            "rec_loss": 0.02,
            "traj_loss": 0.01,
            "ce_loss": 0.03,
            "prediction_cosine": 0.9,
            "reconstruction_cosine": 0.95,
        }

    def fake_validation_metrics(*args, **kwargs):
        return {
            "prediction_cosine_mean": 0.91,
            "reconstruction_cosine_mean": 0.96,
            "trajectory_alignment_mean": 0.61,
        }

    def fake_save_flow_checkpoint(**kwargs):
        torch.save(
            {
                "validation": kwargs["validation"],
                "summary": kwargs["extra_summary"],
                "optimizer_state": kwargs["optimizer"].state_dict(),
                "scheduler_state": kwargs["scheduler"].state_dict(),
            },
            kwargs["path"],
        )

    monkeypatch.setattr(workflows, "build_cifar10_splits", fake_build_cifar10_splits)
    monkeypatch.setattr(workflows, "load_components_from_checkpoint", fake_load_components_from_checkpoint)
    monkeypatch.setattr(workflows, "_build_optimizer", fake_build_optimizer)
    monkeypatch.setattr(workflows, "CosineAnnealingLR", DummyScheduler)
    monkeypatch.setattr(workflows, "_run_branch_epoch", fake_run_branch_epoch)
    monkeypatch.setattr(workflows, "_validation_metrics", fake_validation_metrics)
    monkeypatch.setattr(workflows, "save_flow_checkpoint", fake_save_flow_checkpoint)

    rows = workflows._run_phase_c_milestone_sweep(
        base_config={
            "data": {"data_dir": str(tmp_path / "data"), "batch_size": 2, "num_workers": 0, "seed": 0, "augment_fit": False, "download": False},
            "backbone": {},
            "training": {"lr": 1.0e-3, "weight_decay": 1.0e-4, "grad_clip": 1.0, "validation_images": 4, "alignment_max_pairs": 16},
            "objectives": {"lambda_rec": 0.2},
            "logging": {"checkpoint_dir": str(output_dir)},
        },
        phase_b_checkpoint=phase_b_checkpoint,
        branch_tag="joint",
        output_dir=output_dir,
        max_epochs=20,
        milestones=[10, 20],
        lambda_traj_candidates=[0.05, 0.1, 0.2],
        train_backbone=True,
        ce_weight=1.0,
        backbone_lr_multiplier=0.1,
        force_rerun=False,
    )

    row_keys = {(row["lambda_traj"], row["epoch"]) for row in rows}
    assert (0.05, 20) in row_keys
    assert (0.1, 10) in row_keys
    assert (0.1, 20) in row_keys
    assert (0.2, 10) in row_keys
    assert (0.2, 20) in row_keys
    assert "phase_c_joint_lambda_0p1_epoch_10.pt" in load_calls
    assert "phase_b_frozen.pt" in load_calls
    assert "phase_c_joint_lambda_0p05_epoch_20.pt" not in load_calls
