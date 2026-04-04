from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from flow_circuits.evaluation.q_validation import run_q_checkpoint_validation_experiment


def test_q_validation_selects_best_frozen_and_joint_candidates(tmp_path, monkeypatch):
    frozen_dir = tmp_path / "frozen"
    joint_dir = tmp_path / "joint"
    frozen_dir.mkdir()
    joint_dir.mkdir()
    phase_b = frozen_dir / "phase_b_frozen.pt"
    frozen_epoch = frozen_dir / "phase_c_frozen_lambda_0p1_epoch_10.pt"
    joint_epoch = joint_dir / "phase_c_joint_lambda_0p2_epoch_20.pt"
    for path in (phase_b, frozen_epoch, joint_epoch):
        path.write_bytes(b"stub")

    monkeypatch.setattr(
        "flow_circuits.evaluation.q_validation.build_cifar10_splits",
        lambda **kwargs: {"val": object()},
    )

    def fake_load_components(checkpoint_path, device):
        return SimpleNamespace(checkpoint_path=str(checkpoint_path))

    def fake_neighbor(components, *args, **kwargs):
        stem = Path(components.checkpoint_path).name
        recall = {
            "phase_b_frozen.pt": 0.40,
            "phase_c_frozen_lambda_0p1_epoch_10.pt": 0.60,
            "phase_c_joint_lambda_0p2_epoch_20.pt": 0.55,
        }[stem]
        return {"summary": {"mean_recall_at_k": recall, "mean_jaccard_at_k": recall / 2.0}}

    def fake_collect_outputs(components, *args, **kwargs):
        stem = Path(components.checkpoint_path).name
        token = {
            "phase_b_frozen.pt": 0.1,
            "phase_c_frozen_lambda_0p1_epoch_10.pt": 0.2,
            "phase_c_joint_lambda_0p2_epoch_20.pt": 0.3,
        }[stem]
        z = torch.full((2, 1, 1, 2), token)
        return {
            "z": z,
            "local_features": [torch.zeros(2, 1, 3)],
            "flow_targets": torch.zeros(2, 1, 1, 2),
            "future_descriptors": torch.zeros(2, 1, 1, 2),
            "predicted_next": torch.zeros(2, 0, 1, 2),
            "reconstructed_current": torch.zeros(2, 1, 1, 2),
        }

    def fake_metrics(*args, **kwargs):
        marker = float(args[0].mean().item())
        return SimpleNamespace(
            prediction_cosine_mean=marker,
            reconstruction_cosine_mean=marker + 0.1,
            trajectory_alignment_mean=marker + 0.2,
            trajectory_alignment_std=0.0,
        )

    monkeypatch.setattr("flow_circuits.evaluation.q_validation.load_components_from_checkpoint", fake_load_components)
    monkeypatch.setattr("flow_circuits.evaluation.efficient_validation.run_neighbor_agreement_experiment", fake_neighbor)
    monkeypatch.setattr("flow_circuits.evaluation.q_validation.collect_model_outputs", fake_collect_outputs)
    monkeypatch.setattr("flow_circuits.evaluation.q_validation.evaluate_representation_metrics", fake_metrics)

    result = run_q_checkpoint_validation_experiment(
        base_config={
            "data": {"data_dir": str(tmp_path / "data"), "batch_size": 2, "num_workers": 0, "seed": 0, "augment_fit": False, "download": False},
            "training": {"alignment_max_pairs": 16},
        },
        frozen_checkpoint_dir=frozen_dir,
        joint_checkpoint_dir=joint_dir,
        device=torch.device("cpu"),
        max_images=2,
        anchor_images=2,
    )

    assert result["selected"]["frozen"]["checkpoint_path"] == str(frozen_epoch)
    assert result["selected"]["joint"]["checkpoint_path"] == str(joint_epoch)
    assert result["summary"]["n_frozen_candidates"] == 2
    assert result["summary"]["n_joint_candidates"] == 1
