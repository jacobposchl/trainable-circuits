from __future__ import annotations

import json
from pathlib import Path

from flow_circuits.training.branch_workflows import run_backbone_and_z_training_workflow


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
