from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch

from flow_circuits.evaluation.interpretability_validation import (
    run_linear_probe_suite_experiment,
    run_motif_case_study_experiment,
    run_motif_visual_report_experiment,
    run_phase_motif_comparison_experiment,
    run_probe_confusion_analysis_experiment,
    run_probe_error_analysis_experiment,
)


def _make_outputs(z: torch.Tensor, labels: torch.Tensor, logits: torch.Tensor) -> dict:
    return {
        "z": z,
        "future_descriptors": z.clone(),
        "images": torch.zeros(z.shape[0], 3, 32, 32),
        "logits": logits,
        "labels": labels,
        "indices": torch.arange(z.shape[0]),
    }


def test_motif_visual_reports_return_motif_and_node_cluster_cards(monkeypatch):
    z = torch.zeros(4, 2, 2, 3)
    z[0:2, 0, 0, 0] = 1.0
    z[0:2, 1, 1, 0] = 1.0
    z[2:4, 1, 0, 1] = 1.0
    z = torch.nn.functional.normalize(z, dim=-1)
    labels = torch.tensor([0, 0, 1, 1])
    logits = torch.tensor([[4.0, 1.0], [3.5, 1.0], [1.0, 4.0], [1.0, 3.5]])
    outputs = _make_outputs(z, labels, logits)
    monkeypatch.setattr(
        "flow_circuits.evaluation.interpretability_validation.collect_interpretability_outputs",
        lambda *args, **kwargs: outputs,
    )
    motif_artifact = {
        "metadata": {"grid_size": 2},
        "motifs": [
            {
                "id": 0,
                "image_set": [0, 1],
                "member_row_indices": [0, 1],
                "representative_node": [0, 0],
                "active_nodes": [[0, 0], [1, 1]],
                "layer_support": [0, 1],
                "centroids": {"0:0": z[0, 0, 0].tolist(), "1:1": z[0, 1, 1].tolist()},
                "thresholds": {"0:0": 0.0, "1:1": 0.0},
                "stability": {"mean_cluster_stability": 0.9, "n_node_clusters": 2},
                "purity": {"dominant_class": 0, "fraction": 1.0},
            }
        ],
        "node_clusters": [
            {
                "node": [1, 0],
                "image_set": [2, 3],
                "row_indices": [2, 3],
                "size": 2,
                "stability": 0.8,
            }
        ],
    }

    result = run_motif_visual_report_experiment(
        SimpleNamespace(),
        loader=object(),
        device=torch.device("cpu"),
        checkpoint_tag="phase_b",
        motif_artifact=motif_artifact,
        max_images=4,
        topk=1,
        min_visual_budget=2,
    )

    assert result["summary"]["n_visual_objects"] == 2
    assert {row["object_type"] for row in result["visual_objects"]} == {"motif", "node_cluster"}
    for row in result["visual_objects"]:
        assert row["overlay"]["representative_box"]["x1"] > row["overlay"]["representative_box"]["x0"]
        assert row["overlay"]["representative_box"]["y1"] > row["overlay"]["representative_box"]["y0"]


def test_phase_motif_comparison_uses_matches_and_falls_back():
    phase_b_report = {
        "visual_objects": [
            {"object_type": "motif", "id": "0", "class_purity": 0.9, "stability": 0.8, "supporting_layers": [0, 1], "size": 10},
            {"object_type": "motif", "id": "1", "class_purity": 0.7, "stability": 0.7, "supporting_layers": [1], "size": 8},
        ]
    }
    phase_c_report = {
        "visual_objects": [
            {"object_type": "motif", "id": "2", "class_purity": 0.8, "stability": 0.9, "supporting_layers": [0, 1, 2], "size": 12},
            {"object_type": "motif", "id": "3", "class_purity": 0.6, "stability": 0.6, "supporting_layers": [1], "size": 7},
        ]
    }

    matched = run_phase_motif_comparison_experiment(
        phase_b_report,
        phase_c_report,
        phase_match_artifact={"matched_pairs": [{"phase_b_motif_id": 0, "phase_c_motif_id": 2, "image_set_jaccard": 0.5}]},
    )
    fallback = run_phase_motif_comparison_experiment(phase_b_report, phase_c_report)

    assert matched["comparison_rows"][0]["comparison_type"] == "matched_motif_pair"
    assert matched["comparison_rows"][0]["match_quality"] == 0.5
    assert fallback["comparison_rows"][0]["comparison_type"] == "ranked_visual_pair"


@dataclass
class _DummyInterventionResult:
    circuit_id: int
    n_members: int = 2
    n_controls: int = 2
    mean_member_delta_margin: float = 0.4
    mean_member_delta_true: float = 0.3
    mean_nonmember_delta_margin: float = 0.1
    mean_nonmember_delta_true: float = 0.05
    mean_random_node_delta_margin: float = 0.05
    mean_random_cell_delta_margin: float = 0.04
    p_member_vs_nonmember: float = 0.01
    p_member_vs_random_node: float = 0.01
    p_member_vs_random_cell: float = 0.01
    corrected_p_member_vs_nonmember: float = 0.01
    corrected_p_member_vs_random_node: float = 0.01
    corrected_p_member_vs_random_cell: float = 0.01
    ci_member_vs_nonmember: list[float] = None
    ci_member_vs_random_node: list[float] = None
    ci_member_vs_random_cell: list[float] = None
    validated: bool = True

    def __post_init__(self):
        self.ci_member_vs_nonmember = self.ci_member_vs_nonmember or [0.1, 0.5]
        self.ci_member_vs_random_node = self.ci_member_vs_random_node or [0.1, 0.5]
        self.ci_member_vs_random_cell = self.ci_member_vs_random_cell or [0.1, 0.5]

    def to_dict(self):
        return self.__dict__.copy()


def test_motif_case_studies_are_deterministic(monkeypatch):
    z = torch.zeros(4, 2, 1, 2)
    z[0:2, :, 0, 0] = 1.0
    z[2:4, :, 0, 1] = 1.0
    z = torch.nn.functional.normalize(z, dim=-1)
    labels = torch.tensor([0, 0, 1, 1])
    logits = torch.tensor([[4.0, 1.0], [3.5, 1.0], [1.0, 4.0], [1.0, 3.5]])
    outputs = _make_outputs(z, labels, logits)
    monkeypatch.setattr(
        "flow_circuits.evaluation.interpretability_validation.collect_interpretability_outputs",
        lambda *args, **kwargs: outputs,
    )
    monkeypatch.setattr(
        "flow_circuits.evaluation.interpretability_validation.run_circuit_interventions",
        lambda *args, **kwargs: [_DummyInterventionResult(circuit_id=0)],
    )
    monkeypatch.setattr(
        "flow_circuits.evaluation.interpretability_validation.ResidualPatchAblator.ablate",
        lambda self, images, nodes: torch.tensor([[2.0, 1.0], [1.8, 1.2]], dtype=torch.float32),
    )
    motif_artifact = {
        "metadata": {"grid_size": 1, "n_layers": 2, "n_cells": 1},
        "motifs": [
            {
                "id": 0,
                "image_set": [0, 1],
                "member_row_indices": [0, 1],
                "representative_node": [0, 0],
                "active_nodes": [[0, 0], [1, 0]],
                "layer_support": [0, 1],
                "centroids": {"0:0": [1.0, 0.0], "1:0": [1.0, 0.0]},
                "thresholds": {"0:0": 0.0, "1:0": 0.0},
                "stability": {"mean_cluster_stability": 0.9, "n_node_clusters": 2},
                "purity": {"dominant_class": 0, "fraction": 1.0},
            }
        ],
    }
    components = SimpleNamespace(config={"interventions": {"n_jobs": 1}})

    result = run_motif_case_study_experiment(
        components,
        loader=object(),
        device=torch.device("cpu"),
        checkpoint_tag="phase_b",
        motif_artifact=motif_artifact,
        max_images=4,
        topk=1,
        exemplar_count=2,
    )

    assert result["summary"]["validated_count"] == 1
    assert result["case_studies"][0]["validated"] is True
    assert len(result["case_studies"][0]["exemplars"]) == 2
    assert result["case_studies"][0]["exemplars"][0]["delta_margin"] > 0.0


def test_linear_probe_suite_learns_global_layer_and_node_decoding(monkeypatch):
    fit_z = torch.tensor(
        [
            [[[1.0, 0.0]], [[1.0, 0.0]]],
            [[[1.1, 0.0]], [[0.9, 0.0]]],
            [[[0.0, 1.0]], [[0.0, 1.0]]],
            [[[0.1, 0.9]], [[0.0, 1.1]]],
        ]
    )
    val_z = torch.tensor(
        [
            [[[1.0, 0.0]], [[1.0, 0.0]]],
            [[[0.0, 1.0]], [[0.0, 1.0]]],
        ]
    )
    test_z = val_z.clone()
    fit_outputs = _make_outputs(fit_z, torch.tensor([0, 0, 1, 1]), torch.zeros(4, 2))
    val_outputs = _make_outputs(val_z, torch.tensor([0, 1]), torch.zeros(2, 2))
    test_outputs = _make_outputs(test_z, torch.tensor([0, 1]), torch.zeros(2, 2))
    queued = [fit_outputs, val_outputs, test_outputs]
    monkeypatch.setattr(
        "flow_circuits.evaluation.interpretability_validation.collect_interpretability_outputs",
        lambda *args, **kwargs: queued.pop(0),
    )

    result = run_linear_probe_suite_experiment(
        SimpleNamespace(),
        fit_loader=object(),
        val_loader=object(),
        test_loader=object(),
        device=torch.device("cpu"),
        checkpoint_tag="phase_b",
        fit_max_images=4,
        val_max_images=2,
        test_max_images=2,
    )

    assert result["summary"]["global_accuracy"] == 1.0
    assert len(result["per_layer"]) == 2
    assert len(result["per_node"]) == 2


def test_confusion_analysis_uses_backbone_confusion_pairs(monkeypatch):
    base_z = torch.tensor(
        [
            [[[1.0, 0.0]], [[1.0, 0.0]]],
            [[[1.1, 0.0]], [[0.9, 0.0]]],
            [[[0.0, 1.0]], [[0.0, 1.0]]],
            [[[0.1, 0.9]], [[0.0, 1.1]]],
        ]
    )
    fit_outputs = _make_outputs(base_z, torch.tensor([0, 0, 1, 1]), torch.zeros(4, 2))
    val_outputs = _make_outputs(base_z[:2], torch.tensor([0, 1]), torch.zeros(2, 2))
    test_logits = torch.tensor([[1.0, 3.0], [3.0, 1.0], [2.5, 1.0], [1.0, 2.5]])
    test_outputs = _make_outputs(base_z, torch.tensor([0, 1, 0, 1]), test_logits)
    queued = [fit_outputs, val_outputs, test_outputs]
    monkeypatch.setattr(
        "flow_circuits.evaluation.interpretability_validation.collect_interpretability_outputs",
        lambda *args, **kwargs: queued.pop(0),
    )

    result = run_probe_confusion_analysis_experiment(
        SimpleNamespace(),
        fit_loader=object(),
        val_loader=object(),
        test_loader=object(),
        device=torch.device("cpu"),
        checkpoint_tag="phase_c",
        fit_max_images=4,
        val_max_images=2,
        test_max_images=4,
        top_pairs=1,
    )

    assert result["summary"]["n_pairs"] == 1
    assert result["pair_rows"][0]["left_label_name"] == "airplane"
    assert result["pair_rows"][0]["right_label_name"] == "automobile"
    assert result["pair_rows"][0]["top_nodes"]


def test_error_analysis_reports_margin_drop_for_wrong_examples(monkeypatch):
    fit_z = torch.tensor(
        [
            [[[1.0, 0.0]], [[1.0, 0.0]]],
            [[[1.1, 0.0]], [[1.0, 0.0]]],
            [[[0.0, 1.0]], [[0.0, 1.0]]],
            [[[0.0, 1.1]], [[0.0, 1.0]]],
        ]
    )
    val_z = fit_z[:2]
    test_z = torch.tensor(
        [
            [[[1.0, 0.0]], [[1.0, 0.0]]],
            [[[0.1, 0.9]], [[0.2, 0.8]]],
            [[[0.0, 1.0]], [[0.0, 1.0]]],
            [[[0.8, 0.2]], [[0.7, 0.3]]],
        ]
    )
    fit_outputs = _make_outputs(fit_z, torch.tensor([0, 0, 1, 1]), torch.zeros(4, 2))
    val_outputs = _make_outputs(val_z, torch.tensor([0, 0]), torch.zeros(2, 2))
    test_logits = torch.tensor([[3.0, 1.0], [1.0, 2.5], [1.0, 3.0], [1.0, 2.5]])
    test_outputs = _make_outputs(test_z, torch.tensor([0, 0, 1, 1]), test_logits)
    queued = [fit_outputs, val_outputs, test_outputs]
    monkeypatch.setattr(
        "flow_circuits.evaluation.interpretability_validation.collect_interpretability_outputs",
        lambda *args, **kwargs: queued.pop(0),
    )

    result = run_probe_error_analysis_experiment(
        SimpleNamespace(),
        fit_loader=object(),
        val_loader=object(),
        test_loader=object(),
        device=torch.device("cpu"),
        checkpoint_tag="phase_b",
        fit_max_images=4,
        val_max_images=2,
        test_max_images=4,
        exemplar_count=2,
    )

    assert result["summary"]["n_wrong_examples"] == 1
    assert result["summary"]["global_margin_drop"] > 0.0
    assert result["top_drop_nodes"]
