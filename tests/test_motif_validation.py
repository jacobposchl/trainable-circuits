from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch

from flow_circuits.evaluation.motif_validation import (
    discover_motif_families,
    run_motif_cooccurrence_experiment,
    run_motif_gallery_experiment,
    run_motif_intervention_experiment,
    run_motif_persistence_experiment,
    run_motif_phase_match_experiment,
    run_motif_predictiveness_experiment,
    run_motif_topology_experiment,
    run_motif_transfer_stability_experiment,
)


def test_discover_motif_families_retains_multilayer_family_without_depth_path(monkeypatch):
    z = torch.zeros(8, 3, 2, 4)
    z[:4, 0, 0, 0] = 1.0
    z[4:, 0, 0, 1] = 1.0
    z[:4, 2, 1, 0] = 1.0
    z[4:, 2, 1, 1] = 1.0
    z[:, 1, :, 2] = 1.0
    z = torch.nn.functional.normalize(z, dim=-1)
    outputs = {
        "z": z,
        "local_features": [torch.zeros(8, 2, 5) for _ in range(3)],
        "future_descriptors": z.clone(),
        "labels": torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
        "indices": torch.arange(8),
    }
    monkeypatch.setattr(
        "flow_circuits.evaluation.motif_validation.collect_probe_outputs",
        lambda *args, **kwargs: outputs,
    )
    components = SimpleNamespace(
        config={
            "data": {"seed": 0},
            "tokenization": {"grid_size": 2},
            "discovery": {
                "min_cluster_fraction": 0.2,
                "max_cluster_fraction": 0.8,
                "min_cluster_size": 2,
                "stability_threshold": 0.0,
            },
        }
    )

    artifact = discover_motif_families(
        components,
        loader=object(),
        device=torch.device("cpu"),
        checkpoint_tag="phase_b",
        max_images=8,
        nodes_per_layer=1,
        bootstrap_iterations=1,
        merge_threshold=0.5,
        node_threshold=0.5,
        node_panel=[[0, 0], [2, 1]],
    )

    assert artifact["summary"]["n_motifs"] >= 1
    assert any(motif["active_nodes"] == [[0, 0], [2, 1]] for motif in artifact["motifs"])
    motif = next(motif for motif in artifact["motifs"] if motif["active_nodes"] == [[0, 0], [2, 1]])
    assert motif["representative_node"] in ([0, 0], [2, 1])
    assert motif["centroids"]
    assert motif["thresholds"]


def test_discover_motif_families_can_scan_all_nodes_without_q_panel(monkeypatch):
    z = torch.nn.functional.normalize(torch.randn(6, 2, 2, 4), dim=-1)
    outputs = {
        "z": z,
        "local_features": [torch.zeros(6, 2, 5) for _ in range(2)],
        "future_descriptors": z.clone(),
        "labels": torch.tensor([0, 0, 1, 1, 2, 2]),
        "indices": torch.arange(6),
    }
    monkeypatch.setattr(
        "flow_circuits.evaluation.motif_validation.collect_probe_outputs",
        lambda *args, **kwargs: outputs,
    )
    captured = {}

    def fake_discover(descriptor_grid, dataset_indices, **kwargs):
        captured["node_subset"] = kwargs["node_subset"]
        return []

    monkeypatch.setattr("flow_circuits.evaluation.motif_validation.discover_node_clusters", fake_discover)
    components = SimpleNamespace(
        config={
            "data": {"seed": 0},
            "tokenization": {"grid_size": 2},
            "discovery": {
                "min_cluster_fraction": 0.2,
                "max_cluster_fraction": 0.8,
                "min_cluster_size": 2,
                "stability_threshold": 0.0,
            },
        }
    )

    artifact = discover_motif_families(
        components,
        loader=object(),
        device=torch.device("cpu"),
        checkpoint_tag="joint",
        max_images=6,
        nodes_per_layer=1,
        bootstrap_iterations=1,
        use_all_nodes=True,
    )

    assert artifact["metadata"]["selected_node_panel_strategy"] == "all_nodes"
    assert sorted(map(tuple, captured["node_subset"])) == [(0, 0), (0, 1), (1, 0), (1, 1)]


def test_motif_gallery_ranking_is_deterministic(monkeypatch):
    z = torch.randn(6, 2, 2, 4)
    z = torch.nn.functional.normalize(z, dim=-1)
    outputs = {
        "z": z,
        "local_features": [torch.zeros(6, 2, 5) for _ in range(2)],
        "future_descriptors": z.clone(),
        "labels": torch.tensor([0, 0, 1, 1, 2, 2]),
        "indices": torch.arange(6),
    }
    monkeypatch.setattr(
        "flow_circuits.evaluation.motif_validation.collect_probe_outputs",
        lambda *args, **kwargs: outputs,
    )
    motif_artifact = {
        "motifs": [
            {
                "id": 0,
                "image_set": [0, 1],
                "member_row_indices": [0, 1],
                "representative_node": [0, 0],
                "active_nodes": [[0, 0], [1, 0]],
                "layer_support": [0, 1],
                "centroids": {"0:0": z[0, 0, 0].tolist(), "1:0": z[0, 1, 0].tolist()},
                "thresholds": {"0:0": 0.0, "1:0": 0.0},
                "stability": {"mean_cluster_stability": 0.8, "n_node_clusters": 2},
                "purity": {"dominant_class": 0, "fraction": 1.0},
            },
            {
                "id": 1,
                "image_set": [2, 3, 4],
                "member_row_indices": [2, 3, 4],
                "representative_node": [0, 1],
                "active_nodes": [[0, 1], [1, 1]],
                "layer_support": [0, 1],
                "centroids": {"0:1": z[2, 0, 1].tolist(), "1:1": z[2, 1, 1].tolist()},
                "thresholds": {"0:1": 0.0, "1:1": 0.0},
                "stability": {"mean_cluster_stability": 0.9, "n_node_clusters": 2},
                "purity": {"dominant_class": 1, "fraction": 2 / 3},
            },
        ]
    }

    result = run_motif_gallery_experiment(
        SimpleNamespace(),
        loader=object(),
        device=torch.device("cpu"),
        checkpoint_tag="phase_c",
        motif_artifact=motif_artifact,
        max_images=6,
        topk=2,
    )

    assert [row["motif_id"] for row in result["motif_rows"]] == [1, 0]


def test_motif_persistence_metrics_are_computed():
    result = run_motif_persistence_experiment(
        {
            "motifs": [
                {
                    "id": 0,
                    "active_nodes": [[0, 0], [1, 0], [2, 0]],
                    "layer_support": [0, 1, 2],
                }
            ]
        },
        checkpoint_tag="phase_b",
    )

    assert result["summary"]["mean_depth_span"] == 3.0
    assert result["summary"]["fraction_spanning_ge_3_layers"] == 1.0
    assert result["motif_rows"][0]["dominant_cells_by_layer"]["1"] == [0]


def test_motif_predictiveness_computes_class_lift(monkeypatch):
    z = torch.zeros(4, 1, 1, 3)
    z[:2, 0, 0, 0] = 1.0
    z[2:, 0, 0, 1] = 1.0
    z = torch.nn.functional.normalize(z, dim=-1)
    logits = torch.tensor(
        [
            [0.1, 2.5, 0.0],
            [0.2, 2.4, 0.0],
            [1.2, 0.1, 0.0],
            [1.1, 0.2, 0.0],
        ]
    )
    outputs = {
        "z": z,
        "local_features": [torch.zeros(4, 1, 5)],
        "flow_targets": torch.zeros(4, 1, 1, 3),
        "future_descriptors": z.clone(),
        "predicted_next": torch.zeros(4, 0, 1, 3),
        "reconstructed_current": torch.zeros(4, 1, 1, 3),
        "images": torch.zeros(4, 3, 32, 32),
        "logits": logits,
        "labels": torch.tensor([1, 1, 0, 0]),
        "indices": torch.arange(4),
    }
    monkeypatch.setattr(
        "flow_circuits.evaluation.motif_validation.collect_model_outputs",
        lambda *args, **kwargs: outputs,
    )
    motif_artifact = {
        "motifs": [
            {
                "id": 0,
                "image_set": [0, 1],
                "representative_node": [0, 0],
                "active_nodes": [[0, 0]],
                "layer_support": [0],
                "centroids": {"0:0": [1.0, 0.0, 0.0]},
                "thresholds": {"0:0": 0.5},
                "stability": {"mean_cluster_stability": 0.9, "n_node_clusters": 2},
                "purity": {"dominant_class": 1, "fraction": 1.0},
            }
        ]
    }

    result = run_motif_predictiveness_experiment(
        SimpleNamespace(),
        loader=object(),
        device=torch.device("cpu"),
        checkpoint_tag="phase_b",
        motif_artifact=motif_artifact,
        max_images=4,
        topk=1,
    )

    row = result["motif_rows"][0]
    assert row["precision"] == 1.0
    assert row["recall"] == 1.0
    assert row["lift_over_base_rate"] > 1.0
    assert row["member_margin_lift"] > 0.0


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
    corrected_p_member_vs_nonmember: float = 0.02
    corrected_p_member_vs_random_node: float = 0.02
    corrected_p_member_vs_random_cell: float = 0.02
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


def test_motif_interventions_reuse_motif_artifacts(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "flow_circuits.evaluation.motif_validation.collect_model_outputs",
        lambda *args, **kwargs: {
            "z": torch.zeros(4, 1, 1, 3),
            "local_features": [torch.zeros(4, 1, 5)],
            "flow_targets": torch.zeros(4, 1, 1, 3),
            "future_descriptors": torch.zeros(4, 1, 1, 3),
            "predicted_next": torch.zeros(4, 0, 1, 3),
            "reconstructed_current": torch.zeros(4, 1, 1, 3),
            "images": torch.zeros(4, 3, 32, 32),
            "logits": torch.zeros(4, 3),
            "labels": torch.zeros(4, dtype=torch.long),
            "indices": torch.arange(4),
        },
    )

    def fake_run_circuit_interventions(*args, **kwargs):
        captured["artifact"] = args[1]
        captured["descriptor_key"] = kwargs["descriptor_key"]
        return [_DummyInterventionResult(circuit_id=int(circuit["id"])) for circuit in captured["artifact"]["circuits"]]

    monkeypatch.setattr("flow_circuits.evaluation.motif_validation.run_circuit_interventions", fake_run_circuit_interventions)
    motif_artifact = {
        "metadata": {"grid_size": 2, "n_layers": 2, "n_cells": 4},
        "motifs": [
            {"id": 0, "image_set": list(range(30)), "stability": {"mean_cluster_stability": 0.8, "n_node_clusters": 2}},
            {"id": 1, "image_set": list(range(20)), "stability": {"mean_cluster_stability": 0.95, "n_node_clusters": 5}},
            {"id": 2, "image_set": list(range(40)), "stability": {"mean_cluster_stability": 0.8, "n_node_clusters": 3}},
        ],
    }
    components = SimpleNamespace(config={"interventions": {"n_jobs": 1}})

    result = run_motif_intervention_experiment(
        components,
        loader=object(),
        device=torch.device("cpu"),
        checkpoint_tag="phase_c",
        motif_artifact=motif_artifact,
        max_images=4,
        topk=2,
    )

    assert [motif["id"] for motif in captured["artifact"]["circuits"]] == [2, 0]
    assert captured["descriptor_key"] == "z"
    assert result["summary"]["member_specific_count"] == 2


def test_motif_cooccurrence_summary_is_deterministic():
    motif_artifact = {
        "motifs": [
            {"id": 0, "image_set": [0, 1, 2], "stability": {"mean_cluster_stability": 0.9, "n_node_clusters": 2}},
            {"id": 1, "image_set": [1, 2, 3], "stability": {"mean_cluster_stability": 0.8, "n_node_clusters": 2}},
            {"id": 2, "image_set": [4, 5], "stability": {"mean_cluster_stability": 0.7, "n_node_clusters": 2}},
        ]
    }

    result = run_motif_cooccurrence_experiment(motif_artifact, checkpoint_tag="phase_b", overlap_threshold=0.25)

    assert result["summary"]["n_edges"] == 1
    assert result["strongest_pairs"][0]["left_motif_id"] == 0
    assert result["strongest_pairs"][0]["right_motif_id"] == 1


def test_motif_phase_matching_uses_representative_node_tiebreak():
    phase_b_artifact = {
        "motifs": [
            {
                "id": 0,
                "image_set": [0, 1, 2],
                "representative_node": [0, 0],
                "active_nodes": [[0, 0], [2, 1]],
                "centroids": {"0:0": [1.0, 0.0]},
                "stability": {"mean_cluster_stability": 0.9, "n_node_clusters": 2},
                "layer_support": [0, 2],
            }
        ]
    }
    phase_c_artifact = {
        "motifs": [
            {
                "id": 0,
                "image_set": [0, 1, 3],
                "representative_node": [1, 1],
                "active_nodes": [[0, 0], [2, 1]],
                "centroids": {"1:1": [1.0, 0.0]},
                "stability": {"mean_cluster_stability": 0.8, "n_node_clusters": 2},
                "layer_support": [0, 2],
            },
            {
                "id": 1,
                "image_set": [0, 1, 3],
                "representative_node": [0, 0],
                "active_nodes": [[0, 0], [2, 1]],
                "centroids": {"0:0": [1.0, 0.0]},
                "stability": {"mean_cluster_stability": 0.7, "n_node_clusters": 2},
                "layer_support": [0, 2],
            },
        ]
    }

    result = run_motif_phase_match_experiment(phase_b_artifact, phase_c_artifact)

    assert result["matched_pairs"][0]["phase_c_motif_id"] == 1


def test_motif_topology_distinguishes_depth_like_and_spatial():
    motif_artifact = {
        "metadata": {"grid_size": 2},
        "motifs": [
            {
                "id": 0,
                "active_nodes": [[0, 0], [1, 0], [2, 0]],
                "stability": {"mean_cluster_stability": 0.9, "n_node_clusters": 3},
                "layer_support": [0, 1, 2],
                "image_set": [0, 1],
            },
            {
                "id": 1,
                "active_nodes": [[0, 0], [0, 1]],
                "stability": {"mean_cluster_stability": 0.8, "n_node_clusters": 2},
                "layer_support": [0],
                "image_set": [2, 3],
            },
        ]
    }

    result = run_motif_topology_experiment(motif_artifact, checkpoint_tag="phase_b")

    assert result["motif_rows"][0]["topology_type"] == "depth_like"
    assert result["motif_rows"][1]["topology_type"] == "spatial"


def test_motif_transfer_stability_is_deterministic(monkeypatch):
    outputs = {
        "z": torch.randn(6, 2, 2, 4),
        "local_features": [torch.zeros(6, 2, 5) for _ in range(2)],
        "future_descriptors": torch.randn(6, 2, 2, 4),
        "labels": torch.tensor([0, 0, 1, 1, 2, 2]),
        "indices": torch.arange(6),
    }
    outputs["z"] = torch.nn.functional.normalize(outputs["z"], dim=-1)
    outputs["future_descriptors"] = torch.nn.functional.normalize(outputs["future_descriptors"], dim=-1)
    monkeypatch.setattr(
        "flow_circuits.evaluation.motif_validation.collect_probe_outputs",
        lambda *args, **kwargs: outputs,
    )
    monkeypatch.setattr(
        "flow_circuits.evaluation.motif_validation._select_q_dispersion_node_panel",
        lambda future_descriptors, *, nodes_per_layer, seed, tracker: [[0, 0], [1, 1]],
    )

    def fake_discover(outputs_subset, *, checkpoint_tag, **kwargs):
        if checkpoint_tag.endswith("_left"):
            motifs = [
                {"id": 0, "image_set": [0, 1, 2], "representative_node": [0, 0], "active_nodes": [[0, 0], [1, 1]], "centroids": {"0:0": [1.0, 0.0]}, "stability": {"mean_cluster_stability": 0.9, "n_node_clusters": 2}, "layer_support": [0, 1]}
            ]
        else:
            motifs = [
                {"id": 0, "image_set": [0, 1, 3], "representative_node": [0, 0], "active_nodes": [[0, 0], [1, 1]], "centroids": {"0:0": [1.0, 0.0]}, "stability": {"mean_cluster_stability": 0.9, "n_node_clusters": 2}, "layer_support": [0, 1]}
            ]
        return {"motifs": motifs}

    monkeypatch.setattr("flow_circuits.evaluation.motif_validation._discover_motif_families_from_outputs", fake_discover)
    components = SimpleNamespace(config={"data": {"seed": 0}, "tokenization": {"grid_size": 2}, "discovery": {"min_cluster_fraction": 0.2, "max_cluster_fraction": 0.8, "min_cluster_size": 2, "stability_threshold": 0.0}})

    result = run_motif_transfer_stability_experiment(
        components,
        loader=object(),
        device=torch.device("cpu"),
        checkpoint_tag="phase_b",
        max_images=6,
        nodes_per_layer=2,
        bootstrap_iterations=1,
    )

    assert result["summary"]["matched_motif_rate"] == 1.0
    assert result["summary"]["mean_image_set_stability"] > 0.0
