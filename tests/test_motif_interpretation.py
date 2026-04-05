from __future__ import annotations

import torch

from flow_circuits.evaluation.motif_interpretation import (
    run_motif_borderline_member_experiment,
    run_motif_semantic_report_experiment,
    run_motif_spatial_footprint_experiment,
)


def _toy_outputs() -> dict[str, torch.Tensor]:
    return {
        "z": torch.tensor(
            [
                [[[1.0, 0.0]], [[1.0, 0.0]]],
                [[[0.9, 0.1]], [[0.95, 0.05]]],
                [[[0.2, 0.8]], [[0.2, 0.8]]],
                [[[0.0, 1.0]], [[0.1, 0.9]]],
            ],
            dtype=torch.float32,
        ),
        "future_descriptors": torch.zeros(4, 2, 1, 2),
        "images": torch.zeros(4, 3, 32, 32),
        "logits": torch.zeros(4, 2),
        "labels": torch.tensor([0, 0, 1, 1]),
        "indices": torch.tensor([10, 11, 12, 13]),
    }


def _toy_motif_artifact() -> dict:
    return {
        "metadata": {
            "grid_size": 1,
        },
        "motifs": [
            {
                "id": 0,
                "active_nodes": [[0, 0], [1, 0]],
                "representative_node": [0, 0],
                "layer_support": [0, 1],
                "image_set": [10, 11],
                "member_row_indices": [0, 1],
                "centroids": {"0:0": [1.0, 0.0], "1:0": [1.0, 0.0]},
                "thresholds": {"0:0": 0.0, "1:0": 0.0},
                "stability": {"mean_cluster_stability": 0.95, "n_node_clusters": 2},
                "purity": {"dominant_class": 0, "fraction": 1.0},
            },
            {
                "id": 1,
                "active_nodes": [[0, 0], [1, 0]],
                "representative_node": [1, 0],
                "layer_support": [0, 1],
                "image_set": [12, 13],
                "member_row_indices": [2, 3],
                "centroids": {"0:0": [0.0, 1.0], "1:0": [0.0, 1.0]},
                "thresholds": {"0:0": 0.0, "1:0": 0.0},
                "stability": {"mean_cluster_stability": 0.80, "n_node_clusters": 2},
                "purity": {"dominant_class": 1, "fraction": 1.0},
            },
        ],
    }


def test_motif_semantic_report_generation_returns_ranked_cards_and_exemplars():
    outputs = _toy_outputs()
    motif_artifact = _toy_motif_artifact()

    result = run_motif_semantic_report_experiment(
        motif_artifact,
        outputs,
        checkpoint_tag="joint",
        topk=2,
    )

    assert result["summary"]["n_ranked_motifs"] == 2
    assert [card["motif_id"] for card in result["motif_cards"]] == [0, 1]
    assert result["motif_cards"][0]["dominant_class_name"] == "airplane"
    assert "class_specific" in result["motif_cards"][0]["heuristic_tags"]
    assert result["motif_cards"][0]["top_exemplar_image_indices"]
    assert result["exemplar_sets"][0]["top_exemplars"]["dataset_indices"]


def test_motif_spatial_footprint_generation_returns_overlay_and_crop_specs():
    outputs = _toy_outputs()
    motif_artifact = _toy_motif_artifact()
    semantic_report = run_motif_semantic_report_experiment(
        motif_artifact,
        outputs,
        checkpoint_tag="joint",
        topk=1,
    )

    result = run_motif_spatial_footprint_experiment(
        motif_artifact,
        outputs,
        checkpoint_tag="joint",
        semantic_report=semantic_report,
        image_size=32,
    )

    assert result["summary"]["n_motifs"] == 1
    assert result["overlay_specs"][0]["motif_id"] == 0
    assert result["overlay_specs"][0]["representative_box"]["cell_idx"] == 0
    assert result["crop_specs"][0]["per_layer_crops"][0]["layer_idx"] == 0


def test_motif_borderline_member_report_returns_near_misses():
    outputs = _toy_outputs()
    motif_artifact = _toy_motif_artifact()

    result = run_motif_borderline_member_experiment(
        motif_artifact,
        outputs,
        checkpoint_tag="joint",
        topk=1,
        borderline_count=1,
        near_miss_count=1,
    )

    assert result["motif_rows"][0]["motif_id"] == 0
    assert len(result["motif_rows"][0]["borderline_members"]["dataset_indices"]) == 1
    assert len(result["motif_rows"][0]["near_misses"]["dataset_indices"]) == 1
