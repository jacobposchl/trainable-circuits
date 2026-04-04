from __future__ import annotations

from types import SimpleNamespace

import torch

from flow_circuits.evaluation.motif_utility import run_motif_clean_utility_experiment


def test_motif_clean_utility_uses_motif_features_and_reports_summary(monkeypatch):
    fit_outputs = {
        "z": torch.tensor(
            [
                [[[1.0, 0.0]], [[1.0, 0.0]]],
                [[[0.9, 0.1]], [[0.9, 0.1]]],
                [[[0.0, 1.0]], [[0.0, 1.0]]],
                [[[0.1, 0.9]], [[0.1, 0.9]]],
            ]
        ),
        "future_descriptors": torch.zeros(4, 2, 1, 2),
        "images": torch.zeros(4, 3, 32, 32),
        "logits": torch.tensor([[2.0, 1.0], [1.9, 1.1], [1.0, 2.0], [1.1, 1.9]]),
        "labels": torch.tensor([0, 0, 1, 1]),
        "indices": torch.arange(4),
    }
    val_outputs = {
        "z": fit_outputs["z"].clone(),
        "future_descriptors": fit_outputs["future_descriptors"].clone(),
        "images": fit_outputs["images"].clone(),
        "logits": torch.tensor([[1.1, 1.2], [1.0, 1.1], [1.2, 1.1], [1.1, 1.0]]),
        "labels": torch.tensor([0, 0, 1, 1]),
        "indices": torch.arange(4),
    }
    test_outputs = {
        "z": fit_outputs["z"].clone(),
        "future_descriptors": fit_outputs["future_descriptors"].clone(),
        "images": fit_outputs["images"].clone(),
        "logits": torch.tensor([[1.1, 1.2], [1.0, 1.1], [1.2, 1.1], [1.1, 1.0]]),
        "labels": torch.tensor([0, 0, 1, 1]),
        "indices": torch.arange(4),
    }
    monkeypatch.setattr(
        "flow_circuits.evaluation.motif_utility._collect_probe_splits",
        lambda *args, **kwargs: (fit_outputs, val_outputs, test_outputs),
    )

    motif_artifact = {
        "motifs": [
            {
                "id": 0,
                "active_nodes": [[0, 0], [1, 0]],
                "representative_node": [0, 0],
                "layer_support": [0, 1],
                "image_set": [0, 1],
                "centroids": {"0:0": [1.0, 0.0], "1:0": [1.0, 0.0]},
                "thresholds": {"0:0": 0.0, "1:0": 0.0},
                "stability": {"mean_cluster_stability": 0.9, "n_node_clusters": 2},
            },
            {
                "id": 1,
                "active_nodes": [[0, 0], [1, 0]],
                "representative_node": [0, 0],
                "layer_support": [0, 1],
                "image_set": [2, 3],
                "centroids": {"0:0": [0.0, 1.0], "1:0": [0.0, 1.0]},
                "thresholds": {"0:0": 0.0, "1:0": 0.0},
                "stability": {"mean_cluster_stability": 0.8, "n_node_clusters": 2},
            },
        ]
    }

    result = run_motif_clean_utility_experiment(
        SimpleNamespace(),
        motif_artifact,
        fit_loader=object(),
        val_loader=object(),
        test_loader=object(),
        device=torch.device("cpu"),
        checkpoint_tag="joint",
        fit_max_images=4,
        val_max_images=4,
        test_max_images=4,
        top_pairs=1,
        trigger_mode="hard_pair_top2_and_low_margin",
        margin_quantile=0.5,
        top_motif_fraction=0.5,
        min_top_motifs=1,
        max_top_motifs=2,
    )

    assert result["summary"]["n_pairs"] == 1
    assert "backbone_overall_accuracy" in result["summary"]
    assert "full_motif_overall_accuracy" in result["summary"]
    assert "top_motif_overall_accuracy" in result["summary"]
    assert result["selected_top_motif_ids"]
