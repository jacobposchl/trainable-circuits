from __future__ import annotations

from types import SimpleNamespace

import torch

from flow_circuits.data import apply_cifar10_corruption
from flow_circuits.evaluation.corruption_selective_correction import (
    run_corruption_sweep_experiment,
    run_top_node_subset_sweep_experiment,
)


def test_apply_cifar10_corruption_preserves_shape_and_range():
    image = torch.full((3, 32, 32), 0.5)
    for corruption_name in ("gaussian_noise", "gaussian_blur", "contrast", "brightness", "pixelate", "occlusion"):
        corrupted = apply_cifar10_corruption(
            image,
            corruption_name=corruption_name,
            severity=3,
            seed=123,
        )
        assert corrupted.shape == image.shape
        assert float(corrupted.min()) >= 0.0
        assert float(corrupted.max()) <= 1.0


def test_corruption_sweep_aggregates_rows(monkeypatch):
    components = SimpleNamespace(
        config={"tokenization": {"grid_size": 4}},
        observer=SimpleNamespace(layer_channels=[64] * 8),
    )
    monkeypatch.setattr(
        "flow_circuits.evaluation.corruption_selective_correction.build_cifar10_corruption_splits",
        lambda **kwargs: {"fit": object(), "val": object(), "test": object()},
    )

    def _fake_selective(*args, **kwargs):
        severity = 1 if "severity_1" in str(kwargs["output_path"]) else 3
        return {
            "summary": {
                "backbone_overall_accuracy": 0.70,
                "full_z_hybrid_overall_accuracy": 0.72 + (0.01 * severity),
                "top_node_hybrid_overall_accuracy": 0.71 + (0.005 * severity),
                "backbone_trigger_accuracy": 0.60,
                "full_z_hybrid_trigger_accuracy": 0.62,
                "top_node_hybrid_trigger_accuracy": 0.61,
                "trigger_coverage": 0.10,
                "full_z_net_gain": severity,
                "top_node_net_gain": severity - 1,
            },
            "full_z_hybrid": {"gain_per_100_triggered": 5.0 + severity},
            "top_node_hybrid": {"gain_per_100_triggered": 2.0 + severity},
        }

    monkeypatch.setattr(
        "flow_circuits.evaluation.corruption_selective_correction.run_selective_hybrid_correction_experiment",
        _fake_selective,
    )

    result = run_corruption_sweep_experiment(
        components,
        device=torch.device("cpu"),
        checkpoint_tag="phase_c",
        data_dir="unused",
        batch_size=16,
        corruption_names=["gaussian_noise", "contrast"],
        severities=[1, 3],
        fit_max_images=32,
        val_max_images=16,
        test_max_images=16,
        top_pairs=3,
        top_node_fraction=0.05,
        top_node_min_k=3,
        top_node_max_k=12,
        trigger_mode="hard_pair_top2_and_low_margin",
    )

    assert len(result["rows"]) == 4
    assert result["config"]["resolved_top_k_nodes"] == 7
    assert result["summary"]["best_full_z_gain_corruption"]


def test_top_node_subset_sweep_resolves_fractional_subset(monkeypatch):
    components = SimpleNamespace(
        config={"tokenization": {"grid_size": 4}},
        observer=SimpleNamespace(layer_channels=[64] * 8),
    )
    monkeypatch.setattr(
        "flow_circuits.evaluation.corruption_selective_correction.build_cifar10_corruption_splits",
        lambda **kwargs: {"fit": object(), "val": object(), "test": object()},
    )

    def _fake_selective(*args, **kwargs):
        top_k = int(kwargs["top_k_nodes"])
        return {
            "summary": {
                "backbone_overall_accuracy": 0.70,
                "full_z_hybrid_overall_accuracy": 0.74,
                "top_node_hybrid_overall_accuracy": 0.70 + min(top_k, 8) * 0.003,
                "backbone_trigger_accuracy": 0.60,
                "full_z_hybrid_trigger_accuracy": 0.64,
                "top_node_hybrid_trigger_accuracy": 0.61,
                "trigger_coverage": 0.10,
                "top_node_net_gain": max(0, top_k - 3),
            },
            "top_node_hybrid": {"gain_per_100_triggered": float(top_k)},
        }

    monkeypatch.setattr(
        "flow_circuits.evaluation.corruption_selective_correction.run_selective_hybrid_correction_experiment",
        _fake_selective,
    )

    result = run_top_node_subset_sweep_experiment(
        components,
        device=torch.device("cpu"),
        checkpoint_tag="phase_c",
        data_dir="unused",
        batch_size=16,
        corruption_name="gaussian_noise",
        severity=3,
        top_node_fractions=[0.02, 0.05, 0.10],
        fit_max_images=32,
        val_max_images=16,
        test_max_images=16,
        top_pairs=3,
        top_node_min_k=3,
        top_node_max_k=12,
        trigger_mode="hard_pair_top2_and_low_margin",
    )

    assert [row["top_k_nodes"] for row in result["rows"]] == [3, 7, 12]
    assert result["summary"]["best_top_k_nodes"] in {7, 12}
