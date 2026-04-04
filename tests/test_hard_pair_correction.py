from __future__ import annotations

from types import SimpleNamespace

import torch

from flow_circuits.evaluation.hard_pair_correction import (
    run_hard_pair_hybrid_correction_experiment,
    run_hard_pair_probe_benchmark_experiment,
    run_multiclass_z_probe_audit_experiment,
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


def test_multiclass_probe_audit_reports_extra_metrics(monkeypatch):
    fit_z = torch.tensor(
        [
            [[[2.0, 0.0]], [[2.0, 0.0]]],
            [[[2.1, 0.0]], [[1.9, 0.0]]],
            [[[0.0, 2.0]], [[0.0, 2.0]]],
            [[[0.1, 1.9]], [[0.0, 2.1]]],
        ]
    )
    val_z = torch.tensor([[[[2.0, 0.0]], [[2.0, 0.0]]], [[[0.0, 2.0]], [[0.0, 2.0]]]])
    test_z = val_z.clone()
    queued = [
        _make_outputs(fit_z, torch.tensor([0, 0, 1, 1]), torch.zeros(4, 2)),
        _make_outputs(val_z, torch.tensor([0, 1]), torch.zeros(2, 2)),
        _make_outputs(test_z, torch.tensor([0, 1]), torch.zeros(2, 2)),
    ]
    monkeypatch.setattr(
        "flow_circuits.evaluation.interpretability_validation.collect_interpretability_outputs",
        lambda *args, **kwargs: queued.pop(0),
    )

    result = run_multiclass_z_probe_audit_experiment(
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
    assert "global_log_loss" in result["summary"]
    assert "global_top2_accuracy" in result["summary"]
    assert result["top_nodes_by_accuracy"]


def test_hard_pair_benchmark_selects_pairs_from_validation_confusion(monkeypatch):
    fit_z = torch.tensor(
        [
            [[[2.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]]],
            [[[2.1, 0.0, 0.0]], [[1.9, 0.0, 0.0]]],
            [[[0.0, 2.0, 0.0]], [[0.0, 2.0, 0.0]]],
            [[[0.0, 2.1, 0.0]], [[0.0, 1.9, 0.0]]],
            [[[0.0, 0.0, 2.0]], [[0.0, 0.0, 2.0]]],
            [[[0.0, 0.0, 2.1]], [[0.0, 0.0, 1.9]]],
        ]
    )
    fit_outputs = _make_outputs(fit_z, torch.tensor([0, 0, 1, 1, 2, 2]), torch.zeros(6, 3))

    val_z = fit_z.clone()
    val_logits = torch.tensor(
        [
            [1.0, 3.0, 0.5],  # 0 -> 1 confusion
            [1.0, 3.1, 0.4],  # 0 -> 1 confusion
            [0.2, 3.0, 0.5],
            [0.2, 2.9, 0.4],
            [0.3, 0.5, 3.0],
            [0.3, 0.4, 3.0],
        ]
    )
    val_outputs = _make_outputs(val_z, torch.tensor([0, 0, 1, 1, 2, 2]), val_logits)

    test_z = fit_z.clone()
    test_logits = torch.tensor(
        [
            [1.0, 0.2, 3.0],  # 0 -> 2 confusion, should not be selected
            [1.0, 0.2, 3.1],  # 0 -> 2 confusion
            [0.2, 3.0, 0.5],
            [0.2, 2.9, 0.4],
            [0.3, 0.5, 3.0],
            [0.3, 0.4, 3.0],
        ]
    )
    test_outputs = _make_outputs(test_z, torch.tensor([0, 0, 1, 1, 2, 2]), test_logits)
    queued = [fit_outputs, val_outputs, test_outputs]
    monkeypatch.setattr(
        "flow_circuits.evaluation.interpretability_validation.collect_interpretability_outputs",
        lambda *args, **kwargs: queued.pop(0),
    )

    result = run_hard_pair_probe_benchmark_experiment(
        SimpleNamespace(),
        fit_loader=object(),
        val_loader=object(),
        test_loader=object(),
        device=torch.device("cpu"),
        checkpoint_tag="phase_c",
        fit_max_images=6,
        val_max_images=6,
        test_max_images=6,
        top_pairs=1,
        top_k_nodes=1,
    )

    row = result["pair_rows"][0]
    assert (row["left_label"], row["right_label"]) == (0, 1)
    assert row["selected_top_nodes"]
    assert "full_z_probe_auroc" in row
    assert "top_node_probe_brier" in row


def test_hard_pair_benchmark_ranks_top_nodes_deterministically(monkeypatch):
    fit_z = torch.tensor(
        [
            [[[3.0, 0.0]], [[0.2, 0.0]]],
            [[[2.8, 0.0]], [[0.1, 0.0]]],
            [[[0.0, 3.0]], [[0.0, 0.2]]],
            [[[0.0, 2.8]], [[0.0, 0.1]]],
        ]
    )
    fit_outputs = _make_outputs(fit_z, torch.tensor([0, 0, 1, 1]), torch.zeros(4, 2))
    val_outputs = _make_outputs(
        fit_z.clone(),
        torch.tensor([0, 0, 1, 1]),
        torch.tensor([[1.0, 3.0], [1.0, 3.0], [0.2, 3.0], [0.2, 2.9]]),
    )
    test_outputs = _make_outputs(fit_z.clone(), torch.tensor([0, 0, 1, 1]), torch.zeros(4, 2))
    queued = [fit_outputs, val_outputs, test_outputs]
    monkeypatch.setattr(
        "flow_circuits.evaluation.interpretability_validation.collect_interpretability_outputs",
        lambda *args, **kwargs: queued.pop(0),
    )

    result = run_hard_pair_probe_benchmark_experiment(
        SimpleNamespace(),
        fit_loader=object(),
        val_loader=object(),
        test_loader=object(),
        device=torch.device("cpu"),
        checkpoint_tag="phase_b",
        fit_max_images=4,
        val_max_images=4,
        test_max_images=4,
        top_pairs=1,
        top_k_nodes=2,
    )

    top_nodes = result["pair_rows"][0]["selected_top_nodes"]
    assert [(row["layer_idx"], row["cell_idx"]) for row in top_nodes] == [(0, 0), (1, 0)]


def test_hard_pair_hybrid_only_overrides_trigger_pairs(monkeypatch):
    fit_z = torch.tensor(
        [
            [[[3.0, 0.0]], [[0.0, 0.0]]],
            [[[2.9, 0.0]], [[0.0, 0.0]]],
            [[[0.0, 3.0]], [[0.0, 0.0]]],
            [[[0.0, 2.9]], [[0.0, 0.0]]],
        ]
    )
    fit_outputs = _make_outputs(fit_z, torch.tensor([0, 0, 1, 1]), torch.zeros(4, 3))

    val_z = fit_z.clone()
    val_logits = torch.tensor(
        [
            [1.0, 3.0, 0.5],  # class 0 confused with 1
            [1.0, 3.0, 0.4],
            [0.2, 3.0, 0.5],
            [0.2, 2.9, 0.4],
        ]
    )
    val_outputs = _make_outputs(val_z, torch.tensor([0, 0, 1, 1]), val_logits)

    test_z = torch.tensor(
        [
            [[[3.0, 0.0]], [[0.0, 0.0]]],  # trigger example, should get corrected to 0
            [[[0.0, 3.0]], [[0.0, 0.0]]],  # trigger example, stays 1
            [[[0.0, 0.0]], [[3.0, 0.0]]],  # non-trigger wrong example
            [[[0.0, 0.0]], [[0.0, 3.0]]],  # non-trigger right example
        ]
    )
    test_logits = torch.tensor(
        [
            [2.8, 3.0, 0.1],  # top-2 is 1 vs 0
            [0.1, 3.0, 2.5],  # top-2 is 1 vs 2, non selected if labels differ?
            [0.1, 2.5, 3.0],  # top-2 is 2 vs 1, non-trigger
            [0.1, 0.2, 3.0],  # non-trigger
        ]
    )
    test_outputs = _make_outputs(test_z, torch.tensor([0, 1, 2, 2]), test_logits)
    queued = [fit_outputs, val_outputs, test_outputs]
    monkeypatch.setattr(
        "flow_circuits.evaluation.interpretability_validation.collect_interpretability_outputs",
        lambda *args, **kwargs: queued.pop(0),
    )

    result = run_hard_pair_hybrid_correction_experiment(
        SimpleNamespace(),
        fit_loader=object(),
        val_loader=object(),
        test_loader=object(),
        device=torch.device("cpu"),
        checkpoint_tag="phase_b",
        fit_max_images=4,
        val_max_images=4,
        test_max_images=4,
        top_pairs=1,
        top_k_nodes=1,
    )

    assert result["backbone"]["trigger_subset_count"] == 1
    assert result["top_node_hybrid"]["net_gain"] == 1
    assert result["top_node_hybrid"]["override_coverage"] == 1.0
    assert result["top_node_hybrid"]["pairwise_win_rate_over_backbone"] >= 0.0


def test_hard_pair_benchmark_reuses_cached_bundle(tmp_path, monkeypatch):
    fit_z = torch.tensor(
        [
            [[[2.0, 0.0]], [[2.0, 0.0]]],
            [[[2.1, 0.0]], [[1.9, 0.0]]],
            [[[0.0, 2.0]], [[0.0, 2.0]]],
            [[[0.0, 2.1]], [[0.0, 1.9]]],
        ]
    )
    queued = [
        _make_outputs(fit_z, torch.tensor([0, 0, 1, 1]), torch.zeros(4, 2)),
        _make_outputs(fit_z.clone(), torch.tensor([0, 0, 1, 1]), torch.tensor([[1.0, 3.0], [1.0, 3.0], [0.2, 3.0], [0.2, 2.9]])),
        _make_outputs(fit_z.clone(), torch.tensor([0, 0, 1, 1]), torch.zeros(4, 2)),
    ]
    monkeypatch.setattr(
        "flow_circuits.evaluation.interpretability_validation.collect_interpretability_outputs",
        lambda *args, **kwargs: queued.pop(0),
    )
    output_path = tmp_path / "phase_b" / "hard_pair_probe_benchmark.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    first = run_hard_pair_probe_benchmark_experiment(
        SimpleNamespace(),
        fit_loader=object(),
        val_loader=object(),
        test_loader=object(),
        device=torch.device("cpu"),
        checkpoint_tag="phase_b",
        fit_max_images=4,
        val_max_images=4,
        test_max_images=4,
        top_pairs=1,
        top_k_nodes=1,
        output_path=output_path,
    )

    monkeypatch.setattr(
        "flow_circuits.evaluation.interpretability_validation.collect_interpretability_outputs",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should reuse cached split and bundle")),
    )
    second = run_hard_pair_probe_benchmark_experiment(
        SimpleNamespace(),
        fit_loader=object(),
        val_loader=object(),
        test_loader=object(),
        device=torch.device("cpu"),
        checkpoint_tag="phase_b",
        fit_max_images=4,
        val_max_images=4,
        test_max_images=4,
        top_pairs=1,
        top_k_nodes=1,
        output_path=output_path,
    )

    assert first["pair_rows"] == second["pair_rows"]
