from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch

from flow_circuits.evaluation.efficient_validation import (
    run_activation_probe_experiment,
    run_discovery_pilot_experiment,
    run_neighbor_agreement_experiment,
    run_topk_intervention_experiment,
)


def test_neighbor_agreement_experiment_is_deterministic(monkeypatch):
    z = torch.randn(6, 2, 2, 4)
    z = torch.nn.functional.normalize(z, dim=-1)
    outputs = {
        "z": z,
        "local_features": [torch.zeros(6, 2, 5) for _ in range(2)],
        "future_descriptors": z.clone(),
        "labels": torch.arange(6) % 2,
        "indices": torch.arange(6),
    }
    monkeypatch.setattr(
        "flow_circuits.evaluation.efficient_validation.collect_probe_outputs",
        lambda *args, **kwargs: outputs,
    )
    components = SimpleNamespace(config={"data": {"seed": 0}})

    result = run_neighbor_agreement_experiment(
        components,
        loader=object(),
        device=torch.device("cpu"),
        checkpoint_tag="phase_b",
        max_images=6,
        anchor_images=3,
        topk=2,
        seed=0,
    )

    assert result["summary"]["mean_recall_at_k"] == 1.0
    assert result["summary"]["mean_jaccard_at_k"] == 1.0
    assert len(result["per_node"]) == 4


def test_activation_probe_experiment_fits_linear_decoder(monkeypatch):
    torch.manual_seed(5)
    fit_z = torch.randn(8, 2, 2, 4)
    eval_z = torch.randn(6, 2, 2, 4)
    local_features_fit = []
    local_features_eval = []
    for layer_idx in range(2):
        weight = torch.randn(4, 3)
        fit_y = fit_z[:, layer_idx] @ weight
        eval_y = eval_z[:, layer_idx] @ weight
        local_features_fit.append(torch.cat([fit_y, torch.ones_like(fit_y[..., :1])], dim=-1))
        local_features_eval.append(torch.cat([eval_y, torch.ones_like(eval_y[..., :1])], dim=-1))

    outputs_sequence = iter(
        [
            {
                "z": fit_z,
                "local_features": local_features_fit,
                "future_descriptors": torch.zeros(8, 2, 2, 4),
                "labels": torch.zeros(8, dtype=torch.long),
                "indices": torch.arange(8),
            },
            {
                "z": eval_z,
                "local_features": local_features_eval,
                "future_descriptors": torch.zeros(6, 2, 2, 4),
                "labels": torch.zeros(6, dtype=torch.long),
                "indices": torch.arange(6),
            },
        ]
    )
    monkeypatch.setattr(
        "flow_circuits.evaluation.efficient_validation.collect_probe_outputs",
        lambda *args, **kwargs: next(outputs_sequence),
    )
    components = SimpleNamespace(config={"data": {"seed": 0}})

    result = run_activation_probe_experiment(
        components,
        fit_loader=object(),
        eval_loader=object(),
        device=torch.device("cpu"),
        checkpoint_tag="phase_c",
        fit_max_images=8,
        eval_max_images=6,
    )

    assert result["summary"]["mean_cosine"] > 0.999
    assert result["summary"]["mean_r2"] > 0.99
    assert len(result["per_layer"]) == 2


def test_discovery_pilot_uses_provided_node_panel(monkeypatch):
    outputs = {
        "future_descriptors": torch.zeros(6, 2, 3, 4),
        "predicted_next": torch.zeros(6, 1, 3, 4),
        "flow_targets": torch.zeros(6, 2, 3, 4),
        "labels": torch.zeros(6, dtype=torch.long),
        "indices": torch.arange(6),
    }
    monkeypatch.setattr(
        "flow_circuits.evaluation.efficient_validation.collect_discovery_outputs",
        lambda *args, **kwargs: outputs,
    )
    captured = {}

    class DummyDiscoverer:
        def __init__(self, **kwargs):
            captured["init_kwargs"] = kwargs

        def discover(self, **kwargs):
            captured["discover_kwargs"] = kwargs
            return {
                "metadata": {"n_images": 6, "n_layers": 2, "n_cells": 3, "grid_size": 2, "random_seed": 0},
                "node_clusters": [{"stability": 0.8}],
                "circuits": [
                    {
                        "id": 0,
                        "image_set": list(range(4)),
                        "representative_node": [0, 1],
                        "active_nodes": [[0, 1], [1, 2]],
                        "engagement_profile": {},
                        "centroids": {"0:1": [1.0], "1:2": [1.0]},
                        "thresholds": {"0:1": 0.1, "1:2": 0.2},
                        "stability": {"mean_cluster_stability": 0.8, "n_node_clusters": 1},
                        "purity": 0.5,
                    }
                ],
            }

    monkeypatch.setattr("flow_circuits.evaluation.efficient_validation.CandidateCircuitDiscoverer", DummyDiscoverer)
    components = SimpleNamespace(
        config={
            "data": {"seed": 0},
            "tokenization": {"grid_size": 2},
            "discovery": {
                "min_cluster_fraction": 0.2,
                "max_cluster_fraction": 0.8,
                "min_cluster_size": 2,
                "stability_threshold": 0.0,
                "merge_threshold": 0.5,
                "node_threshold": 0.5,
                "seed": 0,
            },
        }
    )

    result = run_discovery_pilot_experiment(
        components,
        loader=object(),
        device=torch.device("cpu"),
        checkpoint_tag="phase_b",
        max_images=6,
        nodes_per_layer=1,
        bootstrap_iterations=2,
        node_panel=[[0, 1], [1, 2]],
    )

    assert result["selected_node_panel"] == [[0, 1], [1, 2]]
    assert captured["discover_kwargs"]["node_subset"] == [[0, 1], [1, 2]]
    assert captured["init_kwargs"]["bootstrap_iterations"] == 2
    assert result["pilot_config"]["compute_seed_stability"] is False
    assert result["summary"]["n_circuits"] == 1


@dataclass
class _DummyInterventionResult:
    circuit_id: int
    n_members: int
    n_controls: int
    mean_member_delta_margin: float
    mean_member_delta_true: float
    mean_nonmember_delta_margin: float
    mean_nonmember_delta_true: float
    mean_random_node_delta_margin: float
    mean_random_cell_delta_margin: float
    p_member_vs_nonmember: float
    p_member_vs_random_node: float
    p_member_vs_random_cell: float
    corrected_p_member_vs_nonmember: float
    corrected_p_member_vs_random_node: float
    corrected_p_member_vs_random_cell: float
    ci_member_vs_nonmember: list[float]
    ci_member_vs_random_node: list[float]
    ci_member_vs_random_cell: list[float]
    validated: bool

    def to_dict(self):
        return self.__dict__.copy()


def test_topk_intervention_experiment_filters_and_sorts_circuits(monkeypatch):
    selected = {}
    monkeypatch.setattr(
        "flow_circuits.evaluation.efficient_validation.collect_intervention_outputs",
        lambda *args, **kwargs: {
            "future_descriptors": torch.zeros(4, 2, 2, 4),
            "images": torch.zeros(4, 3, 32, 32),
            "logits": torch.zeros(4, 10),
            "labels": torch.zeros(4, dtype=torch.long),
            "indices": torch.arange(4),
        },
    )

    def fake_run_circuit_interventions(*args, **kwargs):
        selected["artifact"] = args[1]
        return [
            _DummyInterventionResult(
                circuit_id=int(circuit["id"]),
                n_members=2,
                n_controls=2,
                mean_member_delta_margin=0.4,
                mean_member_delta_true=0.3,
                mean_nonmember_delta_margin=0.1,
                mean_nonmember_delta_true=0.05,
                mean_random_node_delta_margin=0.05,
                mean_random_cell_delta_margin=0.04,
                p_member_vs_nonmember=0.01,
                p_member_vs_random_node=0.01,
                p_member_vs_random_cell=0.01,
                corrected_p_member_vs_nonmember=0.02,
                corrected_p_member_vs_random_node=0.02,
                corrected_p_member_vs_random_cell=0.02,
                ci_member_vs_nonmember=[0.1, 0.5],
                ci_member_vs_random_node=[0.1, 0.5],
                ci_member_vs_random_cell=[0.1, 0.5],
                validated=True,
            )
            for circuit in selected["artifact"]["circuits"]
        ]

    monkeypatch.setattr("flow_circuits.evaluation.efficient_validation.run_circuit_interventions", fake_run_circuit_interventions)
    components = SimpleNamespace(config={"interventions": {"n_jobs": 1}})
    circuits_artifact = {
        "metadata": {"grid_size": 2, "n_layers": 2, "n_cells": 4},
        "circuits": [
            {"id": 0, "image_set": list(range(30)), "stability": {"mean_cluster_stability": 0.8, "n_node_clusters": 2}},
            {"id": 1, "image_set": list(range(20)), "stability": {"mean_cluster_stability": 0.95, "n_node_clusters": 5}},
            {"id": 2, "image_set": list(range(40)), "stability": {"mean_cluster_stability": 0.8, "n_node_clusters": 3}},
            {"id": 3, "image_set": list(range(35)), "stability": {"mean_cluster_stability": 0.7, "n_node_clusters": 4}},
        ],
    }

    result = run_topk_intervention_experiment(
        components,
        test_loader=object(),
        device=torch.device("cpu"),
        checkpoint_tag="phase_c",
        alpha=0.05,
        topk=2,
        circuits_artifact=circuits_artifact,
    )

    assert [circuit["id"] for circuit in selected["artifact"]["circuits"]] == [2, 0]
    assert result["selection"]["selected_circuit_ids"] == [2, 0]
    assert result["summary"]["member_specific_count"] == 2


def test_topk_interventions_generates_pilot_when_missing(monkeypatch):
    pilot_called = {}
    monkeypatch.setattr(
        "flow_circuits.evaluation.efficient_validation.run_discovery_pilot_experiment",
        lambda *args, **kwargs: pilot_called.setdefault(
            "result",
            {
                "metadata": {"grid_size": 2, "n_layers": 2, "n_cells": 4},
                "circuits": [
                    {"id": 7, "image_set": list(range(30)), "stability": {"mean_cluster_stability": 0.8, "n_node_clusters": 2}}
                ],
            },
        ),
    )
    monkeypatch.setattr(
        "flow_circuits.evaluation.efficient_validation.collect_intervention_outputs",
        lambda *args, **kwargs: {
            "future_descriptors": torch.zeros(4, 2, 2, 4),
            "images": torch.zeros(4, 3, 32, 32),
            "logits": torch.zeros(4, 10),
            "labels": torch.zeros(4, dtype=torch.long),
            "indices": torch.arange(4),
        },
    )
    monkeypatch.setattr(
        "flow_circuits.evaluation.efficient_validation.run_circuit_interventions",
        lambda *args, **kwargs: [],
    )
    components = SimpleNamespace(config={"interventions": {"n_jobs": 1}, "discovery": {"max_images": 6}})

    result = run_topk_intervention_experiment(
        components,
        test_loader=object(),
        device=torch.device("cpu"),
        checkpoint_tag="phase_b",
        alpha=0.05,
        topk=1,
        circuits_artifact=None,
        pilot_loader=object(),
    )

    assert pilot_called["result"]["circuits"][0]["id"] == 7
    assert result["selection"]["n_candidate_circuits"] == 1


def test_progress_callback_reports_monotonic_counts_and_eta(monkeypatch):
    outputs = {
        "z": torch.randn(5, 2, 1, 4),
        "local_features": [torch.zeros(5, 1, 5) for _ in range(2)],
        "future_descriptors": torch.randn(5, 2, 1, 4),
        "labels": torch.zeros(5, dtype=torch.long),
        "indices": torch.arange(5),
    }
    outputs["z"] = torch.nn.functional.normalize(outputs["z"], dim=-1)
    outputs["future_descriptors"] = torch.nn.functional.normalize(outputs["future_descriptors"], dim=-1)
    monkeypatch.setattr(
        "flow_circuits.evaluation.efficient_validation.collect_probe_outputs",
        lambda components, loader, *, device, max_images, progress_callback=None: (
            progress_callback(batch_idx=1, total_batches=1, seen_images=5, target_images=max_images) if progress_callback else None
        )
        or outputs,
    )
    components = SimpleNamespace(config={"data": {"seed": 0}})
    events = []

    run_neighbor_agreement_experiment(
        components,
        loader=object(),
        device=torch.device("cpu"),
        checkpoint_tag="phase_b",
        max_images=5,
        anchor_images=2,
        topk=2,
        progress_callback=lambda **event: events.append(event),
    )

    assert any(event["stage"] == "data_collection" for event in events)
    assert any(event["stage"] == "node_overlap" for event in events)
    for stage in {event["stage"] for event in events}:
        stage_events = [event for event in events if event["stage"] == stage]
        assert [event["completed"] for event in stage_events] == sorted(event["completed"] for event in stage_events)
    eta_events = [event for event in events if event["total"] is not None and event["completed"] > 0]
    assert eta_events
    assert all(event["eta_seconds"] is not None for event in eta_events)
