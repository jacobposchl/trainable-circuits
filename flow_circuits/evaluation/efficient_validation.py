from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F

from flow_circuits.discovery import CandidateCircuitDiscoverer
from flow_circuits.interventions import run_circuit_interventions
from flow_circuits.training import (
    LoadedFlowComponents,
    collect_discovery_outputs,
    collect_intervention_outputs,
    collect_probe_outputs,
)


NEIGHBOR_AGREEMENT_ID = "neighbor_agreement"
ACTIVATION_PROBE_ID = "activation_probe"
DISCOVERY_PILOT_ID = "discovery_pilot"
TOPK_INTERVENTIONS_ID = "topk_interventions"
EFFICIENT_EXPERIMENT_IDS = [
    NEIGHBOR_AGREEMENT_ID,
    ACTIVATION_PROBE_ID,
    DISCOVERY_PILOT_ID,
    TOPK_INTERVENTIONS_ID,
]


def run_neighbor_agreement_experiment(
    components: LoadedFlowComponents,
    loader,
    *,
    device: torch.device,
    checkpoint_tag: str,
    max_images: int,
    anchor_images: int,
    topk: int = 20,
    seed: int = 0,
    output_path: str | Path | None = None,
    progress_callback=None,
) -> dict:
    tracker = _ProgressTracker(
        experiment=NEIGHBOR_AGREEMENT_ID,
        checkpoint_tag=checkpoint_tag,
        progress_callback=progress_callback,
    )
    outputs = collect_probe_outputs(
        components,
        loader,
        device=device,
        max_images=max_images,
        progress_callback=lambda **event: tracker.emit(
            stage="data_collection",
            completed=event["batch_idx"],
            total=event.get("total_batches"),
            message="collecting z and q features",
        ),
    )
    z = F.normalize(outputs["z"], dim=-1)
    q = F.normalize(outputs["future_descriptors"], dim=-1)
    n_images, n_layers, n_cells, _ = z.shape
    effective_topk = min(int(topk), max(0, n_images - 1))
    if effective_topk <= 0:
        result = {
            "experiment": NEIGHBOR_AGREEMENT_ID,
            "checkpoint_tag": checkpoint_tag,
            "n_images": int(n_images),
            "anchor_images": 0,
            "topk": 0,
            "summary": {"mean_recall_at_k": 0.0, "mean_jaccard_at_k": 0.0},
            "layer_summaries": [],
            "per_node": [],
        }
        _maybe_write_json(result, output_path)
        return result

    rng = np.random.default_rng(seed)
    n_anchors = min(int(anchor_images), n_images)
    anchor_indices = np.sort(rng.choice(n_images, size=n_anchors, replace=False))
    per_node = []
    total_nodes = n_layers * n_cells
    for node_idx, (layer_idx, cell_idx) in enumerate(_iter_nodes(n_layers, n_cells), start=1):
        z_node = z[:, layer_idx, cell_idx, :]
        q_node = q[:, layer_idx, cell_idx, :]
        q_neighbors = _topk_neighbor_indices(q_node, anchor_indices, effective_topk)
        z_neighbors = _topk_neighbor_indices(z_node, anchor_indices, effective_topk)
        overlaps = []
        jaccards = []
        for anchor_row in range(n_anchors):
            q_set = set(q_neighbors[anchor_row].tolist())
            z_set = set(z_neighbors[anchor_row].tolist())
            overlap = len(q_set & z_set)
            overlaps.append(overlap / effective_topk)
            union = len(q_set | z_set)
            jaccards.append((overlap / union) if union else 1.0)
        per_node.append(
            {
                "layer_idx": int(layer_idx),
                "cell_idx": int(cell_idx),
                "recall_at_k": float(np.mean(overlaps)),
                "jaccard_at_k": float(np.mean(jaccards)),
            }
        )
        tracker.emit(
            stage="node_overlap",
            completed=node_idx,
            total=total_nodes,
            message=f"recall@{effective_topk} accumulation",
        )

    layer_summaries = _summarize_neighbor_layers(per_node, n_layers)
    result = {
        "experiment": NEIGHBOR_AGREEMENT_ID,
        "checkpoint_tag": checkpoint_tag,
        "n_images": int(n_images),
        "anchor_images": int(n_anchors),
        "topk": int(effective_topk),
        "summary": {
            "mean_recall_at_k": float(np.mean([item["recall_at_k"] for item in per_node])) if per_node else 0.0,
            "mean_jaccard_at_k": float(np.mean([item["jaccard_at_k"] for item in per_node])) if per_node else 0.0,
        },
        "layer_summaries": layer_summaries,
        "per_node": per_node,
    }
    _maybe_write_json(result, output_path)
    return result


def run_activation_probe_experiment(
    components: LoadedFlowComponents,
    fit_loader,
    eval_loader,
    *,
    device: torch.device,
    checkpoint_tag: str,
    fit_max_images: int,
    eval_max_images: int,
    ridge_alpha: float = 1.0,
    output_path: str | Path | None = None,
    progress_callback=None,
) -> dict:
    tracker = _ProgressTracker(
        experiment=ACTIVATION_PROBE_ID,
        checkpoint_tag=checkpoint_tag,
        progress_callback=progress_callback,
    )
    fit_outputs = collect_probe_outputs(
        components,
        fit_loader,
        device=device,
        max_images=fit_max_images,
        progress_callback=lambda **event: tracker.emit(
            stage="fit_data_collection",
            completed=event["batch_idx"],
            total=event.get("total_batches"),
            message="collecting fit probe features",
        ),
    )
    eval_outputs = collect_probe_outputs(
        components,
        eval_loader,
        device=device,
        max_images=eval_max_images,
        progress_callback=lambda **event: tracker.emit(
            stage="eval_data_collection",
            completed=event["batch_idx"],
            total=event.get("total_batches"),
            message="collecting eval probe features",
        ),
    )
    per_layer = []
    n_layers = fit_outputs["z"].shape[1]
    for layer_idx in range(n_layers):
        fit_x = fit_outputs["z"][:, layer_idx].reshape(-1, fit_outputs["z"].shape[-1]).numpy()
        fit_y = fit_outputs["local_features"][layer_idx][..., :-1].reshape(-1, fit_outputs["local_features"][layer_idx].shape[-1] - 1).numpy()
        eval_x = eval_outputs["z"][:, layer_idx].reshape(-1, eval_outputs["z"].shape[-1]).numpy()
        eval_y = eval_outputs["local_features"][layer_idx][..., :-1].reshape(-1, eval_outputs["local_features"][layer_idx].shape[-1] - 1).numpy()
        scaler = StandardScaler()
        fit_x_scaled = scaler.fit_transform(fit_x)
        eval_x_scaled = scaler.transform(eval_x)
        probe = Ridge(alpha=ridge_alpha)
        probe.fit(fit_x_scaled, fit_y)
        pred_y = probe.predict(eval_x_scaled)
        cosine = F.cosine_similarity(
            torch.from_numpy(pred_y).float(),
            torch.from_numpy(eval_y).float(),
            dim=-1,
        ).mean()
        per_layer.append(
            {
                "layer_idx": int(layer_idx),
                "cosine": float(cosine.item()),
                "r2": float(r2_score(eval_y, pred_y, multioutput="variance_weighted")),
            }
        )
        tracker.emit(
            stage="layer_probe",
            completed=layer_idx + 1,
            total=n_layers,
            message="ridge decode",
        )

    result = {
        "experiment": ACTIVATION_PROBE_ID,
        "checkpoint_tag": checkpoint_tag,
        "fit_images": int(fit_outputs["z"].shape[0]),
        "eval_images": int(eval_outputs["z"].shape[0]),
        "ridge_alpha": float(ridge_alpha),
        "summary": {
            "mean_cosine": float(np.mean([item["cosine"] for item in per_layer])) if per_layer else 0.0,
            "mean_r2": float(np.mean([item["r2"] for item in per_layer])) if per_layer else 0.0,
        },
        "per_layer": per_layer,
    }
    _maybe_write_json(result, output_path)
    return result


def run_discovery_pilot_experiment(
    components: LoadedFlowComponents,
    loader,
    *,
    device: torch.device,
    checkpoint_tag: str,
    max_images: int,
    nodes_per_layer: int,
    bootstrap_iterations: int,
    min_cluster_fraction: float | None = None,
    max_cluster_fraction: float | None = None,
    min_cluster_size: int | None = None,
    stability_threshold: float | None = None,
    merge_threshold: float | None = None,
    node_threshold: float | None = None,
    random_seed: int | None = None,
    node_panel: list[list[int]] | list[tuple[int, int]] | None = None,
    output_path: str | Path | None = None,
    progress_callback=None,
) -> dict:
    tracker = _ProgressTracker(
        experiment=DISCOVERY_PILOT_ID,
        checkpoint_tag=checkpoint_tag,
        progress_callback=progress_callback,
    )
    outputs = collect_discovery_outputs(
        components,
        loader,
        device=device,
        max_images=max_images,
        progress_callback=lambda **event: tracker.emit(
            stage="data_collection",
            completed=event["batch_idx"],
            total=event.get("total_batches"),
            message="collecting discovery features",
        ),
    )
    if node_panel is None:
        node_panel = _select_q_dispersion_node_panel(
            outputs["future_descriptors"],
            nodes_per_layer=nodes_per_layer,
            seed=components.config["data"].get("seed", 0),
            tracker=tracker,
        )
    normalized_node_panel = [[int(layer_idx), int(cell_idx)] for layer_idx, cell_idx in node_panel]
    dcfg = components.config["discovery"]
    discoverer = CandidateCircuitDiscoverer(
        grid_size=components.config["tokenization"].get("grid_size", 4),
        min_cluster_fraction=min_cluster_fraction if min_cluster_fraction is not None else dcfg.get("min_cluster_fraction", 0.005),
        max_cluster_fraction=max_cluster_fraction if max_cluster_fraction is not None else dcfg.get("max_cluster_fraction", 0.40),
        min_cluster_size=min_cluster_size if min_cluster_size is not None else dcfg.get("min_cluster_size", 20),
        bootstrap_iterations=int(bootstrap_iterations),
        stability_threshold=stability_threshold if stability_threshold is not None else dcfg.get("stability_threshold", 0.60),
        merge_threshold=merge_threshold if merge_threshold is not None else dcfg.get("merge_threshold", 0.70),
        node_threshold=node_threshold if node_threshold is not None else dcfg.get("node_threshold", 0.70),
        random_seed=int(random_seed if random_seed is not None else dcfg.get("seed", 0)),
    )

    def discovery_progress(**event) -> None:
        stage = event.get("stage")
        if stage == "node_clustering_start":
            tracker.emit(
                stage="pilot_discovery",
                completed=0,
                total=event["total"],
                message="HDBSCAN pilot clustering",
            )
        elif stage == "node_clustering":
            tracker.emit(
                stage="pilot_discovery",
                completed=event["completed"],
                total=event["total"],
                message=f"selected nodes retained={event['n_node_clusters']}",
            )
        elif stage == "node_clustering_done":
            tracker.emit(
                stage="pilot_discovery",
                completed=event["completed"],
                total=event["total"],
                message=f"node clustering complete retained={event['n_node_clusters']}",
            )

    artifact = discoverer.discover(
        future_descriptors=outputs["future_descriptors"].numpy(),
        predicted_next=outputs["predicted_next"].numpy(),
        flow_targets=outputs["flow_targets"].numpy(),
        dataset_indices=outputs["indices"].numpy(),
        labels=outputs["labels"].numpy(),
        progress_callback=discovery_progress,
        node_subset=normalized_node_panel,
    )
    circuit_depth_spans = [_circuit_depth_span(circuit["active_nodes"]) for circuit in artifact["circuits"]]
    result = {
        "experiment": DISCOVERY_PILOT_ID,
        "checkpoint_tag": checkpoint_tag,
        "pilot_config": {
            "max_images": int(max_images),
            "nodes_per_layer": int(nodes_per_layer),
            "bootstrap_iterations": int(bootstrap_iterations),
            "seed_runs": 1,
            "compute_seed_stability": False,
            "compute_node_shuffle_null": False,
        },
        "selected_node_panel": normalized_node_panel,
        "metadata": {
            **artifact["metadata"],
            "pilot_node_count": int(len(normalized_node_panel)),
        },
        "node_clusters": artifact["node_clusters"],
        "circuits": artifact["circuits"],
        "summary": {
            "n_node_clusters": int(len(artifact["node_clusters"])),
            "n_circuits": int(len(artifact["circuits"])),
            "mean_cluster_stability": float(np.mean([item["stability"] for item in artifact["node_clusters"]])) if artifact["node_clusters"] else 0.0,
            "mean_active_node_depth_span": float(np.mean(circuit_depth_spans)) if circuit_depth_spans else 0.0,
        },
    }
    _maybe_write_json(result, output_path)
    return result


def run_topk_intervention_experiment(
    components: LoadedFlowComponents,
    test_loader,
    *,
    device: torch.device,
    checkpoint_tag: str,
    alpha: float,
    topk: int,
    min_image_set_size: int = 25,
    max_images: int | None = None,
    circuits_artifact: dict | None = None,
    pilot_loader=None,
    pilot_node_panel: list[list[int]] | list[tuple[int, int]] | None = None,
    pilot_max_images: int | None = None,
    pilot_nodes_per_layer: int = 2,
    pilot_bootstrap_iterations: int = 5,
    output_path: str | Path | None = None,
    progress_callback=None,
) -> dict:
    tracker = _ProgressTracker(
        experiment=TOPK_INTERVENTIONS_ID,
        checkpoint_tag=checkpoint_tag,
        progress_callback=progress_callback,
    )
    if circuits_artifact is None:
        if pilot_loader is None:
            raise ValueError("pilot_loader is required when circuits_artifact is not provided")
        circuits_artifact = run_discovery_pilot_experiment(
            components,
            pilot_loader,
            device=device,
            checkpoint_tag=checkpoint_tag,
            max_images=int(pilot_max_images or max_images or components.config["discovery"].get("max_images", 5000)),
            nodes_per_layer=int(pilot_nodes_per_layer),
            bootstrap_iterations=int(pilot_bootstrap_iterations),
            node_panel=pilot_node_panel,
            progress_callback=progress_callback,
        )

    filtered_circuits = [
        circuit
        for circuit in circuits_artifact.get("circuits", [])
        if len(circuit.get("image_set", [])) >= int(min_image_set_size)
    ]
    filtered_circuits.sort(
        key=lambda circuit: (
            -float(circuit["stability"]["mean_cluster_stability"]),
            -int(circuit["stability"]["n_node_clusters"]),
            -len(circuit["image_set"]),
            int(circuit["id"]),
        )
    )
    selected_circuits = filtered_circuits[: int(topk)]
    selected_artifact = {
        "metadata": circuits_artifact["metadata"],
        "circuits": selected_circuits,
    }
    outputs = collect_intervention_outputs(
        components,
        test_loader,
        device=device,
        max_images=max_images,
        progress_callback=lambda **event: tracker.emit(
            stage="data_collection",
            completed=event["batch_idx"],
            total=event.get("total_batches"),
            message="collecting intervention features",
        ),
    )
    results = run_circuit_interventions(
        components,
        selected_artifact,
        outputs,
        alpha=alpha,
        progress_callback=lambda **event: tracker.emit(
            stage="circuit_interventions",
            completed=event["completed"],
            total=event["total"],
            message=f"circuit_id={event['circuit_id']} {event['status']}",
        ),
        n_jobs=max(1, int(components.config.get("interventions", {}).get("n_jobs", 1))),
    )
    result_dicts = [result.to_dict() for result in results]
    member_specific_count = sum(
        1
        for item in result_dicts
        if item["corrected_p_member_vs_nonmember"] < alpha
        and item["ci_member_vs_nonmember"][0] > 0.0
        and item["mean_member_delta_margin"] > item["mean_nonmember_delta_margin"]
    )
    result = {
        "experiment": TOPK_INTERVENTIONS_ID,
        "checkpoint_tag": checkpoint_tag,
        "selection": {
            "topk": int(topk),
            "min_image_set_size": int(min_image_set_size),
            "selected_circuit_ids": [int(circuit["id"]) for circuit in selected_circuits],
            "n_candidate_circuits": int(len(circuits_artifact.get("circuits", []))),
            "n_selected_circuits": int(len(selected_circuits)),
        },
        "summary": {
            "member_specific_count": int(member_specific_count),
            "validated_count": int(sum(1 for item in result_dicts if item["validated"])),
            "mean_member_delta_margin": float(np.mean([item["mean_member_delta_margin"] for item in result_dicts])) if result_dicts else 0.0,
            "mean_corrected_p_member_vs_nonmember": float(np.mean([item["corrected_p_member_vs_nonmember"] for item in result_dicts])) if result_dicts else 1.0,
        },
        "selected_circuits": selected_circuits,
        "intervention_results": result_dicts,
    }
    _maybe_write_json(result, output_path)
    return result


class _ProgressTracker:
    def __init__(self, *, experiment: str, checkpoint_tag: str, progress_callback) -> None:
        self.experiment = experiment
        self.checkpoint_tag = checkpoint_tag
        self.progress_callback = progress_callback
        self.stage_started: dict[str, float] = {}

    def emit(
        self,
        *,
        stage: str,
        completed: int,
        total: int | None,
        message: str,
        **extra,
    ) -> None:
        if self.progress_callback is None:
            return
        started_at = self.stage_started.setdefault(stage, time.perf_counter())
        elapsed_seconds = float(max(0.0, time.perf_counter() - started_at))
        eta_seconds = None
        if total is not None and completed > 0:
            rate = elapsed_seconds / max(completed, 1)
            eta_seconds = float(max(0.0, rate * max(total - completed, 0)))
        event = {
            "experiment": self.experiment,
            "checkpoint_tag": self.checkpoint_tag,
            "stage": stage,
            "completed": int(completed),
            "total": int(total) if total is not None else None,
            "elapsed_seconds": elapsed_seconds,
            "eta_seconds": eta_seconds,
            "message": message,
        }
        event.update(extra)
        self.progress_callback(**event)


def _topk_neighbor_indices(vectors: torch.Tensor, anchor_indices: np.ndarray, topk: int) -> torch.Tensor:
    similarities = vectors[anchor_indices] @ vectors.T
    similarities = similarities.clone()
    similarities[torch.arange(anchor_indices.shape[0]), torch.from_numpy(anchor_indices)] = -float("inf")
    return torch.topk(similarities, k=topk, dim=1).indices


def _summarize_neighbor_layers(per_node: list[dict], n_layers: int) -> list[dict]:
    summaries = []
    for layer_idx in range(n_layers):
        layer_nodes = [item for item in per_node if item["layer_idx"] == layer_idx]
        summaries.append(
            {
                "layer_idx": int(layer_idx),
                "mean_recall_at_k": float(np.mean([item["recall_at_k"] for item in layer_nodes])) if layer_nodes else 0.0,
                "mean_jaccard_at_k": float(np.mean([item["jaccard_at_k"] for item in layer_nodes])) if layer_nodes else 0.0,
            }
        )
    return summaries


def _iter_nodes(n_layers: int, n_cells: int):
    for layer_idx in range(n_layers):
        for cell_idx in range(n_cells):
            yield layer_idx, cell_idx


def _select_q_dispersion_node_panel(
    future_descriptors: torch.Tensor,
    *,
    nodes_per_layer: int,
    seed: int,
    tracker: _ProgressTracker,
    pair_samples: int = 2048,
) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    n_images, n_layers, n_cells, _ = future_descriptors.shape
    if n_images < 2:
        return [[layer_idx, cell_idx] for layer_idx in range(n_layers) for cell_idx in range(min(nodes_per_layer, n_cells))]
    sample_count = min(int(pair_samples), n_images * (n_images - 1))
    left = rng.integers(0, n_images, size=sample_count)
    right = rng.integers(0, n_images, size=sample_count)
    right = np.where(left == right, (right + 1) % n_images, right)
    scores_by_layer: dict[int, list[tuple[float, int]]] = {}
    total_nodes = n_layers * n_cells
    for node_idx, (layer_idx, cell_idx) in enumerate(_iter_nodes(n_layers, n_cells), start=1):
        node_vectors = future_descriptors[:, layer_idx, cell_idx, :]
        cosine = (node_vectors[left] * node_vectors[right]).sum(dim=-1).cpu().numpy()
        dispersion = float(1.0 - cosine.mean())
        scores_by_layer.setdefault(layer_idx, []).append((dispersion, cell_idx))
        tracker.emit(
            stage="panel_scoring",
            completed=node_idx,
            total=total_nodes,
            message="q-dispersion node scoring",
        )
    panel = []
    for layer_idx in range(n_layers):
        ranked = sorted(scores_by_layer.get(layer_idx, []), key=lambda item: (-item[0], item[1]))
        panel.extend([[int(layer_idx), int(cell_idx)] for _, cell_idx in ranked[:nodes_per_layer]])
    return panel


def _circuit_depth_span(active_nodes: list[list[int]] | list[tuple[int, int]]) -> int:
    if not active_nodes:
        return 0
    layers = [int(node[0]) for node in active_nodes]
    return int(max(layers) - min(layers) + 1)


def _maybe_write_json(payload: dict, output_path: str | Path | None) -> None:
    if output_path is None:
        return
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
