from __future__ import annotations

from collections import Counter, defaultdict, deque
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from flow_circuits.discovery.node_clustering import discover_node_clusters, jaccard
from flow_circuits.interventions import assign_circuit_members, run_circuit_interventions
from flow_circuits.training import (
    LoadedFlowComponents,
    collect_model_outputs,
    collect_probe_outputs,
)


MOTIF_FAMILIES_ID = "motif_families"
MOTIF_GALLERIES_ID = "motif_galleries"
MOTIF_PERSISTENCE_ID = "motif_persistence"
MOTIF_PREDICTIVENESS_ID = "motif_predictiveness"
MOTIF_INTERVENTIONS_ID = "motif_interventions"
MOTIF_COOCCURRENCE_GRAPH_ID = "motif_cooccurrence_graph"
MOTIF_PHASE_MATCH_ID = "motif_phase_match"
MOTIF_TOPOLOGY_ID = "motif_topology"
MOTIF_TRANSFER_STABILITY_ID = "motif_transfer_stability"

CORE_MOTIF_EXPERIMENT_IDS = [
    MOTIF_FAMILIES_ID,
    MOTIF_GALLERIES_ID,
    MOTIF_PERSISTENCE_ID,
    MOTIF_PREDICTIVENESS_ID,
    MOTIF_INTERVENTIONS_ID,
]

EXTENDED_MOTIF_EXPERIMENT_IDS = [
    MOTIF_COOCCURRENCE_GRAPH_ID,
    MOTIF_PHASE_MATCH_ID,
    MOTIF_TOPOLOGY_ID,
    MOTIF_TRANSFER_STABILITY_ID,
]


def discover_motif_families(
    components: LoadedFlowComponents,
    loader,
    *,
    device: torch.device,
    checkpoint_tag: str,
    max_images: int,
    nodes_per_layer: int,
    bootstrap_iterations: int,
    merge_threshold: float = 0.50,
    node_threshold: float = 0.50,
    min_cluster_fraction: float | None = None,
    max_cluster_fraction: float | None = None,
    min_cluster_size: int | None = None,
    stability_threshold: float | None = None,
    random_seed: int | None = None,
    node_panel: list[list[int]] | list[tuple[int, int]] | None = None,
    output_path: str | Path | None = None,
    progress_callback=None,
) -> dict:
    tracker = _ProgressTracker(
        experiment=MOTIF_FAMILIES_ID,
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
            message="collecting z/q motif features",
        ),
    )
    result = _discover_motif_families_from_outputs(
        outputs,
        grid_size=int(components.config["tokenization"].get("grid_size", 4)),
        checkpoint_tag=checkpoint_tag,
        merge_threshold=merge_threshold,
        node_threshold=node_threshold,
        nodes_per_layer=nodes_per_layer,
        bootstrap_iterations=bootstrap_iterations,
        min_cluster_fraction=(
            float(min_cluster_fraction)
            if min_cluster_fraction is not None
            else float(components.config.get("discovery", {}).get("min_cluster_fraction", 0.005))
        ),
        max_cluster_fraction=(
            float(max_cluster_fraction)
            if max_cluster_fraction is not None
            else float(components.config.get("discovery", {}).get("max_cluster_fraction", 0.40))
        ),
        min_cluster_size=(
            int(min_cluster_size)
            if min_cluster_size is not None
            else int(components.config.get("discovery", {}).get("min_cluster_size", 20))
        ),
        stability_threshold=(
            float(stability_threshold)
            if stability_threshold is not None
            else float(components.config.get("discovery", {}).get("stability_threshold", 0.60))
        ),
        random_seed=int(random_seed if random_seed is not None else components.config["data"].get("seed", 0)),
        node_panel=node_panel,
        tracker=tracker,
    )
    _maybe_write_json(result, output_path)
    return result


def run_motif_gallery_experiment(
    components: LoadedFlowComponents,
    loader,
    *,
    device: torch.device,
    checkpoint_tag: str,
    motif_artifact: dict,
    max_images: int,
    topk: int,
    exemplar_count: int = 9,
    output_path: str | Path | None = None,
    progress_callback=None,
) -> dict:
    tracker = _ProgressTracker(
        experiment=MOTIF_GALLERIES_ID,
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
            message="collecting discovery features for motif galleries",
        ),
    )
    z = F.normalize(outputs["z"], dim=-1)
    labels = outputs["labels"]
    ranked_motifs = _rank_motifs(motif_artifact.get("motifs", []))[: int(topk)]
    galleries = []
    for motif_idx, motif in enumerate(ranked_motifs, start=1):
        member_rows = _member_rows_for_motif(motif, outputs["indices"])
        scores = _motif_scores(motif, z)
        member_scores = scores[member_rows] if member_rows.size else torch.empty(0)
        exemplar_rows = member_rows[torch.argsort(member_scores, descending=True)[: int(exemplar_count)]].tolist() if member_rows.size else []
        member_labels = labels[member_rows].cpu().numpy() if member_rows.size else np.empty(0, dtype=np.int64)
        galleries.append(
            {
                "motif_id": int(motif["id"]),
                "size": int(len(motif.get("image_set", []))),
                "supporting_layers": int(len(motif.get("layer_support", []))),
                "representative_node": motif["representative_node"],
                "dominant_class": int(motif["purity"]["dominant_class"]) if motif.get("purity", {}).get("dominant_class") is not None else None,
                "class_purity": float(motif.get("purity", {}).get("fraction", 0.0)),
                "cohesion": float(member_scores.mean().item()) if member_rows.size else 0.0,
                "exemplar_dataset_indices": outputs["indices"][exemplar_rows].cpu().tolist() if exemplar_rows else [],
                "exemplar_labels": outputs["labels"][exemplar_rows].cpu().tolist() if exemplar_rows else [],
                "class_histogram": _label_histogram(member_labels),
            }
        )
        tracker.emit(
            stage="motif_gallery",
            completed=motif_idx,
            total=len(ranked_motifs),
            message="ranking exemplars and purity",
        )
    result = {
        "experiment": MOTIF_GALLERIES_ID,
        "checkpoint_tag": checkpoint_tag,
        "summary": {
            "n_ranked_motifs": int(len(galleries)),
            "mean_class_purity": float(np.mean([item["class_purity"] for item in galleries])) if galleries else 0.0,
            "mean_cohesion": float(np.mean([item["cohesion"] for item in galleries])) if galleries else 0.0,
        },
        "motif_rows": galleries,
    }
    _maybe_write_json(result, output_path)
    return result


def run_motif_persistence_experiment(
    motif_artifact: dict,
    *,
    checkpoint_tag: str,
    output_path: str | Path | None = None,
    progress_callback=None,
) -> dict:
    tracker = _ProgressTracker(
        experiment=MOTIF_PERSISTENCE_ID,
        checkpoint_tag=checkpoint_tag,
        progress_callback=progress_callback,
    )
    rows = []
    motifs = _rank_motifs(motif_artifact.get("motifs", []))
    for motif_idx, motif in enumerate(motifs, start=1):
        active_nodes = [tuple(node) for node in motif.get("active_nodes", [])]
        layers = sorted({int(node[0]) for node in active_nodes})
        first_layer = min(layers) if layers else None
        last_layer = max(layers) if layers else None
        depth_span = (last_layer - first_layer + 1) if layers else 0
        per_layer_support = _per_layer_support(active_nodes)
        persistence_score = (len(layers) / depth_span) if depth_span else 0.0
        dominant_cells = {
            str(layer_idx): sorted({int(cell_idx) for _, cell_idx in layer_nodes})
            for layer_idx, layer_nodes in _group_nodes_by_layer(active_nodes).items()
        }
        rows.append(
            {
                "motif_id": int(motif["id"]),
                "first_layer": int(first_layer) if first_layer is not None else None,
                "last_layer": int(last_layer) if last_layer is not None else None,
                "depth_span": int(depth_span),
                "n_supporting_layers": int(len(layers)),
                "persistence_score": float(persistence_score),
                "per_layer_support": {str(key): int(value) for key, value in per_layer_support.items()},
                "dominant_cells_by_layer": dominant_cells,
            }
        )
        tracker.emit(
            stage="motif_persistence",
            completed=motif_idx,
            total=len(motifs),
            message="computing depth span",
        )
    result = {
        "experiment": MOTIF_PERSISTENCE_ID,
        "checkpoint_tag": checkpoint_tag,
        "summary": {
            "n_motifs": int(len(rows)),
            "mean_depth_span": float(np.mean([item["depth_span"] for item in rows])) if rows else 0.0,
            "fraction_spanning_ge_3_layers": float(np.mean([item["n_supporting_layers"] >= 3 for item in rows])) if rows else 0.0,
            "mean_persistence_score": float(np.mean([item["persistence_score"] for item in rows])) if rows else 0.0,
        },
        "motif_rows": rows,
    }
    _maybe_write_json(result, output_path)
    return result


def run_motif_predictiveness_experiment(
    components: LoadedFlowComponents,
    loader,
    *,
    device: torch.device,
    checkpoint_tag: str,
    motif_artifact: dict,
    max_images: int,
    topk: int,
    output_path: str | Path | None = None,
    progress_callback=None,
) -> dict:
    tracker = _ProgressTracker(
        experiment=MOTIF_PREDICTIVENESS_ID,
        checkpoint_tag=checkpoint_tag,
        progress_callback=progress_callback,
    )
    outputs = collect_model_outputs(
        components,
        loader,
        device=device,
        max_images=max_images,
        progress_callback=lambda **event: tracker.emit(
            stage="data_collection",
            completed=event["batch_idx"],
            total=event.get("total_batches"),
            message="collecting held-out motif predictiveness features",
        ),
    )
    logits = outputs["logits"]
    labels = outputs["labels"]
    z = outputs["z"]
    ranked_motifs = _rank_motifs(motif_artifact.get("motifs", []))[: int(topk)]
    per_motif = []
    for motif_idx, motif in enumerate(ranked_motifs, start=1):
        dominant_class = motif.get("purity", {}).get("dominant_class")
        member_mask = assign_circuit_members(motif, z, outputs["indices"])
        member_rows = torch.nonzero(member_mask, as_tuple=False).flatten()
        if member_rows.numel() == 0 or dominant_class is None:
            per_motif.append(
                {
                    "motif_id": int(motif["id"]),
                    "dominant_class": None if dominant_class is None else int(dominant_class),
                    "n_members": int(member_rows.numel()),
                    "precision": 0.0,
                    "recall": 0.0,
                    "lift_over_base_rate": 0.0,
                    "member_margin_lift": 0.0,
                }
            )
            tracker.emit(
                stage="motif_predictiveness",
                completed=motif_idx,
                total=len(ranked_motifs),
                message="motif has no held-out members",
            )
            continue
        dominant_class = int(dominant_class)
        member_labels = labels[member_rows]
        positives = labels == dominant_class
        precision = float((member_labels == dominant_class).float().mean().item())
        recall = float(((member_labels == dominant_class).sum().item()) / max(int(positives.sum().item()), 1))
        base_rate = float(positives.float().mean().item())
        lift_over_base_rate = float(precision / max(base_rate, 1.0e-8))
        member_logits = logits[member_rows]
        member_margins = _margin(member_logits)
        control_rows = _matched_rows(
            member_rows=member_rows,
            member_classes=logits[member_rows].argmax(dim=1),
            member_margins=member_margins,
            predicted_classes=logits.argmax(dim=1),
            margins=_margin(logits),
        )
        control_margins = _margin(logits[control_rows]) if control_rows.numel() else torch.zeros(0, dtype=member_margins.dtype)
        member_margin_lift = float(
            (member_margins[: control_margins.shape[0]] - control_margins).mean().item()
        ) if control_margins.numel() else 0.0
        per_motif.append(
            {
                "motif_id": int(motif["id"]),
                "dominant_class": dominant_class,
                "n_members": int(member_rows.numel()),
                "precision": precision,
                "recall": recall,
                "lift_over_base_rate": lift_over_base_rate,
                "member_margin_lift": member_margin_lift,
            }
        )
        tracker.emit(
            stage="motif_predictiveness",
            completed=motif_idx,
            total=len(ranked_motifs),
            message="held-out motif membership diagnostics",
        )
    result = {
        "experiment": MOTIF_PREDICTIVENESS_ID,
        "checkpoint_tag": checkpoint_tag,
        "summary": {
            "n_tested_motifs": int(len(per_motif)),
            "mean_precision": float(np.mean([item["precision"] for item in per_motif])) if per_motif else 0.0,
            "mean_lift_over_base_rate": float(np.mean([item["lift_over_base_rate"] for item in per_motif])) if per_motif else 0.0,
            "mean_member_margin_lift": float(np.mean([item["member_margin_lift"] for item in per_motif])) if per_motif else 0.0,
        },
        "motif_rows": per_motif,
    }
    _maybe_write_json(result, output_path)
    return result


def run_motif_intervention_experiment(
    components: LoadedFlowComponents,
    loader,
    *,
    device: torch.device,
    checkpoint_tag: str,
    motif_artifact: dict,
    max_images: int,
    topk: int,
    alpha: float = 0.05,
    min_image_set_size: int = 25,
    output_path: str | Path | None = None,
    progress_callback=None,
) -> dict:
    tracker = _ProgressTracker(
        experiment=MOTIF_INTERVENTIONS_ID,
        checkpoint_tag=checkpoint_tag,
        progress_callback=progress_callback,
    )
    selected_motifs = [
        motif for motif in _rank_motifs(motif_artifact.get("motifs", []))
        if len(motif.get("image_set", [])) >= int(min_image_set_size)
    ][: int(topk)]
    outputs = collect_model_outputs(
        components,
        loader,
        device=device,
        max_images=max_images,
        progress_callback=lambda **event: tracker.emit(
            stage="data_collection",
            completed=event["batch_idx"],
            total=event.get("total_batches"),
            message="collecting held-out intervention features",
        ),
    )
    selected_artifact = {
        "metadata": {
            "grid_size": int(motif_artifact["metadata"]["grid_size"]),
            "n_layers": int(motif_artifact["metadata"]["n_layers"]),
            "n_cells": int(motif_artifact["metadata"]["n_cells"]),
        },
        "circuits": selected_motifs,
    }
    results = run_circuit_interventions(
        components,
        selected_artifact,
        outputs,
        alpha=alpha,
        descriptor_key="z",
        progress_callback=lambda **event: tracker.emit(
            stage="motif_interventions",
            completed=event["completed"],
            total=event["total"],
            message=f"motif_id={event['circuit_id']} {event['status']}",
        ),
        n_jobs=max(1, int(components.config.get("interventions", {}).get("n_jobs", 1))),
    )
    result_dicts = [item.to_dict() for item in results]
    member_specific_count = sum(
        1
        for item in result_dicts
        if item["corrected_p_member_vs_nonmember"] < alpha
        and item["ci_member_vs_nonmember"][0] > 0.0
        and item["mean_member_delta_margin"] > item["mean_nonmember_delta_margin"]
    )
    result = {
        "experiment": MOTIF_INTERVENTIONS_ID,
        "checkpoint_tag": checkpoint_tag,
        "selection": {
            "topk": int(topk),
            "min_image_set_size": int(min_image_set_size),
            "selected_motif_ids": [int(motif["id"]) for motif in selected_motifs],
            "n_candidate_motifs": int(len(motif_artifact.get("motifs", []))),
            "n_selected_motifs": int(len(selected_motifs)),
        },
        "summary": {
            "member_specific_count": int(member_specific_count),
            "validated_count": int(sum(1 for item in result_dicts if item["validated"])),
            "mean_member_delta_margin": float(np.mean([item["mean_member_delta_margin"] for item in result_dicts])) if result_dicts else 0.0,
        },
        "selected_motifs": selected_motifs,
        "intervention_results": result_dicts,
    }
    _maybe_write_json(result, output_path)
    return result


def run_motif_cooccurrence_experiment(
    motif_artifact: dict,
    *,
    checkpoint_tag: str,
    overlap_threshold: float = 0.25,
    output_path: str | Path | None = None,
    progress_callback=None,
) -> dict:
    tracker = _ProgressTracker(
        experiment=MOTIF_COOCCURRENCE_GRAPH_ID,
        checkpoint_tag=checkpoint_tag,
        progress_callback=progress_callback,
    )
    motifs = _rank_motifs(motif_artifact.get("motifs", []))
    overlaps = []
    adjacency = defaultdict(set)
    total_pairs = max((len(motifs) * (len(motifs) - 1)) // 2, 1)
    pair_idx = 0
    for left in range(len(motifs)):
        for right in range(left + 1, len(motifs)):
            pair_idx += 1
            overlap = jaccard(set(motifs[left]["image_set"]), set(motifs[right]["image_set"]))
            if overlap >= overlap_threshold:
                adjacency[int(motifs[left]["id"])].add(int(motifs[right]["id"]))
                adjacency[int(motifs[right]["id"])].add(int(motifs[left]["id"]))
            overlaps.append(
                {
                    "left_motif_id": int(motifs[left]["id"]),
                    "right_motif_id": int(motifs[right]["id"]),
                    "jaccard": float(overlap),
                }
            )
            tracker.emit(
                stage="motif_pair_overlap",
                completed=pair_idx,
                total=total_pairs,
                message="building motif co-occurrence graph",
            )
    components = _graph_components([int(motif["id"]) for motif in motifs], adjacency)
    edge_count = sum(len(neighbors) for neighbors in adjacency.values()) // 2
    max_edges = max((len(motifs) * (len(motifs) - 1)) // 2, 1)
    result = {
        "experiment": MOTIF_COOCCURRENCE_GRAPH_ID,
        "checkpoint_tag": checkpoint_tag,
        "summary": {
            "n_motifs": int(len(motifs)),
            "n_edges": int(edge_count),
            "graph_density": float(edge_count / max_edges) if len(motifs) > 1 else 0.0,
            "n_connected_components": int(len(components)),
            "largest_component_size": int(max((len(component) for component in components), default=0)),
        },
        "strongest_pairs": sorted(overlaps, key=lambda item: (-item["jaccard"], item["left_motif_id"], item["right_motif_id"]))[:10],
        "components": components,
    }
    _maybe_write_json(result, output_path)
    return result


def run_motif_phase_match_experiment(
    phase_b_artifact: dict,
    phase_c_artifact: dict,
    *,
    output_path: str | Path | None = None,
    progress_callback=None,
) -> dict:
    tracker = _ProgressTracker(
        experiment=MOTIF_PHASE_MATCH_ID,
        checkpoint_tag="phase_b_vs_phase_c",
        progress_callback=progress_callback,
    )
    phase_b_motifs = _rank_motifs(phase_b_artifact.get("motifs", []))
    phase_c_motifs = _rank_motifs(phase_c_artifact.get("motifs", []))
    overlap_table = []
    for left_idx, motif_b in enumerate(phase_b_motifs, start=1):
        for motif_c in phase_c_motifs:
            overlap_table.append(_motif_match_features(motif_b, motif_c))
        tracker.emit(
            stage="phase_match_scoring",
            completed=left_idx,
            total=len(phase_b_motifs),
            message="scoring Phase B -> Phase C motif overlaps",
        )
    matched_pairs = _greedy_match_motifs(phase_b_motifs, phase_c_motifs)
    split_phase_b = sorted(
        int(motif["id"])
        for motif in phase_b_motifs
        if sum(1 for item in overlap_table if item["phase_b_motif_id"] == int(motif["id"]) and item["image_set_jaccard"] >= 0.25) >= 2
    )
    merged_phase_c = sorted(
        int(motif["id"])
        for motif in phase_c_motifs
        if sum(1 for item in overlap_table if item["phase_c_motif_id"] == int(motif["id"]) and item["image_set_jaccard"] >= 0.25) >= 2
    )
    matched_phase_b = {item["phase_b_motif_id"] for item in matched_pairs}
    phase_c_only = sorted(
        int(motif["id"])
        for motif in phase_c_motifs
        if all(
            item["phase_c_motif_id"] != int(motif["id"]) or item["image_set_jaccard"] < 0.25
            for item in overlap_table
        )
    )
    result = {
        "experiment": MOTIF_PHASE_MATCH_ID,
        "summary": {
            "matched_motif_count": int(len(matched_pairs)),
            "mean_match_quality": float(np.mean([item["image_set_jaccard"] for item in matched_pairs])) if matched_pairs else 0.0,
        },
        "matched_pairs": matched_pairs,
        "split_phase_b_motifs": split_phase_b,
        "merged_phase_c_motifs": merged_phase_c,
        "phase_c_only_motifs": phase_c_only,
        "phase_b_unmatched_motifs": sorted(int(motif["id"]) for motif in phase_b_motifs if int(motif["id"]) not in matched_phase_b),
    }
    _maybe_write_json(result, output_path)
    return result


def run_motif_topology_experiment(
    motif_artifact: dict,
    *,
    checkpoint_tag: str,
    output_path: str | Path | None = None,
    progress_callback=None,
) -> dict:
    tracker = _ProgressTracker(
        experiment=MOTIF_TOPOLOGY_ID,
        checkpoint_tag=checkpoint_tag,
        progress_callback=progress_callback,
    )
    grid_size = int(motif_artifact["metadata"]["grid_size"])
    rows = []
    motifs = _rank_motifs(motif_artifact.get("motifs", []))
    for motif_idx, motif in enumerate(motifs, start=1):
        metrics = _motif_topology_metrics(motif.get("active_nodes", []), grid_size=grid_size)
        rows.append(
            {
                "motif_id": int(motif["id"]),
                **metrics,
                "topology_type": _classify_topology(metrics),
            }
        )
        tracker.emit(
            stage="motif_topology",
            completed=motif_idx,
            total=len(motifs),
            message="computing motif structural diagnostics",
        )
    topology_counts = Counter(row["topology_type"] for row in rows)
    result = {
        "experiment": MOTIF_TOPOLOGY_ID,
        "checkpoint_tag": checkpoint_tag,
        "summary": {
            "n_motifs": int(len(rows)),
            "mean_same_layer_adjacent_edges": float(np.mean([row["same_layer_adjacent_edges"] for row in rows])) if rows else 0.0,
            "mean_same_cell_depth_edges": float(np.mean([row["same_cell_depth_edges"] for row in rows])) if rows else 0.0,
            "mean_largest_connected_component": float(np.mean([row["largest_connected_component"] for row in rows])) if rows else 0.0,
            "topology_counts": dict(sorted(topology_counts.items())),
        },
        "motif_rows": rows,
    }
    _maybe_write_json(result, output_path)
    return result


def run_motif_transfer_stability_experiment(
    components: LoadedFlowComponents,
    loader,
    *,
    device: torch.device,
    checkpoint_tag: str,
    max_images: int,
    nodes_per_layer: int,
    bootstrap_iterations: int,
    merge_threshold: float = 0.50,
    node_threshold: float = 0.50,
    subset_fraction: float = 0.80,
    output_path: str | Path | None = None,
    progress_callback=None,
) -> dict:
    tracker = _ProgressTracker(
        experiment=MOTIF_TRANSFER_STABILITY_ID,
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
            message="collecting discovery features for motif transfer stability",
        ),
    )
    seed = int(components.config["data"].get("seed", 0))
    rng = np.random.default_rng(seed)
    total_rows = outputs["z"].shape[0]
    subset_size = max(2, min(total_rows, int(round(total_rows * subset_fraction))))
    left_rows = np.sort(rng.choice(total_rows, size=subset_size, replace=False))
    right_rows = np.sort(rng.choice(total_rows, size=subset_size, replace=False))
    shared_panel = _select_q_dispersion_node_panel(
        outputs["future_descriptors"],
        nodes_per_layer=nodes_per_layer,
        seed=seed,
        tracker=tracker,
    )
    left_artifact = _discover_motif_families_from_outputs(
        _subset_probe_outputs(outputs, left_rows),
        grid_size=int(components.config["tokenization"].get("grid_size", 4)),
        checkpoint_tag=f"{checkpoint_tag}_left",
        merge_threshold=merge_threshold,
        node_threshold=node_threshold,
        nodes_per_layer=nodes_per_layer,
        bootstrap_iterations=bootstrap_iterations,
        min_cluster_fraction=float(components.config.get("discovery", {}).get("min_cluster_fraction", 0.005)),
        max_cluster_fraction=float(components.config.get("discovery", {}).get("max_cluster_fraction", 0.40)),
        min_cluster_size=int(components.config.get("discovery", {}).get("min_cluster_size", 20)),
        stability_threshold=float(components.config.get("discovery", {}).get("stability_threshold", 0.60)),
        random_seed=seed,
        node_panel=shared_panel,
        tracker=tracker,
    )
    right_artifact = _discover_motif_families_from_outputs(
        _subset_probe_outputs(outputs, right_rows),
        grid_size=int(components.config["tokenization"].get("grid_size", 4)),
        checkpoint_tag=f"{checkpoint_tag}_right",
        merge_threshold=merge_threshold,
        node_threshold=node_threshold,
        nodes_per_layer=nodes_per_layer,
        bootstrap_iterations=bootstrap_iterations,
        min_cluster_fraction=float(components.config.get("discovery", {}).get("min_cluster_fraction", 0.005)),
        max_cluster_fraction=float(components.config.get("discovery", {}).get("max_cluster_fraction", 0.40)),
        min_cluster_size=int(components.config.get("discovery", {}).get("min_cluster_size", 20)),
        stability_threshold=float(components.config.get("discovery", {}).get("stability_threshold", 0.60)),
        random_seed=seed + 1,
        node_panel=shared_panel,
        tracker=tracker,
    )
    matched_pairs = _greedy_match_motifs(left_artifact.get("motifs", []), right_artifact.get("motifs", []))
    result = {
        "experiment": MOTIF_TRANSFER_STABILITY_ID,
        "checkpoint_tag": checkpoint_tag,
        "summary": {
            "left_motif_count": int(len(left_artifact.get("motifs", []))),
            "right_motif_count": int(len(right_artifact.get("motifs", []))),
            "matched_motif_rate": float(
                len(matched_pairs) / max(1, min(len(left_artifact.get("motifs", [])), len(right_artifact.get("motifs", []))))
            ),
            "mean_image_set_stability": float(np.mean([item["image_set_jaccard"] for item in matched_pairs])) if matched_pairs else 0.0,
            "mean_support_node_stability": float(np.mean([item["support_node_f1"] for item in matched_pairs])) if matched_pairs else 0.0,
            "mean_exemplar_overlap": float(np.mean([item["exemplar_overlap"] for item in matched_pairs])) if matched_pairs else 0.0,
        },
        "matched_pairs": matched_pairs,
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


def _discover_motif_families_from_outputs(
    outputs: dict[str, torch.Tensor],
    *,
    grid_size: int,
    checkpoint_tag: str,
    merge_threshold: float,
    node_threshold: float,
    nodes_per_layer: int,
    bootstrap_iterations: int,
    min_cluster_fraction: float,
    max_cluster_fraction: float,
    min_cluster_size: int,
    stability_threshold: float,
    random_seed: int,
    node_panel: list[list[int]] | list[tuple[int, int]] | None,
    tracker: _ProgressTracker,
) -> dict:
    if node_panel is None:
        node_panel = _select_q_dispersion_node_panel(
            outputs["future_descriptors"],
            nodes_per_layer=nodes_per_layer,
            seed=random_seed,
            tracker=tracker,
        )
    normalized_node_panel = [[int(layer_idx), int(cell_idx)] for layer_idx, cell_idx in node_panel]
    descriptor_grid = F.normalize(outputs["z"], dim=-1).cpu().numpy()
    node_clusters = discover_node_clusters(
        descriptor_grid,
        outputs["indices"].cpu().numpy(),
        min_cluster_fraction=min_cluster_fraction,
        max_cluster_fraction=max_cluster_fraction,
        min_cluster_size=min_cluster_size,
        bootstrap_iterations=bootstrap_iterations,
        stability_threshold=stability_threshold,
        random_seed=random_seed,
        progress_callback=lambda **event: _relay_node_clustering_progress(tracker, **event),
        node_subset=normalized_node_panel,
    )
    motifs = _merge_node_clusters_into_motifs(
        node_clusters=node_clusters,
        descriptor_grid=descriptor_grid,
        dataset_indices=outputs["indices"].cpu().numpy(),
        labels=outputs["labels"].cpu().numpy(),
        merge_threshold=merge_threshold,
        node_threshold=node_threshold,
        tracker=tracker,
    )
    return {
        "metadata": {
            "checkpoint_tag": checkpoint_tag,
            "n_images": int(outputs["z"].shape[0]),
            "n_layers": int(outputs["z"].shape[1]),
            "n_cells": int(outputs["z"].shape[2]),
            "grid_size": int(grid_size),
            "random_seed": int(random_seed),
            "discovery_space": "z",
            "merge_threshold": float(merge_threshold),
            "node_threshold": float(node_threshold),
            "bootstrap_iterations": int(bootstrap_iterations),
        },
        "selected_node_panel": normalized_node_panel,
        "node_clusters": node_clusters,
        "motifs": motifs,
        "summary": {
            "n_node_clusters": int(len(node_clusters)),
            "n_motifs": int(len(motifs)),
            "mean_motif_stability": float(np.mean([motif["stability"]["mean_cluster_stability"] for motif in motifs])) if motifs else 0.0,
            "mean_supporting_layers": float(np.mean([len(motif["layer_support"]) for motif in motifs])) if motifs else 0.0,
            "mean_motif_size": float(np.mean([len(motif["image_set"]) for motif in motifs])) if motifs else 0.0,
        },
    }


def _relay_node_clustering_progress(tracker: _ProgressTracker, **event) -> None:
    if event.get("stage") == "node_clustering":
        tracker.emit(
            stage="motif_node_clustering",
            completed=event["completed"],
            total=event["total"],
            message=f"retained node clusters={event['n_node_clusters']}",
        )


def _merge_node_clusters_into_motifs(
    *,
    node_clusters: list[dict],
    descriptor_grid: np.ndarray,
    dataset_indices: np.ndarray,
    labels: np.ndarray,
    merge_threshold: float,
    node_threshold: float,
    tracker: _ProgressTracker,
) -> list[dict]:
    if not node_clusters:
        return []

    cluster_sets = [set(cluster["image_set"]) for cluster in node_clusters]
    adjacency = defaultdict(set)
    total_pairs = max((len(node_clusters) * (len(node_clusters) - 1)) // 2, 1)
    pair_idx = 0
    for left in range(len(node_clusters)):
        for right in range(left + 1, len(node_clusters)):
            pair_idx += 1
            if jaccard(cluster_sets[left], cluster_sets[right]) >= merge_threshold:
                adjacency[left].add(right)
                adjacency[right].add(left)
            tracker.emit(
                stage="motif_merge_graph",
                completed=pair_idx,
                total=total_pairs,
                message="building node-cluster family graph",
            )

    components = _graph_components(list(range(len(node_clusters))), adjacency)
    row_lookup = {int(dataset_idx): row_idx for row_idx, dataset_idx in enumerate(dataset_indices.tolist())}
    motifs = []
    for motif_idx, component in enumerate(components, start=1):
        family_clusters = [node_clusters[idx] for idx in component]
        if len(family_clusters) < 2:
            continue
        medoid_idx = max(
            component,
            key=lambda idx: np.mean([jaccard(cluster_sets[idx], cluster_sets[other]) for other in component]),
        )
        canonical_set = set(node_clusters[medoid_idx]["image_set"])
        active_cluster_candidates = [
            cluster for cluster in family_clusters
            if jaccard(set(cluster["image_set"]), canonical_set) >= node_threshold
        ]
        active_nodes = sorted({tuple(cluster["node"]) for cluster in active_cluster_candidates})
        layer_support = sorted({int(layer_idx) for layer_idx, _ in active_nodes})
        if len(active_cluster_candidates) < 2 or len(layer_support) < 2:
            continue
        member_rows = np.array(sorted(row_lookup[idx] for idx in canonical_set if idx in row_lookup), dtype=int)
        if member_rows.size == 0:
            continue
        centroids = {}
        thresholds = {}
        for layer_idx, cell_idx in active_nodes:
            vectors = descriptor_grid[member_rows, layer_idx, cell_idx, :]
            centroid = vectors.mean(axis=0)
            centroid = centroid / np.clip(np.linalg.norm(centroid), 1.0e-8, None)
            similarities = vectors @ centroid
            centroids[f"{layer_idx}:{cell_idx}"] = centroid.tolist()
            thresholds[f"{layer_idx}:{cell_idx}"] = float(np.percentile(similarities, 10.0))
        member_labels = labels[member_rows]
        counts = np.bincount(member_labels.astype(int)) if member_labels.size else np.array([], dtype=int)
        dominant_class = int(np.argmax(counts)) if counts.size else None
        purity = float(counts.max() / max(member_labels.size, 1)) if counts.size else 0.0
        motifs.append(
            {
                "id": int(len(motifs)),
                "image_set": sorted(int(idx) for idx in canonical_set),
                "member_row_indices": member_rows.tolist(),
                "representative_node": list(node_clusters[medoid_idx]["node"]),
                "active_nodes": [list(node) for node in active_nodes],
                "layer_support": [int(layer_idx) for layer_idx in layer_support],
                "centroids": centroids,
                "thresholds": thresholds,
                "stability": {
                    "mean_cluster_stability": float(np.mean([cluster["stability"] for cluster in family_clusters])),
                    "n_node_clusters": int(len(family_clusters)),
                },
                "purity": {
                    "dominant_class": dominant_class,
                    "fraction": purity,
                },
            }
        )
        tracker.emit(
            stage="motif_family_retention",
            completed=motif_idx,
            total=len(components),
            message="assembling retained motif families",
        )
    return motifs


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


def _iter_nodes(n_layers: int, n_cells: int):
    for layer_idx in range(n_layers):
        for cell_idx in range(n_cells):
            yield layer_idx, cell_idx


def _maybe_write_json(payload: dict, output_path: str | Path | None) -> None:
    if output_path is None:
        return
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _rank_motifs(motifs: list[dict]) -> list[dict]:
    return sorted(
        motifs,
        key=lambda motif: (
            -float(motif.get("stability", {}).get("mean_cluster_stability", 0.0)),
            -int(len(motif.get("layer_support", []))),
            -int(len(motif.get("image_set", []))),
            int(motif["id"]),
        ),
    )


def _member_rows_for_motif(motif: dict, dataset_indices: torch.Tensor) -> np.ndarray:
    if motif.get("member_row_indices"):
        return np.asarray(motif["member_row_indices"], dtype=np.int64)
    row_lookup = {int(idx): row for row, idx in enumerate(dataset_indices.cpu().tolist())}
    return np.asarray([row_lookup[idx] for idx in motif.get("image_set", []) if idx in row_lookup], dtype=np.int64)


def _motif_scores(motif: dict, z: torch.Tensor) -> torch.Tensor:
    active_nodes = [tuple(node) for node in motif.get("active_nodes", [])]
    if not active_nodes:
        return torch.zeros(z.shape[0], dtype=z.dtype)
    device = z.device
    dtype = z.dtype
    layer_indices = torch.tensor([node[0] for node in active_nodes], dtype=torch.long, device=device)
    cell_indices = torch.tensor([node[1] for node in active_nodes], dtype=torch.long, device=device)
    centroid_matrix = torch.stack(
        [
            torch.tensor(motif["centroids"][f"{layer_idx}:{cell_idx}"], dtype=dtype, device=device)
            for layer_idx, cell_idx in active_nodes
        ],
        dim=0,
    )
    node_descriptors = z[:, layer_indices, cell_indices, :]
    return (node_descriptors * centroid_matrix.unsqueeze(0)).sum(dim=-1).mean(dim=1)


def _label_histogram(labels: np.ndarray) -> dict[str, int]:
    if labels.size == 0:
        return {}
    counts = np.bincount(labels.astype(int))
    return {str(idx): int(count) for idx, count in enumerate(counts.tolist()) if count > 0}


def _group_nodes_by_layer(active_nodes: list[tuple[int, int]]) -> dict[int, list[tuple[int, int]]]:
    grouped: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for layer_idx, cell_idx in active_nodes:
        grouped[int(layer_idx)].append((int(layer_idx), int(cell_idx)))
    return grouped


def _per_layer_support(active_nodes: list[tuple[int, int]]) -> dict[int, int]:
    return {layer_idx: len(nodes) for layer_idx, nodes in _group_nodes_by_layer(active_nodes).items()}


def _margin(logits: torch.Tensor) -> torch.Tensor:
    top2 = torch.topk(logits, k=2, dim=1).values
    return top2[:, 0] - top2[:, 1]


def _matched_rows(
    *,
    member_rows: torch.Tensor,
    member_classes: torch.Tensor,
    member_margins: torch.Tensor,
    predicted_classes: torch.Tensor,
    margins: torch.Tensor,
) -> torch.Tensor:
    available = torch.ones_like(predicted_classes, dtype=torch.bool)
    available[member_rows] = False
    chosen = []
    for member_class, member_margin in zip(member_classes.tolist(), member_margins.tolist()):
        class_mask = available & (predicted_classes == member_class)
        candidates = torch.nonzero(class_mask, as_tuple=False).flatten()
        if candidates.numel() == 0:
            candidates = torch.nonzero(available, as_tuple=False).flatten()
        if candidates.numel() == 0:
            continue
        distances = (margins[candidates] - member_margin).abs()
        best = candidates[int(torch.argmin(distances).item())]
        chosen.append(int(best.item()))
        available[best] = False
    return torch.tensor(chosen, dtype=torch.long)


def _graph_components(nodes, adjacency) -> list[list]:
    if not nodes:
        return []
    components = []
    seen = set()
    for node in nodes:
        if node in seen:
            continue
        queue = deque([node])
        component = []
        seen.add(node)
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in adjacency.get(current, set()):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        components.append(sorted(component))
    return sorted(components, key=lambda component: (-len(component), component))


def _motif_match_features(left_motif: dict, right_motif: dict) -> dict:
    left_nodes = left_motif.get("active_nodes", [])
    right_nodes = right_motif.get("active_nodes", [])
    rep_match = int(left_motif.get("representative_node") == right_motif.get("representative_node"))
    shared_rep_key = None
    if left_motif.get("representative_node") == right_motif.get("representative_node"):
        layer_idx, cell_idx = left_motif["representative_node"]
        shared_rep_key = f"{layer_idx}:{cell_idx}"
    centroid_cosine = 0.0
    if shared_rep_key is not None:
        left_centroid = np.asarray(left_motif["centroids"][shared_rep_key], dtype=np.float64)
        right_centroid = np.asarray(right_motif["centroids"][shared_rep_key], dtype=np.float64)
        centroid_cosine = float(np.dot(left_centroid, right_centroid))
    exemplar_overlap = 0.0
    if left_motif.get("image_set") and right_motif.get("image_set"):
        exemplar_overlap = float(jaccard(set(left_motif["image_set"][:20]), set(right_motif["image_set"][:20])))
    return {
        "phase_b_motif_id": int(left_motif["id"]),
        "phase_c_motif_id": int(right_motif["id"]),
        "image_set_jaccard": float(jaccard(set(left_motif["image_set"]), set(right_motif["image_set"]))),
        "support_node_f1": float(_node_f1(left_nodes, right_nodes)),
        "representative_node_match": rep_match,
        "centroid_cosine": float(centroid_cosine),
        "exemplar_overlap": exemplar_overlap,
    }


def _greedy_match_motifs(left_motifs: list[dict], right_motifs: list[dict]) -> list[dict]:
    right_by_id = {int(motif["id"]): motif for motif in right_motifs}
    available_right = set(right_by_id)
    matched = []
    for left_motif in _rank_motifs(left_motifs):
        candidates = [
            _motif_match_features(left_motif, right_by_id[right_id])
            for right_id in sorted(available_right)
        ]
        if not candidates:
            continue
        best = max(
            candidates,
            key=lambda item: (
                item["image_set_jaccard"],
                item["representative_node_match"],
                item["centroid_cosine"],
                -item["phase_c_motif_id"],
            ),
        )
        if best["image_set_jaccard"] <= 0.0 and best["support_node_f1"] <= 0.0 and best["representative_node_match"] == 0:
            continue
        available_right.discard(best["phase_c_motif_id"])
        matched.append(best)
    return matched


def _motif_topology_metrics(active_nodes: list[list[int]] | list[tuple[int, int]], *, grid_size: int) -> dict:
    nodes = sorted({(int(node[0]), int(node[1])) for node in active_nodes})
    node_set = set(nodes)
    same_layer_adjacent_edges = 0
    same_cell_depth_edges = 0
    for layer_idx, cell_idx in nodes:
        row = cell_idx // grid_size
        col = cell_idx % grid_size
        if (layer_idx + 1, cell_idx) in node_set:
            same_cell_depth_edges += 1
        for dr, dc in ((1, 0), (0, 1)):
            nr = row + dr
            nc = col + dc
            if 0 <= nr < grid_size and 0 <= nc < grid_size:
                neighbor = (layer_idx, (nr * grid_size) + nc)
                if neighbor in node_set:
                    same_layer_adjacent_edges += 1
    max_depth_run = 0
    best_depth_cell = None
    layers_by_cell: dict[int, list[int]] = defaultdict(list)
    for layer_idx, cell_idx in nodes:
        layers_by_cell[cell_idx].append(layer_idx)
    for cell_idx, layers in layers_by_cell.items():
        ordered = sorted(set(layers))
        current = 1
        best = 1 if ordered else 0
        for left, right in zip(ordered, ordered[1:]):
            if right == left + 1:
                current += 1
                best = max(best, current)
            else:
                current = 1
        if best > max_depth_run:
            max_depth_run = best
            best_depth_cell = int(cell_idx)
    adjacency = defaultdict(set)
    for node in nodes:
        for neighbor in _motif_neighbors(node, grid_size=grid_size):
            if neighbor in node_set:
                adjacency[node].add(neighbor)
                adjacency[neighbor].add(node)
    components = _graph_components(nodes, adjacency)
    largest_component = max((len(component) for component in components), default=0)
    return {
        "unique_retained_nodes": int(len(nodes)),
        "same_layer_adjacent_edges": int(same_layer_adjacent_edges),
        "same_cell_depth_edges": int(same_cell_depth_edges),
        "max_depth_run": int(max_depth_run),
        "best_depth_cell": int(best_depth_cell) if best_depth_cell is not None else None,
        "largest_connected_component": int(largest_component),
        "n_connected_components": int(len(components)),
    }


def _motif_neighbors(node: tuple[int, int], *, grid_size: int) -> list[tuple[int, int]]:
    layer_idx, cell_idx = node
    row = cell_idx // grid_size
    col = cell_idx % grid_size
    neighbors = [(layer_idx - 1, cell_idx), (layer_idx + 1, cell_idx)]
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr = row + dr
        nc = col + dc
        if 0 <= nr < grid_size and 0 <= nc < grid_size:
            neighbors.append((layer_idx, (nr * grid_size) + nc))
    return neighbors


def _classify_topology(metrics: dict) -> str:
    if metrics["max_depth_run"] >= 3 or metrics["same_cell_depth_edges"] >= 2:
        return "depth_like"
    if metrics["same_layer_adjacent_edges"] >= 1 and metrics["largest_connected_component"] >= 2:
        return "spatial"
    return "fragmented"


def _subset_probe_outputs(outputs: dict[str, torch.Tensor], rows: np.ndarray) -> dict[str, torch.Tensor]:
    return {
        "z": outputs["z"][rows],
        "local_features": [layer[rows] for layer in outputs["local_features"]],
        "future_descriptors": outputs["future_descriptors"][rows],
        "labels": outputs["labels"][rows],
        "indices": outputs["indices"][rows],
    }


def _node_f1(left_nodes: list[list[int]] | list[tuple[int, int]], right_nodes: list[list[int]] | list[tuple[int, int]]) -> float:
    left_set = {tuple(node) for node in left_nodes}
    right_set = {tuple(node) for node in right_nodes}
    if not left_set and not right_set:
        return 1.0
    if not left_set or not right_set:
        return 0.0
    true_positive = len(left_set & right_set)
    precision = true_positive / len(right_set)
    recall = true_positive / len(left_set)
    if precision + recall == 0.0:
        return 0.0
    return float((2.0 * precision * recall) / (precision + recall))
