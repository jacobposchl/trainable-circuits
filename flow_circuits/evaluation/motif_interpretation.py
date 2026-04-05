from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from flow_circuits.evaluation.motif_validation import (
    _ProgressTracker,
    _classify_topology,
    _label_histogram,
    _maybe_write_json,
    _member_rows_for_motif,
    _motif_scores,
    _motif_topology_metrics,
    _rank_motifs,
)


MOTIF_SEMANTIC_REPORT_ID = "motif_semantic_report"
MOTIF_SPATIAL_FOOTPRINT_ID = "motif_spatial_footprint"
MOTIF_BORDERLINE_MEMBER_ID = "motif_borderline_members"

MOTIF_INTERPRETATION_EXPERIMENT_IDS = [
    MOTIF_SEMANTIC_REPORT_ID,
    MOTIF_SPATIAL_FOOTPRINT_ID,
    MOTIF_BORDERLINE_MEMBER_ID,
]

CIFAR10_CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def run_motif_semantic_report_experiment(
    motif_artifact: dict,
    outputs: dict[str, torch.Tensor],
    *,
    checkpoint_tag: str,
    motif_ids: str | list[int] = "top",
    topk: int = 8,
    exemplar_count: int = 9,
    borderline_count: int = 6,
    near_miss_count: int = 6,
    class_names: list[str] | None = None,
    output_path: str | Path | None = None,
    progress_callback=None,
) -> dict:
    tracker = _ProgressTracker(
        experiment=MOTIF_SEMANTIC_REPORT_ID,
        checkpoint_tag=checkpoint_tag,
        progress_callback=progress_callback,
    )
    class_names = class_names or CIFAR10_CLASS_NAMES
    selected_motifs = _select_motifs(motif_artifact, motif_ids=motif_ids, topk=topk)
    z = F.normalize(outputs["z"], dim=-1)
    labels = outputs["labels"].cpu().numpy()
    dataset_indices = outputs["indices"]
    grid_size = int(motif_artifact.get("metadata", {}).get("grid_size", 4))

    motif_cards = []
    exemplar_sets = []
    tag_counts: dict[str, int] = {}
    topology_counts: dict[str, int] = {}

    for motif_idx, motif in enumerate(selected_motifs, start=1):
        member_rows = _member_rows_for_motif(motif, dataset_indices)
        scores = _motif_scores(motif, z).detach().cpu()
        selection = _select_exemplar_sets(
            scores=scores,
            member_rows=member_rows,
            labels=outputs["labels"].cpu(),
            dataset_indices=dataset_indices.cpu(),
            exemplar_count=exemplar_count,
            borderline_count=borderline_count,
            near_miss_count=near_miss_count,
        )
        topology_metrics = _motif_topology_metrics(motif.get("active_nodes", []), grid_size=grid_size)
        topology_type = _classify_topology(topology_metrics)
        active_nodes = [list(node) for node in motif.get("active_nodes", [])]
        purity_fraction = float(motif.get("purity", {}).get("fraction", 0.0))
        dominant_class = motif.get("purity", {}).get("dominant_class")
        if dominant_class is None and member_rows.size:
            dominant_class = int(np.bincount(labels[member_rows].astype(int)).argmax())
        dominant_class_name = _class_name(dominant_class, class_names)
        class_hist = _label_histogram(labels[member_rows]) if member_rows.size else {}
        heuristic_tags = _heuristic_tags(
            active_nodes=active_nodes,
            purity_fraction=purity_fraction,
            topology_type=topology_type,
            grid_size=grid_size,
        )
        for tag in heuristic_tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        topology_counts[topology_type] = topology_counts.get(topology_type, 0) + 1
        supporting_layers = sorted({int(node[0]) for node in active_nodes})
        depth_span = (
            int(max(supporting_layers) - min(supporting_layers) + 1)
            if supporting_layers
            else 0
        )
        motif_cards.append(
            {
                "motif_id": int(motif["id"]),
                "checkpoint_tag": checkpoint_tag,
                "motif_size": int(len(motif.get("image_set", []))),
                "mean_cluster_stability": float(motif.get("stability", {}).get("mean_cluster_stability", 0.0)),
                "supporting_layers": int(len(supporting_layers)),
                "layer_support": supporting_layers,
                "depth_span": depth_span,
                "topology_type": topology_type,
                "topology_metrics": topology_metrics,
                "dominant_class": None if dominant_class is None else int(dominant_class),
                "dominant_class_name": dominant_class_name,
                "class_purity": purity_fraction,
                "representative_node": [int(value) for value in motif.get("representative_node", [])],
                "active_nodes": active_nodes,
                "class_histogram": _class_histogram_with_names(class_hist, class_names),
                "heuristic_tags": heuristic_tags,
                "top_exemplar_image_indices": selection["top_exemplars"]["dataset_indices"],
                "borderline_member_image_indices": selection["borderline_members"]["dataset_indices"],
                "near_miss_image_indices": selection["near_misses"]["dataset_indices"],
            }
        )
        exemplar_sets.append(
            {
                "motif_id": int(motif["id"]),
                **selection,
            }
        )
        tracker.emit(
            stage="semantic_cards",
            completed=motif_idx,
            total=len(selected_motifs),
            message="summarizing motif semantics",
        )

    result = {
        "experiment": MOTIF_SEMANTIC_REPORT_ID,
        "checkpoint_tag": checkpoint_tag,
        "selection": {
            "motif_ids": motif_ids if motif_ids != "top" else "top",
            "topk": int(topk),
            "n_selected_motifs": int(len(selected_motifs)),
        },
        "summary": {
            "n_ranked_motifs": int(len(motif_cards)),
            "mean_purity": float(np.mean([card["class_purity"] for card in motif_cards])) if motif_cards else 0.0,
            "mean_supporting_layers": float(np.mean([card["supporting_layers"] for card in motif_cards])) if motif_cards else 0.0,
            "mean_depth_span": float(np.mean([card["depth_span"] for card in motif_cards])) if motif_cards else 0.0,
            "topology_counts": topology_counts,
            "tag_counts": tag_counts,
        },
        "motif_cards": motif_cards,
        "exemplar_sets": exemplar_sets,
    }
    _maybe_write_json(result, output_path)
    return result


def run_motif_spatial_footprint_experiment(
    motif_artifact: dict,
    outputs: dict[str, torch.Tensor],
    *,
    checkpoint_tag: str,
    semantic_report: dict,
    image_size: int = 32,
    output_path: str | Path | None = None,
    progress_callback=None,
) -> dict:
    tracker = _ProgressTracker(
        experiment=MOTIF_SPATIAL_FOOTPRINT_ID,
        checkpoint_tag=checkpoint_tag,
        progress_callback=progress_callback,
    )
    motif_lookup = {
        int(motif["id"]): motif
        for motif in motif_artifact.get("motifs", [])
    }
    grid_size = int(motif_artifact.get("metadata", {}).get("grid_size", 4))
    overlay_specs = []
    crop_specs = []
    exemplar_sets = {
        int(item["motif_id"]): item
        for item in semantic_report.get("exemplar_sets", [])
    }

    motif_cards = semantic_report.get("motif_cards", [])
    for motif_idx, card in enumerate(motif_cards, start=1):
        motif_id = int(card["motif_id"])
        motif = motif_lookup[motif_id]
        exemplars = exemplar_sets.get(motif_id, {})
        top_rows = exemplars.get("top_exemplars", {}).get("row_indices", [])
        active_nodes = [tuple(node) for node in motif.get("active_nodes", [])]
        representative_node = tuple(motif.get("representative_node", [])) if motif.get("representative_node") else None
        active_boxes = [_cell_box(cell_idx, grid_size=grid_size, image_size=image_size) for _, cell_idx in active_nodes]
        unique_active_boxes = _dedupe_boxes(active_boxes)
        representative_box = (
            _cell_box(representative_node[1], grid_size=grid_size, image_size=image_size)
            if representative_node is not None
            else None
        )
        overlay_images = []
        for row_index in top_rows[: min(len(top_rows), 6)]:
            overlay_images.append(
                {
                    "row_index": int(row_index),
                    "dataset_index": int(outputs["indices"][row_index].item()),
                    "label": int(outputs["labels"][row_index].item()),
                    "boxes": [
                        {
                            "kind": "representative",
                            **representative_box,
                        }
                    ] if representative_box is not None else []
                    + [
                        {
                            "kind": "active",
                            **box,
                        }
                        for box in unique_active_boxes
                    ],
                }
            )
        overlay_specs.append(
            {
                "motif_id": motif_id,
                "representative_box": representative_box,
                "active_boxes": unique_active_boxes,
                "images": overlay_images,
            }
        )
        crop_specs.append(
            {
                "motif_id": motif_id,
                "representative_crop": representative_box,
                "union_crop": _union_box(unique_active_boxes),
                "per_layer_crops": _per_layer_crop_specs(active_nodes, grid_size=grid_size, image_size=image_size),
            }
        )
        tracker.emit(
            stage="spatial_footprints",
            completed=motif_idx,
            total=len(motif_cards),
            message="building overlay and crop specs",
        )

    result = {
        "experiment": MOTIF_SPATIAL_FOOTPRINT_ID,
        "checkpoint_tag": checkpoint_tag,
        "summary": {
            "n_motifs": int(len(motif_cards)),
            "image_size": int(image_size),
            "grid_size": grid_size,
        },
        "overlay_specs": overlay_specs,
        "crop_specs": crop_specs,
    }
    _maybe_write_json(result, output_path)
    return result


def run_motif_borderline_member_experiment(
    motif_artifact: dict,
    outputs: dict[str, torch.Tensor],
    *,
    checkpoint_tag: str,
    motif_ids: str | list[int] = "top",
    topk: int = 8,
    borderline_count: int = 6,
    near_miss_count: int = 6,
    output_path: str | Path | None = None,
    progress_callback=None,
) -> dict:
    tracker = _ProgressTracker(
        experiment=MOTIF_BORDERLINE_MEMBER_ID,
        checkpoint_tag=checkpoint_tag,
        progress_callback=progress_callback,
    )
    selected_motifs = _select_motifs(motif_artifact, motif_ids=motif_ids, topk=topk)
    z = F.normalize(outputs["z"], dim=-1)
    rows = []
    for motif_idx, motif in enumerate(selected_motifs, start=1):
        member_rows = _member_rows_for_motif(motif, outputs["indices"])
        scores = _motif_scores(motif, z).detach().cpu()
        selection = _select_exemplar_sets(
            scores=scores,
            member_rows=member_rows,
            labels=outputs["labels"].cpu(),
            dataset_indices=outputs["indices"].cpu(),
            exemplar_count=0,
            borderline_count=borderline_count,
            near_miss_count=near_miss_count,
        )
        rows.append(
            {
                "motif_id": int(motif["id"]),
                "borderline_members": selection["borderline_members"],
                "near_misses": selection["near_misses"],
            }
        )
        tracker.emit(
            stage="borderline_members",
            completed=motif_idx,
            total=len(selected_motifs),
            message="collecting borderline and near-miss sets",
        )
    result = {
        "experiment": MOTIF_BORDERLINE_MEMBER_ID,
        "checkpoint_tag": checkpoint_tag,
        "selection": {
            "motif_ids": motif_ids if motif_ids != "top" else "top",
            "topk": int(topk),
        },
        "motif_rows": rows,
    }
    _maybe_write_json(result, output_path)
    return result


def _select_motifs(motif_artifact: dict, *, motif_ids: str | list[int], topk: int) -> list[dict]:
    ranked = _rank_motifs(motif_artifact.get("motifs", []))
    if motif_ids == "top":
        return ranked[: int(topk)]
    motif_lookup = {int(motif["id"]): motif for motif in ranked}
    selected = [motif_lookup[int(motif_id)] for motif_id in motif_ids if int(motif_id) in motif_lookup]
    return selected


def _select_exemplar_sets(
    *,
    scores: torch.Tensor,
    member_rows: np.ndarray,
    labels: torch.Tensor,
    dataset_indices: torch.Tensor,
    exemplar_count: int,
    borderline_count: int,
    near_miss_count: int,
) -> dict:
    member_tensor = torch.tensor(member_rows.tolist(), dtype=torch.long) if member_rows.size else torch.zeros(0, dtype=torch.long)
    all_rows = torch.arange(scores.shape[0], dtype=torch.long)
    nonmember_mask = torch.ones(scores.shape[0], dtype=torch.bool)
    if member_tensor.numel():
        nonmember_mask[member_tensor] = False
    nonmember_rows = all_rows[nonmember_mask]

    top_exemplar_rows = _ordered_slice(member_tensor, scores[member_tensor] if member_tensor.numel() else torch.zeros(0), count=exemplar_count, descending=True)
    borderline_rows = _ordered_slice(member_tensor, scores[member_tensor] if member_tensor.numel() else torch.zeros(0), count=borderline_count, descending=False)
    near_miss_rows = _ordered_slice(nonmember_rows, scores[nonmember_rows] if nonmember_rows.numel() else torch.zeros(0), count=near_miss_count, descending=True)

    return {
        "top_exemplars": _selection_payload(top_exemplar_rows, scores, labels, dataset_indices),
        "borderline_members": _selection_payload(borderline_rows, scores, labels, dataset_indices),
        "near_misses": _selection_payload(near_miss_rows, scores, labels, dataset_indices),
    }


def _ordered_slice(rows: torch.Tensor, row_scores: torch.Tensor, *, count: int, descending: bool) -> torch.Tensor:
    if rows.numel() == 0 or count <= 0:
        return torch.zeros(0, dtype=torch.long)
    order = torch.argsort(row_scores, descending=descending)
    return rows[order[: min(int(count), rows.numel())]]


def _selection_payload(rows: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor, dataset_indices: torch.Tensor) -> dict:
    if rows.numel() == 0:
        return {
            "row_indices": [],
            "dataset_indices": [],
            "labels": [],
            "scores": [],
        }
    return {
        "row_indices": [int(value) for value in rows.tolist()],
        "dataset_indices": [int(value) for value in dataset_indices[rows].tolist()],
        "labels": [int(value) for value in labels[rows].tolist()],
        "scores": [float(value) for value in scores[rows].tolist()],
    }


def _heuristic_tags(
    *,
    active_nodes: list[list[int]],
    purity_fraction: float,
    topology_type: str,
    grid_size: int,
) -> list[str]:
    tags = [topology_type]
    unique_cells = sorted({int(node[1]) for node in active_nodes})
    border_fraction = _border_fraction(unique_cells, grid_size=grid_size)

    if purity_fraction >= 0.9:
        tags.append("class_specific")
    if purity_fraction < 0.7:
        tags.append("mixed_semantic")
    if purity_fraction >= 0.95:
        tags.append("high_purity")
    if len(unique_cells) <= max(2, math.ceil(len(active_nodes) * 0.5)):
        tags.append("spatially_local")
    if border_fraction >= 0.6:
        tags.append("background_like_candidate")
    elif topology_type in {"depth_like", "spatial"} and purity_fraction >= 0.75:
        tags.append("object_part_candidate")
    return sorted(set(tags))


def _border_fraction(cells: list[int], *, grid_size: int) -> float:
    if not cells:
        return 0.0
    border = 0
    for cell_idx in cells:
        row = cell_idx // grid_size
        col = cell_idx % grid_size
        if row in {0, grid_size - 1} or col in {0, grid_size - 1}:
            border += 1
    return float(border / len(cells))


def _class_name(label: int | None, class_names: list[str]) -> str | None:
    if label is None:
        return None
    if 0 <= int(label) < len(class_names):
        return class_names[int(label)]
    return str(label)


def _class_histogram_with_names(histogram: dict[str, int], class_names: list[str]) -> list[dict]:
    rows = []
    for key, value in sorted(histogram.items(), key=lambda item: int(item[0])):
        label = int(key)
        rows.append(
            {
                "label": label,
                "label_name": _class_name(label, class_names),
                "count": int(value),
            }
        )
    return rows


def _cell_box(cell_idx: int, *, grid_size: int, image_size: int) -> dict:
    cell_size = image_size // grid_size
    row = int(cell_idx) // grid_size
    col = int(cell_idx) % grid_size
    x0 = col * cell_size
    y0 = row * cell_size
    return {
        "cell_idx": int(cell_idx),
        "x0": int(x0),
        "y0": int(y0),
        "x1": int(x0 + cell_size),
        "y1": int(y0 + cell_size),
    }


def _dedupe_boxes(boxes: list[dict]) -> list[dict]:
    seen = set()
    deduped = []
    for box in boxes:
        key = (box["cell_idx"], box["x0"], box["y0"], box["x1"], box["y1"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(box)
    return deduped


def _union_box(boxes: list[dict]) -> dict | None:
    if not boxes:
        return None
    return {
        "x0": int(min(box["x0"] for box in boxes)),
        "y0": int(min(box["y0"] for box in boxes)),
        "x1": int(max(box["x1"] for box in boxes)),
        "y1": int(max(box["y1"] for box in boxes)),
    }


def _per_layer_crop_specs(
    active_nodes: list[tuple[int, int]],
    *,
    grid_size: int,
    image_size: int,
) -> list[dict]:
    by_layer: dict[int, list[dict]] = {}
    for layer_idx, cell_idx in active_nodes:
        by_layer.setdefault(int(layer_idx), []).append(
            _cell_box(cell_idx, grid_size=grid_size, image_size=image_size)
        )
    specs = []
    for layer_idx in sorted(by_layer):
        specs.append(
            {
                "layer_idx": int(layer_idx),
                "union_crop": _union_box(_dedupe_boxes(by_layer[layer_idx])),
            }
        )
    return specs
