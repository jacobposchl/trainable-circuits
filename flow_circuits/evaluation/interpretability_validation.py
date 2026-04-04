from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F

from flow_circuits.evaluation.motif_validation import (
    _ProgressTracker,
    _label_histogram,
    _member_rows_for_motif,
    _maybe_write_json,
    _motif_scores,
    _rank_motifs,
)
from flow_circuits.interventions import ResidualPatchAblator, assign_circuit_members, run_circuit_interventions
from flow_circuits.training import LoadedFlowComponents, collect_interpretability_outputs


MOTIF_REPORTS_ID = "motif_reports"
PHASE_COMPARISON_ID = "phase_comparison"
INTERVENTION_CASES_ID = "intervention_cases"
CLASS_PROBE_SUITE_ID = "class_probe_suite"
CONFUSION_ANALYSIS_ID = "confusion_analysis"
ERROR_ANALYSIS_ID = "error_analysis"

NB05_EXPERIMENT_IDS = [
    MOTIF_REPORTS_ID,
    PHASE_COMPARISON_ID,
    INTERVENTION_CASES_ID,
    CLASS_PROBE_SUITE_ID,
    CONFUSION_ANALYSIS_ID,
    ERROR_ANALYSIS_ID,
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


def run_motif_visual_report_experiment(
    components: LoadedFlowComponents,
    loader,
    *,
    device: torch.device,
    checkpoint_tag: str,
    motif_artifact: dict,
    max_images: int,
    topk: int,
    min_visual_budget: int = 4,
    exemplar_count: int = 9,
    output_path: str | Path | None = None,
    progress_callback=None,
) -> dict:
    tracker = _ProgressTracker(
        experiment=MOTIF_REPORTS_ID,
        checkpoint_tag=checkpoint_tag,
        progress_callback=progress_callback,
    )
    outputs = collect_interpretability_outputs(
        components,
        loader,
        device=device,
        max_images=max_images,
        progress_callback=lambda **event: tracker.emit(
            stage="data_collection",
            completed=event["batch_idx"],
            total=event.get("total_batches"),
            message="collecting images, z, and backbone logits",
        ),
    )
    z = F.normalize(outputs["z"], dim=-1)
    labels = outputs["labels"]
    image_shape = list(outputs["images"].shape[-2:])
    grid_size = int(motif_artifact["metadata"]["grid_size"])

    ranked_motifs = _rank_motifs(motif_artifact.get("motifs", []))[: int(topk)]
    selected_objects: list[dict] = []
    for motif in ranked_motifs:
        selected_objects.append(
            _build_motif_visual_object(
                motif,
                z=z,
                labels=labels,
                dataset_indices=outputs["indices"],
                checkpoint_tag=checkpoint_tag,
                image_shape=image_shape,
                grid_size=grid_size,
                exemplar_count=exemplar_count,
            )
        )

    if len(selected_objects) < int(min_visual_budget):
        needed = int(min_visual_budget) - len(selected_objects)
        existing_cluster_keys = {
            tuple(obj["active_nodes"][0])
            for obj in selected_objects
            if obj["object_type"] == "node_cluster" and obj["active_nodes"]
        }
        for cluster in _rank_node_clusters(motif_artifact.get("node_clusters", [])):
            if needed <= 0:
                break
            node_key = tuple(cluster["node"])
            if node_key in existing_cluster_keys:
                continue
            selected_objects.append(
                _build_node_cluster_visual_object(
                    cluster,
                    z=z,
                    labels=labels,
                    dataset_indices=outputs["indices"],
                    checkpoint_tag=checkpoint_tag,
                    image_shape=image_shape,
                    grid_size=grid_size,
                    exemplar_count=exemplar_count,
                )
            )
            existing_cluster_keys.add(node_key)
            needed -= 1

    for idx, _ in enumerate(selected_objects, start=1):
        tracker.emit(
            stage="visual_object_packaging",
            completed=idx,
            total=len(selected_objects),
            message="building motif visual cards",
        )

    result = {
        "experiment": MOTIF_REPORTS_ID,
        "checkpoint_tag": checkpoint_tag,
        "metadata": {
            "max_images": int(outputs["images"].shape[0]),
            "image_shape": image_shape,
            "grid_size": grid_size,
            "n_available_motifs": int(len(motif_artifact.get("motifs", []))),
            "n_available_node_clusters": int(len(motif_artifact.get("node_clusters", []))),
        },
        "summary": {
            "n_visual_objects": int(len(selected_objects)),
            "n_motif_objects": int(sum(1 for obj in selected_objects if obj["object_type"] == "motif")),
            "n_node_cluster_objects": int(sum(1 for obj in selected_objects if obj["object_type"] == "node_cluster")),
            "mean_class_purity": float(np.mean([obj["class_purity"] for obj in selected_objects])) if selected_objects else 0.0,
            "mean_supporting_layers": float(np.mean([len(obj["supporting_layers"]) for obj in selected_objects])) if selected_objects else 0.0,
        },
        "visual_objects": selected_objects,
    }
    _maybe_write_json(result, output_path)
    return result


def run_phase_motif_comparison_experiment(
    phase_b_report: dict,
    phase_c_report: dict,
    *,
    phase_match_artifact: dict | None = None,
    output_path: str | Path | None = None,
) -> dict:
    phase_b_by_id = {
        (row["object_type"], str(row["id"])): row
        for row in phase_b_report.get("visual_objects", [])
    }
    phase_c_by_id = {
        (row["object_type"], str(row["id"])): row
        for row in phase_c_report.get("visual_objects", [])
    }

    comparisons = []
    if phase_match_artifact and phase_match_artifact.get("matched_pairs"):
        for pair in phase_match_artifact["matched_pairs"]:
            b_key = ("motif", str(int(pair["phase_b_motif_id"])))
            c_key = ("motif", str(int(pair["phase_c_motif_id"])))
            if b_key not in phase_b_by_id or c_key not in phase_c_by_id:
                continue
            comparisons.append(
                _comparison_row(
                    phase_b_by_id[b_key],
                    phase_c_by_id[c_key],
                    match_quality=float(pair["image_set_jaccard"]),
                    comparison_type="matched_motif_pair",
                )
            )
    if not comparisons:
        limit = min(len(phase_b_report.get("visual_objects", [])), len(phase_c_report.get("visual_objects", [])))
        for idx in range(limit):
            comparisons.append(
                _comparison_row(
                    phase_b_report["visual_objects"][idx],
                    phase_c_report["visual_objects"][idx],
                    match_quality=None,
                    comparison_type="ranked_visual_pair",
                )
            )
    result = {
        "experiment": PHASE_COMPARISON_ID,
        "summary": {
            "n_pairs": int(len(comparisons)),
            "mean_phase_b_purity": float(np.mean([row["phase_b"]["class_purity"] for row in comparisons])) if comparisons else 0.0,
            "mean_phase_c_purity": float(np.mean([row["phase_c"]["class_purity"] for row in comparisons])) if comparisons else 0.0,
            "mean_phase_b_layers": float(np.mean([len(row["phase_b"]["supporting_layers"]) for row in comparisons])) if comparisons else 0.0,
            "mean_phase_c_layers": float(np.mean([len(row["phase_c"]["supporting_layers"]) for row in comparisons])) if comparisons else 0.0,
        },
        "comparison_rows": comparisons,
    }
    _maybe_write_json(result, output_path)
    return result


def run_motif_case_study_experiment(
    components: LoadedFlowComponents,
    loader,
    *,
    device: torch.device,
    checkpoint_tag: str,
    motif_artifact: dict,
    max_images: int,
    topk: int,
    exemplar_count: int = 6,
    alpha: float = 0.05,
    output_path: str | Path | None = None,
    progress_callback=None,
) -> dict:
    tracker = _ProgressTracker(
        experiment=INTERVENTION_CASES_ID,
        checkpoint_tag=checkpoint_tag,
        progress_callback=progress_callback,
    )
    outputs = collect_interpretability_outputs(
        components,
        loader,
        device=device,
        max_images=max_images,
        progress_callback=lambda **event: tracker.emit(
            stage="data_collection",
            completed=event["batch_idx"],
            total=event.get("total_batches"),
            message="collecting held-out case-study features",
        ),
    )
    ranked_motifs = _rank_motifs(motif_artifact.get("motifs", []))[: int(topk)]
    selected_artifact = {
        "metadata": {
            "grid_size": int(motif_artifact["metadata"]["grid_size"]),
            "n_layers": int(motif_artifact["metadata"]["n_layers"]),
            "n_cells": int(motif_artifact["metadata"]["n_cells"]),
        },
        "circuits": ranked_motifs,
    }
    intervention_results = run_circuit_interventions(
        components,
        selected_artifact,
        outputs,
        alpha=alpha,
        descriptor_key="z",
        progress_callback=lambda **event: tracker.emit(
            stage="intervention_screen",
            completed=event["completed"],
            total=event["total"],
            message=f"motif_id={event['circuit_id']} {event['status']}",
        ),
        n_jobs=max(1, int(components.config.get("interventions", {}).get("n_jobs", 1))),
    )
    results_by_id = {int(item.circuit_id): item.to_dict() for item in intervention_results}
    ablator = ResidualPatchAblator(components, grid_size=int(motif_artifact["metadata"]["grid_size"]))
    z = outputs["z"]
    logits = outputs["logits"]
    labels = outputs["labels"]
    image_shape = list(outputs["images"].shape[-2:])
    rows = []
    ordered_motifs = sorted(
        ranked_motifs,
        key=lambda motif: (
            -int(results_by_id.get(int(motif["id"]), {}).get("validated", False)),
            -float(motif.get("stability", {}).get("mean_cluster_stability", 0.0)),
            -int(len(motif.get("image_set", []))),
            int(motif["id"]),
        ),
    )
    for motif_idx, motif in enumerate(ordered_motifs, start=1):
        member_mask = assign_circuit_members(motif, z, outputs["indices"])
        member_rows = torch.nonzero(member_mask, as_tuple=False).flatten()
        if member_rows.numel() == 0:
            continue
        scores = _motif_scores(motif, F.normalize(z, dim=-1))
        top_member_rows = member_rows[torch.argsort(scores[member_rows], descending=True)[: int(exemplar_count)]]
        member_images = outputs["images"][top_member_rows].to(device)
        before_logits = logits[top_member_rows]
        after_logits = ablator.ablate(member_images, [tuple(node) for node in motif["active_nodes"]]).cpu()
        margins_before = _margin(before_logits)
        margins_after = _margin(after_logits)
        exemplar_rows = []
        for local_idx, row_idx in enumerate(top_member_rows.tolist()):
            exemplar_rows.append(
                {
                    "dataset_index": int(outputs["indices"][row_idx].item()),
                    "row_index": int(row_idx),
                    "label": int(labels[row_idx].item()),
                    "label_name": _label_name(int(labels[row_idx].item())),
                    "before_pred": int(before_logits[local_idx].argmax().item()),
                    "before_pred_name": _label_name(int(before_logits[local_idx].argmax().item())),
                    "after_pred": int(after_logits[local_idx].argmax().item()),
                    "after_pred_name": _label_name(int(after_logits[local_idx].argmax().item())),
                    "before_margin": float(margins_before[local_idx].item()),
                    "after_margin": float(margins_after[local_idx].item()),
                    "delta_margin": float((margins_before[local_idx] - margins_after[local_idx]).item()),
                    "overlay": _overlay_spec(
                        active_nodes=motif["active_nodes"],
                        representative_node=motif["representative_node"],
                        image_height=image_shape[0],
                        image_width=image_shape[1],
                        grid_size=int(motif_artifact["metadata"]["grid_size"]),
                    ),
                }
            )
        result_row = results_by_id.get(int(motif["id"]), {})
        rows.append(
            {
                "motif_id": int(motif["id"]),
                "validated": bool(result_row.get("validated", False)),
                "member_specific": bool(
                    result_row
                    and result_row.get("corrected_p_member_vs_nonmember", 1.0) < alpha
                    and result_row.get("ci_member_vs_nonmember", [0.0, 0.0])[0] > 0.0
                    and result_row.get("mean_member_delta_margin", 0.0) > result_row.get("mean_nonmember_delta_margin", 0.0)
                ),
                "dominant_class": motif.get("purity", {}).get("dominant_class"),
                "dominant_class_name": _label_name(motif.get("purity", {}).get("dominant_class")),
                "class_purity": float(motif.get("purity", {}).get("fraction", 0.0)),
                "supporting_layers": [int(layer) for layer in motif.get("layer_support", [])],
                "active_nodes": motif["active_nodes"],
                "representative_node": motif["representative_node"],
                "mean_member_delta_margin": float(result_row.get("mean_member_delta_margin", 0.0)),
                "mean_nonmember_delta_margin": float(result_row.get("mean_nonmember_delta_margin", 0.0)),
                "n_members": int(result_row.get("n_members", int(member_rows.numel()))),
                "n_controls": int(result_row.get("n_controls", 0)),
                "exemplars": exemplar_rows,
            }
        )
        tracker.emit(
            stage="case_studies",
            completed=motif_idx,
            total=len(ordered_motifs),
            message="packing top member intervention examples",
        )
    result = {
        "experiment": INTERVENTION_CASES_ID,
        "checkpoint_tag": checkpoint_tag,
        "summary": {
            "n_case_studies": int(len(rows)),
            "validated_count": int(sum(1 for row in rows if row["validated"])),
            "member_specific_count": int(sum(1 for row in rows if row["member_specific"])),
            "mean_member_delta_margin": float(np.mean([row["mean_member_delta_margin"] for row in rows])) if rows else 0.0,
        },
        "case_studies": rows,
    }
    _maybe_write_json(result, output_path)
    return result


def run_linear_probe_suite_experiment(
    components: LoadedFlowComponents,
    fit_loader,
    val_loader,
    test_loader,
    *,
    device: torch.device,
    checkpoint_tag: str,
    fit_max_images: int,
    val_max_images: int,
    test_max_images: int,
    c_grid: tuple[float, ...] = (0.1, 1.0, 10.0),
    output_path: str | Path | None = None,
    progress_callback=None,
) -> dict:
    tracker = _ProgressTracker(
        experiment=CLASS_PROBE_SUITE_ID,
        checkpoint_tag=checkpoint_tag,
        progress_callback=progress_callback,
    )
    fit_outputs, val_outputs, test_outputs = _collect_probe_splits(
        components,
        fit_loader,
        val_loader,
        test_loader,
        device=device,
        fit_max_images=fit_max_images,
        val_max_images=val_max_images,
        test_max_images=test_max_images,
        tracker=tracker,
        stage_prefix="class_probe_suite",
    )
    z_fit = fit_outputs["z"].numpy()
    z_val = val_outputs["z"].numpy()
    z_test = test_outputs["z"].numpy()
    y_fit = fit_outputs["labels"].numpy()
    y_val = val_outputs["labels"].numpy()
    y_test = test_outputs["labels"].numpy()

    global_result = _fit_probe_result(
        _global_probe_features(z_fit),
        y_fit,
        _global_probe_features(z_val),
        y_val,
        _global_probe_features(z_test),
        y_test,
        c_grid=c_grid,
    )

    per_layer = []
    n_layers = z_fit.shape[1]
    for layer_idx in range(n_layers):
        layer_result = _fit_probe_result(
            _layer_probe_features(z_fit, layer_idx),
            y_fit,
            _layer_probe_features(z_val, layer_idx),
            y_val,
            _layer_probe_features(z_test, layer_idx),
            y_test,
            c_grid=c_grid,
        )
        per_layer.append(
            {
                "layer_idx": int(layer_idx),
                "val_accuracy": layer_result["val_accuracy"],
                "accuracy": layer_result["accuracy"],
                "macro_f1": layer_result["macro_f1"],
                "selected_c": layer_result["selected_c"],
            }
        )
        tracker.emit(
            stage="per_layer_probe",
            completed=layer_idx + 1,
            total=n_layers,
            message="fitting per-layer multinomial probe",
        )

    per_node = []
    total_nodes = z_fit.shape[1] * z_fit.shape[2]
    node_counter = 0
    for layer_idx in range(z_fit.shape[1]):
        for cell_idx in range(z_fit.shape[2]):
            node_counter += 1
            node_result = _fit_probe_result(
                _node_probe_features(z_fit, layer_idx, cell_idx),
                y_fit,
                _node_probe_features(z_val, layer_idx, cell_idx),
                y_val,
                _node_probe_features(z_test, layer_idx, cell_idx),
                y_test,
                c_grid=c_grid,
            )
            per_node.append(
                {
                    "layer_idx": int(layer_idx),
                    "cell_idx": int(cell_idx),
                    "val_accuracy": node_result["val_accuracy"],
                    "accuracy": node_result["accuracy"],
                    "macro_f1": node_result["macro_f1"],
                    "selected_c": node_result["selected_c"],
                    "coef_norm": node_result["coef_norm"],
                }
            )
            tracker.emit(
                stage="per_node_probe",
                completed=node_counter,
                total=total_nodes,
                message="fitting multinomial probe",
            )

    result = {
        "experiment": CLASS_PROBE_SUITE_ID,
        "checkpoint_tag": checkpoint_tag,
        "summary": {
            "global_accuracy": global_result["accuracy"],
            "global_macro_f1": global_result["macro_f1"],
            "mean_layer_accuracy": float(np.mean([item["accuracy"] for item in per_layer])) if per_layer else 0.0,
            "mean_node_accuracy": float(np.mean([item["accuracy"] for item in per_node])) if per_node else 0.0,
        },
        "global_probe": {
            "selected_c": global_result["selected_c"],
            "val_accuracy": global_result["val_accuracy"],
            "accuracy": global_result["accuracy"],
            "macro_f1": global_result["macro_f1"],
            "class_coefficient_norms": _class_coefficient_norms(global_result["coef_norms"]),
        },
        "per_layer": per_layer,
        "per_node": per_node,
        "top_nodes_by_coef_norm": sorted(
            per_node,
            key=lambda item: (-item["coef_norm"], -item["accuracy"], item["layer_idx"], item["cell_idx"]),
        )[:10],
    }
    _maybe_write_json(result, output_path)
    return result


def run_probe_confusion_analysis_experiment(
    components: LoadedFlowComponents,
    fit_loader,
    val_loader,
    test_loader,
    *,
    device: torch.device,
    checkpoint_tag: str,
    fit_max_images: int,
    val_max_images: int,
    test_max_images: int,
    top_pairs: int = 3,
    exemplar_count: int = 6,
    c_grid: tuple[float, ...] = (0.1, 1.0, 10.0),
    output_path: str | Path | None = None,
    progress_callback=None,
) -> dict:
    tracker = _ProgressTracker(
        experiment=CONFUSION_ANALYSIS_ID,
        checkpoint_tag=checkpoint_tag,
        progress_callback=progress_callback,
    )
    fit_outputs, val_outputs, test_outputs = _collect_probe_splits(
        components,
        fit_loader,
        val_loader,
        test_loader,
        device=device,
        fit_max_images=fit_max_images,
        val_max_images=val_max_images,
        test_max_images=test_max_images,
        tracker=tracker,
        stage_prefix="confusion_analysis",
    )
    fit_z = fit_outputs["z"].numpy()
    val_z = val_outputs["z"].numpy()
    test_z = test_outputs["z"].numpy()
    fit_labels = fit_outputs["labels"].numpy()
    val_labels = val_outputs["labels"].numpy()
    test_labels = test_outputs["labels"].numpy()
    test_logits = test_outputs["logits"]
    predicted = test_logits.argmax(dim=1).numpy()
    pair_counts = _top_confusion_pairs(test_labels, predicted, top_pairs=int(top_pairs))

    rows = []
    for pair_idx, (left_label, right_label, count) in enumerate(pair_counts, start=1):
        fit_mask = np.isin(fit_labels, [left_label, right_label])
        val_mask = np.isin(val_labels, [left_label, right_label])
        test_mask = np.isin(test_labels, [left_label, right_label])
        pair_fit_labels = (fit_labels[fit_mask] == right_label).astype(int)
        pair_val_labels = (val_labels[val_mask] == right_label).astype(int)
        pair_test_labels = (test_labels[test_mask] == right_label).astype(int)
        global_result = _fit_probe_result(
            _global_probe_features(fit_z[fit_mask]),
            pair_fit_labels,
            _global_probe_features(val_z[val_mask]),
            pair_val_labels,
            _global_probe_features(test_z[test_mask]),
            pair_test_labels,
            c_grid=c_grid,
        )
        per_layer = []
        for layer_idx in range(test_z.shape[1]):
            layer_result = _fit_probe_result(
                _layer_probe_features(fit_z[fit_mask], layer_idx),
                pair_fit_labels,
                _layer_probe_features(val_z[val_mask], layer_idx),
                pair_val_labels,
                _layer_probe_features(test_z[test_mask], layer_idx),
                pair_test_labels,
                c_grid=c_grid,
            )
            per_layer.append(
                {
                    "layer_idx": int(layer_idx),
                    "accuracy": layer_result["accuracy"],
                    "macro_f1": layer_result["macro_f1"],
                    "selected_c": layer_result["selected_c"],
                }
            )
        per_node = []
        for layer_idx in range(test_z.shape[1]):
            for cell_idx in range(test_z.shape[2]):
                node_result = _fit_probe_result(
                    _node_probe_features(fit_z[fit_mask], layer_idx, cell_idx),
                    pair_fit_labels,
                    _node_probe_features(val_z[val_mask], layer_idx, cell_idx),
                    pair_val_labels,
                    _node_probe_features(test_z[test_mask], layer_idx, cell_idx),
                    pair_test_labels,
                    c_grid=c_grid,
                )
                per_node.append(
                    {
                        "layer_idx": int(layer_idx),
                        "cell_idx": int(cell_idx),
                        "accuracy": node_result["accuracy"],
                        "coef_norm": node_result["coef_norm"],
                    }
                )
        test_rows = np.nonzero(test_mask)[0]
        pair_confusions = [
            idx
            for idx in test_rows.tolist()
            if test_labels[idx] != predicted[idx] and {int(test_labels[idx]), int(predicted[idx])} == {int(left_label), int(right_label)}
        ]
        pair_confusions = sorted(
            pair_confusions,
            key=lambda idx: float(_margin(test_logits[idx : idx + 1]).item()),
        )[: int(exemplar_count)]
        rows.append(
            {
                "left_label": int(left_label),
                "left_label_name": _label_name(left_label),
                "right_label": int(right_label),
                "right_label_name": _label_name(right_label),
                "backbone_confusion_count": int(count),
                "global_accuracy": global_result["accuracy"],
                "global_macro_f1": global_result["macro_f1"],
                "top_layers": sorted(per_layer, key=lambda item: (-item["accuracy"], item["layer_idx"]))[:3],
                "top_nodes": sorted(per_node, key=lambda item: (-item["accuracy"], -item["coef_norm"], item["layer_idx"], item["cell_idx"]))[:5],
                "exemplar_errors": [
                    {
                        "dataset_index": int(test_outputs["indices"][idx].item()),
                        "row_index": int(idx),
                        "true_label": int(test_labels[idx]),
                        "true_label_name": _label_name(int(test_labels[idx])),
                        "pred_label": int(predicted[idx]),
                        "pred_label_name": _label_name(int(predicted[idx])),
                        "margin": float(_margin(test_logits[idx : idx + 1]).item()),
                    }
                    for idx in pair_confusions
                ],
            }
        )
        tracker.emit(
            stage="pairwise_probe",
            completed=pair_idx,
            total=len(pair_counts),
            message=f"fitting pairwise probes for {_label_name(left_label)} vs {_label_name(right_label)}",
        )

    result = {
        "experiment": CONFUSION_ANALYSIS_ID,
        "checkpoint_tag": checkpoint_tag,
        "summary": {
            "n_pairs": int(len(rows)),
            "mean_global_accuracy": float(np.mean([row["global_accuracy"] for row in rows])) if rows else 0.0,
            "mean_global_macro_f1": float(np.mean([row["global_macro_f1"] for row in rows])) if rows else 0.0,
        },
        "pair_rows": rows,
    }
    _maybe_write_json(result, output_path)
    return result


def run_probe_error_analysis_experiment(
    components: LoadedFlowComponents,
    fit_loader,
    val_loader,
    test_loader,
    *,
    device: torch.device,
    checkpoint_tag: str,
    fit_max_images: int,
    val_max_images: int,
    test_max_images: int,
    exemplar_count: int = 6,
    c_grid: tuple[float, ...] = (0.1, 1.0, 10.0),
    output_path: str | Path | None = None,
    progress_callback=None,
) -> dict:
    tracker = _ProgressTracker(
        experiment=ERROR_ANALYSIS_ID,
        checkpoint_tag=checkpoint_tag,
        progress_callback=progress_callback,
    )
    fit_outputs, val_outputs, test_outputs = _collect_probe_splits(
        components,
        fit_loader,
        val_loader,
        test_loader,
        device=device,
        fit_max_images=fit_max_images,
        val_max_images=val_max_images,
        test_max_images=test_max_images,
        tracker=tracker,
        stage_prefix="error_analysis",
    )
    fit_z = fit_outputs["z"].numpy()
    val_z = val_outputs["z"].numpy()
    test_z = test_outputs["z"].numpy()
    fit_labels = fit_outputs["labels"].numpy()
    val_labels = val_outputs["labels"].numpy()
    test_labels = test_outputs["labels"].numpy()
    test_logits = test_outputs["logits"]
    backbone_pred = test_logits.argmax(dim=1).numpy()
    correct_mask = backbone_pred == test_labels
    wrong_mask = ~correct_mask

    global_model = _fit_probe_model(
        _global_probe_features(fit_z),
        fit_labels,
        _global_probe_features(val_z),
        val_labels,
        c_grid=c_grid,
    )
    global_scores = global_model["model"].predict_proba(global_model["scaler"].transform(_global_probe_features(test_z)))
    global_true_margin = _true_class_probability_margin(global_scores, test_labels)

    per_layer_rows = []
    for layer_idx in range(test_z.shape[1]):
        layer_model = _fit_probe_model(
            _layer_probe_features(fit_z, layer_idx),
            fit_labels,
            _layer_probe_features(val_z, layer_idx),
            val_labels,
            c_grid=c_grid,
        )
        layer_scores = layer_model["model"].predict_proba(layer_model["scaler"].transform(_layer_probe_features(test_z, layer_idx)))
        layer_margin = _true_class_probability_margin(layer_scores, test_labels)
        per_layer_rows.append(
            {
                "layer_idx": int(layer_idx),
                "mean_correct_margin": float(np.mean(layer_margin[correct_mask])) if np.any(correct_mask) else 0.0,
                "mean_wrong_margin": float(np.mean(layer_margin[wrong_mask])) if np.any(wrong_mask) else 0.0,
                "margin_drop": float(
                    (np.mean(layer_margin[correct_mask]) if np.any(correct_mask) else 0.0)
                    - (np.mean(layer_margin[wrong_mask]) if np.any(wrong_mask) else 0.0)
                ),
            }
        )
        tracker.emit(
            stage="layer_margin_analysis",
            completed=layer_idx + 1,
            total=test_z.shape[1],
            message="measuring correct-vs-wrong margin drop",
        )

    per_node_rows = []
    total_nodes = test_z.shape[1] * test_z.shape[2]
    node_counter = 0
    for layer_idx in range(test_z.shape[1]):
        for cell_idx in range(test_z.shape[2]):
            node_counter += 1
            node_model = _fit_probe_model(
                _node_probe_features(fit_z, layer_idx, cell_idx),
                fit_labels,
                _node_probe_features(val_z, layer_idx, cell_idx),
                val_labels,
                c_grid=c_grid,
            )
            node_scores = node_model["model"].predict_proba(node_model["scaler"].transform(_node_probe_features(test_z, layer_idx, cell_idx)))
            node_margin = _true_class_probability_margin(node_scores, test_labels)
            per_node_rows.append(
                {
                    "layer_idx": int(layer_idx),
                    "cell_idx": int(cell_idx),
                    "mean_correct_margin": float(np.mean(node_margin[correct_mask])) if np.any(correct_mask) else 0.0,
                    "mean_wrong_margin": float(np.mean(node_margin[wrong_mask])) if np.any(wrong_mask) else 0.0,
                    "margin_drop": float(
                        (np.mean(node_margin[correct_mask]) if np.any(correct_mask) else 0.0)
                        - (np.mean(node_margin[wrong_mask]) if np.any(wrong_mask) else 0.0)
                    ),
                }
            )
            tracker.emit(
                stage="node_margin_analysis",
                completed=node_counter,
                total=total_nodes,
                message="scoring node-wise failure signatures",
            )

    wrong_rows = np.nonzero(wrong_mask)[0]
    ranked_wrong_rows = sorted(wrong_rows.tolist(), key=lambda idx: float(global_true_margin[idx]))[: int(exemplar_count)]
    result = {
        "experiment": ERROR_ANALYSIS_ID,
        "checkpoint_tag": checkpoint_tag,
        "summary": {
            "n_wrong_examples": int(wrong_mask.sum()),
            "global_mean_correct_margin": float(np.mean(global_true_margin[correct_mask])) if np.any(correct_mask) else 0.0,
            "global_mean_wrong_margin": float(np.mean(global_true_margin[wrong_mask])) if np.any(wrong_mask) else 0.0,
            "global_margin_drop": float(
                (np.mean(global_true_margin[correct_mask]) if np.any(correct_mask) else 0.0)
                - (np.mean(global_true_margin[wrong_mask]) if np.any(wrong_mask) else 0.0)
            ),
        },
        "per_layer": per_layer_rows,
        "top_drop_nodes": sorted(per_node_rows, key=lambda item: (-item["margin_drop"], item["layer_idx"], item["cell_idx"]))[:10],
        "failure_examples": [
            {
                "dataset_index": int(test_outputs["indices"][idx].item()),
                "row_index": int(idx),
                "true_label": int(test_labels[idx]),
                "true_label_name": _label_name(int(test_labels[idx])),
                "backbone_pred": int(backbone_pred[idx]),
                "backbone_pred_name": _label_name(int(backbone_pred[idx])),
                "probe_true_margin": float(global_true_margin[idx]),
                "backbone_margin": float(_margin(test_logits[idx : idx + 1]).item()),
            }
            for idx in ranked_wrong_rows
        ],
    }
    _maybe_write_json(result, output_path)
    return result


def _build_motif_visual_object(
    motif: dict,
    *,
    z: torch.Tensor,
    labels: torch.Tensor,
    dataset_indices: torch.Tensor,
    checkpoint_tag: str,
    image_shape: list[int],
    grid_size: int,
    exemplar_count: int,
) -> dict:
    member_rows = _member_rows_for_motif(motif, dataset_indices)
    scores = _motif_scores(motif, z)
    member_scores = scores[member_rows] if member_rows.size else torch.empty(0)
    exemplar_rows = (
        member_rows[torch.argsort(member_scores, descending=True)[: int(exemplar_count)]].tolist()
        if member_rows.size
        else []
    )
    member_labels = labels[member_rows].cpu().numpy() if member_rows.size else np.empty(0, dtype=np.int64)
    overlay = _overlay_spec(
        active_nodes=motif["active_nodes"],
        representative_node=motif["representative_node"],
        image_height=image_shape[0],
        image_width=image_shape[1],
        grid_size=grid_size,
    )
    return {
        "object_type": "motif",
        "checkpoint_tag": checkpoint_tag,
        "id": str(int(motif["id"])),
        "dominant_class": int(motif["purity"]["dominant_class"]) if motif.get("purity", {}).get("dominant_class") is not None else None,
        "dominant_class_name": _label_name(motif.get("purity", {}).get("dominant_class")),
        "class_purity": float(motif.get("purity", {}).get("fraction", 0.0)),
        "stability": float(motif.get("stability", {}).get("mean_cluster_stability", 0.0)),
        "size": int(len(motif.get("image_set", []))),
        "supporting_layers": [int(layer) for layer in motif.get("layer_support", [])],
        "active_nodes": [[int(layer), int(cell)] for layer, cell in motif.get("active_nodes", [])],
        "representative_node": [int(value) for value in motif["representative_node"]],
        "class_histogram": _label_histogram(member_labels),
        "exemplar_row_indices": [int(idx) for idx in exemplar_rows],
        "exemplar_dataset_indices": dataset_indices[exemplar_rows].cpu().tolist() if exemplar_rows else [],
        "exemplar_labels": labels[exemplar_rows].cpu().tolist() if exemplar_rows else [],
        "overlay": overlay,
    }


def _build_node_cluster_visual_object(
    cluster: dict,
    *,
    z: torch.Tensor,
    labels: torch.Tensor,
    dataset_indices: torch.Tensor,
    checkpoint_tag: str,
    image_shape: list[int],
    grid_size: int,
    exemplar_count: int,
) -> dict:
    row_indices = np.asarray(cluster.get("row_indices", []), dtype=np.int64)
    layer_idx, cell_idx = [int(value) for value in cluster["node"]]
    member_labels = labels[row_indices].cpu().numpy() if row_indices.size else np.empty(0, dtype=np.int64)
    if row_indices.size:
        member_vectors = z[row_indices, layer_idx, cell_idx, :]
        centroid = member_vectors.mean(dim=0)
        centroid = centroid / torch.clamp(centroid.norm(), min=1.0e-8)
        scores = (z[:, layer_idx, cell_idx, :] * centroid.unsqueeze(0)).sum(dim=-1)
        exemplar_rows = row_indices[torch.argsort(scores[row_indices], descending=True)[: int(exemplar_count)]].tolist()
    else:
        exemplar_rows = []
    counts = np.bincount(member_labels.astype(int), minlength=len(CIFAR10_CLASS_NAMES)) if member_labels.size else np.zeros(len(CIFAR10_CLASS_NAMES), dtype=int)
    dominant_class = int(np.argmax(counts)) if counts.sum() else None
    purity = float(counts.max() / max(counts.sum(), 1)) if counts.sum() else 0.0
    overlay = _overlay_spec(
        active_nodes=[[layer_idx, cell_idx]],
        representative_node=[layer_idx, cell_idx],
        image_height=image_shape[0],
        image_width=image_shape[1],
        grid_size=grid_size,
    )
    return {
        "object_type": "node_cluster",
        "checkpoint_tag": checkpoint_tag,
        "id": f"{layer_idx}:{cell_idx}",
        "dominant_class": dominant_class,
        "dominant_class_name": _label_name(dominant_class),
        "class_purity": purity,
        "stability": float(cluster.get("stability", 0.0)),
        "size": int(cluster.get("size", len(cluster.get("image_set", [])))),
        "supporting_layers": [layer_idx],
        "active_nodes": [[layer_idx, cell_idx]],
        "representative_node": [layer_idx, cell_idx],
        "class_histogram": _label_histogram(member_labels),
        "exemplar_row_indices": [int(idx) for idx in exemplar_rows],
        "exemplar_dataset_indices": dataset_indices[exemplar_rows].cpu().tolist() if exemplar_rows else [],
        "exemplar_labels": labels[exemplar_rows].cpu().tolist() if exemplar_rows else [],
        "overlay": overlay,
    }


def _comparison_row(phase_b_row: dict, phase_c_row: dict, *, match_quality: float | None, comparison_type: str) -> dict:
    return {
        "comparison_type": comparison_type,
        "match_quality": None if match_quality is None else float(match_quality),
        "phase_b": phase_b_row,
        "phase_c": phase_c_row,
        "deltas": {
            "class_purity": float(phase_c_row["class_purity"] - phase_b_row["class_purity"]),
            "stability": float(phase_c_row["stability"] - phase_b_row["stability"]),
            "supporting_layers": int(len(phase_c_row["supporting_layers"]) - len(phase_b_row["supporting_layers"])),
            "size": int(phase_c_row["size"] - phase_b_row["size"]),
        },
    }


def _overlay_spec(
    *,
    active_nodes: list[list[int]] | list[tuple[int, int]],
    representative_node: list[int] | tuple[int, int],
    image_height: int,
    image_width: int,
    grid_size: int,
) -> dict:
    boxes = []
    rep_key = tuple(int(value) for value in representative_node)
    for node in active_nodes:
        layer_idx, cell_idx = int(node[0]), int(node[1])
        box = _cell_box(cell_idx=cell_idx, image_height=image_height, image_width=image_width, grid_size=grid_size)
        boxes.append(
            {
                "layer_idx": layer_idx,
                "cell_idx": cell_idx,
                **box,
                "is_representative": (layer_idx, cell_idx) == rep_key,
            }
        )
    return {
        "active_boxes": boxes,
        "representative_box": next((box for box in boxes if box["is_representative"]), None),
        "union_box": _union_box(boxes),
    }


def _cell_box(*, cell_idx: int, image_height: int, image_width: int, grid_size: int) -> dict:
    row = int(cell_idx) // grid_size
    col = int(cell_idx) % grid_size
    y0 = int(np.floor((row * image_height) / grid_size))
    y1 = int(np.floor(((row + 1) * image_height) / grid_size))
    x0 = int(np.floor((col * image_width) / grid_size))
    x1 = int(np.floor(((col + 1) * image_width) / grid_size))
    return {
        "x0": x0,
        "y0": y0,
        "x1": max(x1, x0 + 1),
        "y1": max(y1, y0 + 1),
    }


def _union_box(boxes: list[dict]) -> dict | None:
    if not boxes:
        return None
    return {
        "x0": int(min(box["x0"] for box in boxes)),
        "y0": int(min(box["y0"] for box in boxes)),
        "x1": int(max(box["x1"] for box in boxes)),
        "y1": int(max(box["y1"] for box in boxes)),
    }


def _rank_node_clusters(node_clusters: list[dict]) -> list[dict]:
    return sorted(
        node_clusters,
        key=lambda cluster: (
            -float(cluster.get("stability", 0.0)),
            -int(cluster.get("size", len(cluster.get("image_set", [])))),
            int(cluster["node"][0]),
            int(cluster["node"][1]),
        ),
    )


def _collect_probe_splits(
    components: LoadedFlowComponents,
    fit_loader,
    val_loader,
    test_loader,
    *,
    device: torch.device,
    fit_max_images: int,
    val_max_images: int,
    test_max_images: int,
    tracker: _ProgressTracker,
    stage_prefix: str,
) -> tuple[dict, dict, dict]:
    fit_outputs = collect_interpretability_outputs(
        components,
        fit_loader,
        device=device,
        max_images=fit_max_images,
        progress_callback=lambda **event: tracker.emit(
            stage=f"{stage_prefix}_fit",
            completed=event["batch_idx"],
            total=event.get("total_batches"),
            message="collecting fit split",
        ),
    )
    val_outputs = collect_interpretability_outputs(
        components,
        val_loader,
        device=device,
        max_images=val_max_images,
        progress_callback=lambda **event: tracker.emit(
            stage=f"{stage_prefix}_val",
            completed=event["batch_idx"],
            total=event.get("total_batches"),
            message="collecting val split",
        ),
    )
    test_outputs = collect_interpretability_outputs(
        components,
        test_loader,
        device=device,
        max_images=test_max_images,
        progress_callback=lambda **event: tracker.emit(
            stage=f"{stage_prefix}_test",
            completed=event["batch_idx"],
            total=event.get("total_batches"),
            message="collecting test split",
        ),
    )
    return fit_outputs, val_outputs, test_outputs


def _fit_probe_result(
    fit_x: np.ndarray,
    fit_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    *,
    c_grid: tuple[float, ...],
) -> dict:
    model_info = _fit_probe_model(fit_x, fit_y, val_x, val_y, c_grid=c_grid)
    scaler = model_info["scaler"]
    model = model_info["model"]
    test_pred = model.predict(scaler.transform(test_x))
    accuracy = float(accuracy_score(test_y, test_pred)) if test_y.size else 0.0
    macro_f1 = float(f1_score(test_y, test_pred, average="macro", zero_division=0)) if test_y.size else 0.0
    coef_norm = float(np.linalg.norm(model.coef_)) if hasattr(model, "coef_") else 0.0
    coef_norms = np.linalg.norm(model.coef_, axis=1).tolist() if hasattr(model, "coef_") else []
    return {
        "selected_c": float(model_info["selected_c"]),
        "val_accuracy": float(model_info["val_accuracy"]),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "coef_norm": coef_norm,
        "coef_norms": [float(value) for value in coef_norms],
    }


def _fit_probe_model(
    fit_x: np.ndarray,
    fit_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    *,
    c_grid: tuple[float, ...],
) -> dict:
    if fit_x.shape[0] == 0 or len(np.unique(fit_y)) < 2:
        return _fallback_probe_model(fit_x, fit_y)
    scaler = StandardScaler()
    fit_scaled = scaler.fit_transform(fit_x)
    val_scaled = scaler.transform(val_x)
    best = None
    for c_value in c_grid:
        model = LogisticRegression(
            C=float(c_value),
            max_iter=1000,
            solver="lbfgs",
        )
        model.fit(fit_scaled, fit_y)
        val_pred = model.predict(val_scaled)
        val_accuracy = float(accuracy_score(val_y, val_pred)) if val_y.size else 0.0
        candidate = {
            "selected_c": float(c_value),
            "val_accuracy": val_accuracy,
            "scaler": scaler,
            "model": model,
        }
        if best is None or val_accuracy > best["val_accuracy"] or (
            val_accuracy == best["val_accuracy"] and float(c_value) < best["selected_c"]
        ):
            best = candidate
    assert best is not None
    return best


def _fallback_probe_model(fit_x: np.ndarray, fit_y: np.ndarray) -> dict:
    class _DummyModel:
        def __init__(self, label: int, n_features: int) -> None:
            self.label = int(label)
            self.coef_ = np.zeros((1, n_features), dtype=np.float64)

        def predict(self, x: np.ndarray) -> np.ndarray:
            return np.full(x.shape[0], self.label, dtype=np.int64)

        def predict_proba(self, x: np.ndarray) -> np.ndarray:
            return np.ones((x.shape[0], 1), dtype=np.float64)

    class _IdentityScaler:
        def transform(self, x: np.ndarray) -> np.ndarray:
            return x

    label = int(fit_y[0]) if fit_y.size else 0
    return {
        "selected_c": 1.0,
        "val_accuracy": 0.0,
        "scaler": _IdentityScaler(),
        "model": _DummyModel(label, fit_x.shape[1] if fit_x.ndim == 2 else 0),
    }


def _global_probe_features(z: np.ndarray) -> np.ndarray:
    return z.mean(axis=2).reshape(z.shape[0], -1)


def _layer_probe_features(z: np.ndarray, layer_idx: int) -> np.ndarray:
    return z[:, int(layer_idx), :, :].mean(axis=1)


def _node_probe_features(z: np.ndarray, layer_idx: int, cell_idx: int) -> np.ndarray:
    return z[:, int(layer_idx), int(cell_idx), :]


def _class_coefficient_norms(coef_norms: list[float]) -> list[dict]:
    return [
        {
            "class_idx": int(class_idx),
            "class_name": _label_name(class_idx),
            "coefficient_norm": float(value),
        }
        for class_idx, value in enumerate(coef_norms)
    ]


def _top_confusion_pairs(labels: np.ndarray, predicted: np.ndarray, *, top_pairs: int) -> list[tuple[int, int, int]]:
    matrix = confusion_matrix(labels, predicted, labels=np.arange(len(CIFAR10_CLASS_NAMES)))
    pair_counts = []
    for left in range(matrix.shape[0]):
        for right in range(left + 1, matrix.shape[1]):
            count = int(matrix[left, right] + matrix[right, left])
            if count > 0:
                pair_counts.append((left, right, count))
    pair_counts.sort(key=lambda item: (-item[2], item[0], item[1]))
    return pair_counts[: int(top_pairs)]


def _true_class_probability_margin(probabilities: np.ndarray, labels: np.ndarray) -> np.ndarray:
    if probabilities.ndim != 2 or probabilities.shape[0] == 0:
        return np.zeros(labels.shape[0], dtype=np.float64)
    n_classes = probabilities.shape[1]
    if n_classes <= 1:
        return np.zeros(probabilities.shape[0], dtype=np.float64)
    one_hot = np.eye(n_classes, dtype=np.float64)[labels.astype(int)]
    true_probs = (probabilities * one_hot).sum(axis=1)
    masked = probabilities.copy()
    masked[np.arange(probabilities.shape[0]), labels.astype(int)] = -np.inf
    other_probs = masked.max(axis=1)
    other_probs = np.where(np.isfinite(other_probs), other_probs, 0.0)
    return true_probs - other_probs


def _margin(logits: torch.Tensor) -> torch.Tensor:
    top2 = torch.topk(logits, k=2, dim=1).values
    return top2[:, 0] - top2[:, 1]


def _label_name(label_idx: int | None) -> str | None:
    if label_idx is None:
        return None
    if 0 <= int(label_idx) < len(CIFAR10_CLASS_NAMES):
        return CIFAR10_CLASS_NAMES[int(label_idx)]
    return str(label_idx)
