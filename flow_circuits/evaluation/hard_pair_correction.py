from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)
import torch

from flow_circuits.evaluation.interpretability_validation import (
    CIFAR10_CLASS_NAMES,
    _ProgressTracker,
    _collect_probe_splits,
    _fit_probe_model,
    _fit_probe_result,
    _global_probe_features,
    _label_name,
    _layer_probe_features,
    _margin,
    _maybe_write_json,
    _node_probe_features,
    _overlay_spec,
    _top_confusion_pairs,
)
from flow_circuits.training import LoadedFlowComponents


MULTICLASS_PROBE_AUDIT_ID = "multiclass_probe_audit"
HARD_PAIR_PROBE_BENCHMARK_ID = "hard_pair_probe_benchmark"
HYBRID_CORRECTION_ID = "hybrid_correction"
CORRECTION_CASE_STUDIES_ID = "correction_case_studies"

NB06_EXPERIMENT_IDS = [
    MULTICLASS_PROBE_AUDIT_ID,
    HARD_PAIR_PROBE_BENCHMARK_ID,
    HYBRID_CORRECTION_ID,
    CORRECTION_CASE_STUDIES_ID,
]


def run_multiclass_z_probe_audit_experiment(
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
        experiment=MULTICLASS_PROBE_AUDIT_ID,
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
        stage_prefix=MULTICLASS_PROBE_AUDIT_ID,
    )
    z_fit = fit_outputs["z"].numpy()
    z_val = val_outputs["z"].numpy()
    z_test = test_outputs["z"].numpy()
    y_fit = fit_outputs["labels"].numpy()
    y_val = val_outputs["labels"].numpy()
    y_test = test_outputs["labels"].numpy()

    global_model = _fit_probe_model(
        _global_probe_features(z_fit),
        y_fit,
        _global_probe_features(z_val),
        y_val,
        c_grid=c_grid,
    )
    global_eval = _evaluate_multiclass_model(global_model, _global_probe_features(z_test), y_test)

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
            message="fitting per-layer multinomial probes",
        )

    per_node = []
    total_nodes = z_fit.shape[1] * z_fit.shape[2]
    node_counter = 0
    for layer_idx in range(z_fit.shape[1]):
        for cell_idx in range(z_fit.shape[2]):
            node_counter += 1
            node_model = _fit_probe_model(
                _node_probe_features(z_fit, layer_idx, cell_idx),
                y_fit,
                _node_probe_features(z_val, layer_idx, cell_idx),
                y_val,
                c_grid=c_grid,
            )
            node_eval = _evaluate_multiclass_model(
                node_model,
                _node_probe_features(z_test, layer_idx, cell_idx),
                y_test,
            )
            per_node.append(
                {
                    "layer_idx": int(layer_idx),
                    "cell_idx": int(cell_idx),
                    "val_accuracy": float(node_model["val_accuracy"]),
                    "accuracy": node_eval["accuracy"],
                    "macro_f1": node_eval["macro_f1"],
                    "log_loss": node_eval["log_loss"],
                    "top2_accuracy": node_eval["top2_accuracy"],
                    "selected_c": float(node_model["selected_c"]),
                    "coef_norm": float(node_eval["coef_norm"]),
                }
            )
            tracker.emit(
                stage="per_node_probe",
                completed=node_counter,
                total=total_nodes,
                message="fitting per-node multinomial probes",
            )

    result = {
        "experiment": MULTICLASS_PROBE_AUDIT_ID,
        "checkpoint_tag": checkpoint_tag,
        "summary": {
            "global_accuracy": global_eval["accuracy"],
            "global_macro_f1": global_eval["macro_f1"],
            "global_log_loss": global_eval["log_loss"],
            "global_top2_accuracy": global_eval["top2_accuracy"],
            "mean_layer_accuracy": float(np.mean([row["accuracy"] for row in per_layer])) if per_layer else 0.0,
            "mean_node_accuracy": float(np.mean([row["accuracy"] for row in per_node])) if per_node else 0.0,
        },
        "global_probe": {
            "selected_c": float(global_model["selected_c"]),
            "val_accuracy": float(global_model["val_accuracy"]),
            "accuracy": global_eval["accuracy"],
            "macro_f1": global_eval["macro_f1"],
            "log_loss": global_eval["log_loss"],
            "top2_accuracy": global_eval["top2_accuracy"],
            "class_coefficient_norms": _class_coefficient_norms(global_eval["coef_norms"]),
        },
        "per_layer": per_layer,
        "per_node": per_node,
        "top_nodes_by_coef_norm": sorted(
            per_node,
            key=lambda row: (-row["coef_norm"], -row["accuracy"], row["layer_idx"], row["cell_idx"]),
        )[:10],
        "top_nodes_by_accuracy": sorted(
            per_node,
            key=lambda row: (-row["accuracy"], -row["macro_f1"], row["layer_idx"], row["cell_idx"]),
        )[:10],
    }
    _maybe_write_json(result, output_path)
    return result


def run_hard_pair_probe_benchmark_experiment(
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
    top_k_nodes: int = 3,
    c_grid: tuple[float, ...] = (0.1, 1.0, 10.0),
    output_path: str | Path | None = None,
    progress_callback=None,
) -> dict:
    pair_bundle = _prepare_hard_pair_bundle(
        components,
        fit_loader,
        val_loader,
        test_loader,
        device=device,
        checkpoint_tag=checkpoint_tag,
        fit_max_images=fit_max_images,
        val_max_images=val_max_images,
        test_max_images=test_max_images,
        top_pairs=top_pairs,
        top_k_nodes=top_k_nodes,
        c_grid=c_grid,
        experiment_id=HARD_PAIR_PROBE_BENCHMARK_ID,
        progress_callback=progress_callback,
    )
    result = {
        "experiment": HARD_PAIR_PROBE_BENCHMARK_ID,
        "checkpoint_tag": checkpoint_tag,
        "summary": {
            "n_pairs": int(len(pair_bundle["pair_rows"])),
            "mean_backbone_pair_accuracy": float(np.mean([row["backbone_accuracy"] for row in pair_bundle["pair_rows"]]))
            if pair_bundle["pair_rows"]
            else 0.0,
            "mean_full_z_probe_accuracy": float(np.mean([row["full_z_probe_accuracy"] for row in pair_bundle["pair_rows"]]))
            if pair_bundle["pair_rows"]
            else 0.0,
            "mean_top_node_probe_accuracy": float(np.mean([row["top_node_probe_accuracy"] for row in pair_bundle["pair_rows"]]))
            if pair_bundle["pair_rows"]
            else 0.0,
        },
        "pair_rows": pair_bundle["pair_rows"],
    }
    _maybe_write_json(result, output_path)
    return result


def run_hard_pair_hybrid_correction_experiment(
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
    top_k_nodes: int = 3,
    c_grid: tuple[float, ...] = (0.1, 1.0, 10.0),
    output_path: str | Path | None = None,
    progress_callback=None,
) -> dict:
    pair_bundle = _prepare_hard_pair_bundle(
        components,
        fit_loader,
        val_loader,
        test_loader,
        device=device,
        checkpoint_tag=checkpoint_tag,
        fit_max_images=fit_max_images,
        val_max_images=val_max_images,
        test_max_images=test_max_images,
        top_pairs=top_pairs,
        top_k_nodes=top_k_nodes,
        c_grid=c_grid,
        experiment_id=HYBRID_CORRECTION_ID,
        progress_callback=progress_callback,
    )
    hybrid_full = _evaluate_hybrid_predictions(pair_bundle=pair_bundle, prediction_mode="full_z")
    hybrid_top_nodes = _evaluate_hybrid_predictions(pair_bundle=pair_bundle, prediction_mode="top_nodes")

    result = {
        "experiment": HYBRID_CORRECTION_ID,
        "checkpoint_tag": checkpoint_tag,
        "summary": {
            "n_pairs": int(len(pair_bundle["pair_rows"])),
            "backbone_overall_accuracy": hybrid_full["backbone"]["overall_accuracy"],
            "full_z_hybrid_overall_accuracy": hybrid_full["hybrid"]["overall_accuracy"],
            "top_node_hybrid_overall_accuracy": hybrid_top_nodes["hybrid"]["overall_accuracy"],
            "backbone_trigger_accuracy": hybrid_full["backbone"]["trigger_subset_accuracy"],
            "full_z_hybrid_trigger_accuracy": hybrid_full["hybrid"]["trigger_subset_accuracy"],
            "top_node_hybrid_trigger_accuracy": hybrid_top_nodes["hybrid"]["trigger_subset_accuracy"],
        },
        "backbone": hybrid_full["backbone"],
        "full_z_hybrid": hybrid_full["hybrid"],
        "top_node_hybrid": hybrid_top_nodes["hybrid"],
        "pair_rows": pair_bundle["pair_rows"],
    }
    _maybe_write_json(result, output_path)
    return result


def run_hard_pair_case_study_experiment(
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
    top_k_nodes: int = 3,
    exemplar_count: int = 6,
    c_grid: tuple[float, ...] = (0.1, 1.0, 10.0),
    output_path: str | Path | None = None,
    progress_callback=None,
) -> dict:
    pair_bundle = _prepare_hard_pair_bundle(
        components,
        fit_loader,
        val_loader,
        test_loader,
        device=device,
        checkpoint_tag=checkpoint_tag,
        fit_max_images=fit_max_images,
        val_max_images=val_max_images,
        test_max_images=test_max_images,
        top_pairs=top_pairs,
        top_k_nodes=top_k_nodes,
        c_grid=c_grid,
        experiment_id=CORRECTION_CASE_STUDIES_ID,
        progress_callback=progress_callback,
    )
    full_eval = _evaluate_hybrid_predictions(pair_bundle=pair_bundle, prediction_mode="full_z")
    top_eval = _evaluate_hybrid_predictions(pair_bundle=pair_bundle, prediction_mode="top_nodes")
    image_shape = list(pair_bundle["test_outputs"]["images"].shape[-2:])
    grid_size = int(components.config["tokenization"].get("grid_size", 4))

    pair_case_studies = [
        _build_pair_case_rows(
            pair_info=pair_info,
            full_eval=full_eval,
            top_eval=top_eval,
            test_outputs=pair_bundle["test_outputs"],
            image_shape=image_shape,
            grid_size=grid_size,
            exemplar_count=exemplar_count,
        )
        for pair_info in pair_bundle["pair_infos"]
    ]
    result = {
        "experiment": CORRECTION_CASE_STUDIES_ID,
        "checkpoint_tag": checkpoint_tag,
        "summary": {
            "n_pairs": int(len(pair_case_studies)),
            "n_corrected_examples": int(sum(len(row["corrected_examples"]) for row in pair_case_studies)),
            "n_harmed_examples": int(sum(len(row["harmed_examples"]) for row in pair_case_studies)),
        },
        "pair_case_studies": pair_case_studies,
    }
    _maybe_write_json(result, output_path)
    return result


def _prepare_hard_pair_bundle(
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
    top_pairs: int,
    top_k_nodes: int,
    c_grid: tuple[float, ...],
    experiment_id: str,
    progress_callback,
) -> dict:
    tracker = _ProgressTracker(
        experiment=experiment_id,
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
        stage_prefix=experiment_id,
    )
    fit_z = fit_outputs["z"].numpy()
    val_z = val_outputs["z"].numpy()
    test_z = test_outputs["z"].numpy()
    fit_labels = fit_outputs["labels"].numpy()
    val_labels = val_outputs["labels"].numpy()
    test_labels = test_outputs["labels"].numpy()
    val_pred = val_outputs["logits"].argmax(dim=1).numpy()
    pair_counts = _top_confusion_pairs(val_labels, val_pred, top_pairs=int(top_pairs))

    pair_rows = []
    pair_infos = []
    for pair_idx, (left_label, right_label, count) in enumerate(pair_counts, start=1):
        pair_info = _fit_pair_models(
            fit_z=fit_z,
            val_z=val_z,
            test_z=test_z,
            fit_labels=fit_labels,
            val_labels=val_labels,
            test_labels=test_labels,
            test_logits=test_outputs["logits"],
            left_label=int(left_label),
            right_label=int(right_label),
            backbone_confusion_count=int(count),
            top_k_nodes=int(top_k_nodes),
            c_grid=c_grid,
        )
        pair_infos.append(pair_info)
        pair_rows.append(_serialize_pair_row(pair_info))
        tracker.emit(
            stage="pairwise_probe",
            completed=pair_idx,
            total=len(pair_counts),
            message=f"fitting hard-pair probes for {_label_name(left_label)} vs {_label_name(right_label)}",
        )
    return {
        "pair_rows": pair_rows,
        "pair_infos": pair_infos,
        "test_outputs": test_outputs,
    }


def _fit_pair_models(
    *,
    fit_z: np.ndarray,
    val_z: np.ndarray,
    test_z: np.ndarray,
    fit_labels: np.ndarray,
    val_labels: np.ndarray,
    test_labels: np.ndarray,
    test_logits: torch.Tensor,
    left_label: int,
    right_label: int,
    backbone_confusion_count: int,
    top_k_nodes: int,
    c_grid: tuple[float, ...],
) -> dict:
    fit_mask = np.isin(fit_labels, [left_label, right_label])
    val_mask = np.isin(val_labels, [left_label, right_label])
    test_mask = np.isin(test_labels, [left_label, right_label])

    fit_y = (fit_labels[fit_mask] == right_label).astype(int)
    val_y = (val_labels[val_mask] == right_label).astype(int)
    test_y = (test_labels[test_mask] == right_label).astype(int)

    full_model = _fit_probe_model(
        _global_probe_features(fit_z[fit_mask]),
        fit_y,
        _global_probe_features(val_z[val_mask]),
        val_y,
        c_grid=c_grid,
    )
    full_eval = _evaluate_binary_model(full_model, _global_probe_features(test_z[test_mask]), test_y)

    node_rows = []
    for layer_idx in range(test_z.shape[1]):
        for cell_idx in range(test_z.shape[2]):
            node_model = _fit_probe_model(
                _node_probe_features(fit_z[fit_mask], layer_idx, cell_idx),
                fit_y,
                _node_probe_features(val_z[val_mask], layer_idx, cell_idx),
                val_y,
                c_grid=c_grid,
            )
            val_eval = _evaluate_binary_model(
                node_model,
                _node_probe_features(val_z[val_mask], layer_idx, cell_idx),
                val_y,
            )
            test_eval = _evaluate_binary_model(
                node_model,
                _node_probe_features(test_z[test_mask], layer_idx, cell_idx),
                test_y,
            )
            node_rows.append(
                {
                    "layer_idx": int(layer_idx),
                    "cell_idx": int(cell_idx),
                    "val_accuracy": val_eval["accuracy"],
                    "val_macro_f1": val_eval["macro_f1"],
                    "test_accuracy": test_eval["accuracy"],
                    "test_macro_f1": test_eval["macro_f1"],
                    "coef_norm": test_eval["coef_norm"],
                }
            )
    node_rows.sort(
        key=lambda row: (
            -row["val_accuracy"],
            -row["val_macro_f1"],
            -row["coef_norm"],
            row["layer_idx"],
            row["cell_idx"],
        )
    )
    selected_top_nodes = node_rows[: int(top_k_nodes)]

    top_node_model = _fit_probe_model(
        _top_node_features(fit_z[fit_mask], selected_top_nodes),
        fit_y,
        _top_node_features(val_z[val_mask], selected_top_nodes),
        val_y,
        c_grid=c_grid,
    )
    top_node_eval = _evaluate_binary_model(
        top_node_model,
        _top_node_features(test_z[test_mask], selected_top_nodes),
        test_y,
    )

    backbone_probs = torch.softmax(test_logits[test_mask], dim=1)[:, [left_label, right_label]].cpu().numpy()
    if backbone_probs.shape[0]:
        backbone_prob_right = backbone_probs[:, 1] / np.clip(backbone_probs.sum(axis=1), 1.0e-8, None)
        backbone_pred = (backbone_prob_right >= 0.5).astype(int)
    else:
        backbone_prob_right = np.empty(0, dtype=np.float64)
        backbone_pred = np.empty(0, dtype=np.int64)
    backbone_eval = _evaluate_binary_predictions(test_y, backbone_pred, backbone_prob_right)

    return {
        "left_label": int(left_label),
        "left_label_name": _label_name(left_label),
        "right_label": int(right_label),
        "right_label_name": _label_name(right_label),
        "backbone_confusion_count": int(backbone_confusion_count),
        "full_model": full_model,
        "full_eval": full_eval,
        "top_node_model": top_node_model,
        "top_node_eval": top_node_eval,
        "backbone_eval": backbone_eval,
        "selected_top_nodes": [
            {
                "layer_idx": int(row["layer_idx"]),
                "cell_idx": int(row["cell_idx"]),
                "val_accuracy": float(row["val_accuracy"]),
                "val_macro_f1": float(row["val_macro_f1"]),
                "test_accuracy": float(row["test_accuracy"]),
                "test_macro_f1": float(row["test_macro_f1"]),
                "coef_norm": float(row["coef_norm"]),
            }
            for row in selected_top_nodes
        ],
        "best_single_node_accuracy": float(node_rows[0]["test_accuracy"]) if node_rows else 0.0,
    }


def _serialize_pair_row(pair_info: dict) -> dict:
    return {
        "left_label": pair_info["left_label"],
        "left_label_name": pair_info["left_label_name"],
        "right_label": pair_info["right_label"],
        "right_label_name": pair_info["right_label_name"],
        "backbone_confusion_count": pair_info["backbone_confusion_count"],
        "backbone_accuracy": pair_info["backbone_eval"]["accuracy"],
        "backbone_macro_f1": pair_info["backbone_eval"]["macro_f1"],
        "backbone_balanced_accuracy": pair_info["backbone_eval"]["balanced_accuracy"],
        "full_z_probe_accuracy": pair_info["full_eval"]["accuracy"],
        "full_z_probe_macro_f1": pair_info["full_eval"]["macro_f1"],
        "full_z_probe_auroc": pair_info["full_eval"]["auroc"],
        "full_z_probe_brier": pair_info["full_eval"]["brier_score"],
        "top_node_probe_accuracy": pair_info["top_node_eval"]["accuracy"],
        "top_node_probe_macro_f1": pair_info["top_node_eval"]["macro_f1"],
        "top_node_probe_auroc": pair_info["top_node_eval"]["auroc"],
        "top_node_probe_brier": pair_info["top_node_eval"]["brier_score"],
        "best_single_node_accuracy": pair_info["best_single_node_accuracy"],
        "selected_top_nodes": pair_info["selected_top_nodes"],
    }


def _evaluate_hybrid_predictions(*, pair_bundle: dict, prediction_mode: str) -> dict:
    test_outputs = pair_bundle["test_outputs"]
    logits = test_outputs["logits"]
    labels = test_outputs["labels"].numpy()
    probabilities = torch.softmax(logits, dim=1).cpu().numpy()
    predictions = probabilities.argmax(axis=1)
    top2_indices = np.argsort(probabilities, axis=1)[:, -2:][:, ::-1]

    hybrid_probs = probabilities.copy()
    hybrid_preds = predictions.copy()
    trigger_mask = np.zeros(labels.shape[0], dtype=bool)
    overrides_mask = np.zeros(labels.shape[0], dtype=bool)
    corrected_mask = np.zeros(labels.shape[0], dtype=bool)
    harmed_mask = np.zeros(labels.shape[0], dtype=bool)
    per_pair_rows = []

    test_z = test_outputs["z"].numpy()
    for pair_info in pair_bundle["pair_infos"]:
        left_label = int(pair_info["left_label"])
        right_label = int(pair_info["right_label"])
        pair_key = {left_label, right_label}
        pair_trigger = np.array(
            [{int(row[0]), int(row[1])} == pair_key for row in top2_indices.tolist()],
            dtype=bool,
        )
        trigger_mask |= pair_trigger
        if not np.any(pair_trigger):
            per_pair_rows.append(_empty_pair_hybrid_row(pair_info))
            continue

        if prediction_mode == "full_z":
            model = pair_info["full_model"]
            features = _global_probe_features(test_z[pair_trigger])
        else:
            model = pair_info["top_node_model"]
            features = _top_node_features(test_z[pair_trigger], pair_info["selected_top_nodes"])
        pair_prob_right = _predict_binary_probabilities(model, features)

        pair_indices = np.nonzero(pair_trigger)[0]
        pair_overrides = 0
        pair_corrected = 0
        pair_harmed = 0
        for local_idx, row_idx in enumerate(pair_indices.tolist()):
            pair_mass = hybrid_probs[row_idx, left_label] + hybrid_probs[row_idx, right_label]
            right_prob = float(pair_prob_right[local_idx])
            hybrid_probs[row_idx, left_label] = pair_mass * (1.0 - right_prob)
            hybrid_probs[row_idx, right_label] = pair_mass * right_prob
            new_pred = int(right_label if right_prob >= 0.5 else left_label)
            old_pred = int(predictions[row_idx])
            hybrid_preds[row_idx] = new_pred
            if new_pred != old_pred:
                overrides_mask[row_idx] = True
                pair_overrides += 1
                if old_pred != int(labels[row_idx]) and new_pred == int(labels[row_idx]):
                    corrected_mask[row_idx] = True
                    pair_corrected += 1
                elif old_pred == int(labels[row_idx]) and new_pred != int(labels[row_idx]):
                    harmed_mask[row_idx] = True
                    pair_harmed += 1

        pair_subset_backbone = _classification_metrics(labels[pair_trigger], predictions[pair_trigger], probabilities[pair_trigger])
        pair_subset_hybrid = _classification_metrics(labels[pair_trigger], hybrid_preds[pair_trigger], hybrid_probs[pair_trigger])
        per_pair_rows.append(
            {
                "left_label": left_label,
                "left_label_name": pair_info["left_label_name"],
                "right_label": right_label,
                "right_label_name": pair_info["right_label_name"],
                "trigger_count": int(pair_trigger.sum()),
                "overrides": int(pair_overrides),
                "corrected": int(pair_corrected),
                "harmed": int(pair_harmed),
                "backbone_subset_accuracy": pair_subset_backbone["accuracy"],
                "hybrid_subset_accuracy": pair_subset_hybrid["accuracy"],
            }
        )

    backbone_metrics = _classification_metrics(labels, predictions, probabilities)
    hybrid_metrics = _classification_metrics(labels, hybrid_preds, hybrid_probs)
    trigger_backbone_metrics = _classification_metrics(labels[trigger_mask], predictions[trigger_mask], probabilities[trigger_mask])
    trigger_hybrid_metrics = _classification_metrics(labels[trigger_mask], hybrid_preds[trigger_mask], hybrid_probs[trigger_mask])
    win_rate = float(np.mean([row["hybrid_subset_accuracy"] > row["backbone_subset_accuracy"] for row in per_pair_rows])) if per_pair_rows else 0.0

    return {
        "backbone": {
            "overall_accuracy": backbone_metrics["accuracy"],
            "overall_macro_f1": backbone_metrics["macro_f1"],
            "overall_log_loss": backbone_metrics["log_loss"],
            "trigger_subset_count": int(trigger_mask.sum()),
            "trigger_subset_accuracy": trigger_backbone_metrics["accuracy"],
            "trigger_subset_macro_f1": trigger_backbone_metrics["macro_f1"],
        },
        "hybrid": {
            "overall_accuracy": hybrid_metrics["accuracy"],
            "overall_macro_f1": hybrid_metrics["macro_f1"],
            "overall_log_loss": hybrid_metrics["log_loss"],
            "trigger_subset_count": int(trigger_mask.sum()),
            "trigger_subset_accuracy": trigger_hybrid_metrics["accuracy"],
            "trigger_subset_macro_f1": trigger_hybrid_metrics["macro_f1"],
            "override_coverage": float(overrides_mask.sum() / max(trigger_mask.sum(), 1)),
            "correction_precision": float(corrected_mask.sum() / max(overrides_mask.sum(), 1)),
            "harm_rate": float(harmed_mask.sum() / max(overrides_mask.sum(), 1)),
            "net_gain": int(corrected_mask.sum() - harmed_mask.sum()),
            "pairwise_win_rate_over_backbone": win_rate,
            "per_pair_rows": per_pair_rows,
        },
        "hybrid_predictions": hybrid_preds,
        "override_mask": overrides_mask,
        "corrected_mask": corrected_mask,
        "harmed_mask": harmed_mask,
    }


def _build_pair_case_rows(
    *,
    pair_info: dict,
    full_eval: dict,
    top_eval: dict,
    test_outputs: dict,
    image_shape: list[int],
    grid_size: int,
    exemplar_count: int,
) -> dict:
    left_label = int(pair_info["left_label"])
    right_label = int(pair_info["right_label"])
    probs = torch.softmax(test_outputs["logits"], dim=1).cpu().numpy()
    top2_indices = np.argsort(probs, axis=1)[:, -2:][:, ::-1]
    pair_key = {left_label, right_label}
    trigger_mask = np.array([{int(row[0]), int(row[1])} == pair_key for row in top2_indices.tolist()], dtype=bool)
    trigger_indices = np.nonzero(trigger_mask)[0]

    def _collect(mask: np.ndarray) -> list[dict]:
        candidates = [idx for idx in trigger_indices.tolist() if mask[idx]]
        candidates = sorted(
            candidates,
            key=lambda idx: -abs(probs[idx, left_label] - probs[idx, right_label]),
        )[: int(exemplar_count)]
        return [
            _case_example(
                row_idx=idx,
                pair_info=pair_info,
                full_eval=full_eval,
                top_eval=top_eval,
                test_outputs=test_outputs,
                image_shape=image_shape,
                grid_size=grid_size,
            )
            for idx in candidates
        ]

    unchanged_mask = trigger_mask & ~top_eval["override_mask"]
    return {
        "left_label": left_label,
        "left_label_name": pair_info["left_label_name"],
        "right_label": right_label,
        "right_label_name": pair_info["right_label_name"],
        "trigger_pair_name": f"{pair_info['left_label_name']} vs {pair_info['right_label_name']}",
        "selected_top_nodes": pair_info["selected_top_nodes"],
        "corrected_examples": _collect(top_eval["corrected_mask"]),
        "harmed_examples": _collect(top_eval["harmed_mask"]),
        "unchanged_examples": _collect(unchanged_mask),
    }


def _case_example(
    *,
    row_idx: int,
    pair_info: dict,
    full_eval: dict,
    top_eval: dict,
    test_outputs: dict,
    image_shape: list[int],
    grid_size: int,
) -> dict:
    logits = test_outputs["logits"][row_idx : row_idx + 1]
    probs = torch.softmax(logits, dim=1)[0]
    backbone_top2 = torch.topk(probs, k=2).indices.tolist()
    top_nodes = pair_info["selected_top_nodes"]
    representative_node = [top_nodes[0]["layer_idx"], top_nodes[0]["cell_idx"]] if top_nodes else [0, 0]
    overlay = _overlay_spec(
        active_nodes=[[row["layer_idx"], row["cell_idx"]] for row in top_nodes],
        representative_node=representative_node,
        image_height=image_shape[0],
        image_width=image_shape[1],
        grid_size=grid_size,
    )
    return {
        "dataset_index": int(test_outputs["indices"][row_idx].item()),
        "row_index": int(row_idx),
        "true_label": int(test_outputs["labels"][row_idx].item()),
        "true_label_name": _label_name(int(test_outputs["labels"][row_idx].item())),
        "backbone_pred": int(logits.argmax(dim=1).item()),
        "backbone_pred_name": _label_name(int(logits.argmax(dim=1).item())),
        "backbone_top2": [int(value) for value in backbone_top2],
        "backbone_top2_names": [_label_name(int(value)) for value in backbone_top2],
        "full_z_hybrid_pred": int(full_eval["hybrid_predictions"][row_idx]),
        "full_z_hybrid_pred_name": _label_name(int(full_eval["hybrid_predictions"][row_idx])),
        "top_node_hybrid_pred": int(top_eval["hybrid_predictions"][row_idx]),
        "top_node_hybrid_pred_name": _label_name(int(top_eval["hybrid_predictions"][row_idx])),
        "trigger_pair_name": f"{pair_info['left_label_name']} vs {pair_info['right_label_name']}",
        "pairwise_probe_choice": _label_name(int(top_eval["hybrid_predictions"][row_idx])),
        "backbone_margin": float(_margin(logits).item()),
        "overlay": overlay,
    }


def _evaluate_multiclass_model(model_info: dict, x: np.ndarray, y: np.ndarray) -> dict:
    scaler = model_info["scaler"]
    model = model_info["model"]
    x_scaled = scaler.transform(x)
    pred = model.predict(x_scaled)
    probabilities = model.predict_proba(x_scaled)
    if probabilities.ndim == 1:
        probabilities = probabilities[:, None]
    return {
        "accuracy": float(accuracy_score(y, pred)) if y.size else 0.0,
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)) if y.size else 0.0,
        "log_loss": _safe_log_loss(y, probabilities),
        "top2_accuracy": _topk_accuracy(y, probabilities, k=2),
        "coef_norm": float(np.linalg.norm(model.coef_)) if hasattr(model, "coef_") else 0.0,
        "coef_norms": np.linalg.norm(model.coef_, axis=1).tolist() if hasattr(model, "coef_") else [],
    }


def _evaluate_binary_model(model_info: dict, x: np.ndarray, y: np.ndarray) -> dict:
    probabilities = _predict_binary_probabilities(model_info, x)
    pred = (probabilities >= 0.5).astype(int)
    result = _evaluate_binary_predictions(y, pred, probabilities)
    result["coef_norm"] = float(np.linalg.norm(model_info["model"].coef_)) if hasattr(model_info["model"], "coef_") else 0.0
    return result


def _evaluate_binary_predictions(y_true: np.ndarray, y_pred: np.ndarray, prob_right: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)) if y_true.size else 0.0,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if y_true.size else 0.0,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if y_true.size else 0.0,
        "auroc": _safe_binary_auroc(y_true, prob_right),
        "brier_score": float(brier_score_loss(y_true, prob_right)) if y_true.size else 0.0,
    }


def _classification_metrics(labels: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray) -> dict:
    if labels.size == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0, "log_loss": 0.0}
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "log_loss": _safe_log_loss(labels, probabilities),
    }


def _predict_binary_probabilities(model_info: dict, x: np.ndarray) -> np.ndarray:
    scaler = model_info["scaler"]
    model = model_info["model"]
    probabilities = model.predict_proba(scaler.transform(x))
    if probabilities.shape[1] == 1:
        return np.zeros(probabilities.shape[0], dtype=np.float64)
    return probabilities[:, -1].astype(np.float64)


def _top_node_features(z: np.ndarray, node_rows: list[dict]) -> np.ndarray:
    if not node_rows:
        return np.zeros((z.shape[0], 0), dtype=np.float64)
    pieces = [_node_probe_features(z, row["layer_idx"], row["cell_idx"]) for row in node_rows]
    return np.concatenate(pieces, axis=1)


def _topk_accuracy(labels: np.ndarray, probabilities: np.ndarray, *, k: int) -> float:
    if labels.size == 0 or probabilities.size == 0:
        return 0.0
    topk = np.argsort(probabilities, axis=1)[:, -int(k) :]
    return float(np.mean([int(label) in row for label, row in zip(labels.tolist(), topk.tolist())]))


def _safe_log_loss(labels: np.ndarray, probabilities: np.ndarray) -> float:
    if labels.size == 0 or probabilities.size == 0:
        return 0.0
    try:
        return float(log_loss(labels, probabilities, labels=np.arange(probabilities.shape[1])))
    except ValueError:
        return 0.0


def _safe_binary_auroc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    if labels.size == 0 or len(np.unique(labels)) < 2:
        return 0.0
    return float(roc_auc_score(labels, probabilities))


def _empty_pair_hybrid_row(pair_info: dict) -> dict:
    return {
        "left_label": int(pair_info["left_label"]),
        "left_label_name": pair_info["left_label_name"],
        "right_label": int(pair_info["right_label"]),
        "right_label_name": pair_info["right_label_name"],
        "trigger_count": 0,
        "overrides": 0,
        "corrected": 0,
        "harmed": 0,
        "backbone_subset_accuracy": 0.0,
        "hybrid_subset_accuracy": 0.0,
    }


def _class_coefficient_norms(coef_norms: list[float]) -> list[dict]:
    return [
        {
            "class_idx": int(class_idx),
            "class_name": CIFAR10_CLASS_NAMES[int(class_idx)],
            "coefficient_norm": float(value),
        }
        for class_idx, value in enumerate(coef_norms)
    ]
