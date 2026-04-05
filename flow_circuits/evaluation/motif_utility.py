from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from flow_circuits.data import build_cifar10_corruption_splits, build_cifar10_splits
from flow_circuits.evaluation.hard_pair_correction import (
    _classification_metrics,
    _entropy,
    _hybrid_summary_from_predictions,
    _predict_binary_probabilities,
    _public_hybrid_summary,
    _top1_margin,
)
from flow_circuits.evaluation.interpretability_validation import (
    _ProgressTracker,
    _collect_probe_splits,
    _fit_probe_model,
    _label_name,
    _margin,
    _maybe_write_json,
    _top_confusion_pairs,
)
from flow_circuits.evaluation.motif_validation import _motif_scores, _rank_motifs
from flow_circuits.training import LoadedFlowComponents, collect_interpretability_outputs


MOTIF_CLEAN_UTILITY_ID = "motif_clean_utility"
MOTIF_CORRUPTION_UTILITY_ID = "motif_corruption_utility"


def run_motif_clean_utility_experiment(
    components: LoadedFlowComponents,
    motif_artifact: dict,
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
    trigger_mode: str,
    margin_quantile: float,
    top_motif_fraction: float,
    min_top_motifs: int,
    max_top_motifs: int,
    c_grid: tuple[float, ...] = (0.1, 1.0, 10.0),
    output_path: str | Path | None = None,
    progress_callback=None,
) -> dict:
    tracker = _ProgressTracker(
        experiment=MOTIF_CLEAN_UTILITY_ID,
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
        stage_prefix=MOTIF_CLEAN_UTILITY_ID,
    )
    result = _run_motif_utility_from_outputs(
        motif_artifact=motif_artifact,
        fit_outputs=fit_outputs,
        val_outputs=val_outputs,
        test_outputs=test_outputs,
        checkpoint_tag=checkpoint_tag,
        top_pairs=top_pairs,
        trigger_mode=trigger_mode,
        margin_quantile=margin_quantile,
        top_motif_fraction=top_motif_fraction,
        min_top_motifs=min_top_motifs,
        max_top_motifs=max_top_motifs,
        c_grid=c_grid,
        tracker=tracker,
    )
    _maybe_write_json(result, output_path)
    return result


def run_motif_corruption_utility_experiment(
    components: LoadedFlowComponents,
    motif_artifact: dict,
    *,
    device: torch.device,
    checkpoint_tag: str,
    data_dir: str,
    batch_size: int,
    corruption_names: list[str] | tuple[str, ...],
    severities: list[int] | tuple[int, ...],
    fit_max_images: int,
    val_max_images: int,
    test_max_images: int,
    top_pairs: int,
    trigger_mode: str,
    margin_quantile: float,
    top_motif_fraction: float,
    min_top_motifs: int,
    max_top_motifs: int,
    num_workers: int = 4,
    seed: int = 0,
    augment_fit: bool = True,
    download: bool = True,
    c_grid: tuple[float, ...] = (0.1, 1.0, 10.0),
    output_path: str | Path | None = None,
    progress_callback=None,
) -> dict:
    tracker = _ProgressTracker(
        experiment=MOTIF_CORRUPTION_UTILITY_ID,
        checkpoint_tag=checkpoint_tag,
        progress_callback=progress_callback,
    )
    clean_loaders = build_cifar10_splits(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        augment_fit=augment_fit,
        download=download,
    )
    fit_outputs = collect_interpretability_outputs(
        components,
        clean_loaders["fit"],
        device=device,
        max_images=fit_max_images,
    )
    rows: list[dict] = []
    total = len(corruption_names) * len(severities)
    completed = 0
    for corruption_name in [str(name) for name in corruption_names]:
        for severity in [int(value) for value in severities]:
            tracker.emit(
                stage="corruption_suite",
                completed=completed,
                total=total,
                message=f"running {corruption_name} severity {severity}",
            )
            loaders = build_cifar10_corruption_splits(
                data_dir=data_dir,
                batch_size=batch_size,
                corruption_name=corruption_name,
                severity=severity,
                num_workers=num_workers,
                seed=seed,
                augment_fit=augment_fit,
                download=download,
            )
            val_outputs = collect_interpretability_outputs(
                components,
                loaders["val"],
                device=device,
                max_images=val_max_images,
            )
            test_outputs = collect_interpretability_outputs(
                components,
                loaders["test"],
                device=device,
                max_images=test_max_images,
            )
            utility = _run_motif_utility_from_outputs(
                motif_artifact=motif_artifact,
                fit_outputs=fit_outputs,
                val_outputs=val_outputs,
                test_outputs=test_outputs,
                checkpoint_tag=checkpoint_tag,
                top_pairs=top_pairs,
                trigger_mode=trigger_mode,
                margin_quantile=margin_quantile,
                top_motif_fraction=top_motif_fraction,
                min_top_motifs=min_top_motifs,
                max_top_motifs=max_top_motifs,
                c_grid=c_grid,
                tracker=tracker,
            )
            rows.append(
                {
                    "corruption_name": corruption_name,
                    "severity": int(severity),
                    "backbone_overall_accuracy": float(utility["summary"]["backbone_overall_accuracy"]),
                    "full_motif_overall_accuracy": float(utility["summary"]["full_motif_overall_accuracy"]),
                    "top_motif_overall_accuracy": float(utility["summary"]["top_motif_overall_accuracy"]),
                    "backbone_trigger_accuracy": float(utility["summary"]["backbone_trigger_accuracy"]),
                    "full_motif_trigger_accuracy": float(utility["summary"]["full_motif_trigger_accuracy"]),
                    "top_motif_trigger_accuracy": float(utility["summary"]["top_motif_trigger_accuracy"]),
                    "trigger_coverage": float(utility["summary"]["trigger_coverage"]),
                    "full_motif_net_gain": int(utility["summary"]["full_motif_net_gain"]),
                    "top_motif_net_gain": int(utility["summary"]["top_motif_net_gain"]),
                    "full_motif_gain_vs_backbone": float(utility["summary"]["full_motif_overall_accuracy"] - utility["summary"]["backbone_overall_accuracy"]),
                    "top_motif_gain_vs_backbone": float(utility["summary"]["top_motif_overall_accuracy"] - utility["summary"]["backbone_overall_accuracy"]),
                }
            )
            completed += 1
            tracker.emit(
                stage="corruption_suite",
                completed=completed,
                total=total,
                message=f"finished {corruption_name} severity {severity}",
            )
    result = {
        "experiment": MOTIF_CORRUPTION_UTILITY_ID,
        "checkpoint_tag": checkpoint_tag,
        "summary": {
            "mean_backbone_overall_accuracy": float(np.mean([row["backbone_overall_accuracy"] for row in rows])) if rows else 0.0,
            "mean_full_motif_overall_accuracy": float(np.mean([row["full_motif_overall_accuracy"] for row in rows])) if rows else 0.0,
            "mean_top_motif_overall_accuracy": float(np.mean([row["top_motif_overall_accuracy"] for row in rows])) if rows else 0.0,
            "mean_full_motif_gain_vs_backbone": float(np.mean([row["full_motif_gain_vs_backbone"] for row in rows])) if rows else 0.0,
            "mean_top_motif_gain_vs_backbone": float(np.mean([row["top_motif_gain_vs_backbone"] for row in rows])) if rows else 0.0,
        },
        "rows": rows,
    }
    _maybe_write_json(result, output_path)
    return result


def _run_motif_utility_from_outputs(
    *,
    motif_artifact: dict,
    fit_outputs: dict,
    val_outputs: dict,
    test_outputs: dict,
    checkpoint_tag: str,
    top_pairs: int,
    trigger_mode: str,
    margin_quantile: float,
    top_motif_fraction: float,
    min_top_motifs: int,
    max_top_motifs: int,
    c_grid: tuple[float, ...],
    tracker: _ProgressTracker | None = None,
) -> dict:
    ranked_motifs = _rank_motifs(motif_artifact.get("motifs", []))
    if not ranked_motifs:
        raise ValueError("Motif utility requires at least one retained motif.")

    if tracker is not None:
        tracker.emit(
            stage="motif_feature_building",
            completed=1,
            total=4,
            message="building fit motif activation features",
        )
    fit_features = _motif_feature_matrix(ranked_motifs, fit_outputs["z"])
    if tracker is not None:
        tracker.emit(
            stage="motif_feature_building",
            completed=2,
            total=4,
            message="building validation motif activation features",
        )
    val_features = _motif_feature_matrix(ranked_motifs, val_outputs["z"])
    if tracker is not None:
        tracker.emit(
            stage="motif_feature_building",
            completed=3,
            total=4,
            message="building test motif activation features",
        )
    test_features = _motif_feature_matrix(ranked_motifs, test_outputs["z"])

    val_labels = val_outputs["labels"].numpy()
    val_probs = torch.softmax(val_outputs["logits"], dim=1).cpu().numpy()
    val_pred = val_probs.argmax(axis=1)
    pair_counts = _top_confusion_pairs(val_labels, val_pred, top_pairs=int(top_pairs))
    if not pair_counts:
        raise ValueError("Motif utility requires at least one non-zero hard pair.")
    if tracker is not None:
        pair_names = ", ".join(f"{_label_name(left)} vs {_label_name(right)}" for left, right, _ in pair_counts)
        tracker.emit(
            stage="hard_pair_selection",
            completed=len(pair_counts),
            total=len(pair_counts),
            message=f"selected hard pairs: {pair_names}",
        )

    margin_cutoff = float(np.quantile(_top1_margin(val_probs), float(margin_quantile)))
    selected_top_indices = _select_top_motifs(
        motifs=ranked_motifs,
        fit_features=fit_features,
        fit_labels=fit_outputs["labels"].numpy(),
        val_features=val_features,
        val_outputs=val_outputs,
        pair_counts=pair_counts,
        margin_cutoff=margin_cutoff,
        trigger_mode=trigger_mode,
        top_motif_fraction=top_motif_fraction,
        min_top_motifs=min_top_motifs,
        max_top_motifs=max_top_motifs,
        c_grid=c_grid,
        tracker=tracker,
    )
    if tracker is not None:
        tracker.emit(
            stage="motif_feature_building",
            completed=4,
            total=4,
            message=f"selected {len(selected_top_indices)} top motifs for compact hybrid",
        )
    pair_infos = _fit_pair_models(
        ranked_motifs=ranked_motifs,
        fit_features=fit_features,
        val_features=val_features,
        test_features=test_features,
        fit_outputs=fit_outputs,
        val_outputs=val_outputs,
        test_outputs=test_outputs,
        pair_counts=pair_counts,
        selected_top_indices=selected_top_indices,
        c_grid=c_grid,
        tracker=tracker,
    )
    evaluation = _evaluate_motif_hybrids(
        pair_infos=pair_infos,
        val_outputs=val_outputs,
        test_outputs=test_outputs,
        test_features=test_features,
        trigger_mode=trigger_mode,
        margin_quantile=margin_quantile,
        tracker=tracker,
    )
    return {
        "experiment": MOTIF_CLEAN_UTILITY_ID,
        "checkpoint_tag": checkpoint_tag,
        "summary": {
            "n_pairs": int(len(pair_infos)),
            "backbone_overall_accuracy": evaluation["backbone"]["overall_accuracy"],
            "full_motif_overall_accuracy": evaluation["full_motif"]["overall_accuracy"],
            "top_motif_overall_accuracy": evaluation["top_motif"]["overall_accuracy"],
            "backbone_trigger_accuracy": evaluation["backbone"]["trigger_subset_accuracy"],
            "full_motif_trigger_accuracy": evaluation["full_motif"]["trigger_subset_accuracy"],
            "top_motif_trigger_accuracy": evaluation["top_motif"]["trigger_subset_accuracy"],
            "trigger_coverage": evaluation["top_motif"]["trigger_coverage"],
            "full_motif_net_gain": evaluation["full_motif"]["net_gain"],
            "top_motif_net_gain": evaluation["top_motif"]["net_gain"],
        },
        "selected_top_motif_ids": [int(ranked_motifs[idx]["id"]) for idx in selected_top_indices],
        "selected_top_motif_indices": [int(idx) for idx in selected_top_indices],
        "full_motif": _public_hybrid_summary(evaluation["full_motif"]),
        "top_motif": _public_hybrid_summary(evaluation["top_motif"]),
        "backbone": _public_hybrid_summary(evaluation["backbone"]),
        "pair_rows": [row["summary"] for row in pair_infos],
    }


def _motif_feature_matrix(ranked_motifs: list[dict], z_tensor: torch.Tensor) -> np.ndarray:
    normalized = torch.nn.functional.normalize(z_tensor, dim=-1)
    columns = [(_motif_scores(motif, normalized).detach().cpu().numpy()) for motif in ranked_motifs]
    return np.stack(columns, axis=1) if columns else np.zeros((z_tensor.shape[0], 0), dtype=np.float32)


def _select_top_motifs(
    *,
    motifs: list[dict],
    fit_features: np.ndarray,
    fit_labels: np.ndarray,
    val_features: np.ndarray,
    val_outputs: dict,
    pair_counts: list[tuple[int, int, int]],
    margin_cutoff: float,
    trigger_mode: str,
    top_motif_fraction: float,
    min_top_motifs: int,
    max_top_motifs: int,
    c_grid: tuple[float, ...],
    tracker: _ProgressTracker | None = None,
) -> list[int]:
    val_probs = torch.softmax(val_outputs["logits"], dim=1).cpu().numpy()
    val_predictions = val_probs.argmax(axis=1)
    val_labels = val_outputs["labels"].numpy()
    top2_indices = np.argsort(val_probs, axis=1)[:, -2:][:, ::-1]
    motif_stats: list[dict] = []
    for motif_idx, motif in enumerate(motifs, start=1):
        motif_col = motif_idx - 1
        corrected = 0
        harmed = 0
        overrides = 0
        trigger_count = 0
        for left_label, right_label, _ in pair_counts:
            fit_mask = np.isin(fit_labels, [left_label, right_label])
            val_mask = np.isin(val_labels, [left_label, right_label])
            fit_y = (fit_labels[fit_mask] == right_label).astype(np.int64)
            val_y = (val_labels[val_mask] == right_label).astype(np.int64)
            model = _fit_probe_model(
                fit_features[fit_mask, motif_col : motif_col + 1],
                fit_y,
                val_features[val_mask, motif_col : motif_col + 1],
                val_y,
                c_grid=c_grid,
            )
            pair_key = {int(left_label), int(right_label)}
            pair_top2 = np.array([{int(row[0]), int(row[1])} == pair_key for row in top2_indices.tolist()], dtype=bool)
            pair_trigger = pair_top2 & (_top1_margin(val_probs) <= margin_cutoff) if trigger_mode == "hard_pair_top2_and_low_margin" else pair_top2
            if not np.any(pair_trigger):
                continue
            pair_indices = np.nonzero(pair_trigger)[0]
            pair_prob_right = _predict_binary_probabilities(model, val_features[pair_trigger, motif_col : motif_col + 1])
            trigger_count += int(pair_trigger.sum())
            for local_idx, row_idx in enumerate(pair_indices.tolist()):
                right_prob = float(pair_prob_right[local_idx])
                new_pred = int(right_label if right_prob >= 0.5 else left_label)
                old_pred = int(val_predictions[row_idx])
                if new_pred != old_pred:
                    overrides += 1
                    if old_pred != int(val_labels[row_idx]) and new_pred == int(val_labels[row_idx]):
                        corrected += 1
                    elif old_pred == int(val_labels[row_idx]) and new_pred != int(val_labels[row_idx]):
                        harmed += 1
        precision = float(corrected / max(overrides, 1))
        motif_stats.append(
            {
                "motif_idx": int(motif_col),
                "net_gain": int(corrected - harmed),
                "correction_precision": precision,
                "stability": float(motif.get("stability", {}).get("mean_cluster_stability", 0.0)),
                "layer_span": int(len(motif.get("layer_support", []))),
                "trigger_count": int(trigger_count),
            }
        )
        if tracker is not None:
            tracker.emit(
                stage="top_motif_selection",
                completed=motif_idx,
                total=len(motifs),
                message="scoring motifs by validation correction utility",
            )
    requested = int(np.ceil(len(motifs) * float(top_motif_fraction)))
    top_k = max(int(min_top_motifs), min(int(max_top_motifs), max(1, requested)))
    top_k = min(top_k, len(motifs))
    ranked = sorted(
        motif_stats,
        key=lambda row: (
            -row["net_gain"],
            -row["correction_precision"],
            -row["stability"],
            -row["layer_span"],
            row["motif_idx"],
        ),
    )
    return [int(row["motif_idx"]) for row in ranked[:top_k]]


def _fit_pair_models(
    *,
    ranked_motifs: list[dict],
    fit_features: np.ndarray,
    val_features: np.ndarray,
    test_features: np.ndarray,
    fit_outputs: dict,
    val_outputs: dict,
    test_outputs: dict,
    pair_counts: list[tuple[int, int, int]],
    selected_top_indices: list[int],
    c_grid: tuple[float, ...],
    tracker: _ProgressTracker | None = None,
) -> list[dict]:
    fit_labels = fit_outputs["labels"].numpy()
    val_labels = val_outputs["labels"].numpy()
    test_labels = test_outputs["labels"].numpy()
    rows = []
    for pair_idx, (left_label, right_label, count) in enumerate(pair_counts, start=1):
        fit_mask = np.isin(fit_labels, [left_label, right_label])
        val_mask = np.isin(val_labels, [left_label, right_label])
        test_mask = np.isin(test_labels, [left_label, right_label])
        fit_y = (fit_labels[fit_mask] == right_label).astype(np.int64)
        val_y = (val_labels[val_mask] == right_label).astype(np.int64)
        test_y = (test_labels[test_mask] == right_label).astype(np.int64)
        full_model = _fit_probe_model(
            fit_features[fit_mask],
            fit_y,
            val_features[val_mask],
            val_y,
            c_grid=c_grid,
        )
        top_model = _fit_probe_model(
            fit_features[fit_mask][:, selected_top_indices],
            fit_y,
            val_features[val_mask][:, selected_top_indices],
            val_y,
            c_grid=c_grid,
        )
        full_prob = _predict_binary_probabilities(full_model, test_features[test_mask]) if np.any(test_mask) else np.zeros(0, dtype=np.float64)
        top_prob = _predict_binary_probabilities(top_model, test_features[test_mask][:, selected_top_indices]) if np.any(test_mask) else np.zeros(0, dtype=np.float64)
        full_pred = (full_prob >= 0.5).astype(np.int64)
        top_pred = (top_prob >= 0.5).astype(np.int64)
        rows.append(
            {
                "left_label": int(left_label),
                "left_label_name": _label_name(left_label),
                "right_label": int(right_label),
                "right_label_name": _label_name(right_label),
                "backbone_confusion_count": int(count),
                "full_model": full_model,
                "top_model": top_model,
                "selected_top_indices": [int(idx) for idx in selected_top_indices],
                "summary": {
                    "left_label": int(left_label),
                    "left_label_name": _label_name(left_label),
                    "right_label": int(right_label),
                    "right_label_name": _label_name(right_label),
                    "backbone_confusion_count": int(count),
                    "backbone_accuracy": _classification_metrics(
                        test_labels[test_mask],
                        test_outputs["logits"].argmax(dim=1).numpy()[test_mask],
                        torch.softmax(test_outputs["logits"], dim=1).cpu().numpy()[test_mask],
                    )["accuracy"]
                    if np.any(test_mask)
                    else 0.0,
                    "full_motif_accuracy": float(np.mean(full_pred == test_y)) if test_y.size else 0.0,
                    "top_motif_accuracy": float(np.mean(top_pred == test_y)) if test_y.size else 0.0,
                },
            }
        )
        if tracker is not None:
            tracker.emit(
                stage="pair_model_fitting",
                completed=pair_idx,
                total=len(pair_counts),
                message=f"fitting motif probes for {_label_name(left_label)} vs {_label_name(right_label)}",
            )
    return rows


def _evaluate_motif_hybrids(
    *,
    pair_infos: list[dict],
    val_outputs: dict,
    test_outputs: dict,
    test_features: np.ndarray,
    trigger_mode: str,
    margin_quantile: float,
    tracker: _ProgressTracker | None = None,
) -> dict:
    test_logits = test_outputs["logits"]
    test_probs = torch.softmax(test_logits, dim=1).cpu().numpy()
    val_probs = torch.softmax(val_outputs["logits"], dim=1).cpu().numpy()
    labels = test_outputs["labels"].numpy()
    base_predictions = test_probs.argmax(axis=1)
    top2_indices = np.argsort(test_probs, axis=1)[:, -2:][:, ::-1]
    margin_cutoff = float(np.quantile(_top1_margin(val_probs), float(margin_quantile)))
    test_margins = _top1_margin(test_probs)

    results = {}
    for mode_idx, mode_name in enumerate(("full_motif", "top_motif"), start=1):
        hybrid_probs = test_probs.copy()
        hybrid_preds = base_predictions.copy()
        trigger_mask = np.zeros(labels.shape[0], dtype=bool)
        overrides_mask = np.zeros(labels.shape[0], dtype=bool)
        corrected_mask = np.zeros(labels.shape[0], dtype=bool)
        harmed_mask = np.zeros(labels.shape[0], dtype=bool)
        pair_confidences = np.ones(labels.shape[0], dtype=np.float64)
        per_pair_rows = []
        for pair_info in pair_infos:
            left_label = int(pair_info["left_label"])
            right_label = int(pair_info["right_label"])
            pair_key = {left_label, right_label}
            pair_top2 = np.array([{int(row[0]), int(row[1])} == pair_key for row in top2_indices.tolist()], dtype=bool)
            if trigger_mode == "hard_pair_top2_and_low_margin":
                pair_trigger = pair_top2 & (test_margins <= margin_cutoff)
            elif trigger_mode == "hard_pair_top2":
                pair_trigger = pair_top2
            else:
                pair_trigger = pair_top2 & (_entropy(test_probs) >= float(np.quantile(_entropy(val_probs), 0.75)))
            trigger_mask |= pair_trigger
            if not np.any(pair_trigger):
                per_pair_rows.append(
                    {
                        "left_label": left_label,
                        "left_label_name": pair_info["left_label_name"],
                        "right_label": right_label,
                        "right_label_name": pair_info["right_label_name"],
                        "trigger_count": 0,
                        "overrides": 0,
                        "corrected": 0,
                        "harmed": 0,
                        "backbone_subset_accuracy": 0.0,
                        "hybrid_subset_accuracy": 0.0,
                    }
                )
                continue
            pair_indices = np.nonzero(pair_trigger)[0]
            if mode_name == "full_motif":
                pair_prob_right = _predict_binary_probabilities(pair_info["full_model"], test_features[pair_trigger])
            else:
                pair_prob_right = _predict_binary_probabilities(
                    pair_info["top_model"],
                    test_features[pair_trigger][:, pair_info["selected_top_indices"]],
                )
            pair_overrides = 0
            pair_corrected = 0
            pair_harmed = 0
            for local_idx, row_idx in enumerate(pair_indices.tolist()):
                right_prob = float(pair_prob_right[local_idx])
                pair_confidences[row_idx] = max(right_prob, 1.0 - right_prob)
                pair_mass = hybrid_probs[row_idx, left_label] + hybrid_probs[row_idx, right_label]
                hybrid_probs[row_idx, left_label] = pair_mass * (1.0 - right_prob)
                hybrid_probs[row_idx, right_label] = pair_mass * right_prob
                new_pred = int(right_label if right_prob >= 0.5 else left_label)
                old_pred = int(base_predictions[row_idx])
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
            pair_subset_backbone = _classification_metrics(labels[pair_trigger], base_predictions[pair_trigger], test_probs[pair_trigger])
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
        results[mode_name] = _hybrid_summary_from_predictions(
            labels=labels,
            predictions=hybrid_preds,
            probabilities=hybrid_probs,
            trigger_mask=trigger_mask,
            overrides_mask=overrides_mask,
            corrected_mask=corrected_mask,
            harmed_mask=harmed_mask,
            per_pair_rows=per_pair_rows,
            pair_confidences=pair_confidences,
        )
        if tracker is not None:
            tracker.emit(
                stage="hybrid_evaluation",
                completed=mode_idx,
                total=2,
                message=f"evaluated {mode_name.replace('_', ' ')} hybrid on held-out data",
            )
    backbone_per_pair_rows = [
        {
            "left_label": row["summary"]["left_label"],
            "left_label_name": row["summary"]["left_label_name"],
            "right_label": row["summary"]["right_label"],
            "right_label_name": row["summary"]["right_label_name"],
            "trigger_count": pair_row["trigger_count"],
            "overrides": 0,
            "corrected": 0,
            "harmed": 0,
            "backbone_subset_accuracy": pair_row["backbone_subset_accuracy"],
            "hybrid_subset_accuracy": pair_row["backbone_subset_accuracy"],
        }
        for row, pair_row in zip(pair_infos, results["top_motif"]["per_pair_rows"])
    ]
    backbone = _hybrid_summary_from_predictions(
        labels=labels,
        predictions=base_predictions,
        probabilities=test_probs,
        trigger_mask=results["top_motif"]["trigger_mask"],
        overrides_mask=np.zeros(labels.shape[0], dtype=bool),
        corrected_mask=np.zeros(labels.shape[0], dtype=bool),
        harmed_mask=np.zeros(labels.shape[0], dtype=bool),
        per_pair_rows=backbone_per_pair_rows,
        pair_confidences=test_probs.max(axis=1),
    )
    return {
        "backbone": backbone,
        "full_motif": results["full_motif"],
        "top_motif": results["top_motif"],
    }
