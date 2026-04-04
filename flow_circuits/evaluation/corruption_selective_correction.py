from __future__ import annotations

import math
from pathlib import Path

from flow_circuits.data import CIFAR10_CORRUPTION_NAMES, build_cifar10_corruption_splits
from flow_circuits.evaluation.hard_pair_correction import run_selective_hybrid_correction_experiment
from flow_circuits.evaluation.motif_validation import _ProgressTracker, _maybe_write_json
from flow_circuits.training import LoadedFlowComponents


CORRUPTION_SWEEP_ID = "corruption_sweep"
TOP_NODE_SUBSET_SWEEP_ID = "top_node_subset_sweep"

NB07_EXPERIMENT_IDS = [
    CORRUPTION_SWEEP_ID,
    TOP_NODE_SUBSET_SWEEP_ID,
]


def run_corruption_sweep_experiment(
    components: LoadedFlowComponents,
    *,
    device,
    checkpoint_tag: str,
    data_dir: str,
    batch_size: int,
    corruption_names: list[str] | tuple[str, ...],
    severities: list[int] | tuple[int, ...],
    fit_max_images: int,
    val_max_images: int,
    test_max_images: int,
    top_pairs: int,
    top_node_fraction: float,
    top_node_min_k: int,
    top_node_max_k: int,
    trigger_mode: str,
    margin_threshold: float | None = None,
    margin_quantile: float = 0.25,
    entropy_threshold: float | None = None,
    entropy_quantile: float = 0.75,
    num_workers: int = 4,
    seed: int = 0,
    augment_fit: bool = True,
    download: bool = True,
    n_jobs: int | None = None,
    output_path: str | Path | None = None,
    progress_callback=None,
) -> dict:
    tracker = _ProgressTracker(
        experiment=CORRUPTION_SWEEP_ID,
        checkpoint_tag=checkpoint_tag,
        progress_callback=progress_callback,
    )
    resolved_corruptions = _validate_corruptions(corruption_names)
    resolved_severities = [int(severity) for severity in severities]
    top_k_nodes = _resolve_top_k_nodes(
        components,
        top_node_fraction=top_node_fraction,
        top_node_min_k=top_node_min_k,
        top_node_max_k=top_node_max_k,
    )
    rows: list[dict] = []
    total = len(resolved_corruptions) * len(resolved_severities)
    completed = 0
    for corruption_name in resolved_corruptions:
        for severity in resolved_severities:
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
            nested_output_path = _nested_output_path(
                output_path,
                subdir=f"{corruption_name}_severity_{severity}",
            )
            result = run_selective_hybrid_correction_experiment(
                components,
                loaders["fit"],
                loaders["val"],
                loaders["test"],
                device=device,
                checkpoint_tag=checkpoint_tag,
                fit_max_images=fit_max_images,
                val_max_images=val_max_images,
                test_max_images=test_max_images,
                top_pairs=top_pairs,
                top_k_nodes=top_k_nodes,
                trigger_mode=trigger_mode,
                margin_threshold=margin_threshold,
                margin_quantile=margin_quantile,
                entropy_threshold=entropy_threshold,
                entropy_quantile=entropy_quantile,
                n_jobs=n_jobs,
                output_path=nested_output_path,
                progress_callback=None,
            )
            rows.append(
                {
                    "corruption_name": corruption_name,
                    "severity": int(severity),
                    "top_k_nodes": int(top_k_nodes),
                    "backbone_overall_accuracy": float(result["summary"]["backbone_overall_accuracy"]),
                    "full_z_hybrid_overall_accuracy": float(result["summary"]["full_z_hybrid_overall_accuracy"]),
                    "top_node_subset_overall_accuracy": float(result["summary"]["top_node_hybrid_overall_accuracy"]),
                    "backbone_trigger_accuracy": float(result["summary"]["backbone_trigger_accuracy"]),
                    "full_z_hybrid_trigger_accuracy": float(result["summary"]["full_z_hybrid_trigger_accuracy"]),
                    "top_node_subset_trigger_accuracy": float(result["summary"]["top_node_hybrid_trigger_accuracy"]),
                    "trigger_coverage": float(result["summary"]["trigger_coverage"]),
                    "full_z_net_gain": int(result["summary"]["full_z_net_gain"]),
                    "top_node_subset_net_gain": int(result["summary"]["top_node_net_gain"]),
                    "full_z_gain_vs_backbone": float(result["summary"]["full_z_hybrid_overall_accuracy"] - result["summary"]["backbone_overall_accuracy"]),
                    "top_node_subset_gain_vs_backbone": float(result["summary"]["top_node_hybrid_overall_accuracy"] - result["summary"]["backbone_overall_accuracy"]),
                    "full_z_gain_per_100_triggered": float(result["full_z_hybrid"]["gain_per_100_triggered"]),
                    "top_node_subset_gain_per_100_triggered": float(result["top_node_hybrid"]["gain_per_100_triggered"]),
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
        "experiment": CORRUPTION_SWEEP_ID,
        "checkpoint_tag": checkpoint_tag,
        "summary": _corruption_summary(rows),
        "rows": rows,
        "config": {
            "corruption_names": list(resolved_corruptions),
            "severities": resolved_severities,
            "top_node_fraction": float(top_node_fraction),
            "top_node_min_k": int(top_node_min_k),
            "top_node_max_k": int(top_node_max_k),
            "resolved_top_k_nodes": int(top_k_nodes),
            "trigger_mode": trigger_mode,
        },
    }
    _maybe_write_json(result, output_path)
    return result


def run_top_node_subset_sweep_experiment(
    components: LoadedFlowComponents,
    *,
    device,
    checkpoint_tag: str,
    data_dir: str,
    batch_size: int,
    corruption_name: str,
    severity: int,
    top_node_fractions: list[float] | tuple[float, ...],
    fit_max_images: int,
    val_max_images: int,
    test_max_images: int,
    top_pairs: int,
    top_node_min_k: int,
    top_node_max_k: int,
    trigger_mode: str,
    margin_threshold: float | None = None,
    margin_quantile: float = 0.25,
    entropy_threshold: float | None = None,
    entropy_quantile: float = 0.75,
    num_workers: int = 4,
    seed: int = 0,
    augment_fit: bool = True,
    download: bool = True,
    n_jobs: int | None = None,
    output_path: str | Path | None = None,
    progress_callback=None,
) -> dict:
    tracker = _ProgressTracker(
        experiment=TOP_NODE_SUBSET_SWEEP_ID,
        checkpoint_tag=checkpoint_tag,
        progress_callback=progress_callback,
    )
    corruption_name = _validate_corruptions([corruption_name])[0]
    severity = int(severity)
    fractions = [float(value) for value in top_node_fractions]
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
    rows: list[dict] = []
    total = len(fractions)
    for completed, fraction in enumerate(fractions, start=1):
        top_k_nodes = _resolve_top_k_nodes(
            components,
            top_node_fraction=fraction,
            top_node_min_k=top_node_min_k,
            top_node_max_k=top_node_max_k,
        )
        tracker.emit(
            stage="top_node_subset",
            completed=completed - 1,
            total=total,
            message=f"running top-node subset fraction {fraction:.3f}",
        )
        nested_output_path = _nested_output_path(
            output_path,
            subdir=f"{corruption_name}_severity_{severity}_fraction_{_fraction_key(fraction)}",
        )
        result = run_selective_hybrid_correction_experiment(
            components,
            loaders["fit"],
            loaders["val"],
            loaders["test"],
            device=device,
            checkpoint_tag=checkpoint_tag,
            fit_max_images=fit_max_images,
            val_max_images=val_max_images,
            test_max_images=test_max_images,
            top_pairs=top_pairs,
            top_k_nodes=top_k_nodes,
            trigger_mode=trigger_mode,
            margin_threshold=margin_threshold,
            margin_quantile=margin_quantile,
            entropy_threshold=entropy_threshold,
            entropy_quantile=entropy_quantile,
            n_jobs=n_jobs,
            output_path=nested_output_path,
            progress_callback=None,
        )
        rows.append(
            {
                "top_node_fraction": float(fraction),
                "top_k_nodes": int(top_k_nodes),
                "backbone_overall_accuracy": float(result["summary"]["backbone_overall_accuracy"]),
                "full_z_hybrid_overall_accuracy": float(result["summary"]["full_z_hybrid_overall_accuracy"]),
                "top_node_subset_overall_accuracy": float(result["summary"]["top_node_hybrid_overall_accuracy"]),
                "backbone_trigger_accuracy": float(result["summary"]["backbone_trigger_accuracy"]),
                "full_z_hybrid_trigger_accuracy": float(result["summary"]["full_z_hybrid_trigger_accuracy"]),
                "top_node_subset_trigger_accuracy": float(result["summary"]["top_node_hybrid_trigger_accuracy"]),
                "trigger_coverage": float(result["summary"]["trigger_coverage"]),
                "top_node_subset_net_gain": int(result["summary"]["top_node_net_gain"]),
                "top_node_subset_gain_vs_backbone": float(result["summary"]["top_node_hybrid_overall_accuracy"] - result["summary"]["backbone_overall_accuracy"]),
                "top_node_subset_gap_to_full_z": float(result["summary"]["full_z_hybrid_overall_accuracy"] - result["summary"]["top_node_hybrid_overall_accuracy"]),
                "top_node_subset_gain_per_100_triggered": float(result["top_node_hybrid"]["gain_per_100_triggered"]),
            }
        )
        tracker.emit(
            stage="top_node_subset",
            completed=completed,
            total=total,
            message=f"finished top-node subset fraction {fraction:.3f}",
        )
    result = {
        "experiment": TOP_NODE_SUBSET_SWEEP_ID,
        "checkpoint_tag": checkpoint_tag,
        "summary": _top_node_subset_summary(rows),
        "rows": rows,
        "config": {
            "corruption_name": corruption_name,
            "severity": int(severity),
            "top_node_fractions": fractions,
            "top_node_min_k": int(top_node_min_k),
            "top_node_max_k": int(top_node_max_k),
            "trigger_mode": trigger_mode,
        },
    }
    _maybe_write_json(result, output_path)
    return result


def _resolve_top_k_nodes(
    components: LoadedFlowComponents,
    *,
    top_node_fraction: float,
    top_node_min_k: int,
    top_node_max_k: int,
) -> int:
    grid_size = int(components.config["tokenization"].get("grid_size", 4))
    n_layers = len(components.observer.layer_channels)
    total_nodes = int(n_layers * grid_size * grid_size)
    requested = int(math.ceil(total_nodes * float(top_node_fraction)))
    return max(int(top_node_min_k), min(int(top_node_max_k), max(1, requested)))


def _validate_corruptions(corruption_names: list[str] | tuple[str, ...]) -> list[str]:
    resolved = [str(name) for name in corruption_names]
    invalid = sorted(set(resolved) - set(CIFAR10_CORRUPTION_NAMES))
    if invalid:
        raise ValueError(f"Unknown corruption names: {invalid}. Valid names: {list(CIFAR10_CORRUPTION_NAMES)}")
    return resolved


def _nested_output_path(output_path: str | Path | None, *, subdir: str) -> Path | None:
    if output_path is None:
        return None
    output_path = Path(output_path)
    nested = output_path.parent / "_nested_runs" / subdir / "selective_hybrid_correction.json"
    nested.parent.mkdir(parents=True, exist_ok=True)
    return nested


def _fraction_key(value: float) -> str:
    return f"{value:.3f}".replace(".", "p")


def _corruption_summary(rows: list[dict]) -> dict:
    if not rows:
        return {
            "mean_backbone_overall_accuracy": 0.0,
            "mean_full_z_hybrid_overall_accuracy": 0.0,
            "mean_top_node_subset_overall_accuracy": 0.0,
            "mean_full_z_gain_vs_backbone": 0.0,
            "mean_top_node_subset_gain_vs_backbone": 0.0,
            "best_full_z_gain_corruption": "",
            "best_top_node_gain_corruption": "",
        }
    best_full = max(rows, key=lambda row: row["full_z_gain_vs_backbone"])
    best_top = max(rows, key=lambda row: row["top_node_subset_gain_vs_backbone"])
    return {
        "mean_backbone_overall_accuracy": float(sum(row["backbone_overall_accuracy"] for row in rows) / len(rows)),
        "mean_full_z_hybrid_overall_accuracy": float(sum(row["full_z_hybrid_overall_accuracy"] for row in rows) / len(rows)),
        "mean_top_node_subset_overall_accuracy": float(sum(row["top_node_subset_overall_accuracy"] for row in rows) / len(rows)),
        "mean_full_z_gain_vs_backbone": float(sum(row["full_z_gain_vs_backbone"] for row in rows) / len(rows)),
        "mean_top_node_subset_gain_vs_backbone": float(sum(row["top_node_subset_gain_vs_backbone"] for row in rows) / len(rows)),
        "best_full_z_gain_corruption": f"{best_full['corruption_name']}@{best_full['severity']}",
        "best_top_node_gain_corruption": f"{best_top['corruption_name']}@{best_top['severity']}",
    }


def _top_node_subset_summary(rows: list[dict]) -> dict:
    if not rows:
        return {
            "best_fraction": 0.0,
            "best_top_k_nodes": 0,
            "best_top_node_subset_accuracy": 0.0,
            "best_gap_to_full_z": 0.0,
        }
    best = max(rows, key=lambda row: (row["top_node_subset_overall_accuracy"], row["top_node_subset_gain_vs_backbone"]))
    return {
        "best_fraction": float(best["top_node_fraction"]),
        "best_top_k_nodes": int(best["top_k_nodes"]),
        "best_top_node_subset_accuracy": float(best["top_node_subset_overall_accuracy"]),
        "best_gap_to_full_z": float(best["top_node_subset_gap_to_full_z"]),
    }
