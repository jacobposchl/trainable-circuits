from __future__ import annotations

from pathlib import Path
import re
import json
import time

import numpy as np
import torch

from flow_circuits.evaluation.metrics import evaluate_representation_metrics
from flow_circuits.training import collect_model_outputs, load_components_from_checkpoint
from flow_circuits.data import build_cifar10_splits


Q_VALIDATION_EXPERIMENT_ID = "q_checkpoint_validation"


class _ProgressTracker:
    def __init__(self, *, experiment: str, checkpoint_tag: str, progress_callback=None) -> None:
        self.experiment = experiment
        self.checkpoint_tag = checkpoint_tag
        self.progress_callback = progress_callback
        self.started = time.perf_counter()

    def emit(self, *, stage: str, completed: int, total: int | None, message: str) -> None:
        if self.progress_callback is None:
            return
        elapsed = time.perf_counter() - self.started
        eta = None
        if total and completed:
            eta = max((elapsed / completed) * max(total - completed, 0), 0.0)
        self.progress_callback(
            experiment=self.experiment,
            checkpoint_tag=self.checkpoint_tag,
            stage=stage,
            completed=completed,
            total=total,
            elapsed_seconds=elapsed,
            eta_seconds=eta,
            message=message,
        )


def run_q_checkpoint_validation_experiment(
    *,
    base_config: dict,
    frozen_checkpoint_dir: str | Path,
    joint_checkpoint_dir: str | Path,
    device: torch.device,
    validation_split: str = "val",
    max_images: int = 1024,
    anchor_images: int = 256,
    neighbor_topk: int = 20,
    top_k_candidates_to_summarize: int = 5,
    output_path: str | Path | None = None,
    progress_callback=None,
) -> dict:
    tracker = _ProgressTracker(
        experiment=Q_VALIDATION_EXPERIMENT_ID,
        checkpoint_tag="selection",
        progress_callback=progress_callback,
    )
    loaders = build_cifar10_splits(
        data_dir=base_config["data"]["data_dir"],
        batch_size=base_config["data"]["batch_size"],
        num_workers=base_config["data"].get("num_workers", 4),
        seed=base_config["data"].get("seed", 0),
        augment_fit=base_config["data"].get("augment_fit", True),
        download=base_config["data"].get("download", True),
    )
    loader = loaders[str(validation_split)]
    candidates = _discover_candidates(frozen_checkpoint_dir, branch_tag="frozen") + _discover_candidates(joint_checkpoint_dir, branch_tag="joint")
    rows: list[dict] = []
    total = len(candidates)
    for completed, candidate in enumerate(candidates, start=1):
        from flow_circuits.evaluation.efficient_validation import run_neighbor_agreement_experiment

        tracker.emit(
            stage="candidate_loading",
            completed=completed - 1,
            total=total,
            message=f"scoring {candidate['branch_tag']} {Path(candidate['checkpoint_path']).name}",
        )
        components = load_components_from_checkpoint(candidate["checkpoint_path"], device)
        neighbor = run_neighbor_agreement_experiment(
            components,
            loader,
            device=device,
            checkpoint_tag=candidate["branch_tag"],
            max_images=max_images,
            anchor_images=anchor_images,
            topk=neighbor_topk,
            output_path=None,
            progress_callback=None,
        )
        outputs = collect_model_outputs(
            components,
            loader,
            device=device,
            max_images=max_images,
        )
        metrics = evaluate_representation_metrics(
            outputs["z"],
            outputs["local_features"],
            outputs["flow_targets"],
            outputs["future_descriptors"],
            outputs["predicted_next"],
            outputs["reconstructed_current"],
            max_alignment_pairs=int(base_config["training"].get("alignment_max_pairs", 2048)),
            alignment_seed=int(base_config["data"].get("seed", 0)),
        )
        rows.append(
            {
                **candidate,
                "neighbor_recall_at_k": float(neighbor["summary"]["mean_recall_at_k"]),
                "neighbor_jaccard_at_k": float(neighbor["summary"]["mean_jaccard_at_k"]),
                "prediction_cosine_mean": float(metrics.prediction_cosine_mean),
                "reconstruction_cosine_mean": float(metrics.reconstruction_cosine_mean),
                "trajectory_alignment_mean": float(metrics.trajectory_alignment_mean),
                "trajectory_alignment_std": float(metrics.trajectory_alignment_std),
            }
        )
        tracker.emit(
            stage="candidate_loading",
            completed=completed,
            total=total,
            message=f"finished {candidate['branch_tag']} {Path(candidate['checkpoint_path']).name}",
        )

    frozen_rows = [row for row in rows if row["branch_tag"] == "frozen"]
    joint_rows = [row for row in rows if row["branch_tag"] == "joint"]
    selected_frozen = _select_best_candidate(frozen_rows)
    selected_joint = _select_best_candidate(joint_rows)
    result = {
        "experiment": Q_VALIDATION_EXPERIMENT_ID,
        "summary": {
            "n_frozen_candidates": int(len(frozen_rows)),
            "n_joint_candidates": int(len(joint_rows)),
            "selected_frozen_checkpoint": selected_frozen["checkpoint_path"] if selected_frozen else None,
            "selected_joint_checkpoint": selected_joint["checkpoint_path"] if selected_joint else None,
        },
        "selected": {
            "frozen": selected_frozen,
            "joint": selected_joint,
        },
        "top_frozen_rows": _top_rows(frozen_rows, top_k_candidates_to_summarize),
        "top_joint_rows": _top_rows(joint_rows, top_k_candidates_to_summarize),
        "rows": rows,
    }
    _maybe_write_json(result, output_path)
    return result


def _discover_candidates(checkpoint_dir: str | Path, *, branch_tag: str) -> list[dict]:
    checkpoint_dir = Path(checkpoint_dir)
    candidates: list[dict] = []
    if branch_tag == "frozen":
        phase_b = checkpoint_dir / "phase_b_frozen.pt"
        if phase_b.exists():
            candidates.append(
                {
                    "branch_tag": branch_tag,
                    "candidate_type": "phase_b",
                    "lambda_traj": None,
                    "epoch": 0,
                    "checkpoint_path": str(phase_b),
                }
            )
    pattern = f"phase_c_{branch_tag}_lambda_*_epoch_*.pt"
    for path in sorted(checkpoint_dir.glob(pattern)):
        meta = _parse_candidate_filename(path.name)
        candidates.append(
            {
                "branch_tag": branch_tag,
                "candidate_type": "phase_c",
                "lambda_traj": meta["lambda_traj"],
                "epoch": meta["epoch"],
                "checkpoint_path": str(path),
            }
        )
    return candidates


def _parse_candidate_filename(name: str) -> dict:
    match = re.search(r"lambda_(?P<lambda>[0-9p]+)_epoch_(?P<epoch>\d+)\.pt$", name)
    if not match:
        return {"lambda_traj": None, "epoch": 0}
    return {
        "lambda_traj": float(match.group("lambda").replace("p", ".")),
        "epoch": int(match.group("epoch")),
    }


def _candidate_sort_key(row: dict) -> tuple:
    return (
        -float(row["neighbor_recall_at_k"]),
        -float(row["trajectory_alignment_mean"]),
        -float(row["reconstruction_cosine_mean"]),
        int(row.get("epoch", 0)),
        float(row.get("lambda_traj") or 0.0),
        str(row["checkpoint_path"]),
    )


def _select_best_candidate(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    return sorted(rows, key=_candidate_sort_key)[0]


def _top_rows(rows: list[dict], top_k: int) -> list[dict]:
    if not rows:
        return []
    return sorted(rows, key=_candidate_sort_key)[: int(top_k)]


def _maybe_write_json(payload: dict, output_path: str | Path | None) -> None:
    if output_path is None:
        return
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
