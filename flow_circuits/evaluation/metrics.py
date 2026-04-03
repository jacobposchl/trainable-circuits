from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import spearmanr
import torch
import torch.nn.functional as F


@dataclass
class BaselineComparison:
    mean_baseline: float
    local_baseline: float
    flow_baseline: float
    best_baseline: float
    best_baseline_name: str = "mean_baseline"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ConfirmatoryCheck:
    model_value: float
    baseline_value: float
    baseline_name: str
    improvement: float
    ci_lower: float
    ci_upper: float
    passes: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RepresentationMetrics:
    n_images: int = 0
    prediction_cosine_mean: float = 0.0
    prediction_cosine_sem: float = 0.0
    reconstruction_cosine_mean: float = 0.0
    reconstruction_cosine_sem: float = 0.0
    trajectory_alignment_mean: float = 0.0
    trajectory_alignment_std: float = 0.0
    local_trajectory_alignment_mean: float = 0.0
    flow_trajectory_alignment_mean: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_representation_metrics(
    z: torch.Tensor,
    local_features: list[torch.Tensor],
    flow_targets: torch.Tensor,
    future_descriptors: torch.Tensor,
    predicted_next: torch.Tensor,
    reconstructed_current: torch.Tensor,
    *,
    max_alignment_pairs: int = 2048,
    alignment_seed: int = 0,
) -> RepresentationMetrics:
    pred_cos = (predicted_next * flow_targets[:, 1:]).sum(dim=-1).mean(dim=(1, 2)).detach().cpu().numpy()
    rec_cos = (reconstructed_current * flow_targets).sum(dim=-1).mean(dim=(1, 2)).detach().cpu().numpy()
    alignment = compute_alignment_scores(
        z=z,
        local_features=local_features,
        flow_targets=flow_targets,
        future_descriptors=future_descriptors,
        max_alignment_pairs=max_alignment_pairs,
        seed=alignment_seed,
    )

    return RepresentationMetrics(
        n_images=int(z.shape[0]),
        prediction_cosine_mean=float(pred_cos.mean()) if pred_cos.size else 0.0,
        prediction_cosine_sem=_sem(pred_cos),
        reconstruction_cosine_mean=float(rec_cos.mean()) if rec_cos.size else 0.0,
        reconstruction_cosine_sem=_sem(rec_cos),
        trajectory_alignment_mean=float(alignment["model_node_scores"].mean()) if alignment["model_node_scores"].size else 0.0,
        trajectory_alignment_std=float(alignment["model_node_scores"].std()) if alignment["model_node_scores"].size else 0.0,
        local_trajectory_alignment_mean=float(alignment["local_node_scores"].mean()) if alignment["local_node_scores"].size else 0.0,
        flow_trajectory_alignment_mean=float(alignment["flow_node_scores"].mean()) if alignment["flow_node_scores"].size else 0.0,
    )


def compute_alignment_scores(
    *,
    z: torch.Tensor,
    local_features: list[torch.Tensor],
    flow_targets: torch.Tensor,
    future_descriptors: torch.Tensor,
    max_alignment_pairs: int = 2048,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    model_scores = []
    local_scores = []
    flow_scores = []
    _, n_layers, n_cells, _ = z.shape
    for layer_idx in range(n_layers):
        local_layer = F.normalize(local_features[layer_idx], dim=-1)
        for cell_idx in range(n_cells):
            q_node = future_descriptors[:, layer_idx, cell_idx].detach().cpu()
            if q_node.shape[0] < 3:
                continue
            idx_a, idx_b = _sample_pair_indices(q_node.shape[0], max_alignment_pairs=max_alignment_pairs, rng=rng)
            q_sim = _pairwise_cosine(q_node, idx_a, idx_b)
            model_scores.append(
                _spearman_on_pairs(z[:, layer_idx, cell_idx].detach().cpu(), q_sim, idx_a, idx_b)
            )
            local_scores.append(
                _spearman_on_pairs(local_layer[:, cell_idx].detach().cpu(), q_sim, idx_a, idx_b)
            )
            flow_scores.append(
                _spearman_on_pairs(flow_targets[:, layer_idx, cell_idx].detach().cpu(), q_sim, idx_a, idx_b)
            )

    if not model_scores:
        zeros = np.zeros(1, dtype=np.float64)
        return {
            "model_node_scores": zeros,
            "local_node_scores": zeros,
            "flow_node_scores": zeros,
        }
    return {
        "model_node_scores": np.asarray(model_scores, dtype=np.float64),
        "local_node_scores": np.asarray(local_scores, dtype=np.float64),
        "flow_node_scores": np.asarray(flow_scores, dtype=np.float64),
    }


def evaluate_prediction_check(
    *,
    model_scores: np.ndarray,
    baseline_scores: dict[str, np.ndarray],
    bootstrap_iterations: int = 500,
    seed: int = 0,
) -> ConfirmatoryCheck:
    baseline_name, baseline_values = _select_best_baseline(baseline_scores)
    diff = np.asarray(model_scores, dtype=np.float64) - np.asarray(baseline_values, dtype=np.float64)
    ci_lower, ci_upper = bootstrap_mean_ci(diff, n_bootstrap=bootstrap_iterations, seed=seed)
    return ConfirmatoryCheck(
        model_value=float(np.mean(model_scores)) if model_scores.size else 0.0,
        baseline_value=float(np.mean(baseline_values)) if baseline_values.size else 0.0,
        baseline_name=baseline_name,
        improvement=float(np.mean(diff)) if diff.size else 0.0,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        passes=ci_lower > 0.0,
    )


def evaluate_alignment_check(
    *,
    alignment_scores: dict[str, np.ndarray],
    bootstrap_iterations: int = 500,
    seed: int = 0,
) -> ConfirmatoryCheck:
    baseline_name, baseline_values = _select_best_baseline(
        {
            "local_baseline": alignment_scores["local_node_scores"],
            "flow_baseline": alignment_scores["flow_node_scores"],
        }
    )
    model_values = alignment_scores["model_node_scores"]
    diff = model_values - baseline_values
    ci_lower, ci_upper = bootstrap_mean_ci(diff, n_bootstrap=bootstrap_iterations, seed=seed)
    return ConfirmatoryCheck(
        model_value=float(np.mean(model_values)) if model_values.size else 0.0,
        baseline_value=float(np.mean(baseline_values)) if baseline_values.size else 0.0,
        baseline_name=baseline_name,
        improvement=float(np.mean(diff)) if diff.size else 0.0,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        passes=ci_lower > 0.0,
    )


def compute_prediction_scores_by_image(
    predicted_next: torch.Tensor,
    flow_targets: torch.Tensor,
) -> np.ndarray:
    return (
        (predicted_next * flow_targets[:, 1:])
        .sum(dim=-1)
        .mean(dim=(1, 2))
        .detach()
        .cpu()
        .numpy()
    )


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    n_bootstrap: int = 500,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    if values.size == 0:
        return 0.0, 0.0
    if values.size == 1:
        value = float(values.item())
        return value, value
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_bootstrap, dtype=np.float64)
    for idx in range(n_bootstrap):
        sample = rng.choice(values, size=values.size, replace=True)
        estimates[idx] = sample.mean()
    lower = float(np.quantile(estimates, alpha / 2.0))
    upper = float(np.quantile(estimates, 1.0 - (alpha / 2.0)))
    return lower, upper


def _select_best_baseline(baseline_scores: dict[str, np.ndarray]) -> tuple[str, np.ndarray]:
    best_name = max(baseline_scores, key=lambda name: float(np.mean(baseline_scores[name])))
    return best_name, np.asarray(baseline_scores[best_name], dtype=np.float64)


def _spearman_on_pairs(
    node_values: torch.Tensor,
    q_sim: np.ndarray,
    idx_a: torch.Tensor,
    idx_b: torch.Tensor,
) -> float:
    value_sim = _pairwise_cosine(node_values, idx_a, idx_b)
    rho, _ = spearmanr(value_sim, q_sim)
    return 0.0 if np.isnan(rho) else float(rho)


def _pairwise_cosine(node_values: torch.Tensor, idx_a: torch.Tensor, idx_b: torch.Tensor) -> np.ndarray:
    normalized = F.normalize(node_values, dim=-1)
    similarities = (normalized[idx_a] * normalized[idx_b]).sum(dim=-1)
    return similarities.numpy()


def _sample_pair_indices(
    n_items: int,
    *,
    max_alignment_pairs: int,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    idx_a, idx_b = torch.triu_indices(n_items, n_items, offset=1)
    if idx_a.numel() > max_alignment_pairs:
        chosen = rng.choice(idx_a.numel(), size=max_alignment_pairs, replace=False)
        idx_a = idx_a[chosen]
        idx_b = idx_b[chosen]
    return idx_a, idx_b


def _sem(values: np.ndarray) -> float:
    if values.size <= 1:
        return 0.0
    return float(values.std(ddof=1) / max(values.size**0.5, 1.0))
