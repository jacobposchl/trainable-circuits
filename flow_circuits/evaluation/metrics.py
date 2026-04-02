from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import spearmanr
import torch


@dataclass
class BaselineComparison:
    mean_baseline: float
    local_baseline: float
    flow_baseline: float
    best_baseline: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RepresentationMetrics:
    prediction_cosine_mean: float
    prediction_cosine_sem: float
    reconstruction_cosine_mean: float
    reconstruction_cosine_sem: float
    trajectory_alignment_mean: float
    trajectory_alignment_std: float

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_representation_metrics(
    z: torch.Tensor,
    flow_targets: torch.Tensor,
    future_descriptors: torch.Tensor,
    predicted_next: torch.Tensor,
    reconstructed_current: torch.Tensor,
    *,
    max_alignment_pairs: int = 2048,
) -> RepresentationMetrics:
    pred_cos = (predicted_next * flow_targets[:, 1:]).sum(dim=-1).reshape(-1).detach().cpu().numpy()
    rec_cos = (reconstructed_current * flow_targets).sum(dim=-1).reshape(-1).detach().cpu().numpy()

    alignments = []
    _, n_layers, n_cells, _ = z.shape
    for layer_idx in range(n_layers):
        for cell_idx in range(n_cells):
            z_node = z[:, layer_idx, cell_idx].detach().cpu()
            q_node = future_descriptors[:, layer_idx, cell_idx].detach().cpu()
            if z_node.shape[0] < 3:
                continue
            z_sim = z_node @ z_node.T
            q_sim = q_node @ q_node.T
            idx_a, idx_b = torch.triu_indices(z_node.shape[0], z_node.shape[0], offset=1)
            if idx_a.numel() > max_alignment_pairs:
                perm = torch.randperm(idx_a.numel())[:max_alignment_pairs]
                idx_a = idx_a[perm]
                idx_b = idx_b[perm]
            rho, _ = spearmanr(z_sim[idx_a, idx_b].numpy(), q_sim[idx_a, idx_b].numpy())
            alignments.append(0.0 if np.isnan(rho) else float(rho))
    if not alignments:
        alignments = [0.0]

    return RepresentationMetrics(
        prediction_cosine_mean=float(pred_cos.mean()) if pred_cos.size else 0.0,
        prediction_cosine_sem=_sem(pred_cos),
        reconstruction_cosine_mean=float(rec_cos.mean()) if rec_cos.size else 0.0,
        reconstruction_cosine_sem=_sem(rec_cos),
        trajectory_alignment_mean=float(np.mean(alignments)),
        trajectory_alignment_std=float(np.std(alignments)),
    )


def _sem(values: np.ndarray) -> float:
    if values.size <= 1:
        return 0.0
    return float(values.std(ddof=1) / max(values.size**0.5, 1.0))
