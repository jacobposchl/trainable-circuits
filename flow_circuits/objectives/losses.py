from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def _normalized_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).pow(2).mean()


class _LayerDecoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or input_dim * 2
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


@dataclass
class FlowObjectiveOutput:
    total_loss: torch.Tensor
    pred_loss: torch.Tensor
    rec_loss: torch.Tensor
    traj_loss: torch.Tensor
    prediction_cosine: torch.Tensor
    reconstruction_cosine: torch.Tensor
    predicted_next: torch.Tensor
    reconstructed_current: torch.Tensor


class FlowObjective(nn.Module):
    def __init__(
        self,
        n_layers: int,
        token_dim: int,
        flow_dim: int,
        pred_hidden_dim: int | None = None,
        rec_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.prediction_decoders = nn.ModuleList(
            _LayerDecoder(token_dim, flow_dim, hidden_dim=pred_hidden_dim)
            for _ in range(max(n_layers - 1, 1))
        )
        self.reconstruction_decoders = nn.ModuleList(
            _LayerDecoder(token_dim, flow_dim, hidden_dim=rec_hidden_dim)
            for _ in range(n_layers)
        )

    def forward(
        self,
        z: torch.Tensor,
        flow_targets: torch.Tensor,
        future_descriptors: torch.Tensor,
        *,
        lambda_pred: float = 1.0,
        lambda_rec: float = 0.0,
        lambda_traj: float = 0.0,
        traj_topk: int = 8,
        traj_gamma: float = 0.2,
        traj_tau: float = 0.1,
    ) -> FlowObjectiveOutput:
        batch_size, n_layers, n_cells, token_dim = z.shape
        flow_dim = flow_targets.shape[-1]

        predicted_next = torch.zeros(
            batch_size,
            n_layers - 1,
            n_cells,
            flow_dim,
            device=z.device,
            dtype=z.dtype,
        )
        reconstructed_current = torch.zeros_like(flow_targets)

        pred_terms = []
        rec_terms = []
        pred_cosines = []
        rec_cosines = []

        for layer_idx in range(n_layers):
            z_layer = z[:, layer_idx].reshape(batch_size * n_cells, token_dim)
            rec = self.reconstruction_decoders[layer_idx](z_layer).view(batch_size, n_cells, flow_dim)
            reconstructed_current[:, layer_idx] = rec
            rec_target = flow_targets[:, layer_idx]
            rec_terms.append(_normalized_mse(rec, rec_target))
            rec_cosines.append((rec * rec_target).sum(dim=-1).mean())

            if layer_idx < n_layers - 1:
                pred = self.prediction_decoders[layer_idx](z_layer).view(batch_size, n_cells, flow_dim)
                predicted_next[:, layer_idx] = pred
                pred_target = flow_targets[:, layer_idx + 1]
                pred_terms.append(_normalized_mse(pred, pred_target))
                pred_cosines.append((pred * pred_target).sum(dim=-1).mean())

        pred_loss = torch.stack(pred_terms).mean() if pred_terms else torch.zeros((), device=z.device)
        rec_loss = torch.stack(rec_terms).mean() if rec_terms else torch.zeros((), device=z.device)
        prediction_cosine = (
            torch.stack(pred_cosines).mean() if pred_cosines else torch.zeros((), device=z.device)
        )
        reconstruction_cosine = (
            torch.stack(rec_cosines).mean() if rec_cosines else torch.zeros((), device=z.device)
        )

        traj_loss = (
            self._trajectory_alignment_loss(
                z=z,
                future_descriptors=future_descriptors,
                topk=traj_topk,
                gamma=traj_gamma,
                tau=traj_tau,
            )
            if lambda_traj > 0.0
            else torch.zeros((), device=z.device)
        )
        total_loss = (lambda_pred * pred_loss) + (lambda_rec * rec_loss) + (lambda_traj * traj_loss)
        return FlowObjectiveOutput(
            total_loss=total_loss,
            pred_loss=pred_loss,
            rec_loss=rec_loss,
            traj_loss=traj_loss,
            prediction_cosine=prediction_cosine,
            reconstruction_cosine=reconstruction_cosine,
            predicted_next=predicted_next,
            reconstructed_current=reconstructed_current,
        )

    def _trajectory_alignment_loss(
        self,
        *,
        z: torch.Tensor,
        future_descriptors: torch.Tensor,
        topk: int,
        gamma: float,
        tau: float,
    ) -> torch.Tensor:
        batch_size, n_layers, n_cells, _ = z.shape
        losses = []
        for layer_idx in range(n_layers):
            for cell_idx in range(n_cells):
                z_node = z[:, layer_idx, cell_idx, :]
                q_node = future_descriptors[:, layer_idx, cell_idx, :]
                q_sim = q_node @ q_node.T
                z_sim = (z_node @ z_node.T) / tau
                mask = ~torch.eye(batch_size, dtype=torch.bool, device=z.device)
                z_sim = z_sim.masked_fill(~mask, float("-inf"))

                for anchor_idx in range(batch_size):
                    weights = q_sim[anchor_idx].clone()
                    weights[anchor_idx] = -1.0
                    valid = weights >= gamma
                    if valid.sum() == 0:
                        continue
                    candidate_weights = weights.masked_fill(~valid, float("-inf"))
                    n_keep = min(int(valid.sum().item()), topk)
                    top_values, top_indices = torch.topk(candidate_weights, k=n_keep)
                    pos_weights = torch.clamp(top_values, min=0.0)
                    denom = torch.logsumexp(z_sim[anchor_idx], dim=0)
                    numerators = z_sim[anchor_idx, top_indices]
                    weighted = pos_weights * (numerators - denom)
                    losses.append(-(weighted.sum() / pos_weights.sum().clamp_min(1.0e-8)))
        if not losses:
            return torch.zeros((), device=z.device)
        return torch.stack(losses).mean()
