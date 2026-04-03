from __future__ import annotations

import torch

from flow_circuits.objectives import FlowObjective


def test_flow_objective_returns_finite_losses_and_expected_shapes():
    objective = FlowObjective(n_layers=4, token_dim=16, flow_dim=8)
    z = torch.randn(3, 4, 4, 16)
    z = torch.nn.functional.normalize(z, dim=-1)
    flow = torch.randn(3, 4, 4, 8)
    flow = torch.nn.functional.normalize(flow, dim=-1)
    q = torch.randn(3, 4, 4, 8)
    q = torch.nn.functional.normalize(q, dim=-1)

    output = objective(z, flow, q, lambda_pred=1.0, lambda_rec=0.2, lambda_traj=0.1, traj_topk=2, traj_gamma=0.0)

    assert output.total_loss.dim() == 0
    assert torch.isfinite(output.total_loss)
    assert output.predicted_next.shape == (3, 3, 4, 8)
    assert output.reconstructed_current.shape == (3, 4, 4, 8)


def test_traj_loss_is_zero_when_disabled():
    objective = FlowObjective(n_layers=3, token_dim=12, flow_dim=6)
    z = torch.nn.functional.normalize(torch.randn(4, 3, 4, 12), dim=-1)
    flow = torch.nn.functional.normalize(torch.randn(4, 3, 4, 6), dim=-1)
    q = torch.nn.functional.normalize(torch.randn(4, 3, 4, 6), dim=-1)

    output = objective(z, flow, q, lambda_pred=1.0, lambda_rec=0.1, lambda_traj=0.0)

    assert output.traj_loss.item() == 0.0


def test_traj_loss_matches_naive_reference():
    objective = FlowObjective(n_layers=2, token_dim=10, flow_dim=5)
    z = torch.nn.functional.normalize(torch.randn(5, 2, 3, 10), dim=-1)
    q = torch.nn.functional.normalize(torch.randn(5, 2, 3, 5), dim=-1)

    actual = objective._trajectory_alignment_loss(
        z=z,
        future_descriptors=q,
        topk=3,
        gamma=0.0,
        tau=0.2,
    )
    expected = _naive_traj_loss(
        z=z,
        future_descriptors=q,
        topk=3,
        gamma=0.0,
        tau=0.2,
    )

    assert torch.allclose(actual, expected, atol=1.0e-6)


def _naive_traj_loss(
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
