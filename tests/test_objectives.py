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
