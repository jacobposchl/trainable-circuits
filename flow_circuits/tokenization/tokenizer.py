from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from flow_circuits.backbones import ResNetObservations


@dataclass
class TokenizedBatch:
    token_inputs: torch.Tensor
    local_features: list[torch.Tensor]
    flow_targets: torch.Tensor
    future_descriptors: torch.Tensor
    layer_indices: torch.Tensor
    spatial_indices: torch.Tensor


class FlowTokenizer(nn.Module):
    """Converts raw backbone maps into token inputs, flow targets, and q descriptors."""

    def __init__(
        self,
        layer_channels: list[int],
        token_dim: int = 128,
        flow_dim: int = 256,
        traj_dim: int = 256,
        grid_size: int = 4,
        eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        self.layer_channels = list(layer_channels)
        self.token_dim = token_dim
        self.flow_dim = flow_dim
        self.traj_dim = traj_dim
        self.grid_size = grid_size
        self.eps = eps
        self.n_layers = len(layer_channels)
        self.n_cells = grid_size * grid_size

        self.content_projectors = nn.ModuleList(
            nn.Linear(channels + 1, token_dim) for channels in layer_channels
        )
        self.position_embedding = nn.Parameter(torch.randn(self.n_cells, token_dim) * 0.02)
        self.depth_embedding = nn.Parameter(torch.randn(self.n_layers, token_dim) * 0.02)
        self.layer_norms = nn.ModuleList(nn.LayerNorm(token_dim) for _ in layer_channels)

        self.flow_projectors = nn.ModuleList()
        for channels in layer_channels:
            projector = nn.Linear(channels, flow_dim, bias=False)
            nn.init.normal_(projector.weight, mean=0.0, std=1.0 / (channels**0.5))
            projector.requires_grad_(False)
            self.flow_projectors.append(projector)

        self.future_projectors = nn.ModuleList()
        for layer_idx in range(self.n_layers):
            input_dim = (self.n_layers - layer_idx) * flow_dim
            projector = nn.Linear(input_dim, traj_dim, bias=False)
            nn.init.normal_(projector.weight, mean=0.0, std=1.0 / (input_dim**0.5))
            projector.requires_grad_(False)
            self.future_projectors.append(projector)

        layer_indices = torch.arange(self.n_layers).repeat_interleave(self.n_cells)
        spatial_indices = torch.arange(self.n_cells).repeat(self.n_layers)
        self.register_buffer("layer_indices", layer_indices, persistent=False)
        self.register_buffer("spatial_indices", spatial_indices, persistent=False)

    def forward(self, observations: ResNetObservations) -> TokenizedBatch:
        if len(observations.states) != self.n_layers or len(observations.residuals) != self.n_layers:
            raise ValueError("Observation depth does not match tokenizer depth")

        local_features: list[torch.Tensor] = []
        token_inputs: list[torch.Tensor] = []
        flow_targets: list[torch.Tensor] = []

        for layer_idx, (state, residual) in enumerate(zip(observations.states, observations.residuals)):
            state_cells = self._pool_cells(state)
            residual_cells = self._pool_cells(residual)
            magnitudes = torch.log(state_cells.norm(dim=-1, keepdim=True) + self.eps)
            layer_features = torch.cat([state_cells, magnitudes], dim=-1)
            local_features.append(layer_features)

            projected = self.content_projectors[layer_idx](layer_features)
            embedded = projected + self.position_embedding.unsqueeze(0) + self.depth_embedding[layer_idx].view(1, 1, -1)
            token_inputs.append(self.layer_norms[layer_idx](embedded))

            flow = self.flow_projectors[layer_idx](residual_cells)
            flow_targets.append(F.normalize(flow, dim=-1))

        token_tensor = torch.stack(token_inputs, dim=1)
        flow_tensor = torch.stack(flow_targets, dim=1)
        future_tensor = self._build_future_descriptors(flow_tensor)

        return TokenizedBatch(
            token_inputs=token_tensor,
            local_features=local_features,
            flow_targets=flow_tensor,
            future_descriptors=future_tensor,
            layer_indices=self.layer_indices.clone(),
            spatial_indices=self.spatial_indices.clone(),
        )

    def _pool_cells(self, x: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_max_pool2d(x, (self.grid_size, self.grid_size))
        pooled = pooled.permute(0, 2, 3, 1).contiguous()
        return pooled.view(x.shape[0], self.n_cells, x.shape[1])

    def _build_future_descriptors(self, flow_tensor: torch.Tensor) -> torch.Tensor:
        descriptors: list[torch.Tensor] = []
        batch_size = flow_tensor.shape[0]
        for layer_idx in range(self.n_layers):
            future = flow_tensor[:, layer_idx:, :, :]
            future = future.permute(0, 2, 1, 3).contiguous()
            future = future.view(batch_size, self.n_cells, -1)
            q = self.future_projectors[layer_idx](future)
            descriptors.append(F.normalize(q, dim=-1))
        return torch.stack(descriptors, dim=1)
