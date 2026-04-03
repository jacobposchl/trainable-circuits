from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from flow_circuits.evaluation.metrics import BaselineComparison
from flow_circuits.utils import seed_everything


class _BaselineMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


@dataclass
class BaselineRegressors:
    mean_targets: list[np.ndarray]
    local_models: list[_BaselineMLP]
    flow_models: list[_BaselineMLP]

    @classmethod
    def fit(
        cls,
        *,
        local_features: list[np.ndarray],
        flow_features: list[np.ndarray],
        next_targets: list[np.ndarray],
        hidden_dim: int | None = None,
        epochs: int = 10,
        batch_size: int = 1024,
        lr: float = 1.0e-3,
        weight_decay: float = 1.0e-4,
        seed: int = 0,
        device: torch.device | str = "cpu",
    ) -> "BaselineRegressors":
        device = torch.device(device)
        mean_targets = [target.mean(axis=0) for target in next_targets]
        local_models = []
        flow_models = []
        for layer_idx, (local_x, flow_x, target) in enumerate(zip(local_features, flow_features, next_targets)):
            output_dim = int(target.shape[-1])
            layer_hidden_dim = hidden_dim or max(output_dim * 2, 64)
            local_models.append(
                _fit_layer_model(
                    features=local_x,
                    target=target,
                    hidden_dim=layer_hidden_dim,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                    weight_decay=weight_decay,
                    seed=seed + (2 * layer_idx),
                    device=device,
                )
            )
            flow_models.append(
                _fit_layer_model(
                    features=flow_x,
                    target=target,
                    hidden_dim=layer_hidden_dim,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                    weight_decay=weight_decay,
                    seed=seed + (2 * layer_idx) + 1,
                    device=device,
                )
            )
        return cls(mean_targets=mean_targets, local_models=local_models, flow_models=flow_models)

    def evaluate(
        self,
        *,
        local_features: list[np.ndarray],
        flow_features: list[np.ndarray],
        next_targets: list[np.ndarray],
    ) -> BaselineComparison:
        scores = self.score_predictions(
            local_features=local_features,
            flow_features=flow_features,
            next_targets=next_targets,
        )
        mean_value = float(scores["mean_baseline"].mean())
        local_value = float(scores["local_baseline"].mean())
        flow_value = float(scores["flow_baseline"].mean())
        best_name = max(
            ("mean_baseline", "local_baseline", "flow_baseline"),
            key=lambda name: float(scores[name].mean()),
        )
        best_value = float(scores[best_name].mean())
        return BaselineComparison(
            mean_baseline=mean_value,
            local_baseline=local_value,
            flow_baseline=flow_value,
            best_baseline=best_value,
            best_baseline_name=best_name,
        )

    def score_predictions(
        self,
        *,
        local_features: list[np.ndarray],
        flow_features: list[np.ndarray],
        next_targets: list[np.ndarray],
    ) -> dict[str, np.ndarray]:
        mean_scores = []
        local_scores = []
        flow_scores = []
        for layer_idx, target in enumerate(next_targets):
            mean_pred = np.broadcast_to(self.mean_targets[layer_idx][None, :, :], target.shape)
            local_pred = _predict_layer_model(self.local_models[layer_idx], local_features[layer_idx])
            flow_pred = _predict_layer_model(self.flow_models[layer_idx], flow_features[layer_idx])
            mean_scores.append(_cosine_by_image(mean_pred, target))
            local_scores.append(_cosine_by_image(local_pred, target))
            flow_scores.append(_cosine_by_image(flow_pred, target))
        return {
            "mean_baseline": np.mean(np.stack(mean_scores, axis=0), axis=0),
            "local_baseline": np.mean(np.stack(local_scores, axis=0), axis=0),
            "flow_baseline": np.mean(np.stack(flow_scores, axis=0), axis=0),
        }


def _fit_layer_model(
    *,
    features: np.ndarray,
    target: np.ndarray,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> _BaselineMLP:
    seed_everything(seed)
    model = _BaselineMLP(features.shape[-1], target.shape[-1], hidden_dim=hidden_dim).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    feature_tensor = torch.from_numpy(features.reshape(-1, features.shape[-1])).float().to(device)
    target_tensor = torch.from_numpy(target.reshape(-1, target.shape[-1])).float().to(device)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    model.train()
    for _ in range(epochs):
        order = torch.randperm(feature_tensor.shape[0], generator=generator, device="cpu").to(feature_tensor.device)
        for start_idx in range(0, feature_tensor.shape[0], batch_size):
            batch_indices = order[start_idx:start_idx + batch_size]
            batch_x = feature_tensor[batch_indices]
            batch_y = target_tensor[batch_indices]
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = (pred - batch_y).pow(2).mean()
            loss.backward()
            optimizer.step()
    model.eval()
    return model.cpu()


def _predict_layer_model(model: _BaselineMLP, features: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        feature_tensor = torch.from_numpy(features.reshape(-1, features.shape[-1])).float()
        pred = model(feature_tensor).view(features.shape[0], features.shape[1], -1)
    return pred.numpy()


def _cosine_by_image(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred_norm = pred / np.clip(np.linalg.norm(pred, axis=-1, keepdims=True), 1.0e-8, None)
    target_norm = target / np.clip(np.linalg.norm(target, axis=-1, keepdims=True), 1.0e-8, None)
    return np.mean(np.sum(pred_norm * target_norm, axis=-1), axis=1)
