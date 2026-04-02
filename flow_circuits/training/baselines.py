from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge

from flow_circuits.evaluation import BaselineComparison


@dataclass
class BaselineRegressors:
    mean_targets: list[np.ndarray]
    local_models: list[Ridge]
    flow_models: list[Ridge]

    @classmethod
    def fit(
        cls,
        *,
        local_features: list[np.ndarray],
        flow_features: list[np.ndarray],
        next_targets: list[np.ndarray],
        alpha: float = 1.0,
    ) -> "BaselineRegressors":
        mean_targets = [target.mean(axis=0) for target in next_targets]
        local_models = [Ridge(alpha=alpha).fit(x, y) for x, y in zip(local_features, next_targets)]
        flow_models = [Ridge(alpha=alpha).fit(x, y) for x, y in zip(flow_features, next_targets)]
        return cls(mean_targets=mean_targets, local_models=local_models, flow_models=flow_models)

    def evaluate(
        self,
        *,
        local_features: list[np.ndarray],
        flow_features: list[np.ndarray],
        next_targets: list[np.ndarray],
    ) -> BaselineComparison:
        mean_scores = []
        local_scores = []
        flow_scores = []
        for layer_idx, target in enumerate(next_targets):
            mean_pred = np.broadcast_to(self.mean_targets[layer_idx], target.shape)
            local_pred = self.local_models[layer_idx].predict(local_features[layer_idx])
            flow_pred = self.flow_models[layer_idx].predict(flow_features[layer_idx])
            mean_scores.append(_mean_cosine(mean_pred, target))
            local_scores.append(_mean_cosine(local_pred, target))
            flow_scores.append(_mean_cosine(flow_pred, target))
        mean_value = float(np.mean(mean_scores))
        local_value = float(np.mean(local_scores))
        flow_value = float(np.mean(flow_scores))
        return BaselineComparison(
            mean_baseline=mean_value,
            local_baseline=local_value,
            flow_baseline=flow_value,
            best_baseline=max(mean_value, local_value, flow_value),
        )


def _mean_cosine(pred: np.ndarray, target: np.ndarray) -> float:
    pred_norm = pred / np.clip(np.linalg.norm(pred, axis=1, keepdims=True), 1.0e-8, None)
    target_norm = target / np.clip(np.linalg.norm(target, axis=1, keepdims=True), 1.0e-8, None)
    return float(np.mean(np.sum(pred_norm * target_norm, axis=1)))
