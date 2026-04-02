from __future__ import annotations

import copy

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset


def make_tiny_loader(n_samples: int = 6, batch_size: int = 2) -> DataLoader:
    torch.manual_seed(7)
    images = torch.randn(n_samples, 3, 32, 32)
    labels = torch.arange(n_samples) % 10
    indices = torch.arange(n_samples)
    dataset = TensorDataset(images, labels, indices)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


@pytest.fixture
def tiny_loader():
    return make_tiny_loader()


@pytest.fixture
def minimal_config(tmp_path):
    return {
        "experiment": {"name": "pytest", "mode": "base"},
        "data": {
            "data_dir": str(tmp_path / "data"),
            "batch_size": 2,
            "num_workers": 0,
            "seed": 0,
            "augment_fit": False,
            "download": False,
        },
        "backbone": {
            "arch": "resnet18",
            "pretrained": False,
            "num_classes": 10,
        },
        "tokenization": {
            "grid_size": 2,
            "token_dim": 32,
            "flow_dim": 16,
            "traj_dim": 16,
            "eps": 1.0e-6,
        },
        "encoder": {
            "n_heads": 4,
            "n_transformer_layers": 1,
            "mlp_dim": 64,
            "dropout": 0.0,
        },
        "objectives": {
            "lambda_pred": 1.0,
            "lambda_rec": 0.2,
            "lambda_traj_candidates": [0.1],
            "traj_topk": 2,
            "traj_gamma": 0.0,
            "traj_tau": 0.2,
        },
        "training": {
            "lr": 1.0e-3,
            "weight_decay": 1.0e-4,
            "grad_clip": 1.0,
            "phase_epochs": {"phase_a": 1, "phase_b": 1, "phase_c": 1},
            "baseline_fit_images": 4,
            "baseline_eval_images": 4,
            "validation_images": 4,
            "alignment_max_pairs": 16,
        },
        "discovery": {
            "min_cluster_fraction": 0.2,
            "max_cluster_fraction": 0.8,
            "min_cluster_size": 2,
            "bootstrap_iterations": 2,
            "stability_threshold": 0.0,
            "merge_threshold": 0.5,
            "node_threshold": 0.5,
            "seed": 0,
            "max_images": 6,
        },
        "interventions": {
            "alpha": 0.1,
            "max_images": 6,
        },
        "logging": {
            "checkpoint_dir": str(tmp_path / "checkpoints"),
        },
    }


@pytest.fixture
def aligned_config(minimal_config):
    config = copy.deepcopy(minimal_config)
    config["experiment"]["mode"] = "aligned"
    return config
