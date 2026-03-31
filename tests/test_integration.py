import copy

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from evaluation.circuit_analysis import CircuitAnalyzer
from evaluation.discovery import SpanCentricDiscovery
from training import Phase1Trainer


def _make_loader(n_samples=8, batch_size=4):
    torch.manual_seed(7)
    images = torch.randn(n_samples, 3, 32, 32)
    labels = torch.arange(n_samples) % 10
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def test_backbone_meta_encoder_info_loss_pipeline_returns_finite_scalar(
    backbone,
    meta_encoder,
    info_loss,
    random_images,
):
    trajectory = backbone(random_images)
    z_list = meta_encoder(trajectory)
    idx_a, idx_b = torch.triu_indices(random_images.shape[0], random_images.shape[0], offset=1)
    z_pairs_a = [z[idx_a] for z in z_list]
    z_pairs_b = [z[idx_b] for z in z_list]
    rich_targets = [
        flow[idx_a] * flow[idx_b]
        for flow in backbone._flow_targets
    ]

    loss = info_loss(z_pairs_a, z_pairs_b, rich_targets)

    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_circuit_analyzer_and_discovery_fit_together(backbone, meta_encoder):
    loader = _make_loader(n_samples=6, batch_size=3)
    analyzer = CircuitAnalyzer(backbone, meta_encoder, loader, torch.device("cpu"))
    representations = analyzer.collect_representations(max_samples=6)

    discovery = SpanCentricDiscovery(
        n_layers=len(representations["z_list"]),
        umap_n_components=2,
        umap_n_neighbors=3,
        min_cluster_fraction=0.3,
        max_cluster_fraction=0.7,
        min_cluster_size=2,
    )
    discovery._reduce = lambda x: x[:, :2]
    discovery._cluster = lambda x: np.array([0, 0, 0, -1, -1, -1])

    z_numpy = [z.numpy() for z in representations["z_list"]]
    circuits = discovery.discover_all(z_numpy)

    assert circuits
    assert all("span" in circuit and "image_mask" in circuit for circuit in circuits)


def test_phase1_trainer_train_epoch_returns_loss_metrics(monkeypatch, minimal_config):
    loader = _make_loader()
    monkeypatch.setattr(
        "training.unified_trainer.get_standard_loaders",
        lambda **kwargs: (loader, loader),
    )

    trainer = Phase1Trainer(copy.deepcopy(minimal_config))
    metrics = trainer._train_epoch(epoch=0, log_interval=1000)

    assert set(metrics) == {"loss", "info_loss"}
    assert metrics["loss"] > 0
    assert metrics["info_loss"] > 0


def test_phase1_trainer_val_epoch_returns_expected_metrics(monkeypatch, minimal_config):
    loader = _make_loader()
    monkeypatch.setattr(
        "training.unified_trainer.get_standard_loaders",
        lambda **kwargs: (loader, loader),
    )

    trainer = Phase1Trainer(copy.deepcopy(minimal_config))
    metrics = trainer._val_epoch()

    assert set(metrics) == {"r2", "mean_rho", "per_layer_rho"}
    assert len(metrics["per_layer_rho"]) == len(trainer.backbone.layer_dims)
    assert np.isfinite(metrics["r2"])
    assert np.isfinite(metrics["mean_rho"])


def test_phase1_trainer_checkpoint_round_trip(monkeypatch, minimal_config, tmp_path):
    loader = _make_loader()
    monkeypatch.setattr(
        "training.unified_trainer.get_standard_loaders",
        lambda **kwargs: (loader, loader),
    )

    config = copy.deepcopy(minimal_config)
    config["logging"]["checkpoint_dir"] = str(tmp_path / "ckpts")

    trainer = Phase1Trainer(config)
    trainer._save_checkpoint(1, {"r2": 0.4, "mean_rho": 0.2}, "roundtrip.pt")

    reloaded = Phase1Trainer(copy.deepcopy(config))
    next_epoch = reloaded._load_checkpoint(str(tmp_path / "ckpts" / "roundtrip.pt"))

    assert next_epoch == 2
    for key, value in trainer.backbone.state_dict().items():
        torch.testing.assert_close(reloaded.backbone.state_dict()[key], value)
    for key, value in trainer.meta_encoder.state_dict().items():
        torch.testing.assert_close(reloaded.meta_encoder.state_dict()[key], value)
    for key, value in trainer.info_loss.state_dict().items():
        torch.testing.assert_close(reloaded.info_loss.state_dict()[key], value)


def test_phase1_trainer_early_stops_when_validation_stalls(monkeypatch, minimal_config):
    loader = _make_loader()
    monkeypatch.setattr(
        "training.unified_trainer.get_standard_loaders",
        lambda **kwargs: (loader, loader),
    )

    config = copy.deepcopy(minimal_config)
    config["training"]["epochs"] = 10
    config["training"]["early_stopping_patience"] = 2
    config["training"]["early_stopping_min_delta"] = 0.01

    trainer = Phase1Trainer(config)

    train_calls = []
    val_sequence = iter(
        [
            {"r2": 0.10, "mean_rho": 0.10, "per_layer_rho": [0.1] * len(trainer.backbone.layer_dims)},
            {"r2": 0.25, "mean_rho": 0.12, "per_layer_rho": [0.1] * len(trainer.backbone.layer_dims)},
            {"r2": 0.255, "mean_rho": 0.12, "per_layer_rho": [0.1] * len(trainer.backbone.layer_dims)},
            {"r2": 0.251, "mean_rho": 0.12, "per_layer_rho": [0.1] * len(trainer.backbone.layer_dims)},
            {"r2": 0.30, "mean_rho": 0.12, "per_layer_rho": [0.1] * len(trainer.backbone.layer_dims)},
        ]
    )
    saved = []

    monkeypatch.setattr(
        trainer,
        "_train_epoch",
        lambda epoch, log_interval: train_calls.append(epoch) or {"loss": 1.0, "info_loss": 0.2},
    )
    monkeypatch.setattr(trainer, "_val_epoch", lambda: next(val_sequence))
    monkeypatch.setattr(trainer, "_save_checkpoint", lambda epoch, val_metrics, name: saved.append((epoch, name)))

    trainer.train()

    assert train_calls == [0, 1, 2, 3]
    assert (1, "best.pt") in saved
