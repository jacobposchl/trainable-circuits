import copy

import matplotlib
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from losses import InfoLoss
from models import FrozenBackbone, MetaEncoder

matplotlib.use("Agg")

ARCH = "resnet18"
GRID_SIZE = 2
FLOW_DIM = 32
PROJECTION_DIM = 64
N_HEADS = 4
HIDDEN_DIM = 32
BATCH_SIZE = 4


@pytest.fixture(scope="session")
def backbone():
    torch.manual_seed(0)
    model = FrozenBackbone(
        arch=ARCH,
        num_classes=10,
        pretrained=False,
        grid_size=GRID_SIZE,
        flow_dim=FLOW_DIM,
    )
    model.eval()
    return model


@pytest.fixture(scope="session")
def meta_encoder(backbone):
    torch.manual_seed(0)
    model = MetaEncoder(
        layer_dims=backbone.layer_dims,
        projection_dim=PROJECTION_DIM,
        n_heads=N_HEADS,
        n_transformer_layers=1,
        dropout=0.0,
    )
    model.eval()
    return model


@pytest.fixture(scope="session")
def info_loss(backbone):
    torch.manual_seed(0)
    module = InfoLoss(
        layer_dims=backbone.layer_dims,
        projection_dim=PROJECTION_DIM,
        hidden_dim=HIDDEN_DIM,
    )
    module.eval()
    return module


@pytest.fixture
def random_images():
    torch.manual_seed(1)
    return torch.randn(BATCH_SIZE, 3, 32, 32)


@pytest.fixture
def trajectory(backbone, random_images):
    return backbone(random_images)


@pytest.fixture
def fake_loader():
    torch.manual_seed(2)
    images = torch.randn(12, 3, 32, 32)
    labels = torch.arange(12) % 10
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)


@pytest.fixture
def minimal_config(tmp_path):
    return {
        "experiment": {"name": "pytest"},
        "model": {
            "arch": ARCH,
            "num_classes": 10,
            "pretrained": False,
            "flow_compression": {
                "grid_size": GRID_SIZE,
                "flow_dim": FLOW_DIM,
            },
            "meta_encoder": {
                "projection_dim": PROJECTION_DIM,
                "n_heads": N_HEADS,
                "n_transformer_layers": 1,
                "dropout": 0.0,
            },
            "regressor": {
                "hidden_dim": HIDDEN_DIM,
            },
        },
        "data": {
            "data_dir": str(tmp_path / "data"),
            "batch_size": BATCH_SIZE,
            "num_workers": 0,
            "augment": False,
        },
        "training": {
            "epochs": 2,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "info_loss_weight": 1.0,
        },
        "logging": {
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "log_interval": 100,
            "save_every": 1,
        },
    }


@pytest.fixture
def make_config():
    def factory(config):
        return copy.deepcopy(config)

    return factory
