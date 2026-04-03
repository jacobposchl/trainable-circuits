from __future__ import annotations

import pytest
import torch

from flow_circuits.backbones import FrozenResNetObserver
from flow_circuits.tokenization import FlowTokenizer
from flow_circuits.encoders import SpatiotemporalEncoder


def test_resnet18_observer_collects_state_and_residual_maps():
    model = FrozenResNetObserver(arch="resnet18", pretrained=False, num_classes=10, grid_size=2)
    observations = model(torch.randn(2, 3, 32, 32))

    assert observations.n_layers == 8
    assert len(observations.states) == 8
    assert len(observations.residuals) == 8
    assert observations.layer_channels == [64, 64, 128, 128, 256, 256, 512, 512]
    assert observations.states[0].shape[:2] == (2, 64)


def test_resnet34_and_resnet50_metadata_are_supported():
    resnet34 = FrozenResNetObserver(arch="resnet34", pretrained=False, num_classes=10)
    resnet50 = FrozenResNetObserver(arch="resnet50", pretrained=False, num_classes=10)

    assert len(resnet34.layer_channels) == 16
    assert len(resnet50.layer_channels) == 16
    assert resnet50.layer_channels[-1] == 2048


def test_observer_requires_trained_checkpoint_when_requested():
    with pytest.raises(ValueError, match="requires a trained supervised backbone checkpoint"):
        FrozenResNetObserver(
            arch="resnet18",
            pretrained=False,
            num_classes=10,
            require_trained_checkpoint=True,
        )


def test_tokenizer_builds_tokens_targets_and_future_descriptors():
    observer = FrozenResNetObserver(arch="resnet18", pretrained=False, num_classes=10, grid_size=2)
    observations = observer(torch.randn(2, 3, 32, 32))
    torch.manual_seed(0)
    tokenizer = FlowTokenizer(
        layer_channels=observer.layer_channels,
        token_dim=32,
        flow_dim=16,
        traj_dim=16,
        grid_size=2,
    )
    batch = tokenizer(observations)

    assert batch.token_inputs.shape == (2, 8, 4, 32)
    assert batch.flow_targets.shape == (2, 8, 4, 16)
    assert batch.future_descriptors.shape == (2, 8, 4, 16)
    assert len(batch.local_features) == 8


def test_fixed_projection_initialization_is_deterministic():
    torch.manual_seed(11)
    first = FlowTokenizer([64, 64], token_dim=16, flow_dim=8, traj_dim=8, grid_size=2)
    torch.manual_seed(11)
    second = FlowTokenizer([64, 64], token_dim=16, flow_dim=8, traj_dim=8, grid_size=2)

    for left, right in zip(first.flow_projectors, second.flow_projectors):
        torch.testing.assert_close(left.weight, right.weight)
    for left, right in zip(first.future_projectors, second.future_projectors):
        torch.testing.assert_close(left.weight, right.weight)


def test_encoder_mask_is_causal_in_depth():
    encoder = SpatiotemporalEncoder(n_layers=3, grid_size=2, token_dim=16, n_heads=4, n_transformer_layers=1)
    mask = encoder.attn_mask

    assert mask[0, 0]
    assert not mask[0, 4]
    assert mask[4, 0]
