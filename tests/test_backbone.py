import torch
import pytest

from models import FrozenBackbone


def _linear_weight(module):
    if hasattr(module, "weight"):
        return module.weight
    for child in reversed(list(module.children())):
        if hasattr(child, "weight"):
            return child.weight
    raise AssertionError("No weight-bearing submodule found")


def test_unsupported_architecture_raises():
    with pytest.raises(ValueError):
        FrozenBackbone("mlp", num_classes=10, pretrained=False)


def test_resnet18_layer_dims_match_expected_length(backbone):
    assert len(backbone.layer_dims) == 8
    assert backbone.layer_dims == [32] * 8


def test_forward_populates_trajectory_and_flow_targets(backbone, random_images):
    trajectory = backbone(random_images)

    assert len(trajectory) == len(backbone.layer_dims)
    assert len(backbone._flow_targets) == len(backbone.layer_dims)

    for tensor in trajectory + backbone._flow_targets:
        assert tensor.shape == (random_images.shape[0], 32)
        assert not tensor.requires_grad
        norms = tensor.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4, rtol=1e-4)


def test_forward_resets_cached_outputs(backbone, random_images):
    first = backbone(random_images)
    second = backbone(random_images[:2])

    assert len(first) == len(second) == len(backbone.layer_dims)
    assert all(t.shape[0] == 2 for t in second)
    assert all(t.shape[0] == 2 for t in backbone._flow_targets)


def test_train_is_no_op(backbone):
    assert backbone.train() is backbone
    assert backbone.training is False


def test_compressors_are_frozen(backbone):
    assert all(not p.requires_grad for p in backbone.traj_compressors.parameters())
    assert all(not p.requires_grad for p in backbone.flow_compressors.parameters())


def test_resnet_uses_distinct_trajectory_and_flow_compressors(backbone):
    assert backbone.traj_compressors is not backbone.flow_compressors
    assert len(backbone.traj_compressors) == len(backbone.flow_compressors) == 8


def test_compressor_weights_are_reproducible():
    first = FrozenBackbone("resnet18", num_classes=10, pretrained=False, grid_size=2, flow_dim=32)
    second = FrozenBackbone("resnet18", num_classes=10, pretrained=False, grid_size=2, flow_dim=32)

    torch.testing.assert_close(
        _linear_weight(first.traj_compressors[0]),
        _linear_weight(second.traj_compressors[0]),
    )
    torch.testing.assert_close(
        _linear_weight(first.flow_compressors[0]),
        _linear_weight(second.flow_compressors[0]),
    )


def test_model_head_matches_requested_class_count():
    backbone = FrozenBackbone("resnet18", num_classes=17, pretrained=False, grid_size=2, flow_dim=32)
    assert backbone.model.fc.out_features == 17


def test_trainable_stem_is_the_only_unfrozen_backbone_component():
    backbone = FrozenBackbone(
        "resnet18",
        num_classes=10,
        pretrained=False,
        grid_size=2,
        flow_dim=32,
        trainable_stem=True,
    )

    trainable = [name for name, param in backbone.named_parameters() if param.requires_grad]

    assert trainable == ["model.conv1.weight"]


def test_trainable_stem_preserves_grad_through_trajectory_only():
    backbone = FrozenBackbone(
        "resnet18",
        num_classes=10,
        pretrained=False,
        grid_size=2,
        flow_dim=32,
        trainable_stem=True,
    )
    images = torch.randn(2, 3, 32, 32, requires_grad=True)

    trajectory = backbone(images)

    assert trajectory[0].requires_grad
    assert not backbone._flow_targets[0].requires_grad
