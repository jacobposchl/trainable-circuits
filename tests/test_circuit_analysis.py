import copy

import torch

from evaluation.circuit_analysis import CircuitAnalyzer, denormalize, load_checkpoint


def test_denormalize_preserves_shape_for_single_image():
    image = torch.zeros(3, 8, 8)
    output = denormalize(image)
    assert output.shape == image.shape
    assert torch.all((0.0 <= output) & (output <= 1.0))


def test_denormalize_preserves_shape_for_batch():
    images = torch.zeros(2, 3, 8, 8)
    output = denormalize(images)
    assert output.shape == images.shape
    assert torch.all((0.0 <= output) & (output <= 1.0))


def test_collect_representations_truncates_to_max_samples(backbone, meta_encoder, fake_loader):
    analyzer = CircuitAnalyzer(backbone, meta_encoder, fake_loader, torch.device("cpu"))
    result = analyzer.collect_representations(max_samples=6)

    assert set(result) == {"trajectories", "flow_targets", "z_list", "labels", "images"}
    assert result["labels"].shape[0] == 6
    assert result["images"].shape[0] == 6
    assert len(result["trajectories"]) == len(backbone.layer_dims)
    assert len(result["flow_targets"]) == len(backbone.layer_dims)
    assert len(result["z_list"]) == len(backbone.layer_dims)
    assert all(t.shape == (6, 32) for t in result["trajectories"])
    assert all(f.shape == (6, 32) for f in result["flow_targets"])
    assert all(z.shape == (6, 64) for z in result["z_list"])


def test_compute_all_profiles_matches_manual_matrix():
    flow_targets = [
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]),
        torch.tensor([[1.0, 1.0], [1.0, -1.0], [1.0, 1.0]]) / (2.0**0.5),
    ]

    profiles = CircuitAnalyzer.compute_all_profiles(flow_targets, chunk_size=2)

    expected_layer0 = flow_targets[0] @ flow_targets[0].t()
    expected_layer1 = flow_targets[1] @ flow_targets[1].t()
    torch.testing.assert_close(profiles[:, :, 0], expected_layer0)
    torch.testing.assert_close(profiles[:, :, 1], expected_layer1)


def test_compute_pair_profiles_matches_manual_values():
    flow_targets = [
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]),
        torch.tensor([[0.5, 0.5], [0.5, -0.5], [0.5, 0.5]]),
    ]
    idx_a = torch.tensor([0, 0, 1])
    idx_b = torch.tensor([1, 2, 2])

    profiles = CircuitAnalyzer.compute_pair_profiles(flow_targets, idx_a, idx_b)

    expected = torch.tensor([
        [0.0, 0.0],
        [1.0, 0.5],
        [0.0, 0.0],
    ])
    torch.testing.assert_close(profiles, expected)


def test_compute_pair_rich_profiles_matches_elementwise_products():
    flow_targets = [
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        torch.tensor([[2.0, 1.0], [5.0, 6.0]]),
    ]
    idx_a = torch.tensor([0])
    idx_b = torch.tensor([1])

    rich = CircuitAnalyzer.compute_pair_rich_profiles(flow_targets, idx_a, idx_b)

    torch.testing.assert_close(rich[0], torch.tensor([[3.0, 8.0]]))
    torch.testing.assert_close(rich[1], torch.tensor([[10.0, 6.0]]))


def test_compute_class_purity_uses_unique_inputs():
    pair_indices = torch.tensor([[0, 1], [1, 2], [2, 3]])
    labels = torch.tensor([1, 1, 2, 2])
    mask = torch.tensor([True, True, False])

    purity = CircuitAnalyzer.compute_class_purity(pair_indices, labels, mask)

    assert purity == 2 / 3


def test_compute_class_purity_returns_zero_for_empty_selection():
    pair_indices = torch.tensor([[0, 1], [1, 2]])
    labels = torch.tensor([0, 1, 2])
    mask = torch.tensor([False, False])
    assert CircuitAnalyzer.compute_class_purity(pair_indices, labels, mask) == 0.0


def test_load_checkpoint_restores_meta_encoder_and_info_loss(minimal_config, meta_encoder, info_loss, tmp_path):
    checkpoint_path = tmp_path / "checkpoint.pt"
    config = copy.deepcopy(minimal_config)

    torch.save(
        {
            "epoch": 3,
            "val_metrics": {"r2": 0.8, "mean_rho": 0.7},
            "meta_encoder_state": meta_encoder.state_dict(),
            "info_loss_state": info_loss.state_dict(),
            "optimizer_state": {},
            "config": config,
        },
        checkpoint_path,
    )

    _, loaded_meta_encoder, loaded_info_loss = load_checkpoint(
        config,
        str(checkpoint_path),
        torch.device("cpu"),
    )

    for key, value in meta_encoder.state_dict().items():
        torch.testing.assert_close(loaded_meta_encoder.state_dict()[key], value)
    for key, value in info_loss.state_dict().items():
        torch.testing.assert_close(loaded_info_loss.state_dict()[key], value)
    assert loaded_meta_encoder.training is False
    assert loaded_info_loss.training is False
