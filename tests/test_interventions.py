import numpy as np
import torch

from evaluation.interventions import (
    build_control_prototypes,
    build_circuit_library,
    build_circuit_prototype,
    compute_circuit_score,
    fit_linear_probe_from_features,
    forward_ctls_with_grad,
    optimize_images_for_score,
    random_direction_prototype,
    run_intervention_batch,
    select_circuit_set,
    summarize_intervention_results,
)


def test_forward_ctls_with_grad_matches_frozen_eval(backbone, meta_encoder, random_images):
    expected_trajectory = backbone(random_images)
    expected_flow = list(backbone._flow_targets)
    expected_z = meta_encoder(expected_trajectory)

    actual = forward_ctls_with_grad(backbone, meta_encoder, random_images.clone().requires_grad_(True))

    assert len(actual["trajectory"]) == len(expected_trajectory)
    assert len(actual["flow_targets"]) == len(expected_flow)
    assert len(actual["z_list"]) == len(expected_z)

    for exp, act in zip(expected_trajectory, actual["trajectory"]):
        torch.testing.assert_close(act.detach(), exp, atol=1e-5, rtol=1e-5)
    for exp, act in zip(expected_flow, actual["flow_targets"]):
        torch.testing.assert_close(act.detach(), exp, atol=1e-5, rtol=1e-5)
    for exp, act in zip(expected_z, actual["z_list"]):
        torch.testing.assert_close(act.detach(), exp, atol=1e-5, rtol=1e-5)

    assert actual["penultimate"].shape[0] == random_images.shape[0]


def test_build_circuit_prototype_and_score_are_deterministic():
    z_list = [
        np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
        np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    ]
    circuit = {
        "span": (0, 1),
        "image_mask": np.array([True, False, True]),
        "size": 2,
        "cluster_id": 1,
    }

    prototype = build_circuit_prototype(circuit, z_list, name="toy")
    scores = compute_circuit_score(
        [torch.from_numpy(layer) for layer in z_list],
        prototype,
    )

    torch.testing.assert_close(prototype.vectors[0], torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(prototype.vectors[1], torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(scores, torch.tensor([1.0, 0.0, 1.0]))


def test_fit_linear_probe_from_features_learns_separable_data():
    train_features = torch.tensor(
        [
            [2.0, 0.0],
            [1.5, 0.0],
            [0.0, 2.0],
            [0.0, 1.5],
        ]
    )
    train_labels = torch.tensor([0, 0, 1, 1])
    val_features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    val_labels = torch.tensor([0, 1])

    probe, metrics = fit_linear_probe_from_features(
        train_features,
        train_labels,
        val_features,
        val_labels,
        epochs=60,
        lr=0.1,
        batch_size=2,
    )

    with torch.no_grad():
        preds = probe(val_features).argmax(dim=1)

    assert metrics["best_val_acc"] >= 1.0
    torch.testing.assert_close(preds, val_labels)


def test_optimize_images_for_score_activation_increases_score():
    images = torch.zeros(2, 1, 1, 1)

    def score_fn(x):
        return x.view(x.shape[0], -1).sum(dim=1)

    optimized = optimize_images_for_score(
        images,
        score_fn,
        mode="activate",
        eps=torch.tensor([0.5]),
        step_size=torch.tensor([0.1]),
        n_steps=5,
        clamp_bounds=torch.tensor([[-1.0], [1.0]]),
    )

    assert optimized.mean() > images.mean()


def test_optimize_images_for_score_suppression_decreases_score():
    images = torch.zeros(2, 1, 1, 1)

    def score_fn(x):
        return x.view(x.shape[0], -1).sum(dim=1)

    optimized = optimize_images_for_score(
        images,
        score_fn,
        mode="suppress",
        eps=torch.tensor([0.5]),
        step_size=torch.tensor([0.1]),
        n_steps=5,
        clamp_bounds=torch.tensor([[-1.0], [1.0]]),
    )

    assert optimized.mean() < images.mean()


def test_select_circuit_set_and_controls_are_built():
    z_list = [
        np.random.randn(10, 4).astype(np.float32),
        np.random.randn(10, 4).astype(np.float32),
    ]
    labels = np.array([0, 0, 0, 0, 1, 2, 3, 4, 5, 6])
    circuits = [
        {
            "span": (0, 0),
            "image_mask": np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=bool),
            "size": 4,
            "cluster_id": 0,
            "elevation_sigma": 2.0,
        },
        {
            "span": (0, 0),
            "image_mask": np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0], dtype=bool),
            "size": 4,
            "cluster_id": 1,
            "elevation_sigma": 1.5,
        },
        {
            "span": (0, 1),
            "image_mask": np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=bool),
            "size": 5,
            "cluster_id": 2,
            "elevation_sigma": 1.0,
        },
    ]

    library = build_circuit_library(circuits, z_list, labels, purity_specific=0.7, purity_agnostic=0.3)
    selected = select_circuit_set(library, n_specific=1, n_agnostic=1, min_size=4)
    controls = build_control_prototypes(selected[0], library, z_list)

    assert len(selected) >= 1
    assert {"matched_random", "wrong_circuit", "random_direction"} <= set(controls)


def test_run_intervention_batch_changes_circuit_score(backbone, meta_encoder, random_images):
    probe = torch.nn.Linear(backbone.model.fc.in_features, 10)
    base = forward_ctls_with_grad(backbone, meta_encoder, random_images)
    target = random_direction_prototype(
        (0, 0),
        dim=base["z_list"][0].shape[1],
        name="control",
    )

    result = run_intervention_batch(
        backbone,
        meta_encoder,
        probe,
        random_images,
        target,
        mode="activate",
        eps_pixels=2.0 / 255.0,
        step_pixels=1.0 / 255.0,
        n_steps=2,
    )

    assert result["scores_after"].mean() != result["scores_before"].mean()


def test_summarize_intervention_results_flattens_rows():
    rows = summarize_intervention_results(
        [
            {"summary": {"circuit_name": "c1", "delta_score": 0.1}},
            {"summary": {"circuit_name": "c2", "delta_score": -0.2}},
        ]
    )

    assert rows == [
        {"circuit_name": "c1", "delta_score": 0.1},
        {"circuit_name": "c2", "delta_score": -0.2},
    ]
