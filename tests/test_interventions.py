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
    select_intervention_images,
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


def test_build_circuit_library_adds_selectivity_stats():
    z_list = [
        np.array(
            [
                [1.0, 0.0],
                [0.95, 0.05],
                [0.9, 0.1],
                [-1.0, 0.0],
                [-0.95, -0.05],
                [-0.9, -0.1],
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                [0.0, 1.0],
                [0.05, 0.95],
                [0.1, 0.9],
                [0.0, -1.0],
                [-0.05, -0.95],
                [-0.1, -0.9],
            ],
            dtype=np.float32,
        ),
    ]
    labels = np.array([0, 0, 0, 1, 1, 1])
    circuits = [
        {
            "span": (0, 1),
            "image_mask": np.array([1, 1, 1, 0, 0, 0], dtype=bool),
            "size": 3,
            "cluster_id": 0,
            "elevation_sigma": 2.0,
        }
    ]

    library = build_circuit_library(circuits, z_list, labels, purity_specific=0.7, purity_agnostic=0.3)

    assert len(library) == 1
    circuit = library[0]
    assert circuit["circuit_type"] == "class_specific"
    assert circuit["score_gap"] > 1.5
    assert circuit["score_auc"] > 0.99
    assert circuit["mean_member_score"] > circuit["mean_nonmember_score"]


def test_select_circuit_set_filters_by_selectivity_and_depth():
    library = [
        {
            "name": "shallow",
            "span": (0, 0),
            "size": 40,
            "purity": 0.95,
            "circuit_type": "class_specific",
            "associated_class": 0,
            "score_gap": 0.20,
            "score_auc": 0.95,
            "elevation_sigma": 2.0,
        },
        {
            "name": "weak_gap",
            "span": (1, 2),
            "size": 40,
            "purity": 0.96,
            "circuit_type": "class_specific",
            "associated_class": 1,
            "score_gap": 0.01,
            "score_auc": 0.95,
            "elevation_sigma": 3.0,
        },
        {
            "name": "nan_auc",
            "span": (1, 2),
            "size": 40,
            "purity": 0.97,
            "circuit_type": "class_specific",
            "associated_class": 2,
            "score_gap": 0.08,
            "score_auc": float("nan"),
            "elevation_sigma": 3.0,
        },
        {
            "name": "strong",
            "span": (1, 3),
            "size": 45,
            "purity": 0.98,
            "circuit_type": "class_specific",
            "associated_class": 3,
            "score_gap": 0.12,
            "score_auc": 0.92,
            "elevation_sigma": 2.5,
        },
    ]

    selected = select_circuit_set(
        library,
        n_circuits=2,
        min_size=20,
        min_purity=0.9,
        min_score_gap=0.05,
        min_score_auc=0.8,
        min_span_length=2,
        min_start_layer=1,
    )

    assert [c["name"] for c in selected] == ["strong"]


def test_select_intervention_images_prefers_low_and_high_score_examples():
    z_list = [
        torch.tensor(
            [
                [1.0, 0.0],
                [0.8, 0.2],
                [0.3, 0.7],
                [-0.2, 1.0],
                [-1.0, 0.0],
            ]
        )
    ]
    labels = torch.tensor([1, 0, 0, 1, 1])
    prototype = random_direction_prototype((0, 0), dim=2, generator=torch.Generator().manual_seed(0))
    prototype.vectors = [torch.tensor([1.0, 0.0])]
    prototype.associated_class = 1

    activate_idx = select_intervention_images(z_list, labels, prototype, mode="activate", n_images=2)
    suppress_idx = select_intervention_images(z_list, labels, prototype, mode="suppress", n_images=2)

    torch.testing.assert_close(activate_idx, torch.tensor([2, 1]))
    torch.testing.assert_close(suppress_idx, torch.tensor([0, 3]))


def test_build_control_prototypes_exposes_all_control_types():
    z_list = [
        np.array(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.8, 0.2],
                [-1.0, 0.0],
                [-0.9, -0.1],
                [-0.8, -0.2],
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                [0.0, 1.0],
                [0.1, 0.9],
                [0.2, 0.8],
                [0.0, -1.0],
                [-0.1, -0.9],
                [-0.2, -0.8],
            ],
            dtype=np.float32,
        ),
    ]
    labels = np.array([0, 0, 0, 1, 1, 1])
    circuits = [
        {
            "span": (0, 1),
            "image_mask": np.array([1, 1, 1, 0, 0, 0], dtype=bool),
            "size": 3,
            "cluster_id": 0,
            "elevation_sigma": 2.0,
        },
        {
            "span": (0, 1),
            "image_mask": np.array([0, 0, 0, 1, 1, 1], dtype=bool),
            "size": 3,
            "cluster_id": 1,
            "elevation_sigma": 1.8,
        },
    ]

    library = build_circuit_library(circuits, z_list, labels, purity_specific=0.7, purity_agnostic=0.3)
    controls = build_control_prototypes(library[0], library, z_list)

    assert {"matched_random", "wrong_circuit", "random_direction"} <= set(controls)
    assert controls["wrong_circuit"].name == library[1]["prototype"].name


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


def test_run_intervention_batch_uses_selected_circuit_readout(backbone, meta_encoder, random_images):
    probe = torch.nn.Linear(backbone.model.fc.in_features, 10)
    base = forward_ctls_with_grad(backbone, meta_encoder, random_images)
    target = random_direction_prototype(
        (0, 0),
        dim=base["z_list"][0].shape[1],
        name="control",
    )
    target.associated_class = 1
    target.associated_label = "automobile"

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
        readout_class=3,
        readout_label="cat",
    )

    assert result["summary"]["associated_class"] == 3
    assert result["summary"]["associated_label"] == "cat"
    assert "delta_associated_logit" in result["summary"]


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
