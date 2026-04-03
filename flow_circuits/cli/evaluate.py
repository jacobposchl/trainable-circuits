from __future__ import annotations

import argparse
import json
from pathlib import Path

from flow_circuits.data import build_cifar10_splits
from flow_circuits.evaluation import (
    compute_alignment_scores,
    compute_prediction_scores_by_image,
    evaluate_alignment_check,
    evaluate_prediction_check,
    evaluate_representation_metrics,
)
from flow_circuits.training import (
    BaselineRegressors,
    collect_baseline_features,
    collect_model_outputs,
    load_components_from_checkpoint,
)
from flow_circuits.utils import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate representation metrics for the flow model")
    parser.add_argument("--checkpoint", required=True, help="Path to a flow-circuits checkpoint")
    parser.add_argument("--split", default="test", choices=["val", "test"], help="Evaluation split")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    components = load_components_from_checkpoint(args.checkpoint, device=device)
    config = components.config
    seed = config["data"].get("seed", 0)
    seed_everything(seed)
    loaders = build_cifar10_splits(
        data_dir=config["data"]["data_dir"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"].get("num_workers", 4),
        seed=seed,
        augment_fit=config["data"].get("augment_fit", True),
        download=config["data"].get("download", True),
    )

    split_max_images = _max_images_for_split(config, args.split)
    outputs = collect_model_outputs(
        components,
        loaders[args.split],
        device=device,
        max_images=split_max_images,
    )
    metrics = evaluate_representation_metrics(
        outputs["z"],
        outputs["local_features"],
        outputs["flow_targets"],
        outputs["future_descriptors"],
        outputs["predicted_next"],
        outputs["reconstructed_current"],
        max_alignment_pairs=config["training"].get("alignment_max_pairs", 2048),
        alignment_seed=seed,
    )

    fit_local, fit_flow, fit_next = collect_baseline_features(
        components,
        loaders["fit"],
        device=device,
        max_images=config["training"].get("baseline_fit_images", 1024),
    )
    baseline_regressors = BaselineRegressors.fit(
        local_features=fit_local,
        flow_features=fit_flow,
        next_targets=fit_next,
        hidden_dim=config["training"].get("baseline_hidden_dim"),
        epochs=config["training"].get("baseline_epochs", 10),
        batch_size=config["training"].get("baseline_batch_size", 1024),
        lr=config["training"].get("baseline_lr", 1.0e-3),
        weight_decay=config["training"].get("baseline_weight_decay", 1.0e-4),
        seed=seed,
        device=device,
    )

    eval_local, eval_flow, eval_next = collect_baseline_features(
        components,
        loaders[args.split],
        device=device,
        max_images=split_max_images,
    )
    baseline_metrics = baseline_regressors.evaluate(
        local_features=eval_local,
        flow_features=eval_flow,
        next_targets=eval_next,
    )
    baseline_scores = baseline_regressors.score_predictions(
        local_features=eval_local,
        flow_features=eval_flow,
        next_targets=eval_next,
    )
    prediction_check = evaluate_prediction_check(
        model_scores=compute_prediction_scores_by_image(outputs["predicted_next"], outputs["flow_targets"]),
        baseline_scores=baseline_scores,
        bootstrap_iterations=config["training"].get("confirmatory_bootstrap_iterations", 500),
        seed=seed,
    )
    alignment_scores = compute_alignment_scores(
        z=outputs["z"],
        local_features=outputs["local_features"],
        flow_targets=outputs["flow_targets"],
        future_descriptors=outputs["future_descriptors"],
        max_alignment_pairs=config["training"].get("alignment_max_pairs", 2048),
        seed=seed,
    )
    alignment_check = evaluate_alignment_check(
        alignment_scores=alignment_scores,
        bootstrap_iterations=config["training"].get("confirmatory_bootstrap_iterations", 500),
        seed=seed,
    )
    null_checks = {
        "future_shuffle_prediction": _future_shuffle_prediction_null(
            outputs["predicted_next"],
            outputs["flow_targets"],
            seed=seed + 31,
        ),
        "depth_order_alignment": _depth_order_alignment_null(
            components=components,
            flow_targets=outputs["flow_targets"],
            z=outputs["z"],
            local_features=outputs["local_features"],
            max_alignment_pairs=config["training"].get("alignment_max_pairs", 2048),
            seed=seed + 61,
        ),
    }

    summary = {
        "split": args.split,
        "n_images": int(outputs["z"].shape[0]),
        "representation_metrics": metrics.to_dict(),
        "baseline_comparison": baseline_metrics.to_dict(),
        "confirmatory_checks": {
            "p1_prediction_vs_best_baseline": prediction_check.to_dict(),
            "p2_alignment_vs_best_baseline": alignment_check.to_dict(),
        },
        "null_checks": null_checks,
    }
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(args.checkpoint).with_name(f"{args.split}_evaluation.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def _max_images_for_split(config: dict, split: str) -> int | None:
    if split == "test":
        return config["training"].get("test_evaluation_images")
    return config["training"].get("validation_images", 512)


def _future_shuffle_prediction_null(
    predicted_next,
    flow_targets,
    *,
    seed: int,
) -> dict:
    import torch

    rng = torch.Generator().manual_seed(seed)
    shuffled_targets = flow_targets[:, 1:].clone()
    for layer_idx in range(shuffled_targets.shape[1]):
        for cell_idx in range(shuffled_targets.shape[2]):
            order = torch.randperm(shuffled_targets.shape[0], generator=rng)
            shuffled_targets[:, layer_idx, cell_idx] = shuffled_targets[order, layer_idx, cell_idx]
    observed = float((predicted_next * flow_targets[:, 1:]).sum(dim=-1).mean().item())
    shuffled = float((predicted_next * shuffled_targets).sum(dim=-1).mean().item())
    return {
        "observed_prediction_cosine_mean": observed,
        "shuffled_target_prediction_cosine_mean": shuffled,
        "drop": observed - shuffled,
    }


def _depth_order_alignment_null(
    *,
    components,
    flow_targets,
    z,
    local_features,
    max_alignment_pairs: int,
    seed: int,
) -> dict:
    import torch

    rng = torch.Generator().manual_seed(seed)
    depth_permutations = []
    for layer_idx in range(flow_targets.shape[1]):
        future_length = flow_targets.shape[1] - layer_idx
        depth_permutations.append(torch.randperm(future_length, generator=rng))
    with torch.no_grad():
        permuted = components.tokenizer.build_future_descriptors(
            flow_targets.to(components.encoder.final_norm.weight.device),
            depth_permutations=depth_permutations,
        ).cpu()
    permuted_alignment = compute_alignment_scores(
        z=z,
        local_features=local_features,
        flow_targets=flow_targets,
        future_descriptors=permuted,
        max_alignment_pairs=max_alignment_pairs,
        seed=seed,
    )
    observed_mean = float(compute_alignment_scores(
        z=z,
        local_features=local_features,
        flow_targets=flow_targets,
        future_descriptors=components.tokenizer.build_future_descriptors(
            flow_targets.to(components.encoder.final_norm.weight.device)
        ).cpu(),
        max_alignment_pairs=max_alignment_pairs,
        seed=seed,
    )["model_node_scores"].mean())
    permuted_mean = float(permuted_alignment["model_node_scores"].mean())
    return {
        "observed_alignment_mean": observed_mean,
        "depth_permuted_alignment_mean": permuted_mean,
        "drop": observed_mean - permuted_mean,
    }


if __name__ == "__main__":
    main()
