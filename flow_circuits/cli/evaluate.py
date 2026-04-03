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
    _print_evaluation_summary(summary)


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


def _print_evaluation_summary(s: dict) -> None:
    bar = "=" * 64
    rm = s["representation_metrics"]
    bl = s["baseline_comparison"]
    cc = s.get("confirmatory_checks", {})
    nc = s.get("null_checks", {})

    print(f"\n{bar}", flush=True)
    print(f"Evaluation Results  (split={s['split']}, n={s['n_images']:,} images)", flush=True)
    print(bar, flush=True)

    pred_mean = rm.get("prediction_cosine_mean", float("nan"))
    pred_sem  = rm.get("prediction_cosine_sem", float("nan"))
    recon     = rm.get("reconstruction_cosine_mean", float("nan"))
    recon_sem = rm.get("reconstruction_cosine_sem", float("nan"))
    traj_mean = rm.get("trajectory_alignment_mean", float("nan"))
    traj_std  = rm.get("trajectory_alignment_std", float("nan"))
    best_bl   = bl.get("best_baseline", float("nan"))

    print("\nRepresentation Quality", flush=True)
    print(f"  Prediction cosine  : {pred_mean:.4f} +/- {pred_sem:.4f}", flush=True)
    print(f"    How well the encoder predicts next-layer flow targets.", flush=True)
    print(f"    1.0 = perfect. Best simple baseline = {best_bl:.4f}.", flush=True)
    print(f"  Reconstruction cos : {recon:.4f} +/- {recon_sem:.4f}", flush=True)
    print(f"    How well the encoder reconstructs the current layer's flow.", flush=True)
    print(f"  Trajectory align   : {traj_mean:.4f} +/- {traj_std:.4f} std", flush=True)
    print(f"    Spatial consistency of representations across images.", flush=True)

    print("\nBaseline Comparison  (three simple predictors)", flush=True)
    best_name = bl.get("best_baseline_name", "")
    for name, label in [("mean_baseline", "Mean predictor "), ("local_baseline", "Local CNN MLP  "), ("flow_baseline", "Flow target MLP")]:
        best_marker = "  <- best" if best_name == name else ""
        print(f"  {label}: {bl.get(name, float('nan')):.4f}{best_marker}", flush=True)
    print(f"  Our encoder    : {pred_mean:.4f}  (+{pred_mean - best_bl:.4f} vs best baseline)", flush=True)

    if cc:
        print("\nConfirmatory Checks", flush=True)
        for key, label in [("p1_prediction_vs_best_baseline", "P1 prediction > baseline"),
                            ("p2_alignment_vs_best_baseline", "P2 alignment  > baseline")]:
            c = cc.get(key, {})
            status = "PASS" if c.get("passes") else "FAIL"
            ci_lo = c.get("ci_lower", 0)
            ci_hi = c.get("ci_upper", 0)
            print(f"  {label} : {status}  CI [{ci_lo:.4f}, {ci_hi:.4f}]", flush=True)

    if nc:
        print("\nNull Checks  (sanity — expect meaningful drops below)", flush=True)
        fut = nc.get("future_shuffle_prediction", {})
        dep = nc.get("depth_order_alignment", {})
        fut_drop = fut.get("drop", 0)
        dep_drop = dep.get("drop", 0)
        fut_ok = "OK" if fut_drop > 0.1 else "WARN"
        dep_ok = "OK" if dep_drop > 0.0 else "WARN"
        print(f"  Shuffle future targets  : drop={fut_drop:.4f}  [{fut_ok}]", flush=True)
        print(f"    Large drop means the encoder genuinely uses future structure.", flush=True)
        print(f"  Permute depth order     : drop={dep_drop:.4f}  [{dep_ok}]", flush=True)
        print(f"    Positive drop means the encoder uses depth ordering.", flush=True)

    print(f"\nFull results saved to file.\n{bar}\n", flush=True)


if __name__ == "__main__":
    main()
