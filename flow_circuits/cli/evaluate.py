from __future__ import annotations

import argparse
import json
from pathlib import Path

from flow_circuits.data import build_cifar10_splits
from flow_circuits.training import collect_model_outputs, load_components_from_checkpoint
from flow_circuits.evaluation import evaluate_representation_metrics
from flow_circuits.training.trainer import FlowCircuitTrainer


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
    loaders = build_cifar10_splits(
        data_dir=config["data"]["data_dir"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"].get("num_workers", 4),
        seed=config["data"].get("seed", 0),
        augment_fit=config["data"].get("augment_fit", True),
        download=config["data"].get("download", True),
    )
    outputs = collect_model_outputs(
        components,
        loaders[args.split],
        device=device,
        max_images=config["training"].get("validation_images", 512),
    )
    metrics = evaluate_representation_metrics(
        outputs["z"],
        outputs["flow_targets"],
        outputs["future_descriptors"],
        outputs["predicted_next"],
        outputs["reconstructed_current"],
    )
    baseline_trainer = FlowCircuitTrainer(config)
    baseline_regressors = baseline_trainer._fit_baselines()
    baseline_metrics = baseline_trainer._evaluate_baselines(baseline_regressors)
    summary = {
        "split": args.split,
        "representation_metrics": metrics.to_dict(),
        "baseline_comparison": baseline_metrics.to_dict(),
    }
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(args.checkpoint).with_name(f"{args.split}_evaluation.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
