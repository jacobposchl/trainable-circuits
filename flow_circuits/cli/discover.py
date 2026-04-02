from __future__ import annotations

import argparse
import json
from pathlib import Path

from flow_circuits.data import build_cifar10_splits
from flow_circuits.discovery import CandidateCircuitDiscoverer
from flow_circuits.training import collect_model_outputs, load_components_from_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run q-based candidate-circuit discovery")
    parser.add_argument("--checkpoint", required=True, help="Path to a flow-circuits checkpoint")
    parser.add_argument("--output", default=None, help="Optional circuit artifact JSON path")
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
        loaders["discovery"],
        device=device,
        max_images=config["discovery"].get("max_images"),
    )
    discoverer = CandidateCircuitDiscoverer(
        grid_size=config["tokenization"].get("grid_size", 4),
        min_cluster_fraction=config["discovery"].get("min_cluster_fraction", 0.005),
        max_cluster_fraction=config["discovery"].get("max_cluster_fraction", 0.40),
        min_cluster_size=config["discovery"].get("min_cluster_size", 20),
        bootstrap_iterations=config["discovery"].get("bootstrap_iterations", 20),
        stability_threshold=config["discovery"].get("stability_threshold", 0.60),
        merge_threshold=config["discovery"].get("merge_threshold", 0.70),
        node_threshold=config["discovery"].get("node_threshold", 0.70),
        random_seed=config["discovery"].get("seed", 0),
    )
    artifact = discoverer.discover(
        future_descriptors=outputs["future_descriptors"].numpy(),
        predicted_next=outputs["predicted_next"].numpy(),
        flow_targets=outputs["flow_targets"].numpy(),
        dataset_indices=outputs["indices"].numpy(),
        labels=outputs["labels"].numpy(),
    )
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(args.checkpoint).with_name("candidate_circuits.json")
    discoverer.save(artifact, output_path)
    print(json.dumps({"n_circuits": len(artifact["circuits"]), "output": str(output_path)}, indent=2))


if __name__ == "__main__":
    main()
