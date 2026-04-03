from __future__ import annotations

import argparse
import json
from pathlib import Path

from flow_circuits.data import build_cifar10_splits
from flow_circuits.discovery import (
    CandidateCircuitDiscoverer,
    run_node_shuffle_null,
    summarize_seed_stability,
)
from flow_circuits.training import collect_model_outputs, load_components_from_checkpoint
from flow_circuits.utils import seed_everything


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
    seed_everything(config["data"].get("seed", 0))
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

    discoverer_kwargs = {
        "grid_size": config["tokenization"].get("grid_size", 4),
        "min_cluster_fraction": config["discovery"].get("min_cluster_fraction", 0.005),
        "max_cluster_fraction": config["discovery"].get("max_cluster_fraction", 0.40),
        "min_cluster_size": config["discovery"].get("min_cluster_size", 20),
        "bootstrap_iterations": config["discovery"].get("bootstrap_iterations", 20),
        "stability_threshold": config["discovery"].get("stability_threshold", 0.60),
        "merge_threshold": config["discovery"].get("merge_threshold", 0.70),
        "node_threshold": config["discovery"].get("node_threshold", 0.70),
    }
    discovery_seeds = config["discovery"].get("seeds") or [config["discovery"].get("seed", 0)]

    seed_runs = []
    for discovery_seed in discovery_seeds:
        discoverer = CandidateCircuitDiscoverer(**discoverer_kwargs, random_seed=discovery_seed)
        artifact = discoverer.discover(
            future_descriptors=outputs["future_descriptors"].numpy(),
            predicted_next=outputs["predicted_next"].numpy(),
            flow_targets=outputs["flow_targets"].numpy(),
            dataset_indices=outputs["indices"].numpy(),
            labels=outputs["labels"].numpy(),
        )
        seed_runs.append(
            {
                "seed": int(discovery_seed),
                "node_clusters": artifact.get("node_clusters", []),
                "circuits": artifact.get("circuits", []),
            }
        )

    primary_run = seed_runs[0] if seed_runs else {"seed": config["discovery"].get("seed", 0), "node_clusters": [], "circuits": []}
    artifact = {
        "metadata": {
            "n_images": int(outputs["future_descriptors"].shape[0]),
            "n_layers": int(outputs["future_descriptors"].shape[1]),
            "n_cells": int(outputs["future_descriptors"].shape[2]),
            "grid_size": int(config["tokenization"].get("grid_size", 4)),
            "random_seed": int(primary_run["seed"]),
            "discovery_seeds": [int(seed) for seed in discovery_seeds],
        },
        "node_clusters": primary_run["node_clusters"],
        "circuits": primary_run["circuits"],
        "seed_runs": seed_runs,
        "stability_summary": summarize_seed_stability(
            seed_runs,
            bootstrap_iterations=config["discovery"].get("stability_bootstrap_iterations", 500),
            seed=config["discovery"].get("seed", 0),
        ),
        "null_checks": {
            "node_shuffle": run_node_shuffle_null(
                future_descriptors=outputs["future_descriptors"].numpy(),
                predicted_next=outputs["predicted_next"].numpy(),
                flow_targets=outputs["flow_targets"].numpy(),
                dataset_indices=outputs["indices"].numpy(),
                labels=outputs["labels"].numpy(),
                discoverer_kwargs=discoverer_kwargs,
                seed=config["discovery"].get("node_shuffle_seed", config["discovery"].get("seed", 0) + 101),
            ),
        },
    }
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(args.checkpoint).with_name("candidate_circuits.json")
    CandidateCircuitDiscoverer(**discoverer_kwargs, random_seed=primary_run["seed"]).save(artifact, output_path)
    n_circuits = len(artifact["circuits"])
    n_nodes = artifact["metadata"]["n_layers"] * artifact["metadata"]["n_cells"]
    active_nodes = {tuple(node) for c in artifact["circuits"] for node in c["active_nodes"]}
    coverage = len(active_nodes) / max(n_nodes, 1)
    bar = "=" * 64
    print(f"\n{bar}", flush=True)
    print("Circuit Discovery Complete", flush=True)
    print(bar, flush=True)
    print(f"  Candidate circuits found : {n_circuits}", flush=True)
    print(f"  Active node coverage     : {len(active_nodes)}/{n_nodes} nodes ({coverage:.1%})", flush=True)
    if n_circuits > 0:
        sizes = [len(c["image_set"]) for c in artifact["circuits"]]
        print(f"  Circuit size (images)    : min={min(sizes)}  max={max(sizes)}  mean={sum(sizes)/len(sizes):.1f}", flush=True)
    print(f"  Results saved to         : {output_path}", flush=True)
    print(f"{bar}\n", flush=True)


if __name__ == "__main__":
    main()
