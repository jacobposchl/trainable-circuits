from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time

from flow_circuits.data import build_cifar10_splits
from flow_circuits.discovery import (
    CandidateCircuitDiscoverer,
    run_node_shuffle_null,
    summarize_seed_stability,
)
from flow_circuits.training import collect_discovery_outputs, load_components_from_checkpoint
from flow_circuits.utils import seed_everything


def _format_seconds(seconds: float) -> str:
    seconds = int(max(0, round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


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
    discovery_batch_size = config["discovery"].get("batch_size", config["data"]["batch_size"])
    loaders = build_cifar10_splits(
        data_dir=config["data"]["data_dir"],
        batch_size=discovery_batch_size,
        num_workers=config["data"].get("num_workers", 4),
        seed=config["data"].get("seed", 0),
        augment_fit=config["data"].get("augment_fit", True),
        download=config["data"].get("download", True),
    )

    def log(message: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)

    def collect_progress(*, batch_idx: int, total_batches: int | None, seen_images: int, target_images: int | None) -> None:
        target = target_images if target_images is not None else "all"
        total = total_batches if total_batches is not None else "?"
        print(
            f"[{time.strftime('%H:%M:%S')}] Discovery data pass: batch {batch_idx}/{total}  images {seen_images}/{target}",
            flush=True,
        )

    log("Loading discovery features from checkpoint outputs...")
    outputs = collect_discovery_outputs(
        components,
        loaders["discovery"],
        device=device,
        max_images=config["discovery"].get("max_images"),
        progress_callback=collect_progress,
    )
    log("Discovery feature collection complete.")

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
    n_jobs = max(1, int(config["discovery"].get("n_jobs", 1)))
    compute_seed_stability = bool(config["discovery"].get("compute_seed_stability", True))
    compute_node_shuffle_null = bool(config["discovery"].get("compute_node_shuffle_null", True))

    if n_jobs > 1 and len(discovery_seeds) > 1:
        log(f"Running discovery seeds with thread parallelism (n_jobs={n_jobs}).")

    def run_seed(seed_idx: int, discovery_seed: int) -> dict:
        log(f"Discovery seed {seed_idx}/{len(discovery_seeds)} (seed={discovery_seed})")
        discoverer = CandidateCircuitDiscoverer(**discoverer_kwargs, random_seed=discovery_seed)
        cluster_t0: list[float | None] = [None]

        def discovery_progress(**event) -> None:
            stage = event.get("stage")
            if stage == "node_clustering_start":
                cluster_t0[0] = time.time()
                log(
                    f"Discovery seed {seed_idx}/{len(discovery_seeds)}:"
                    f" starting node clustering over {event['total']} nodes"
                    f" (HDBSCAN + {discoverer.bootstrap_iterations} bootstrap stability runs per node"
                    f" - first node may take a minute or more)..."
                )
            elif stage == "node_clustering":
                completed = event["completed"]
                total = event["total"]
                eta_str = ""
                if cluster_t0[0] is not None and completed > 0:
                    elapsed = time.time() - cluster_t0[0]
                    rate = elapsed / completed
                    remaining = rate * (total - completed)
                    eta_str = f"  ETA ~{_format_seconds(remaining)}"
                print(
                    f"[{time.strftime('%H:%M:%S')}] Discovery seed {seed_idx}/{len(discovery_seeds)}:"
                    f" node {completed}/{total}"
                    f" (layer {event['layer_idx']}, cell {event['cell_idx']})"
                    f"  clusters={event['n_node_clusters']}{eta_str}",
                    flush=True,
                )
            elif stage == "node_clustering_done":
                elapsed = (time.time() - cluster_t0[0]) if cluster_t0[0] is not None else 0.0
                log(
                    f"Discovery seed {seed_idx}/{len(discovery_seeds)}:"
                    f" node clustering complete - {event['n_node_clusters']} node clusters retained"
                    f" in {_format_seconds(elapsed)}"
                )

        artifact = discoverer.discover(
            future_descriptors=outputs["future_descriptors"].numpy(),
            predicted_next=outputs["predicted_next"].numpy(),
            flow_targets=outputs["flow_targets"].numpy(),
            dataset_indices=outputs["indices"].numpy(),
            labels=outputs["labels"].numpy(),
            progress_callback=discovery_progress,
        )
        log(
            f"Discovery seed {seed_idx}/{len(discovery_seeds)} complete:"
            f" {len(artifact.get('circuits', []))} candidate circuits"
        )
        return {
            "seed": int(discovery_seed),
            "node_clusters": artifact.get("node_clusters", []),
            "circuits": artifact.get("circuits", []),
        }

    seed_runs_by_index: dict[int, dict] = {}
    if n_jobs > 1 and len(discovery_seeds) > 1:
        max_workers = min(n_jobs, len(discovery_seeds))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(run_seed, seed_idx, discovery_seed): seed_idx
                for seed_idx, discovery_seed in enumerate(discovery_seeds, start=1)
            }
            for future in as_completed(futures):
                seed_runs_by_index[futures[future]] = future.result()
    else:
        for seed_idx, discovery_seed in enumerate(discovery_seeds, start=1):
            seed_runs_by_index[seed_idx] = run_seed(seed_idx, discovery_seed)

    seed_runs = [seed_runs_by_index[idx] for idx in sorted(seed_runs_by_index)]
    primary_run = seed_runs[0] if seed_runs else {"seed": config["discovery"].get("seed", 0), "node_clusters": [], "circuits": []}

    n_reference_circuits = len(primary_run["circuits"])
    if compute_seed_stability and len(seed_runs) > 1 and n_reference_circuits > 0:
        log(
            f"Cross-seed stability: measuring reproducibility of {n_reference_circuits} circuit(s)"
            f" across {len(seed_runs)} seeds (bootstrap iterations:"
            f" {config['discovery'].get('stability_bootstrap_iterations', 500)})..."
        )
    elif compute_seed_stability:
        log("Skipping cross-seed stability (only one seed or no circuits found).")
    else:
        log("Cross-seed stability disabled by config.")

    stability_t0: list[float | None] = [None]

    def stability_progress(**event) -> None:
        if stability_t0[0] is None:
            stability_t0[0] = time.time()
        completed = event["completed"]
        total = event["total"]
        eta_str = ""
        if completed > 0:
            elapsed = time.time() - stability_t0[0]
            rate = elapsed / completed
            remaining = rate * (total - completed)
            eta_str = f"  ETA ~{_format_seconds(remaining)}"
        print(
            f"[{time.strftime('%H:%M:%S')}] Seed stability:"
            f" circuit {completed}/{total}"
            f" (circuit_id={event['circuit_id']}){eta_str}",
            flush=True,
        )

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
        "stability_summary": {},
        "null_checks": {},
    }
    if compute_seed_stability:
        artifact["stability_summary"] = summarize_seed_stability(
            seed_runs,
            bootstrap_iterations=config["discovery"].get("stability_bootstrap_iterations", 500),
            seed=config["discovery"].get("seed", 0),
            progress_callback=stability_progress,
        )
    else:
        artifact["stability_summary"] = {
            "n_seed_runs": int(len(seed_runs)),
            "reference_seed": int(primary_run["seed"]),
            "per_circuit": [],
            "skipped": True,
            "reason": "disabled_by_config",
        }

    if compute_node_shuffle_null:
        log("Node-shuffle null: re-running discovery with shuffled node labels (sanity check - circuits should disappear)...")
        artifact["null_checks"]["node_shuffle"] = run_node_shuffle_null(
            future_descriptors=outputs["future_descriptors"].numpy(),
            predicted_next=outputs["predicted_next"].numpy(),
            flow_targets=outputs["flow_targets"].numpy(),
            dataset_indices=outputs["indices"].numpy(),
            labels=outputs["labels"].numpy(),
            discoverer_kwargs=discoverer_kwargs,
            seed=config["discovery"].get("node_shuffle_seed", config["discovery"].get("seed", 0) + 101),
        )
        log("Node-shuffle null complete.")
    else:
        log("Node-shuffle null disabled by config.")
        artifact["null_checks"]["node_shuffle"] = {
            "seed": int(config["discovery"].get("node_shuffle_seed", config["discovery"].get("seed", 0) + 101)),
            "n_node_clusters": 0,
            "n_circuits": 0,
            "skipped": True,
            "reason": "disabled_by_config",
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
