"""
CLI entry point for Phase 1 evaluation.

Loads a trained meta-encoder checkpoint and runs the 5 success criteria,
optionally with circuit discovery and visualizations.

Examples:
    # Run all criteria metrics
    python scripts/evaluate.py --config configs/phase1.yaml \\
        --checkpoint experiments/phase1/best.pt

    # Run circuit discovery
    python scripts/evaluate.py --config configs/phase1.yaml \\
        --checkpoint experiments/phase1/best.pt --discover

    # Generate UMAP visualizations
    python scripts/evaluate.py --config configs/phase1.yaml \\
        --checkpoint experiments/phase1/best.pt --viz
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import yaml
from pathlib import Path

from models.backbone import FrozenBackbone
from models.meta_encoder import MetaEncoder, ProfileRegressor
from data.cifar import get_standard_loaders


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Phase 1 meta-encoder")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--discover", action="store_true",
                        help="Run span-centric circuit discovery")
    parser.add_argument("--viz", action="store_true",
                        help="Generate UMAP visualizations")
    parser.add_argument("--output-dir", default=None,
                        help="Directory to save outputs (defaults to checkpoint dir)")
    parser.add_argument("--max-samples", type=int, default=2000,
                        help="Max samples for evaluation metrics")
    return parser.parse_args()


def load_model(config: dict, checkpoint_path: str, device: torch.device):
    """Build models and load checkpoint."""
    mcfg = config["model"]
    ecfg = mcfg["meta_encoder"]
    rcfg = mcfg.get("regressor", {})

    backbone = FrozenBackbone(
        arch=mcfg["arch"],
        num_classes=mcfg.get("num_classes", 10),
        pretrained=mcfg.get("pretrained", True),
        pool_mode=mcfg.get("pool_mode", "gap"),
    ).to(device)

    meta_encoder = MetaEncoder(
        layer_dims=backbone.layer_dims,
        projection_dim=ecfg.get("projection_dim", 128),
        n_heads=ecfg.get("n_heads", 4),
        n_transformer_layers=ecfg.get("n_transformer_layers", 2),
        dropout=ecfg.get("dropout", 0.0),
    ).to(device)

    regressor = ProfileRegressor(
        input_dim=ecfg.get("projection_dim", 128),
        hidden_dim=rcfg.get("hidden_dim", 64),
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    meta_encoder.load_state_dict(ckpt["meta_encoder_state"])
    regressor.load_state_dict(ckpt["regressor_state"])
    metrics = ckpt.get("val_metrics", {})
    r2_val = metrics.get("r2")
    r2_str = f"{r2_val:.4f}" if r2_val is not None else "N/A"
    print(f"Loaded: {checkpoint_path} (epoch {ckpt['epoch']}, R2={r2_str})")

    return backbone, meta_encoder, regressor


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dcfg = config["data"]

    backbone, meta_encoder, regressor = load_model(config, args.checkpoint, device)
    meta_encoder.eval()
    regressor.eval()

    _, val_loader = get_standard_loaders(
        data_dir=dcfg.get("data_dir", "data/cifar10"),
        batch_size=dcfg.get("batch_size", 256),
        num_workers=dcfg.get("num_workers", 4),
        augment=False,
    )

    output_dir = Path(args.output_dir or Path(args.checkpoint).parent)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Collect representations
    # ------------------------------------------------------------------ #
    from evaluation.circuit_analysis import CircuitAnalyzer

    analyzer = CircuitAnalyzer(backbone, meta_encoder, val_loader, device)
    print(f"Collecting representations (max {args.max_samples} samples)...")
    data = analyzer.collect_representations(max_samples=args.max_samples)

    trajectories = data["trajectories"]
    z_list = data["z_list"]
    labels = data["labels"]
    N = labels.shape[0]
    L = len(z_list)
    print(f"Collected {N} samples, {L} layers")

    # ------------------------------------------------------------------ #
    # Criterion 1: Profile Reconstruction R^2
    # ------------------------------------------------------------------ #
    from evaluation.metrics import (
        profile_reconstruction_r2,
        geometric_consistency,
    )

    print("\n--- Criterion 1: Profile Reconstruction ---")
    idx_a, idx_b = torch.triu_indices(N, N, offset=1)
    # Subsample pairs for tractability
    n_pairs = idx_a.shape[0]
    max_eval_pairs = 50000
    if n_pairs > max_eval_pairs:
        perm = torch.randperm(n_pairs)[:max_eval_pairs]
        idx_a, idx_b = idx_a[perm], idx_b[perm]

    true_sims = CircuitAnalyzer.compute_pair_profiles(trajectories, idx_a, idx_b)

    # Predict
    predicted = []
    with torch.no_grad():
        for l in range(L):
            z_a = z_list[l][idx_a]
            z_b = z_list[l][idx_b]
            pred_l = regressor(z_a * z_b)
            predicted.append(pred_l)
    predicted = torch.stack(predicted, dim=1).numpy()
    true_np = true_sims.numpy()

    r2_result = profile_reconstruction_r2(predicted, true_np)
    print(f"  R^2 = {r2_result['r2']:.4f}  (target >= 0.7, "
          f"{'PASS' if r2_result['passes'] else 'FAIL'})")

    # ------------------------------------------------------------------ #
    # Criterion 2: Geometric Consistency
    # ------------------------------------------------------------------ #
    print("\n--- Criterion 2: Geometric Consistency ---")
    z_sims = np.zeros((idx_a.shape[0], L))
    for l in range(L):
        z_a = z_list[l][idx_a].numpy()
        z_b = z_list[l][idx_b].numpy()
        z_sims[:, l] = (z_a * z_b).sum(axis=1)

    gc_result = geometric_consistency(z_sims, true_np, L)
    print(f"  Per-layer rho: {[f'{r:.3f}' for r in gc_result['per_layer_rho']]}")
    print(f"  Mean rho = {gc_result['mean_rho']:.4f}  (target > 0.65, "
          f"{'PASS' if gc_result['passes'] else 'FAIL'})")

    # ------------------------------------------------------------------ #
    # Circuit Discovery (optional)
    # ------------------------------------------------------------------ #
    if args.discover:
        from evaluation.discovery import SpanCentricDiscovery
        from evaluation.metrics import circuit_diversity, class_purity_distribution

        disc_cfg = config.get("discovery", {})
        discovery = SpanCentricDiscovery(
            n_layers=L,
            min_cluster_fraction=disc_cfg.get("min_cluster_fraction", 0.01),
            max_cluster_fraction=disc_cfg.get("max_cluster_fraction", 0.40),
            min_cluster_size=disc_cfg.get("hdbscan_min_cluster_size", 5),
        )

        print("\n--- Circuit Discovery ---")
        pair_indices = np.stack([idx_a.numpy(), idx_b.numpy()], axis=1)

        # Discovery runs on z-space similarity profiles — the representation
        # trained to organise pairs by circuit co-activation structure.
        circuits = discovery.discover_all(z_sims, pair_indices)
        print(f"  Found {len(circuits)} canonical circuits")

        for i, c in enumerate(circuits):
            purity = SpanCentricDiscovery.compute_class_purity(
                c, pair_indices, labels.numpy()
            )
            c["purity"] = purity
            print(f"  Circuit {i}: span={c['span']}, size={c['size']}, "
                  f"mean_sim={c['mean_similarity']:.3f}, purity={purity:.3f}")

        # Criterion 4: Diversity
        spans = [c["span"] for c in circuits]
        div_result = circuit_diversity(spans, L)
        print(f"\n  Criterion 4 — Coverage: {div_result['coverage']:.2f} "
              f"(target >= 0.6, {'PASS' if div_result['passes'] else 'FAIL'})")

        # Criterion 5: Class Purity Distribution
        purities = [c["purity"] for c in circuits]
        if purities:
            pur_result = class_purity_distribution(purities)
            print(f"  Criterion 5 — Agnostic(<0.3): {pur_result['n_agnostic']}, "
                  f"Specific(>0.7): {pur_result['n_specific']} "
                  f"({'PASS' if pur_result['passes'] else 'FAIL'})")

        # Multi-circuit membership
        membership = discovery.multi_circuit_membership(circuits, n_pairs=z_sims.shape[0])
        print(f"\n  Multi-circuit membership: "
              f"mean={membership.mean():.1f}, max={membership.max()}")

    # ------------------------------------------------------------------ #
    # Visualizations (optional)
    # ------------------------------------------------------------------ #
    if args.viz:
        from evaluation.circuit_viz import plot_per_layer_umap

        print("\n--- Generating Visualizations ---")
        z_np = [z.numpy() for z in z_list]
        labels_np = labels.numpy()

        fig = plot_per_layer_umap(z_np, labels_np)
        path = output_dir / "umap_per_layer.png"
        fig.savefig(str(path), dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")


if __name__ == "__main__":
    main()
