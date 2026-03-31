# Phase 1: Meta-Encoder Validation

A self-supervised framework for learning interpretable representations of neural network computational structure. The meta-encoder reads a frozen backbone's activation trajectories and maps them into a **circuit space** where geometric proximity reflects shared internal computation.

## Core Idea

Neural networks reuse recurring computational pathways — stable patterns of activation across contiguous layers. We call these **circuits**. The meta-encoder learns per-layer representations `z_1, ..., z_L` such that:

- Inputs processed similarly by the backbone at a given layer are close in z-space at that layer
- The layer-by-layer structure of z-space reveals *where* in the network similarity occurs
- Discovered circuits span identifiable, contiguous depth ranges

## Architecture

```
Input x
    |
[Frozen Backbone (ResNet18)]
    |
h_l(x) — AdaptiveMaxPool + Linear → L2-normalize → detach  (trajectory)
f_l(x) — AdaptiveMaxPool + Linear → L2-normalize → detach  (flow targets, pre-skip)
    |
[Per-layer projectors: Linear → GELU → LayerNorm]
    |
p_1, ..., p_L
    |
[RoPE Transformer Encoder]
    |
z_1, ..., z_L  (L2-normalized per-layer circuit representations)
```

## Training Objective

```
L = L_info
```

- **L_info** (fidelity): a per-layer MLP predicts the flow co-activation product `f_l^a ⊙ f_l^b` from `z_l^a * z_l^b`, minimizing a normalized (1 − R²) loss that is ~1.0 at initialization and approaches 0 at perfect reconstruction

No class labels are used in training. The signal comes entirely from the backbone's internal flow targets.

## Circuit Discovery

Post-training, circuits are discovered via **span-centric, image-centric clustering**:

1. Enumerate all `L(L+1)/2` contiguous layer spans `[l_start, l_end]`
2. For each span: concatenate per-image z-vectors across span layers, reduce with UMAP (cosine metric), cluster with HDBSCAN
3. Canonical circuits = clusters containing 1%–40% of all images
4. One image can belong to circuits at multiple spans (multi-circuit membership)

## Success Criteria

| # | Criterion | Target |
|---|-----------|--------|
| 1 | Profile Reconstruction R² | >= 0.7 |
| 2 | Geometric Consistency (Spearman ρ) | > 0.5/layer, > 0.65 mean |
| 3 | Within-Span Similarity Elevation | cluster mean > pop mean + 1σ |
| 4 | Circuit Diversity | >= 60% layer coverage |
| 5 | Class Purity Distribution | bimodal (agnostic + specific) |

## Repository Structure

```
models/
  backbone.py          # Frozen backbone with dual hooks (trajectory + flow targets)
  meta_encoder.py      # RoPE transformer, per-layer projectors, ProfileRegressor
losses/
  info_loss.py         # L_info: normalized flow co-activation reconstruction loss
training/
  unified_trainer.py   # Phase 1 training loop
evaluation/
  metrics.py           # 5 success criteria functions
  discovery.py         # Span-centric image-centric circuit discovery (UMAP + HDBSCAN)
  circuit_analysis.py  # Data collection and profile computation
  circuit_viz.py       # UMAP, circuit member grids, span coverage visualizations
data/
  cifar.py             # CIFAR-10 data loading
configs/
  phase1.yaml          # Main training config
  ablations/
    info_only.yaml     # Ablation: different checkpoint dir, same loss
scripts/
  train.py             # CLI training entry point (ctls-train)
  evaluate.py          # CLI evaluation entry point (ctls-evaluate)
notebooks/
  experiments/         # Experiment notebooks nb00–nb07
documents/
  project_context.md   # Detailed technical specification and design decisions
  preliminary_results.md  # Results on earlier checkpoints
tests/
  test_meta_encoder.py # RoPE, MetaEncoder, ProfileRegressor tests
  test_losses.py       # InfoLoss tests
  test_discovery.py    # Span enumeration, discovery pipeline tests
```

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd model_interpretability

# Install (editable, core deps only)
pip install -e .

# Install with ViT architecture support
pip install -e ".[vit]"

# Install with dev/test dependencies
pip install -e ".[dev]"
```

## Quickstart

```bash
# Train the meta-encoder (installed entry point)
ctls-train --config configs/phase1.yaml

# Or run directly
python scripts/train.py --config configs/phase1.yaml

# Resume from checkpoint
ctls-train --config configs/phase1.yaml --resume experiments/phase1/epoch_50.pt

# Evaluate with C1–C2 success criteria
ctls-evaluate --config configs/phase1.yaml \
    --checkpoint experiments/phase1/best.pt

# Run circuit discovery (C3–C5)
ctls-evaluate --config configs/phase1.yaml \
    --checkpoint experiments/phase1/best.pt --discover

# Generate UMAP visualizations
ctls-evaluate --config configs/phase1.yaml \
    --checkpoint experiments/phase1/best.pt --viz
```

## Using as a Library

After `pip install -e .`, all subpackages are importable directly:

```python
from models import FrozenBackbone, MetaEncoder
from losses import InfoLoss
from training import Phase1Trainer
from evaluation import CircuitAnalyzer, SpanCentricDiscovery
from evaluation import profile_reconstruction_r2, geometric_consistency
```

## Validation Experiments

1. **Profile Reconstruction Fidelity** — R² of per-layer MLP regressors (info-only ablation)
2. **Geometric Consistency** — Per-layer Spearman ρ + UMAP visualization
3. **Circuit Discovery & Span Validation** — Span-centric clustering + multi-circuit membership
4. **Temperature Sensitivity** — UMAP n_neighbors and HDBSCAN min_cluster_size sweep
5. **Transfer Across Backbone Depth** — ResNet18/34/50
6. **Dataset Generalization** — CIFAR-100, STL-10
