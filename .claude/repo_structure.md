# Repository Structure — model_interpretability

Mechanistic interpretability project that discovers **circuits** (recurring computational pathways) in frozen neural networks using a meta-encoder trained with self-supervised learning. No class labels are used — the training signal comes entirely from flow co-activation profiles.

Installable as a Python package via `pip install -e .` (see `pyproject.toml`).

---

## Architecture & Data Flow

```
Input Image [B, 3, 32, 32]
    ↓
[FrozenBackbone (ResNet18)]
    ├─ trajectory: post-ReLU block outputs → h_1...h_L  [B, D_flow] × L
    └─ flow_targets: bn2/bn3 pre-skip → f_1...f_L  [B, D_flow] × L
         (L2-normalized, AdaptiveMaxPool → Flatten → Linear)
    ↓
[MetaEncoder]
    Per-layer projectors → RoPE Transformer (2 layers) → z_1...z_L  [B, d] × L
    ↓
[InfoLoss]
    Per-pair z_l^a * z_l^b → MLP_l → predicted f_l^a ⊙ f_l^b
    Loss: normalized MSE (SS_res / SS_tot) ≈ (1 - R²), ~1.0 at init → 0 at perfect fit
```

Training objective: `L_total = info_loss_weight * L_info`

---

## File Map

### `models/`

| File | Purpose |
|------|---------|
| `backbone.py` | `FrozenBackbone`: frozen ResNet18/34/50 or ViT with dual hooks. Populates `self._trajectory` (block outputs) and `self._flow_targets` (bn2/bn3 pre-skip, compressed). Exposes `layer_dims`. |
| `meta_encoder.py` | `MetaEncoder`: per-layer projectors + RoPE Transformer → L2-normalized z-vectors. Also `RotaryPositionEmbedding`, `RoPEMultiHeadAttention`, `RoPETransformerLayer`, `ProfileRegressor` (MLP used inside InfoLoss). |
| `__init__.py` | Exports: `FrozenBackbone`, `MetaEncoder`, `ProfileRegressor` |

### `losses/`

| File | Purpose |
|------|---------|
| `info_loss.py` | `InfoLoss`: holds L per-layer `ProfileRegressor` regressors (accessible via `self.regressors[l]`). Computes normalized MSE over flow co-activation targets. |
| `__init__.py` | Exports: `InfoLoss` |

### `training/`

| File | Purpose |
|------|---------|
| `unified_trainer.py` | `Phase1Trainer`: full training loop. Builds backbone, meta_encoder, info_loss. Saves checkpoints with keys: `epoch`, `val_metrics`, `meta_encoder_state`, `info_loss_state`, `optimizer_state`, `config`. |
| `__init__.py` | Exports: `Phase1Trainer` |

### `data/`

| File | Purpose |
|------|---------|
| `cifar.py` | `get_train_transform()`, `get_val_transform()`, `get_standard_loaders()` → CIFAR-10 train/val DataLoaders. |
| `datasets.py` | `get_loaders(dataset, ...)` → multi-dataset loaders for cifar10, cifar100, stl10 (all resized to 32×32). `n_classes(dataset)` → class count. |
| `__init__.py` | Exports: `get_standard_loaders`, `get_train_transform`, `get_val_transform`, `get_loaders`, `n_classes` |

### `evaluation/`

| File | Purpose |
|------|---------|
| `metrics.py` | C1–C5 success criteria: `profile_reconstruction_r2`, `rich_profile_reconstruction_r2`, `geometric_consistency`, `within_span_elevation`, `circuit_diversity`, `class_purity_distribution`. |
| `discovery.py` | `SpanCentricDiscovery`: enumerates layer spans, embeds with UMAP (cosine, 15 components), clusters with HDBSCAN, filters canonical circuits (1%–40% of images). Also `compute_span_similarities`, `compute_class_purity`, `multi_circuit_membership`, `compute_prototypes`. |
| `circuit_analysis.py` | `CircuitAnalyzer`: batch-collects trajectories/flows/z-vectors/labels/images. `compute_pair_profiles`, `compute_pair_rich_profiles`, `compute_all_profiles`, `compute_class_purity`. `load_checkpoint(config, path, device)` builds all models and loads weights. |
| `interventions.py` | Causal evaluation helpers: grad-enabled CTLS forward pass, frozen-feature linear probes, circuit prototype construction, control generation, circuit selection, and PGD-style input-space interventions with summary metrics. |
| `circuit_viz.py` | `plot_per_layer_umap`, `plot_circuit_members`, `plot_span_coverage`, `plot_span_heatmap`, `plot_multi_circuit_histogram`. |
| `__init__.py` | Exports: `CircuitAnalyzer`, `load_checkpoint`, `SpanCentricDiscovery`, plotting helpers, C1?C5 metrics, and the notebook-facing causal intervention utilities. |

### `scripts/`

| File | Purpose |
|------|---------|
| `train.py` | CLI: loads YAML config → `Phase1Trainer` → `trainer.train()`. Entry point: `ctls-train`. |
| `evaluate.py` | CLI: loads checkpoint, collects representations, runs C1–C2; C3–C5 with `--discover`; UMAP with `--viz`. Entry point: `ctls-evaluate`. |
| `__init__.py` | Empty — makes `scripts/` a package for `pyproject.toml` entry points. |

### `configs/`

| File | Purpose |
|------|---------|
| `phase1.yaml` | Legacy main config (resnet18). Superseded by `configs/models/`. |
| `ablations/info_only.yaml` | Ablation with different checkpoint dir. |
| `models/resnet18.yaml` | ResNet18 model config (primary). |
| `models/resnet34.yaml` | ResNet34 model config. |
| `models/resnet50.yaml` | ResNet50 model config. |

**Notebook pattern**: load a model config with `MODEL_CONFIG = CONFIG_DIR + '/models/resnet18.yaml'`, then override `data_dir` and `checkpoint_dir` at runtime.

### `notebooks/`

| File | Purpose |
|------|---------|
| `nb01_training_and_validation.ipynb` | Trains the meta-encoder and evaluates C1 (R²) and C2 (Spearman ρ). Single config cell selects the model. |
| `nb02_analysis.ipynb` | Thorough analysis: circuit discovery (C3–C5), architecture transfer, dataset generalization, discovery parameter sensitivity. |

### `tests/`

| File | Purpose |
|------|---------|
| `test_meta_encoder.py` | Tests `RotaryPositionEmbedding`, `RoPEMultiHeadAttention`, `MetaEncoder`, `ProfileRegressor`. |
| `test_losses.py` | Tests `InfoLoss`. |
| `test_discovery.py` | Tests `SpanCentricDiscovery`. |
| `test_interventions.py` | Tests grad-enabled forward passes, linear-probe fitting, circuit prototype/control utilities, and intervention optimization behavior. |

### `documents/`

| File | Purpose |
|------|---------|
| `project_context.md` | Full technical spec: architecture, training objective, design decisions, known limitations. |
| `preliminary_results.md` | Results on older checkpoint. C3–C5 analysis with example circuits. |

---

## Packaging

`pyproject.toml` makes the project installable:

```bash
pip install -e .           # core deps
pip install -e ".[vit]"    # + timm for ViT architectures
pip install -e ".[dev]"    # + pytest
```

After installation, all subpackages are importable without `sys.path` hacks. CLI commands `ctls-train` and `ctls-evaluate` are available globally.

---

## Checkpoint Format

Saved by `Phase1Trainer._save_checkpoint`:

```python
{
    "epoch":              int,
    "val_metrics":        {"r2": float, "mean_rho": float, "per_layer_rho": [...]},
    "meta_encoder_state": state_dict,
    "info_loss_state":    state_dict,   # contains all L per-layer regressors
    "optimizer_state":    state_dict,
    "config":             dict,
}
```

Load with `from evaluation import load_checkpoint; bb, me, il = load_checkpoint(config, path, device)`.

---

## Key Design Decisions

- **Geometry loss removed**: was dead from epoch 1 (near-uniform soft targets in 256-d flow space due to concentration of measure). Only `InfoLoss` is used.
- **Flow targets**: bn2/bn3 pre-skip outputs isolate pure block contribution from accumulated skip-connection history.
- **Image-centric discovery**: circuits are sets of images sharing computation over a contiguous span, not pairwise relationships.
- **Fixed compression**: seed=42 random projection matrices in backbone, non-trainable, for reproducibility.
