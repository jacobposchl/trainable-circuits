# Repository Structure - flow-circuits

This repository is organized around the `flow_circuits/` package and the current flow-based candidate-circuit workflow.

---

## Package Layout

### `flow_circuits/backbones/`

| File | Purpose |
|---|---|
| `resnet.py` | `FrozenResNetObserver`: frozen ResNet18/34/50 with explicit post-skip state hooks and pre-skip residual hooks. Returns raw observation maps plus architecture metadata. |

### `flow_circuits/tokenization/`

| File | Purpose |
|---|---|
| `tokenizer.py` | `FlowTokenizer`: fixed-grid pooling, log-magnitude feature construction, learned token embeddings, fixed random flow projections, and fixed future descriptor projections. |

### `flow_circuits/encoders/`

| File | Purpose |
|---|---|
| `spatiotemporal_transformer.py` | `SpatiotemporalEncoder`: token-level transformer with causal depth masking, depth-only RoPE, and L2-normalized outputs. |

### `flow_circuits/objectives/`

| File | Purpose |
|---|---|
| `losses.py` | `FlowObjective`: per-layer next-step decoders, same-layer reconstruction decoders, and optional external trajectory alignment loss. |

### `flow_circuits/training/`

| File | Purpose |
|---|---|
| `baselines.py` | Fits and evaluates node-wise mean, local-MLP, and flow-MLP baselines used for P1 comparison and Phase C gating. |
| `trainer.py` | `FlowCircuitTrainer`, checkpoint loading, staged training, validation metric collection, and artifact-oriented output assembly. |

### `flow_circuits/discovery/`

| File | Purpose |
|---|---|
| `candidate_discovery.py` | `CandidateCircuitDiscoverer`: node-wise HDBSCAN clustering on `q_{l,i}`, bootstrap stability filtering, cluster-family merge, connectivity pruning, centroid/threshold persistence, multi-seed stability summaries, and node-shuffle nulls. |

### `flow_circuits/evaluation/`

| File | Purpose |
|---|---|
| `metrics.py` | Confirmatory/descriptive representation metrics, including one-step prediction summaries, same-split baseline comparisons, bootstrap confidence intervals, and external trajectory alignment against raw pooled-state and raw flow baselines. |

### `flow_circuits/interventions/`

| File | Purpose |
|---|---|
| `residual_ablation.py` | `ResidualPatchAblator`, held-out circuit assignment, matched control selection, layer-matched random controls, residual-patch ablation, and intervention summary generation with confidence intervals. |

### `flow_circuits/data/`

| File | Purpose |
|---|---|
| `cifar10.py` | Deterministic CIFAR-10 split builder: `40k fit / 5k val / 5k discovery / 10k test`. Returns indexed loaders so artifacts can persist stable dataset ids. |

### `flow_circuits/cli/`

| File | Purpose |
|---|---|
| `train.py` | CLI entry point for staged training: `flow-train`. |
| `evaluate.py` | CLI entry point for representation metrics and baseline comparison: `flow-evaluate`. |
| `discover.py` | CLI entry point for candidate-circuit discovery: `flow-discover`. |
| `intervene.py` | CLI entry point for held-out residual-patch interventions: `flow-intervene`. |

---

## Configs

Canonical configs live under `configs/flow/`:

- `resnet18_base.yaml`
- `resnet18_aligned.yaml`
- `resnet34_base.yaml`
- `resnet50_base.yaml`

These follow the repo-standard schema:

```text
experiment
data
backbone
tokenization
encoder
objectives
training
discovery
interventions
logging
```

---

## Notebooks

The supported notebook suite is:

- `nb01_training_and_representation_metrics.ipynb`
- `nb02_candidate_circuit_discovery_and_stability.ipynb`
- `nb03_interventions_and_qualitative_analysis.ipynb`

Each notebook uses the current `flow_circuits` package and/or the `flow-*` CLIs only.

---

## Documents

- `documents/project_context.md`: theoretical and methodological spec
- `documents/experiment_guide.md`: operational workflow for configs, artifacts, notebooks, and CLIs
- `documents/artifact_contracts.md`: stable saved-output schemas and path conventions
- `documents/dev_workflows.md`: development runbook and validation habits
- `documents/decision_log.md`: non-obvious project decisions that should remain explicit
- `AGENTS.md`: shared top-level guide for coding agents

---

## Tests

The test suite covers:

- observer/tokenizer/encoder behavior
- objective numerics
- staged training behavior
- discovery artifact generation
- residual-patch interventions
- CLI smoke tests
- end-to-end integration
- repo cleanliness and notebook static validation
