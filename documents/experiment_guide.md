# Experiment Guide

This guide explains how to run the current flow-circuits workflow end to end and how the generated artifacts connect to the notebooks and evaluation criteria.

Related references:

- `AGENTS.md` for the shared coding-agent contract
- `documents/artifact_contracts.md` for saved-output schemas
- `documents/dev_workflows.md` for development-time validation and editing habits
- `documents/decision_log.md` for intentional project choices

## Workflow Overview

The supported workflow is:

1. `flow-train`
2. `flow-evaluate`
3. `flow-discover`
4. `flow-intervene`

These stages map onto the current notebook suite:

- `nb01_training_and_representation_metrics.ipynb`
- `nb02_efficient_representation_and_circuit_validation.ipynb`

Each notebook is designed for Google Colab. The setup cell:

- clones or updates `https://github.com/jacobposchl/model-interpretability.git`
- installs the repo in editable mode inside the Colab runtime
- mounts Google Drive
- stores data, checkpoints, and notebook outputs under `MyDrive/flow_circuits/`

## Dataset Protocol

All current configs assume CIFAR-10 with a deterministic split:

- `40k` fit images
- `5k` validation images
- `5k` discovery images
- `10k` test images

Labels are not used during representation learning or candidate-circuit discovery. They are used only for descriptive analysis, class-matched intervention controls, and optional secondary reporting.

## Training Modes

There are two supported training modes:

Before either mode is run, set `backbone.weights_path` in the chosen config to a supervised CIFAR-10 checkpoint for the frozen ResNet backbone. Canonical configs now fail loudly if that checkpoint is not provided.

### Base

- Phase A: `L_pred`
- Phase B: `L_pred + L_rec`
- Final checkpoint defaults to the accepted Phase B model

Use:

```bash
flow-train --config configs/flow/resnet18_base.yaml
```

### Aligned

- Phase A: `L_pred`
- Phase B: `L_pred + L_rec`
- Phase C: optional `L_traj`, enabled only after the Phase B baseline gate is passed
- The canonical aligned config now evaluates one Phase C lambda candidate for 20 epochs and always saves the resulting `phase_c.pt` checkpoint for downstream comparison

Phase C is retained only if external trajectory alignment improves and one-step prediction remains within the Phase B acceptance window.

Use:

```bash
flow-train --config configs/flow/resnet18_aligned.yaml
```

## Artifact Flow

### 1. Training Checkpoints

`flow-train` writes versioned `.pt` checkpoints containing:

- config
- accepted training phase
- observer/tokenizer/encoder/objective state
- optimizer and scheduler state
- validation summaries

Important outputs:

- `phase_b.pt`
- `phase_c.pt`
- `final.pt`

`phase_b.pt` is the predictive anchor checkpoint.
`phase_c.pt` is always kept as the trajectory-aligned exploratory checkpoint.
`final.pt` remains the accepted model checkpoint after the Phase C selection rule is applied.

If an aligned run is interrupted after Phase B has already been saved, you can resume from that checkpoint instead of retraining Phase A+B:

```bash
flow-train --config configs/flow/resnet18_aligned.yaml --resume experiments/flow/resnet18_aligned/phase_b.pt
```

### 2. Evaluation Summary

`flow-evaluate` writes a JSON summary containing:

- representation metrics
- baseline comparison
- confirmatory checks with bootstrap confidence intervals
- evaluation null checks

Use:

```bash
flow-evaluate --checkpoint experiments/flow/resnet18_base/final.pt
```

This stage supports the P1/P2-facing checks described in `project_context.md`.

### 3. Candidate-Circuit Artifact

`flow-discover` writes a JSON artifact containing:

- discovery metadata
- retained node clusters
- candidate circuits
- centroids
- thresholds
- multi-seed stability statistics
- discovery null-check summaries

Use:

```bash
flow-discover --checkpoint experiments/flow/resnet18_base/final.pt
```

This remains the advanced exhaustive discovery path. The unified Notebook 2 now uses a smaller pilot-discovery workflow implemented through package APIs instead of calling `flow-discover`.

### 4. Intervention Summary

`flow-intervene` writes:

- intervention JSON
- intervention CSV

Use:

```bash
flow-intervene \
  --checkpoint experiments/flow/resnet18_base/final.pt \
  --circuits experiments/flow/resnet18_base/candidate_circuits.json
```

This remains the advanced exhaustive intervention path. The unified Notebook 2 now uses a top-k pilot intervention workflow implemented through package APIs instead of calling `flow-intervene`.

## Notebook Roles

### Notebook 1: Training and Representation Metrics

Use this notebook to:

- choose a Base or Aligned config
- train a quick or full run
- inspect evaluation summaries
- compare the model against baselines
- inspect a compact qualitative view of token-level outputs

### Notebook 2: Efficient Representation and Circuit Validation

Use this notebook to:

- load `phase_b.pt` and `phase_c.pt`
- run fast side-by-side validation experiments without retraining
- compare neighbor agreement, activation decoding, pilot discovery, and top-k interventions
- reuse notebook-local cached experiment outputs across Colab sessions
- decide whether Phase C is promising enough to justify the exhaustive CLI workflow

## Interpreting Outputs

### Confirmatory Analyses

Treat these as the main checks:

- one-step prediction against non-contextual baselines
- latent geometry alignment to external future similarity
- candidate-circuit stability
- held-out causal specificity

The evaluation and discovery artifacts now include the confirmatory/statistical summaries needed to support these checks directly.

### Descriptive Analyses

Treat these as characterization, not gatekeeping:

- reconstruction fidelity
- multi-step prediction decay
- active-node coverage
- post hoc class purity
- transfer experiments

## Recommended Run Order

1. Train `resnet18_base`
2. Evaluate the resulting `final.pt`
3. Train `resnet18_aligned` and keep both `phase_b.pt` and `phase_c.pt`
4. Run `nb02_efficient_representation_and_circuit_validation.ipynb`
5. Use the exhaustive `flow-discover` / `flow-intervene` CLI path only if the efficient notebook indicates the aligned representation is promising

## Operational Notes

- The repo only supports the `flow_circuits` package and `flow-*` CLIs.
- The notebooks are analysis surfaces, not alternative implementations.
- If you change configs for a quick notebook run, keep checkpoint and artifact outputs inside a notebook-specific output directory so they do not overwrite your main experiment artifacts.
