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
- `nb02_candidate_circuit_discovery_and_stability.ipynb`
- `nb03_interventions_and_qualitative_analysis.ipynb`

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
- `final.pt`

### 2. Evaluation Summary

`flow-evaluate` writes a JSON summary containing:

- representation metrics
- baseline comparison

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
- stability statistics

Use:

```bash
flow-discover --checkpoint experiments/flow/resnet18_base/final.pt
```

This stage is the primary input to Notebook 2.

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

This stage supports the held-out causal specificity analysis and is the primary input to Notebook 3.

## Notebook Roles

### Notebook 1: Training and Representation Metrics

Use this notebook to:

- choose a Base or Aligned config
- train a quick or full run
- inspect evaluation summaries
- compare the model against baselines
- inspect a compact qualitative view of token-level outputs

### Notebook 2: Candidate Circuit Discovery and Stability

Use this notebook to:

- run or load candidate-circuit discovery
- inspect active nodes, engagement profiles, centroids, and thresholds
- summarize descriptive analyses such as active-node coverage and post hoc purity
- inspect stability across already-generated discovery artifacts when available

### Notebook 3: Interventions and Qualitative Analysis

Use this notebook to:

- load a checkpoint and candidate-circuit artifact
- run or load held-out interventions
- compare member vs control outcomes
- inspect corrected significance values
- visualize circuit footprints and intervention summaries

## Interpreting Outputs

### Confirmatory Analyses

Treat these as the main checks:

- one-step prediction against non-contextual baselines
- latent geometry alignment to external future similarity
- candidate-circuit stability
- held-out causal specificity

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
3. Run discovery on that checkpoint
4. Run interventions on the discovered candidate circuits
5. Repeat with `resnet18_aligned` only if the Base model is already healthy

## Operational Notes

- The repo only supports the `flow_circuits` package and `flow-*` CLIs.
- The notebooks are analysis surfaces, not alternative implementations.
- If you change configs for a quick notebook run, keep checkpoint and artifact outputs inside a notebook-specific output directory so they do not overwrite your main experiment artifacts.
