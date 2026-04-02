# Notebooks

The notebooks in this repo are analysis and orchestration surfaces, not the primary implementation layer.

## Current Suite

- `nb01_training_and_representation_metrics.ipynb`
- `nb02_candidate_circuit_discovery_and_stability.ipynb`
- `nb03_interventions_and_qualitative_analysis.ipynb`

## Intended Role

Each notebook should:

- bootstrap the repo in Google Colab
- mount Google Drive
- reuse saved checkpoints and derived artifacts
- call `flow_circuits` package APIs or `flow-*` CLIs
- visualize or summarize results

Each notebook should not:

- contain unique core implementation logic
- redefine model components already present in `flow_circuits/`
- become the only place where a workflow is runnable

## Persistence Model

The notebooks are designed to persist artifacts under:

```text
MyDrive/flow_circuits/
```

This keeps checkpoints and outputs available across Colab sessions.

## When to Refactor Notebook Code into the Package

Move notebook code into `flow_circuits/` when it:

- computes reusable metrics
- implements reusable artifact loading or transformation logic
- becomes more than a small plotting/helper snippet
- is needed by more than one notebook
