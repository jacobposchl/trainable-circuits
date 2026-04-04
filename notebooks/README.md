# Notebooks

The notebooks in this repo are analysis and orchestration surfaces, not the primary implementation layer.

## Current Suite

- `nb01_backbone_and_z_training.ipynb`
- `nb02_q_validation.ipynb`
- `nb03_z_motif_discovery_and_analysis.ipynb`
- `nb04_motif_utility_and_robustness.ipynb`

## Intended Role

Each notebook should:

- bootstrap the repo in Google Colab
- mount Google Drive
- reuse saved checkpoints and derived artifacts
- call `flow_circuits` package APIs or `flow-*` CLIs
- visualize or summarize results

Current notebook roles:

- `nb01`: supervised backbone training plus frozen/joint `z` branch training with milestone Phase C checkpoints
- `nb02`: the only `q` notebook; ranks frozen and joint checkpoints and selects the downstream pair
- `nb03`: `z`-only motif discovery and motif-family analysis on clean data
- `nb04`: motif-based prediction utility on clean hard examples and corrupted inputs

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
