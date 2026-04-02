# Flow Circuits

`flow-circuits` is a mechanistic-interpretability research codebase for discovering candidate circuits in frozen ResNets using a flow-based spatial-token analysis model.

The supported workflow is:

1. train a flow-based meta-encoder on frozen backbone observations
2. evaluate predictive and geometric representation metrics
3. discover candidate circuits on the held-out discovery split
4. validate candidate circuits with held-out residual-patch interventions

`flow_circuits/` is the only supported code path in this repository.

## Current Project

The current iteration follows the theory and methodology in [documents/project_context.md](documents/project_context.md). The project treats a residual network as a flow system:

- token inputs come from post-skip state maps
- prediction targets come from pre-skip residual contributions
- external future descriptors are built from frozen backbone flow targets
- discovery is node-wise and image-set based, not span-centric
- causal validation uses residual-patch ablations, not input-space optimization

Older code paths were removed from the tracked repo so the public surface matches the current project.

## Supported Surface

- Package: `flow_circuits`
- CLIs: `flow-train`, `flow-evaluate`, `flow-discover`, `flow-intervene`
- Configs: `configs/flow/*.yaml`
- Backbones: `resnet18`, `resnet34`, `resnet50`
- Dataset protocol: CIFAR-10 with deterministic `40k fit / 5k val / 5k discovery / 10k test`

## Install

```bash
pip install -e .
pip install -e ".[dev]"
```

## Quickstart

Train a base model:

```bash
flow-train --config configs/flow/resnet18_base.yaml
```

Evaluate representation metrics and baseline comparison:

```bash
flow-evaluate --checkpoint experiments/flow/resnet18_base/final.pt
```

Discover candidate circuits:

```bash
flow-discover --checkpoint experiments/flow/resnet18_base/final.pt
```

Run held-out interventions:

```bash
flow-intervene \
  --checkpoint experiments/flow/resnet18_base/final.pt \
  --circuits experiments/flow/resnet18_base/candidate_circuits.json
```

## Artifact Contract

The repo now standardizes four artifact types:

- Training checkpoints: versioned `.pt` files containing config, phase metadata, model state, optimizer state, scheduler state, and validation summaries.
- Evaluation summary: JSON from `flow-evaluate` containing representation metrics and baseline comparison.
- Candidate-circuit artifact: JSON from `flow-discover` containing discovery metadata, retained node clusters, candidate circuits, centroids, thresholds, and stability stats.
- Intervention summary: JSON and CSV from `flow-intervene` containing per-circuit member/control effects and corrected significance values.

## Notebook Workflow

The first-class notebook suite is:

- [notebooks/nb01_training_and_representation_metrics.ipynb](notebooks/nb01_training_and_representation_metrics.ipynb)
- [notebooks/nb02_candidate_circuit_discovery_and_stability.ipynb](notebooks/nb02_candidate_circuit_discovery_and_stability.ipynb)
- [notebooks/nb03_interventions_and_qualitative_analysis.ipynb](notebooks/nb03_interventions_and_qualitative_analysis.ipynb)

Each notebook is Colab-ready: the setup cell clones or updates this GitHub repo under `/content/model_interpretability`, installs the package, mounts Google Drive, and stores checkpoints plus derived artifacts under `MyDrive/flow_circuits/`.

Use [documents/experiment_guide.md](documents/experiment_guide.md) for the operational workflow that connects configs, CLIs, notebooks, and saved artifacts.

## Repository Layout

```text
flow_circuits/
  backbones/
  tokenization/
  encoders/
  objectives/
  training/
  discovery/
  evaluation/
  interventions/
  data/
  cli/
configs/flow/
documents/
notebooks/
tests/
```

## Notes

- `documents/project_context.md` is the theory/spec document.
- `documents/experiment_guide.md` is the operational experiment guide.
- Notebook cells are written against the current `flow_circuits` surface and the `flow-*` CLI workflow only.
