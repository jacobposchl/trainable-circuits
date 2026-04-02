# Flow Configs

This directory contains the canonical experiment configs for the current `flow-circuits` workflow.

## Available Configs

- `resnet18_base.yaml`
- `resnet18_aligned.yaml`
- `resnet34_base.yaml`
- `resnet50_base.yaml`

## Recommended Starting Point

Start with:

```text
configs/flow/resnet18_base.yaml
```

Use `resnet18_aligned.yaml` only after the base workflow is healthy.

## Schema

Each config follows this top-level structure:

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

## Common Safe Edits

These fields are the most common to change during development:

- `experiment.name`
- `data.batch_size`
- `data.num_workers`
- `training.phase_epochs`
- `training.validation_images`
- `training.baseline_fit_images`
- `training.baseline_eval_images`
- `training.alignment_max_pairs`
- `discovery.max_images`
- `discovery.bootstrap_iterations`
- `interventions.max_images`
- `logging.checkpoint_dir`

## Guidance

- prefer notebook-local derived configs for quick-mode experimentation
- keep canonical configs stable and readable
- if you add a new canonical config, update this file and the experiment guide
