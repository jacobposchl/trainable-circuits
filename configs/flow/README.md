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
- `discovery.batch_size`
- `discovery.n_jobs`
- `discovery.compute_seed_stability`
- `discovery.compute_node_shuffle_null`
- `interventions.max_images`
- `interventions.batch_size`
- `interventions.n_jobs`
- `logging.checkpoint_dir`

## Guidance

- canonical configs require `backbone.weights_path` to be set to a supervised CIFAR-10 checkpoint before use
- prefer notebook-local derived configs for quick-mode experimentation
- keep canonical configs stable and readable
- if you add a new canonical config, update this file and the experiment guide
