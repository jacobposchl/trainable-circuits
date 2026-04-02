# AGENTS

This file is the shared top-level contract for coding agents working in this repository. It is intended to be the canonical project guide for both Codex and Claude Code.

## Purpose

`flow-circuits` is a research codebase for discovering and validating candidate circuits in frozen ResNets using a flow-based spatial-token analysis model.

The supported end-to-end workflow is:

1. `flow-train`
2. `flow-evaluate`
3. `flow-discover`
4. `flow-intervene`

## Canonical Sources of Truth

Read these before making substantial changes:

1. [`documents/project_context.md`](C:/Users/Jacob%20Poschl/Desktop/model_interpretability/documents/project_context.md)
2. [`documents/experiment_guide.md`](C:/Users/Jacob%20Poschl/Desktop/model_interpretability/documents/experiment_guide.md)
3. [`documents/repo_structure.md`](C:/Users/Jacob%20Poschl/Desktop/model_interpretability/documents/repo_structure.md)
4. [`documents/artifact_contracts.md`](C:/Users/Jacob%20Poschl/Desktop/model_interpretability/documents/artifact_contracts.md)
5. [`documents/decision_log.md`](C:/Users/Jacob%20Poschl/Desktop/model_interpretability/documents/decision_log.md)

`AGENTS.md` is the canonical shared agent guide. Tool-specific files such as `.claude/CLAUDE.md` and `CODEX.md` should point back here instead of redefining project truth.

## Package Boundaries

- `flow_circuits/backbones/`: frozen backbone observation
- `flow_circuits/tokenization/`: pooling, token construction, flow/future descriptors
- `flow_circuits/encoders/`: token-level transformer
- `flow_circuits/objectives/`: prediction, reconstruction, trajectory objectives
- `flow_circuits/training/`: staged training, checkpoints, baseline fitting, output collection
- `flow_circuits/discovery/`: node-wise candidate-circuit discovery
- `flow_circuits/evaluation/`: confirmatory and descriptive representation metrics
- `flow_circuits/interventions/`: held-out member assignment and residual-patch ablations
- `flow_circuits/data/`: CIFAR-10 split builder
- `flow_circuits/cli/`: `flow-*` entry points

## Edit Policy

- Put core logic in `flow_circuits/`, not in notebooks.
- Treat notebooks as thin orchestration and analysis surfaces.
- Do not silently change checkpoint schemas or artifact JSON/CSV shapes without updating docs and tests.
- Prefer adding or updating tests when behavior changes.
- Preserve the current clean-break architecture; do not reintroduce legacy CTLS paths or old CLI naming from the pre-flow-circuits iteration.

## Validation Policy

After non-trivial changes, run:

```bash
python -m pytest -q
```

If you change docs, configs, notebooks, or artifact formats, also sanity-check:

Use the existing repo hygiene test as the source of truth for forbidden legacy names rather than manually re-listing them in docs.

## Notebook Policy

- Notebooks are Google Colab-first.
- Setup cells may clone the GitHub repo, install the package, mount Google Drive, and reuse saved checkpoints/artifacts.
- Notebook code should call library APIs or `flow-*` CLIs rather than duplicating implementation logic.

## Artifact Policy

Current first-class artifacts are:

- training checkpoints
- evaluation summary JSON
- candidate-circuit artifact JSON
- intervention summary JSON/CSV

See [`documents/artifact_contracts.md`](C:/Users/Jacob%20Poschl/Desktop/model_interpretability/documents/artifact_contracts.md) for details.

## High-Signal Working Pattern

When making a change:

1. Read the relevant theory/workflow docs.
2. Inspect the target module boundary in `flow_circuits/`.
3. Implement the smallest coherent change in code, not notebooks.
4. Update docs if repo behavior or expectations changed.
5. Run tests.

## Supporting Docs

- [`documents/dev_workflows.md`](C:/Users/Jacob%20Poschl/Desktop/model_interpretability/documents/dev_workflows.md): operational development runbook
- [`configs/flow/README.md`](C:/Users/Jacob%20Poschl/Desktop/model_interpretability/configs/flow/README.md): config usage guide
- [`notebooks/README.md`](C:/Users/Jacob%20Poschl/Desktop/model_interpretability/notebooks/README.md): notebook role and usage guide
