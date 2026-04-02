# Development Workflows

This document is the operator-facing runbook for working on the current `flow-circuits` repo.

## Quick Validation

Run this after substantive code or doc changes:

```bash
python -m pytest -q
```

For a legacy-reference sweep, rely on the repo hygiene test and a targeted `rg` query for the specific file or namespace you are changing, rather than copying the forbidden-name list into docs.

## Core CLI Workflows

### Train

```bash
flow-train --config configs/flow/resnet18_base.yaml
```

### Evaluate

```bash
flow-evaluate --checkpoint experiments/flow/resnet18_base/final.pt
```

### Discover

```bash
flow-discover --checkpoint experiments/flow/resnet18_base/final.pt
```

### Intervene

```bash
flow-intervene \
  --checkpoint experiments/flow/resnet18_base/final.pt \
  --circuits experiments/flow/resnet18_base/candidate_circuits.json
```

## Typical Research Loop

1. Start with `resnet18_base.yaml`
2. Train to `final.pt`
3. Evaluate prediction/alignment against baselines
4. Run candidate-circuit discovery
5. Run held-out interventions
6. Only then consider `resnet18_aligned.yaml`

## Quick-Mode Development

For faster iteration:

- reduce `phase_epochs`
- reduce `validation_images`
- reduce `baseline_fit_images`
- reduce `baseline_eval_images`
- reduce `alignment_max_pairs`
- reduce `discovery.max_images`
- reduce `interventions.max_images`

Keep these quick changes in notebook-local generated configs or temporary working configs rather than rewriting canonical configs unless the defaults should really change.

## Where to Put Changes

- model/training/discovery/intervention logic: `flow_circuits/`
- canonical experiment settings: `configs/flow/`
- theory/spec changes: `documents/project_context.md`
- workflow changes: `documents/experiment_guide.md`
- artifact schema changes: `documents/artifact_contracts.md`
- architecture map changes: `documents/repo_structure.md`

## Notebook Workflow

The notebooks are Colab-first and should remain thin:

- bootstrap the repo from GitHub
- mount Drive
- reuse saved checkpoints and derived artifacts
- call `flow_circuits` library code or `flow-*` CLIs
- avoid embedding core implementation logic

If a notebook needs custom analysis code that might be reused, move it into the package first and import it back into the notebook.

## Multi-Agent Workflow

Recommended shared workflow for Codex and Claude:

1. Read `AGENTS.md`
2. Read the relevant section of `project_context.md`
3. Check `documents/repo_structure.md`
4. Implement in the appropriate package boundary
5. Update the matching doc if contracts or workflows changed
6. Run tests

## When to Update Which Doc

- update `project_context.md` when the scientific framing or methodological spec changes
- update `experiment_guide.md` when the user-facing run sequence changes
- update `artifact_contracts.md` when saved outputs change shape or meaning
- update `decision_log.md` when a non-obvious choice is made and should not be “optimized away” later
