# Notebooks

The notebooks in this repo are analysis and orchestration surfaces, not the primary implementation layer.

## Current Suite

- `nb01_training_and_representation_metrics.ipynb`
- `nb02_efficient_representation_and_circuit_validation.ipynb`
- `nb03_recurring_motif_core_validation.ipynb`
- `nb04_motif_extended_characterization.ipynb`
- `nb05_motif_visual_interpretability_and_probe_analysis.ipynb`
- `nb06_hard_pair_correction_from_z.ipynb`
- `nb07_phase_c_corruption_selective_correction.ipynb`

## Intended Role

Each notebook should:

- bootstrap the repo in Google Colab
- mount Google Drive
- reuse saved checkpoints and derived artifacts
- call `flow_circuits` package APIs or `flow-*` CLIs
- visualize or summarize results

Current notebook roles:

- `nb01`: training, evaluation, and representation-metric inspection
- `nb02`: efficient Phase B vs Phase C validation across neighbor agreement, activation probes, pilot discovery, and top-k interventions
- `nb03`: recurring motif discovery and the most decision-driving motif validation experiments
- `nb04`: extended motif characterization, Phase B vs Phase C motif matching, and motif transfer/topology analysis
- `nb05`: image-first motif interpretation, intervention case studies, and linear-probe semantics/error/confusion analysis
- `nb06`: Phase-C-only selective correction, confidence, and hard-example auditing from `z`
- `nb07`: Phase-C-only corruption stress testing with backbone, full-`z`, and top-node-subset selective correction

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
