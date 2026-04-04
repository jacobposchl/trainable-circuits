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
- `nb03_recurring_motif_core_validation.ipynb`
- `nb04_motif_extended_characterization.ipynb`
- `nb05_motif_visual_interpretability_and_probe_analysis.ipynb`
- `nb06_hard_pair_correction_from_z.ipynb`
- `nb07_phase_c_corruption_selective_correction.ipynb`

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

### Notebook 3: Recurring Motif Core Validation

Use this notebook to:

- discover recurring motif families directly in `z`
- compare motif galleries, persistence, predictiveness, and interventions for `phase_b.pt` vs `phase_c.pt`
- reuse notebook-local motif artifacts instead of re-running exhaustive circuit discovery
- answer the main “does alignment create better recurring multi-layer motifs?” question

### Notebook 4: Motif Extended Characterization

Use this notebook to:

- inspect motif co-occurrence structure within each checkpoint
- match Phase B motifs to Phase C motifs
- characterize motif topology as spatial, depth-like, or fragmented
- test whether motifs remain similar across overlapping rediscovery subsets

`nb04` requires cached motif-family artifacts from `nb03` and will error if they are missing.

### Notebook 5: Motif Visual Interpretability and Probe Analysis

Use this notebook to:

- inspect motif and node-cluster exemplar grids directly on CIFAR-10 images
- compare matched Phase B vs Phase C motifs visually
- review intervention case studies
- measure what class/error/confusion information is linearly decodable from `z`

### Notebook 6: Hard-Pair Correction from z

Use this notebook to:

- audit multiclass linear decodability from `z`
- benchmark full-`z` and top-node probes on the backbone's hardest validation-selected confusion pairs
- test whether pairwise `z` probes improve the frozen backbone when used as top-2 tie-breakers
- inspect corrected and harmed examples with top-node overlays

### Notebook 7: Phase-C Corruption Selective Correction

Use this notebook to:

- stress-test `phase_c.pt` on a deterministic CIFAR-10 corruption suite
- compare backbone, backbone + full-`z`, and backbone + top-node-subset selective correction in the same experiment cells
- inspect whether `z` becomes more useful as corruption severity increases
- sweep the size of the top-node subset to test whether useful signal is concentrated or distributed

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
5. Run `nb03_recurring_motif_core_validation.ipynb`
6. Run `nb04_motif_extended_characterization.ipynb` if you want the broader motif diagnostics
7. Run `nb05_motif_visual_interpretability_and_probe_analysis.ipynb` to inspect motifs and probe-readable semantics directly
8. Run `nb06_hard_pair_correction_from_z.ipynb` if you want a Phase-C-only selective-correction readout for hard examples, confidence quality, and actionable hard-pair support from `z`
9. Run `nb07_phase_c_corruption_selective_correction.ipynb` if you want to test whether the same Phase-C correction signal becomes more useful under corruption stress
10. Use the exhaustive `flow-discover` / `flow-intervene` CLI path only if the notebook suite indicates the aligned representation is promising

## Operational Notes

- The repo only supports the `flow_circuits` package and `flow-*` CLIs.
- The notebooks are analysis surfaces, not alternative implementations.
- If you change configs for a quick notebook run, keep checkpoint and artifact outputs inside a notebook-specific output directory so they do not overwrite your main experiment artifacts.
