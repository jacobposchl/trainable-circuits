# Experiment Guide

This guide explains the current five-notebook workflow for training, validating, discovering, testing, and interpreting `z`-space motif flows in `flow-circuits`.

Related references:

- `AGENTS.md` for the shared coding-agent contract
- `documents/artifact_contracts.md` for saved-output schemas
- `documents/dev_workflows.md` for development-time validation and editing habits
- `documents/decision_log.md` for intentional project choices

## Workflow Overview

The supported workflow is still grounded in:

1. `flow-train`
2. `flow-evaluate`
3. `flow-discover`
4. `flow-intervene`

But the canonical notebook story is now:

- `nb01_backbone_and_z_training.ipynb`
- `nb02_q_validation.ipynb`
- `nb03_z_motif_discovery_and_analysis.ipynb`
- `nb04_motif_utility_and_robustness.ipynb`
- `nb05_motif_semantic_interpretation.ipynb`

Each notebook is Google Colab-first. The setup cell:

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

Labels are not used during representation learning or clean motif discovery. They are used only for descriptive analysis, motif utility ranking, hard-example selection, and corruption reporting.

## Training Workflow

The notebook-first workflow now trains two `z` branches after the supervised backbone is available:

- a canonical **frozen-backbone** branch
- an experimental **joint-backbone** branch

Notebook 1 is responsible for:

- training or reusing the supervised backbone checkpoint
- training the frozen branch through Phase A and Phase B
- warm-starting the joint branch from the frozen Phase B checkpoint
- running one Phase C continuation per `lambda_traj` candidate and saving only milestone checkpoints for later selection

This is intentionally compute efficient:

- one Phase C continuation per lambda candidate
- milestone checkpoints only
- no retraining to compare epoch counts
- no joint branch training from scratch

## Artifact Flow

### 1. Notebook 1 Training Artifacts

Notebook 1 writes notebook-managed checkpoint artifacts such as:

- `backbone_supervised.pt`
- `phase_b_frozen.pt`
- `phase_c_frozen_lambda_<x>_epoch_<y>.pt`
- `phase_c_joint_lambda_<x>_epoch_<y>.pt`
- `training_candidates.json`

Notebook 1 does not pick the final downstream checkpoints. Notebook 2 performs that selection.

### 2. Notebook 2 Selection Artifact

Notebook 2 writes:

- a ranking summary over frozen and joint checkpoint candidates
- `selected_checkpoints.json`

This is the only notebook that uses `q` directly. All later notebooks should be `z`-only.

### 3. Notebook 3 Motif Artifacts

Notebook 3 writes clean motif artifacts for the chosen frozen and joint checkpoints:

- `frozen_clean_motifs.json`
- `joint_clean_motifs.json`
- motif analysis summaries and rendering metadata

These artifacts are intended to be reused in notebook 4 instead of rediscovering clean motifs.

### 4. Notebook 4 Utility Artifacts

Notebook 4 writes:

- clean hard-example motif utility summaries
- corruption utility summaries
- selected top-motif subset summaries
- case-study-ready metadata

By default it transfers clean motifs to corrupted inputs before considering any corruption-specific rediscovery.

### 5. Notebook 5 Interpretation Artifacts

Notebook 5 writes:

- branch-local semantic motif reports
- borderline-member / near-miss summaries
- spatial overlay and crop metadata

These artifacts are meant for qualitative inspection rather than new statistical validation.

## Notebook Roles

### Notebook 1: Backbone and `z` Training

Use this notebook to:

- train or reload the supervised backbone
- train the frozen `z` branch
- warm-start and train the joint `z` branch
- save milestone Phase C checkpoints without retraining per epoch comparison

### Notebook 2: `q` Validation

Use this notebook to:

- load the saved frozen and joint milestone checkpoints
- score them against `q`-alignment / future-structure metrics
- choose one downstream checkpoint per branch
- keep all later notebooks `z`-only

### Notebook 3: `z` Motif Discovery and Analysis

Use this notebook to:

- discover motif families directly in `z`
- cache clean motif artifacts once for the frozen and joint branches
- analyze multi-layer support, persistence, purity, and topology
- prepare transfer-ready motif artifacts for notebook 4

### Notebook 4: Motif Utility and Robustness

Use this notebook to:

- test motif-based hybrids instead of raw-node hybrids
- compare backbone, full-motif, and top-motif correction for the frozen and joint branches
- evaluate clean hard-example utility
- evaluate corruption robustness with transferred clean motifs

### Notebook 5: Motif Semantic Interpretation

Use this notebook to:

- inspect the cached clean motifs from notebook 3 without rediscovering them
- focus on the joint branch by default, with optional lightweight frozen reference context
- render image-first motif cards with exemplars, borderline members, and near misses
- inspect active-cell overlays and approximate crops
- produce a cautious plain-English interpretation of what the motifs seem to encode

## Interpreting Outputs

Primary success signals now come from three linked checks:

1. `q` validation:
   - does `z` preserve future-flow structure well enough to justify downstream use?
2. motif quality:
   - are the discovered `z` motifs stable, multi-layer, and interpretable?
3. motif utility:
   - do those same motifs improve decisions on hard or corrupted inputs?
4. motif meaning:
   - do the top motifs look semantically coherent enough to inspect, describe, and reuse?

The intended optimization target is not raw clean CIFAR-10 accuracy alone. The stronger target is:

- future-structured `z`
- better multi-layer motif families
- useful motif-based correction under stress

## Recommended Run Order

1. Run `nb01_backbone_and_z_training.ipynb`
2. Run `nb02_q_validation.ipynb`
3. Run `nb03_z_motif_discovery_and_analysis.ipynb`
4. Run `nb04_motif_utility_and_robustness.ipynb`
5. Run `nb05_motif_semantic_interpretation.ipynb`
6. Use the exhaustive `flow-discover` / `flow-intervene` CLI path only if the notebook suite indicates the chosen branch is promising

## Operational Notes

- The repo still supports the `flow_circuits` package and `flow-*` CLIs.
- The notebooks should remain thin orchestration surfaces over package APIs.
- If you change checkpoint shapes, notebook artifact layouts, or motif utility logic, update docs and tests together.
