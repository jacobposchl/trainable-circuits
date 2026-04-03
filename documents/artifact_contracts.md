# Artifact Contracts

This document defines the stable artifact types used by the current `flow-circuits` workflow. If an artifact shape changes, update this file, the relevant code, and the tests together.

## 1. Training Checkpoints

Producer:

- `flow-train`
- `FlowCircuitTrainer._save_checkpoint`

Typical filenames:

- `phase_b.pt`
- `phase_c.pt`
- `final.pt`

Current required top-level fields:

- `version`
- `phase`
- `config`
- `observer_state`
- `tokenizer_state`
- `encoder_state`
- `objective_state`
- `optimizer_state`
- `scheduler_state`
- `validation`
- `summary`

Meaning:

- `version`: checkpoint schema version
- `phase`: checkpoint model phase, usually `phase_b` or `phase_c`
- `config`: full experiment config used to build components
- `*_state`: serialized PyTorch state dicts
- `validation`: final validation metrics snapshot
- `summary`: training summary including phase selection outcomes

Compatibility rule:

- Treat checkpoints as versioned artifacts.
- Do not change field names casually.
- If fields are added or removed, increment or reinterpret `version` explicitly and update loading logic.

## 2. Evaluation Summary JSON

Producer:

- `flow-evaluate`

Typical filename:

- `test_evaluation.json`
- `val_evaluation.json`
- notebook-local `evaluation.json`

Current required top-level fields:

- `split`
- `n_images`
- `representation_metrics`
- `baseline_comparison`
- `confirmatory_checks`
- `null_checks`

`representation_metrics` currently includes:

- `n_images`
- `prediction_cosine_mean`
- `prediction_cosine_sem`
- `reconstruction_cosine_mean`
- `reconstruction_cosine_sem`
- `trajectory_alignment_mean`
- `trajectory_alignment_std`
- `local_trajectory_alignment_mean`
- `flow_trajectory_alignment_mean`

`baseline_comparison` currently includes:

- `mean_baseline`
- `local_baseline`
- `flow_baseline`
- `best_baseline`
- `best_baseline_name`

`confirmatory_checks` currently includes:

- `p1_prediction_vs_best_baseline`
- `p2_alignment_vs_best_baseline`

Each confirmatory check currently includes:

- `model_value`
- `baseline_value`
- `baseline_name`
- `improvement`
- `ci_lower`
- `ci_upper`
- `passes`

`null_checks` currently includes:

- `future_shuffle_prediction`
- `depth_order_alignment`

## 3. Candidate-Circuit Artifact JSON

Producer:

- `flow-discover`
- `CandidateCircuitDiscoverer.save`

Typical filename:

- `candidate_circuits.json`

Current required top-level fields:

- `metadata`
- `node_clusters`
- `circuits`
- `seed_runs`
- `stability_summary`
- `null_checks`

`metadata` currently includes:

- `n_images`
- `n_layers`
- `n_cells`
- `grid_size`
- `random_seed`
- `discovery_seeds`

Each `node_clusters` item currently includes:

- `node`
- `image_set`
- `row_indices`
- `size`
- `stability`

Each `circuits` item currently includes:

- `id`
- `image_set`
- `representative_node`
- `active_nodes`
- `engagement_profile`
- `centroids`
- `thresholds`
- `stability`
- `purity`

Each `seed_runs` item currently includes:

- `seed`
- `node_clusters`
- `circuits`

`stability_summary` currently includes:

- `n_seed_runs`
- `reference_seed`
- `per_circuit`

When cross-seed stability is disabled by config, `stability_summary` still exists and additionally includes:

- `skipped`
- `reason`

Each `stability_summary.per_circuit` item currently includes:

- `circuit_id`
- `n_matches`
- `mean_image_jaccard`
- `mean_active_node_f1`
- `mean_null_image_jaccard`
- `mean_null_active_node_f1`
- `image_jaccard_improvement_ci`
- `active_node_f1_improvement_ci`
- `stable`

`null_checks` currently includes:

- `node_shuffle`

When the node-shuffle null is disabled by config, `null_checks.node_shuffle` still exists and additionally includes:

- `skipped`
- `reason`

Contract notes:

- `image_set` uses stable dataset indices, not row positions from a temporary batch.
- `centroids` and `thresholds` are keyed by `\"layer:cell\"`.
- `active_nodes` and `representative_node` are backbone-aligned node references.

## 4. Intervention Summary JSON

Producer:

- `flow-intervene`
- `run_circuit_interventions`

Typical filename:

- `intervention_summary.json`

Current JSON shape:

- list of per-circuit result objects

Each result currently includes:

- `circuit_id`
- `n_members`
- `n_controls`
- `mean_member_delta_margin`
- `mean_member_delta_true`
- `mean_nonmember_delta_margin`
- `mean_nonmember_delta_true`
- `mean_random_node_delta_margin`
- `mean_random_cell_delta_margin`
- `p_member_vs_nonmember`
- `p_member_vs_random_node`
- `p_member_vs_random_cell`
- `corrected_p_member_vs_nonmember`
- `corrected_p_member_vs_random_node`
- `corrected_p_member_vs_random_cell`
- `ci_member_vs_nonmember`
- `ci_member_vs_random_node`
- `ci_member_vs_random_cell`
- `validated`

## 5. Intervention Summary CSV

Producer:

- `flow-intervene`

Typical filename:

- `intervention_summary.csv`

Contract:

- row-wise mirror of the intervention JSON
- columns should remain aligned with JSON field names unless a documented migration is made

## 6. Path Conventions

Typical experiment-local output layout:

```text
experiments/flow/<experiment_name>/
  phase_b.pt
  phase_c.pt
  final.pt
  candidate_circuits.json
  test_evaluation.json
  intervention_summary.json
  intervention_summary.csv
```

For Colab notebooks, persisted artifacts are typically stored under:

```text
MyDrive/flow_circuits/
  data/
  experiments/
  notebook_runs/
```

## 7. Change Checklist

If you change an artifact contract:

1. update code
2. update this document
3. update tests
4. update notebook assumptions if relevant
5. update the experiment guide if user workflow changes
