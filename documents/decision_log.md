# Decision Log

This file records high-value project decisions that agents and collaborators should preserve unless there is a deliberate redesign.

## 2026-04 Current Flow-Circuits Iteration

### 1. Clean Break from Legacy CTLS

Decision:

- the repo supports only `flow_circuits` and `flow-*` commands

Why:

- reduces ambiguity
- keeps docs, code, configs, and notebooks aligned
- avoids agents reviving old assumptions from legacy paths

### 2. External Trajectory Descriptor for `L_traj`

Decision:

- trajectory alignment uses external future-flow descriptors, not live `z`-space self-targets

Why:

- avoids circular self-reinforcement
- makes alignment metrics interpretable
- preserves a clearer methodological story

### 3. Candidate-Circuit Discovery Is Node-Wise

Decision:

- discovery is based on node-wise clustering over `q_{l,i}` and circuit assembly by overlap/connectivity

Why:

- matches the current spatial-token formulation
- makes circuit membership and centroids artifact-friendly
- replaces older span-centric assumptions that no longer fit the method

### 4. Attention Is Visualization, Not Evidence

Decision:

- encoder attention summaries are not treated as confirmatory mechanistic evidence

Why:

- attention belongs to the analysis model, not the frozen backbone computation itself
- avoids overstating causal claims

### 5. Residual-Patch Interventions over Input-Space Optimization

Decision:

- interventions zero residual-branch regions at active nodes instead of perturbing input pixels

Why:

- better matches discovered object granularity
- gives a more interpretable causal test
- aligns intervention mechanism with the current backbone observation model

### 6. Notebooks Are Colab-First Thin Clients

Decision:

- notebooks clone the GitHub repo, mount Drive, and reuse saved artifacts

Why:

- supports lightweight iteration in Colab
- keeps outputs persistent across sessions
- keeps implementation in the package rather than notebook cells

### 7. One Shared Agent Contract

Decision:

- `AGENTS.md` is the shared top-level guide for coding agents

Why:

- prevents Codex and Claude from following diverging project rules
- keeps tool-specific files thin and referential rather than duplicative

### 8. Canonical Runs Require a Supervised Backbone Checkpoint

Decision:

- canonical CIFAR-10 configs must set `backbone.weights_path`
- the code fails loudly if a canonical run would otherwise fall back to an untrained classifier head

Why:

- keeps intervention logits scientifically meaningful
- prevents silent degradation from a pretrained-feature-extractor plus random-head setup

### 9. Confirmatory Outputs Are First-Class Artifacts

Decision:

- evaluation artifacts include same-split confirmatory checks with bootstrap confidence intervals
- discovery artifacts include multi-seed stability summaries and node-shuffle nulls
- intervention artifacts include matched-control confidence intervals and all control comparisons used for validation

Why:

- makes the confirmatory story inspectable from saved outputs rather than implicit notebook logic
- reduces the risk of point-estimate over-interpretation
