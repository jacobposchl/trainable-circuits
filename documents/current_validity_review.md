# Current Validity Review

Date: 2026-04-02

Note:
This document is a point-in-time audit of the pre-fix repository state. Several items here were subsequently addressed in code and docs after this review was written, so treat it as a historical review rather than a live status page.

Scope:
This document records concrete methodological and implementation problems in the current repository, based on a review of the code against the intended scientific protocol in `documents/project_context.md`.

Status:
The codebase is internally consistent enough to run, and `python -m pytest -q` passes at the time of writing. The issues below are therefore mostly not "the code crashes" problems. They are problems of scientific validity, rigor, or implementation/spec mismatch that could make results easier to over-interpret than the current code justifies.

How to read this document:
- "Severity" is about scientific risk, not code style.
- "Problem" describes the current issue.
- "Why this matters" explains the scientific consequence.
- "Evidence" points to the relevant docs and implementation.
- "Recommended remediation" describes the smallest credible fix.

## Summary

The current repository is strongest as an exploratory research scaffold. It is not yet strong enough to support the full confirmatory story described in `documents/project_context.md`.

The highest-risk problems are:
- the default backbone setup does not match the documented "pretrained CIFAR-10 ResNet" assumption
- the main causal metric uses logits from that mismatched backbone head
- `flow-evaluate` mixes test-set model metrics with validation-set baselines
- baseline models are weaker than the written protocol
- the confirmatory statistics and replication procedures described in the project docs are not implemented

Until those are fixed, outputs should be treated as exploratory/descriptive rather than confirmatory/mechanistic.

## Problem 1: The Default Backbone Does Not Match the Documented Scientific Object

Severity: Critical

### Problem

The project context describes the backbone as a frozen pretrained CIFAR-10 ResNet18, but the checked-in configs and implementation do not instantiate that object by default.

Instead, the current code path:
- loads ImageNet torchvision weights when `pretrained: true`
- replaces the classifier head with a fresh 10-way linear layer
- freezes that model without training the new head

That means the backbone used by default is not a pretrained CIFAR-10 classifier. It is an ImageNet-pretrained feature extractor with a randomly initialized CIFAR-10 head.

### Why This Matters

This is not a small documentation mismatch. It changes the scientific target.

The project context relies on backbone logits for the intervention metric and for control matching. If the classifier head is random, then:
- predicted classes are not trustworthy
- top-1/top-2 logit margins are not meaningful as a measure of task performance
- member/control matching by predicted class is poorly grounded
- "validated circuits" based on margin changes become scientifically weak or potentially misleading

In other words, the current default setup undermines the main confirmatory causal readout.

### Evidence

Project claim:
- `documents/project_context.md` states "The backbone `B` is a pretrained ResNet18 for CIFAR-10" and uses backbone logit margin as the primary intervention metric.

Current config:
- `configs/flow/resnet18_base.yaml` sets `pretrained: true` and `weights_path: null`.
- `configs/flow/resnet18_aligned.yaml` does the same.

Current implementation:
- `flow_circuits/backbones/resnet.py` loads `IMAGENET1K_V1` weights.
- `flow_circuits/backbones/resnet.py` then replaces `model.fc` with `nn.Linear(in_features, num_classes)`.
- No subsequent code trains that replacement classifier head.

Downstream dependence on logits:
- `flow_circuits/interventions/residual_ablation.py` uses backbone logits to compute predicted classes and logit margins.

### Recommended Remediation

Minimum acceptable fix:
- require `weights_path` to point to a trained CIFAR-10 checkpoint for any run that will be used for intervention claims

Safer fix:
- fail loudly when `num_classes != 1000` and `weights_path is null`, instead of silently constructing a random classifier head

If this is not fixed:
- remove or explicitly downgrade any claims based on predicted-class margin
- treat interventions as exploratory perturbation analyses only

## Problem 2: `flow-evaluate` Does Not Produce a Clean Held-Out Baseline Comparison

Severity: High

### Problem

The evaluation CLI scores the learned model on the user-selected split, but it does not evaluate the baselines on that same split.

Instead:
- model metrics are computed on `args.split`
- baseline regressors are fit on the fit split
- baseline regressors are always evaluated on the validation split

On top of that, the evaluation code truncates both validation and test evaluation to `training.validation_images`, which defaults to 1024.

So the default "test evaluation" report is not a true test-set model-vs-baseline comparison. It is a mixture of:
- model metrics on the first 1024 test images
- baseline metrics on validation images

### Why This Matters

This breaks the core logic of P1 and weakens any held-out claim.

If the repo writes `test_evaluation.json`, readers can reasonably assume:
- model and baselines were compared on the same test split
- the full test split was used unless otherwise stated

That is not what the current code does.

This is particularly risky because the output artifact shape looks clean and final, even though the underlying comparison is not cleanly held out.

### Evidence

Current CLI behavior:
- `flow_circuits/cli/evaluate.py` collects model outputs from `loaders[args.split]`
- `flow_circuits/cli/evaluate.py` caps that collection with `config["training"].get("validation_images", 512)`
- `flow_circuits/cli/evaluate.py` then instantiates `FlowCircuitTrainer(config)` and calls `_fit_baselines()` and `_evaluate_baselines(...)`
- `flow_circuits/training/trainer.py` evaluates those baselines on `self.loaders["val"]`

### Recommended Remediation

Minimum fix:
- when `--split test` is used, evaluate the learned model and all baselines on the same test subset

Better fix:
- make evaluation output explicit about sample counts and actual split used for each metric
- add separate config keys for `val_evaluation_images` and `test_evaluation_images`
- default test evaluation to the full 10k test set unless the user opts into a cap

Best fix:
- compute paired per-example deltas between model and baseline on the same held-out split and report bootstrap confidence intervals

## Problem 3: The Implemented Baselines Are Weaker Than the Written Protocol

Severity: High

### Problem

The written methodology defines `B_local` and `B_flow` as per-layer MLP baselines, but the code implements them as linear Ridge regressors.

That means the current baselines are not the same baselines described in the project context.

### Why This Matters

This matters for both evaluation and model selection.

If the baseline is weaker than promised, then:
- P1 is easier to pass
- the aligned-model Phase C gate is easier to satisfy
- the project can appear to outperform "strong baselines" when it has actually only beaten linear ones

For a method whose central claim is "the contextual encoder learns more than local non-contextual predictors," the strength of the baseline is part of the scientific result, not a convenience detail.

### Evidence

Written protocol:
- `documents/project_context.md` defines `B_local` as a per-layer MLP on pooled state
- `documents/project_context.md` defines `B_flow` as a per-layer MLP on current flow target

Current implementation:
- `flow_circuits/training/baselines.py` imports `sklearn.linear_model.Ridge`
- `flow_circuits/training/baselines.py` fits `Ridge(alpha=alpha)` models for both local and flow baselines

Downstream dependence:
- `flow_circuits/training/trainer.py` uses those baselines for the aligned-model Phase C gate

### Recommended Remediation

Minimum fix:
- rename the baselines in docs and artifacts to "linear local baseline" and "linear flow baseline" so the claim matches the code

Scientifically stronger fix:
- implement the actual per-layer MLP baselines from the project context
- evaluate them on the same split and sample budget as the main model
- use those stronger baselines for both P1 and Phase C gating

## Problem 4: The Confirmatory Statistical Story in the Docs Is Not Yet Implemented

Severity: High

### Problem

The project context describes a confirmatory analysis protocol with confidence intervals, strongest-baseline comparisons, replication across seeds, and stability/null controls. The current implementation only covers a subset of that protocol.

Specifically, the code currently does not implement:
- P1 as a paired bootstrap CI against the strongest baseline
- P2 as improvement over raw pooled-state and raw flow-similarity baselines with CI
- discovery replication over 5 random seeds with matched-null comparisons
- paired bootstrap confidence intervals for intervention effect sizes

The current code mostly reports point estimates, SEMs, and permutation-test p-values.

### Why This Matters

This is the difference between:
- "we ran the pipeline and got suggestive numbers"
- "we ran the confirmatory procedure described in the theory/spec"

Without the confirmatory statistics and replication pieces, the repo can support exploratory analyses, but it does not yet support the stronger success/failure logic written into the project context.

### Evidence

Project context requires:
- P1: 95% paired bootstrap CI over strongest baseline
- P2: CI-based comparison against raw pooled-state and raw flow similarity baselines
- P3: discovery stability over 5 seeds plus bootstrap/null comparisons
- P4: paired bootstrap CIs for intervention effect sizes, plus permutation testing and Holm correction

Current implementation:
- `flow_circuits/evaluation/metrics.py` reports means, SEMs, and alignment standard deviation only
- `flow_circuits/cli/discover.py` runs discovery once for a single configured seed
- `flow_circuits/interventions/residual_ablation.py` computes permutation p-values and Holm-corrected p-values, but not effect-size bootstrap CIs

Notably missing for P2:
- `flow_circuits/evaluation/metrics.py` computes latent-to-future alignment only for `z` vs `q`
- it does not compute the corresponding raw pooled-state similarity baseline or raw flow-similarity baseline described in the docs

### Recommended Remediation

Minimum fix:
- explicitly label current evaluation outputs as exploratory/descriptive rather than confirmatory

Required fix for confirmatory use:
- implement paired bootstrap deltas for P1 and P2
- add raw pooled-state and raw flow alignment baselines for P2
- add multi-seed discovery orchestration and matched-null stability summaries
- add bootstrap confidence intervals for member-vs-control intervention effects

## Problem 5: The Intervention Controls Are Weaker Than the Protocol

Severity: Medium-High

### Problem

The intervention code includes matched non-members, random-node controls, and same-layer random-cell controls, but the random-node control does not match the written protocol, and the same-layer random-cell control is not part of the validation decision.

The project context specifies size-matched random-node interventions at the same layers. The current implementation samples random nodes from arbitrary layers and allows duplicates.

Also, the `validated` flag only depends on:
- member vs non-member
- member vs random-node

It does not require success against the same-layer random-cell control, even though that control is part of the written causal protocol.

### Why This Matters

A good control should isolate the intended causal claim.

If random-node controls can move to different layers:
- the control no longer cleanly isolates "which cells at the same layers matter"
- effects can be confounded by layer selection rather than circuit specificity

If same-layer random-cell results are computed but ignored in validation:
- the pipeline reports a stronger standard than it actually enforces

### Evidence

Written protocol:
- `documents/project_context.md` requires size-matched random-node interventions at the same layers
- `documents/project_context.md` also requires same-layer random-cell interventions with identical mask area

Current implementation:
- `flow_circuits/interventions/residual_ablation.py` uses `_random_nodes_like(...)`, which samples random `(layer, cell)` pairs from the full layer/cell range
- `_random_nodes_like(...)` does not preserve the original layer multiset
- `validated` is computed from corrected p-values for non-member and random-node comparisons only
- same-layer random-cell deltas are computed and saved, but they are not used in the validation decision

### Recommended Remediation

Minimum fix:
- preserve the original circuit's layer multiset when sampling random-node controls
- sample without replacement within each layer when possible

Stronger fix:
- require the same-layer random-cell control to be passed before setting `validated = true`
- save all control-comparison decisions explicitly in the intervention artifact

## Problem 6: The Discovery Connectivity Criterion Is Looser Than the Written Spec

Severity: Medium

### Problem

The project context requires "at least one depth-adjacent path of length at least two" among engaged nodes. The current implementation only checks whether any engaged node has an engaged neighbor one layer deeper at the same spatial cell.

That is a weaker condition.

### Why This Matters

Connectivity and path constraints are part of what keeps a discovered object from collapsing into a loosely connected or weakly cross-layer cluster family.

With the current implementation, a candidate can satisfy the depth-path rule with only a single adjacent pair, as long as the overall active-node count is high enough from spatial neighbors.

That makes the retained objects easier to pass through discovery than the doc implies.

### Evidence

Written protocol:
- `documents/project_context.md` requires "at least one depth-adjacent path of length at least two"

Current implementation:
- `flow_circuits/discovery/candidate_discovery.py` defines `_has_depth_path(...)`
- `_has_depth_path(...)` returns `True` as soon as it finds a node `(l, i)` with `(l + 1, i)` also engaged

### Recommended Remediation

Minimum fix:
- clarify the docs if a single adjacent depth pair is the intended criterion

Scientifically stronger fix:
- enforce the written criterion directly, for example by requiring a same-cell chain across three engaged nodes or an explicit graph path of depth length at least two

## Problem 7: Reproducibility and Falsification Checks Are Incomplete

Severity: Medium

### Problem

The project context emphasizes stability across seeds and several null/sanity checks, but the current repo does not provide a full implementation of those safeguards.

There are two separate gaps here:

1. Reproducibility is only partially controlled.
- the data split is seeded
- discovery bootstrap sampling is seeded
- but the training path does not set global PyTorch/NumPy/random seeds
- fit-loader shuffling is not driven by an explicit generator
- tokenizer parameters and fixed random projections are sampled from the ambient global RNG state

2. The null experiments described in the project context are not implemented as runnable repo workflows.
- future-shuffle null
- depth-order null
- node-shuffle null

### Why This Matters

For a method that is explicitly making stability and falsification part of its validity story, these checks are not optional polish.

Without robust seeding:
- apparent seed-to-seed differences are harder to interpret
- reproduction of a given run is weaker than the config suggests

Without the nulls:
- it is harder to show that the method is not winning through a shortcut
- positive results are less diagnostic than the project context claims

### Evidence

Partial reproducibility controls:
- `flow_circuits/data/cifar10.py` seeds the train/val/discovery split
- `flow_circuits/discovery/candidate_discovery.py` seeds bootstrap resampling

Current gaps:
- tokenizer embeddings and frozen random projectors are initialized randomly in `flow_circuits/tokenization/tokenizer.py`
- there is no corresponding global seeding setup in `flow_circuits/training/trainer.py`
- the fit loader in `flow_circuits/data/cifar10.py` uses `shuffle=True` without an explicit generator

Null experiments:
- `documents/project_context.md` specifies future-shuffle, depth-order, and node-shuffle nulls
- a repo-wide search during review found those nulls described in docs, but not implemented as code paths in `flow_circuits/` or the CLI surface

### Recommended Remediation

Minimum fix:
- add a central seeding utility used by training, evaluation, discovery, and intervention entry points
- thread an explicit generator into the fit loader

Stronger fix:
- add first-class null-experiment entry points or notebook-backed library helpers
- store null results in artifacts next to the main experiment outputs

## Practical Interpretation

If the repo is used in its current state, the safest interpretation is:
- the pipeline is a promising exploratory framework for learning predictive token representations and surfacing candidate circuit-like structures
- it is not yet sufficient to support the strongest confirmatory claims written in `documents/project_context.md`
- intervention-based "validated circuit" language should be used only after the backbone/logit issue and the evaluation/control/statistics issues above are fixed

## Suggested Priority Order

If these problems are going to be fixed in stages, the highest-value order is:

1. Fix the backbone/checkpoint story so the logits mean something.
2. Fix `flow-evaluate` so model-vs-baseline comparisons are performed on the same held-out split.
3. Strengthen or relabel the baselines so the claim matches the implementation.
4. Tighten intervention controls and validation criteria.
5. Add the missing confirmatory statistics and multi-seed discovery protocol.
6. Add seeding hygiene and the null experiments.
