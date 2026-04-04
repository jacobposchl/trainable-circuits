# Research Results So Far

This document summarizes what the current flow-circuits iteration has established so far, what remains uncertain, and which next experiments are most likely to clarify where `z` is actually useful.

## Executive Summary

The project has clearly succeeded at learning a nontrivial latent space `z` over a frozen CIFAR-10 ResNet18. The strongest evidence is now:

- `Phase C` makes `z` more aligned with future-flow structure than `Phase B`
- `z` contains recurring multi-layer motifs that can be visualized, probed, and in some cases causally validated
- `z` contains linearly decodable discriminative information that the frozen backbone head does not always fully exploit
- under corrupted inputs, a compact top-node subset from `z` can outperform both the backbone alone and the full-`z` hybrid correction path

The project has **not** yet shown strong evidence for strict connected-depth candidate circuits as originally defined by the old discovery path. The current evidence fits a weaker but still meaningful conclusion better:

- the backbone appears to reuse recurring, partially distributed processing motifs
- `z` makes those motifs easier to detect and, in some settings, more useful for selective correction

## Main Questions

The work so far has been testing four linked questions:

1. Does `z` encode meaningful forward-computational structure?
2. Does that structure organize into recurring motifs across inputs and depth?
3. Can we interpret those motifs at the image level?
4. Is `z` useful for decision support when the frozen backbone struggles?

## What We Built

The current notebook and package suite now supports:

- `nb01`: backbone training plus frozen/joint `z` branch training with milestone Phase C checkpoints
- `nb02`: `q` validation and checkpoint selection
- `nb03`: `z` motif discovery and analysis
- `nb04`: motif-based clean/corruption utility testing

This progression shifted the project from:

- strict candidate-circuit validation only

to a broader and more empirically grounded workflow:

- representation learning
- motif discovery
- image-level interpretability
- selective correction under stress

## Findings By Stage

### 1. Representation Learning

Strongest finding:

- `Phase C` improves alignment between `z` geometry and frozen future-flow descriptor geometry

Interpretation:

- `z` is not trivial noise
- `Phase C` is doing something real and directionally intended
- the latent is becoming more future-oriented rather than simply preserving present local activations

Tradeoff:

- `Phase B` is better at reconstructing current local backbone state
- `Phase C` is better at future-oriented alignment

This looks like a genuine tradeoff, not a random fluctuation.

### 2. Strict Circuit Discovery

The strict pilot discovery path found stable local node clusters but failed to assemble them into retained circuits.

Inspection showed:

- the discovered structure was sparse
- it rarely formed same-cell depth chains
- the connected components were tiny

Interpretation:

- the backbone may contain recurring structure without satisfying the current strict circuit object
- the old connected depth-path rule is likely too rigid for the phenomenon being learned

This was an important negative result, but it refined the project in a productive direction.

### 3. Recurring Motifs

After reframing the object from strict circuits to recurring motifs discovered directly in `z`, the project started working much better.

Key motif results:

- recurring motif families can be found directly in `z`
- `Phase B` tends to produce cleaner, more class-pure motifs
- `Phase C` tends to produce broader, less class-pure, more distributed motifs
- at least some `Phase C` motifs look more causally meaningful under intervention

Interpretation:

- `Phase B` looks more semantically tidy
- `Phase C` looks more computationally broad

This supports a meaningful distinction between the phases rather than redundancy.

### 4. Visual Interpretability and Probes

The image-first notebook showed that motifs and node-clusters can be inspected directly on CIFAR-10 images.

Probe results showed:

- class information is highly linearly decodable from `z`
- some individual `(layer, cell)` nodes carry surprisingly strong discriminative signal
- hard-pair distinctions such as `cat vs dog` are sometimes more separable in `z` than in the frozen backbone head

Interpretation:

- `z` can “know” more than the final classifier currently uses
- this makes `z` scientifically interesting even when intervention gains are still small

### 5. Clean-Data Selective Correction

On clean CIFAR-10:

- `Phase C z` gave only a small selective-correction gain
- the clearest benefit was on the genuinely hard `cat vs dog` pair
- calibration and confidence quality did not improve

Interpretation:

- `z` is not a large clean-data accuracy booster
- `z` is not currently a better general confidence signal than the backbone
- `z` may still contain useful specialist information for brittle ambiguities

### 6. Corruption-Focused Selective Correction

This is currently the strongest practical result.

On corrupted CIFAR-10:

- full-`z` selective correction helped slightly on average
- a compact top-node subset helped more on average
- the gains were especially visible for some mid-to-high severity blur, noise, and contrast settings

Interpretation:

- `z` becomes more useful when the backbone is stressed
- the useful correction signal may be concentrated in a compact subset of top pair-discriminative nodes
- filtering to strong nodes can outperform using the full latent

This is the clearest current evidence that `z` is useful as a selective robustness-support signal rather than just a descriptive latent.

## Current Strongest Claims

The strongest claims supported by the current evidence are:

1. `Phase C` changes the latent in a real, future-oriented direction.
2. `z` contains recurring multi-layer structure that is easier to analyze as motifs than as strict connected-depth circuits.
3. `z` contains discriminative information that the backbone head does not always fully exploit.
4. `z` is more useful under corruption than on clean CIFAR-10.
5. A compact top-node subset can sometimes carry more useful selective-correction signal than the full latent.

## Current Weaknesses and Open Questions

Several important claims are still unproven:

- that strict connected-depth circuits are the right object for this backbone
- that `z` improves confidence calibration in a robust way
- that `z` gives large in-distribution accuracy gains
- that the best top-node subsets are stable across seeds, checkpoints, and corruption families
- that the useful nodes correspond to easily nameable human concepts rather than just reliable discriminative patches

## What `z` Seems Most Useful For Right Now

The current evidence suggests that `z` is most useful for:

- analyzing future-oriented latent structure
- identifying recurring motifs in a frozen model
- probing which nodes carry discriminative information
- selective correction on hard or degraded inputs
- studying where the backbone fails to exploit available evidence

The evidence is much weaker for:

- large clean-data accuracy improvements
- general confidence replacement
- strong claims of mechanistic circuit recovery under the old strict definition

## How To Further Test Where `z` Is Most Useful

The most informative next experiments would target **regimes with real headroom**, not already-solved clean inputs.

### 1. Expand Corruption Families

The current corruption suite is already informative, but it is still narrow.

Good next additions:

- brightness
- occlusion
- pixelation
- mixed corruptions

This would test whether `z` helps specifically with blur/noise-like degradations or more broadly across distribution shift.

### 2. Pair- and Class-Level Corruption Breakdowns

The corruption notebook currently shows aggregate gains. The next useful breakdown is:

- which classes benefit most
- which confusion pairs benefit most
- which corruptions favor full `z` vs top-node subsets

This would make the usefulness story much more concrete.

### 3. Stability of Top-Node Subsets

If top-node subsets are really the best correction signal, we should test:

- how stable those subsets are across reruns
- whether the same nodes recur across corruption severities
- whether the same nodes recur across corruption families

This would tell us whether the useful subset is a robust object or just an opportunistic fit.

### 4. Case Studies On Corrected Corrupted Inputs

A strong qualitative next step is:

- show corrected corrupted images
- show the top-node subset overlays
- compare backbone vs corrected predictions

This would help answer whether the top-node subset is tracking semantically plausible evidence.

## Should We Train `z` Harder?

Probably yes, but in a **controlled sweep**, not just by blindly increasing Phase C pressure.

The current evidence suggests that better training of `z` could help, but there are real risks.

### Why More Phase C Could Help

If the current `Phase C` objective is under-optimized, then:

- future-flow alignment could improve further
- corruption robustness support might get stronger
- top-node subset signal might become cleaner and more stable

This is plausible because the corruption notebook already suggests that the useful `z` signal emerges more clearly under stress.

### Why More Phase C Could Also Hurt

The project has already observed tradeoffs:

- `Phase C` reduced activation-probe faithfulness relative to `Phase B`
- `Phase C` motifs became less class-pure
- clean-data calibration did not improve

That means pushing Phase C too hard could:

- over-compress toward future similarity
- wash out useful current-state information
- make the latent less semantically crisp
- worsen calibration further

So the correct hypothesis is not “more Phase C is always better.”
It is:

- there may be a better Phase C regime than the current one, but it likely sits on a tradeoff frontier

## Best Training Sweep To Run Next

The most useful next training experiment would be a **Phase C hyperparameter sweep** evaluated by the corruption notebook, not just by standard validation loss.

Recommended sweep axes:

- Phase C epochs: for example `5, 10, 20, 40`
- `lambda_traj`: for example `0.05, 0.1, 0.2, 0.4`

And evaluate each checkpoint with:

1. `nb02` representation metrics
   - especially neighbor agreement
2. `nb03` motif metrics
   - especially intervention-relevant motif signals
3. `nb06` corruption selective correction
   - average gain over backbone
   - best corruption gain
   - top-node subset advantage over full `z`
4. calibration metrics
   - so improvements do not come from a pathological confidence regime

### Best Single Selection Criterion

If only one downstream criterion is used, the best current one is:

- **corruption selective-correction gain with a penalty for calibration collapse**

This is closer to the actual current use case of `z` than plain clean-data accuracy.

## Current Best Interpretation

The project has moved from:

- “Can we recover strict candidate circuits from a future-predictive latent?”

to:

- “Can we learn a future-oriented latent that exposes recurring motifs and becomes selectively useful when the frozen backbone is brittle?”

That revised question is better supported by the evidence so far.

At this point, the strongest story is:

- `z` is a meaningful analysis representation
- `Phase C` pushes it toward future-computational structure
- recurring motifs are real and inspectable
- the most promising practical use of `z` is selective correction under stress
- the most promising next optimization is a controlled Phase C sweep judged by corruption usefulness, not just clean accuracy
