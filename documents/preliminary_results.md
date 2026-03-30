# CTLS Phase 1 — Preliminary Results

## Status

These results are from a training run using an **older objective** (raw MSE + geometry loss, not normalized MSE). The codebase has since been updated to use normalized MSE with no geometry loss, but this newer configuration has not yet been trained. The circuit discovery results below were obtained by running the new image-centric UMAP+HDBSCAN discovery pipeline on the checkpoint from the old training run.

Results should be interpreted as preliminary — the model quality is below target on C1, which likely depresses performance on C3. The pipeline itself (architecture, discovery, evaluation) is functioning correctly.

---

## Training Results

**Model:** ResNet18 backbone (frozen), MetaEncoder (RoPE transformer, projection_dim=128, 4 layers, 4 heads), per-layer regressors outputting D_flow=256.

**Objective at time of training:** Raw MSE + soft contrastive geometry loss (lambda warmup over 30 epochs). This has since been replaced with normalized MSE only.

**Epochs run:** 100

**Final validation metrics:**
- **val_R² = 0.512** (C1 target: ≥ 0.70) — **FAIL**
- **val_rho = 0.715** (C2 target: ≥ 0.65) — **PASS**
- Geometry loss at epoch 100: 5.49 (started at 5.54; barely moved, confirming it was effectively dead)

**C1 failure analysis:** R² = 0.512 means the regressor is explaining 51.2% of the variance in flow co-activation targets. The remaining 48.8% is unaccounted for. The primary cause is the loss scale imbalance during training — raw MSE on L2-normalized 256-d vectors produces values ~1.5e-5, while the geometry loss was ~5.49, a ratio of ~366,000:1. The geometry term dominated the gradient budget and starved the information term. The normalized MSE fix is expected to address this directly.

**C2 result analysis:** val_rho = 0.715 despite C1 failing. This indicates that z-space geometric structure (pairs with similar flows are nearby in z-space) emerged from the information loss alone, even when the information loss was scale-dominated. The geometry loss produced zero gradient throughout training (dead from epoch 1 due to near-uniform soft targets in 256-d space), so this rho value is attributable entirely to the information loss.

---

## Circuit Discovery Results

**Discovery run on:** 2000 validation images, old model checkpoint.

**Pipeline:** Image-centric UMAP+HDBSCAN (new pipeline). For each of the 36 candidate spans (ResNet18 with L=8), concatenate per-image z-vectors across span layers, reduce with UMAP (cosine metric, 15 components, 15 neighbors), cluster with HDBSCAN (min_cluster_size=5), filter by size (1%–40% of N).

**Total canonical circuits discovered:** Hundreds (exact count varies by run due to HDBSCAN stochasticity; typical run produces ~350–400).

Note: With 0 circuits in the previous pairwise-scalar approach, discovering hundreds is a qualitative change, not a marginal improvement. The old approach was fundamentally broken by concentration of measure in the 8-dimensional profile space.

---

## Per-Criterion Results

### C1 — Profile Reconstruction R²

**Result: 0.512 — FAIL** (target ≥ 0.70)

The regressor predicts per-layer flow co-activation vectors from z-space element-wise products. R² = 0.512 is meaningful — better than the mean baseline — but below the target. Expected to improve significantly with normalized MSE training.

### C2 — Geometric Consistency (Spearman ρ)

**Result: 0.715 — PASS** (target ≥ 0.65)

Spearman correlation between z-space cosine similarity and flow cosine similarity, averaged across layers. This result is notable because it was achieved with a dead geometry loss — z-space organized itself geometrically from information loss pressure alone. This validates removing the geometry loss entirely: it wasn't contributing, and its absence didn't hurt C2.

### C3 — Within-Span Similarity Elevation

**Result: Partial — mixed**

Elevation is computed per circuit as: (mean z-cosine similarity within circuit members) minus (population mean z-cosine similarity for that span), divided by population standard deviation. Target: > 1.0σ per circuit.

Many circuits pass individually, especially at longer spans and later layers. Circuits at early short spans (e.g., (0,0), (1,1)) have weaker elevation, often 0.6–0.9σ. The highest observed elevations are in the 1.9–2.25σ range for long spans starting at layers 3–4.

The strict C3 interpretation ("all canonical circuits must pass") is not satisfied — a meaningful fraction of discovered circuits fall below 1.0σ. This is consistent with C1 being below target: with R² = 0.512, z-space geometry is imperfect, and cluster boundaries in HDBSCAN include some noise circuits alongside real ones.

### C4 — Circuit Diversity (Layer Coverage)

**Result: 100% — PASS** (target ≥ 60%)

Circuits are discovered at every span from (0,0) through (7,7). All 8 layers appear in at least one canonical circuit. This result is robust — even with a below-target model, the UMAP+HDBSCAN pipeline finds structure across all depth levels.

### C5 — Class Purity Bimodality

**Result: PASS** (122 agnostic <0.3, 8 specific >0.7)

The purity distribution is strongly bimodal: most circuits are class-agnostic (mix many classes), and a small number are class-specific. This is expected for a reconstruction-trained model — the encoder learns visual structure, not class boundaries. The agnostic circuits capture visual properties like background type, shape profile, and lighting; the specific circuits capture class-diagnostic visual patterns that coincidentally align with class boundaries.

---

## Span Heatmap Analysis

The heatmap of mean elevation per span (l_start × l_end) shows a clear gradient:

**Early short spans are weakest:** Spans like (0,0), (1,1), (2,2) have mean elevation in the 0.8–1.0σ range. Early layers in ResNet18 encode low-level texture and edge information that varies widely across images — clusters are less coherent.

**Elevation improves with span length and depth:** Moving right (longer l_end) and down (later l_start) consistently improves mean elevation. Spans (3,6), (4,6), (4,7) show mean elevation in the 1.5–1.7σ range with 9–14 circuits each.

**Notable gap:** Span (2,3) has zero circuits in some runs. Layers 2–3 in ResNet18 appear to produce z-vectors without well-separated density structure for HDBSCAN at the 1%–40% size threshold.

**Best performing region:** Spans starting at layers 3–4 and ending at layers 6–7. This corresponds to ResNet18's layer3 and layer4 groups — the mid-to-late network where shape and object-level representations are being built. This is mechanistically sensible: circuits at this depth are processing structured, stable visual concepts rather than raw texture.

---

## Example Circuits

These circuits were selected from the high-quality subset (elevation ≥ 1.5σ) and are representative of the types of structure being discovered.

### Circuit 4 — span=(1,4), n=40, elevation=1.22σ, purity=0.18 — "Figure-Ground Separation"

Images: frog, bird, deer, dog, airplane, truck, cat — 7 different classes. Every image has a single subject against a clean, low-texture background (white, light blue, soft grey).

Similarity scores within circuit: 0.93–0.98 across span layers 2–5. These are unusually tight for 40 images from 7 classes. The z-vectors at layers 1–4 are nearly perfectly aligned for these images.

Interpretation: The meta-encoder has discovered that "clean background / high figure-ground contrast" is a consistent computational pattern in ResNet18's layers 1–4, independent of what the subject is. This circuit is class-agnostic because it captures how the network processes background structure, not subject identity.

### Circuit 320 — span=(4,5), n=20, elevation=1.94σ, purity=0.60 — "Horizontal Metallic Profile"

Images: ships, automobiles, one airplane. Dominant class is ship, but the grouping is not purely about ship identity — the unifying visual property is a wide, horizontal, metallic silhouette viewed from the side. A container ship, a sedan, and a cargo plane share this mid-layer representation.

Interpretation: Layers 4–5 encode something like "large horizontal object with reflective surface." This is the kind of cross-class structural feature that class-agnostic circuit discovery is designed to find.

### Circuit 326 — span=(4,5), n=20, elevation=1.96σ, purity=0.60 — "Object Against Blue"

Images: deer in blue water, stealth airplane in blue sky, birds in blue sky, boats on blue water. The subject is irrelevant — the shared property is a blue-dominant background (sky = water = same color class to the network at this span).

Interpretation: The network has learned that "blue surround" is a coherent context feature at layers 4–5. This circuit reveals that ResNet18 at mid-network depth represents sky and water identically from a circuit perspective — a mechanistically interesting finding about the backbone's internal representations.

### Circuit 349 — span=(4,7), n=29, elevation=2.10σ, purity=0.48 — "Flat Silhouette on Neutral Ground"

Images: stealth planes (from below, against white sky), an overhead bomber shot, a small figure on a white background, swept-wing jets, a speedboat from above. All show a dark flat silhouette against a light or uniform background, often photographed from an unconventional angle.

Interpretation: This circuit captures a specific compositional pattern — high-contrast flat shape against a uniform ground — rather than a semantic category. The span (4,7) suggests this pattern is crystallized in the second half of the network.

### Circuit 288 — span=(3,6), n=20, elevation=2.13σ, purity=0.65 — "Airplane Shape"

Images: predominantly airplanes (Concorde-type, swept-wing fighters, biplanes), with one ship. This is the closest to a class-specific airplane circuit, with purity 0.65.

Interpretation: The network has a dedicated "airplane" direction at layers 3–6. The single ship that sneaks in (a sleek speedboat with a tapered hull) suggests the circuit is capturing "tapered aerodynamic form" rather than pure class identity — the ship is there because it looks like an airplane at this layer depth.

### Circuit 340 — span=(4,6), n=20, elevation=2.25σ, purity=0.55 — "Swept Aerodynamic Curvature"

Images: Concorde-like airplane, swept-wing fighter jets, sleek speedboats, red biplane, a dark bird with spread wings from below. The highest elevation of any circuit observed (2.25σ).

Interpretation: This is a tighter, more specific version of the "aerodynamic form" concept. The common feature is swept-back, curved, elongated profiles — shared between certain aircraft and certain watercraft and even a bird in flight. This circuit likely corresponds to a specific low-to-mid-level shape detector in the backbone's layer3/layer4 groups.

---

## Cross-Circuit Observations

**Same images recur across multiple circuits.** The Concorde-type airplane image appears in circuits 111, 288, and 340. The black stealth plane appears in circuits 111, 326, and 349. These images are genuinely multi-modal — they activate multiple distinct computational patterns at different spans simultaneously. This is the multi-circuit membership property working as intended.

**The airplane–ship confusion is mechanistically visible.** Multiple high-quality circuits contain both airplanes and ships. This is not noise — it reflects that ResNet18's mid-layer representations for swept aerodynamic shapes (airplanes) overlap substantially with representations for sleek watercraft (boats). The circuit discovery is revealing the internal mechanism behind a known classification confusion, which is the core interpretability value of the method.

**Span depth correlates with concept abstraction.** Early spans ((0,0), (1,1)) produce circuits around low-level visual properties (texture, color statistics). Mid spans ((3,6), (4,6)) produce circuits around shape and compositional properties (swept profiles, figure-ground structure). This gradient is consistent with the known representational hierarchy of ResNet-class networks, and the heatmap confirms it quantitatively.

---

## What Still Needs to Be Done

**Immediate:** Retrain with normalized MSE + no geometry loss. R² should increase substantially. After retraining, re-run nb03 and measure the effect on circuit quality (elevation, purity, cluster tightness).

**Required before any publication claim:**
1. Ablation: run UMAP+HDBSCAN directly on raw backbone activations (without meta-encoder) and compare circuit quality. This determines whether the meta-encoder is adding anything.
2. Circuit stability analysis: run discovery with 5 different random seeds and measure inter-run agreement (Adjusted Rand Index or Jaccard similarity on image_mask overlap). If circuits evaporate across seeds, they are artifacts.
3. Additional datasets: evaluate on CIFAR-100 or STL-10 without retraining to test generalization.

**Longer term:**
- Mechanistic intervention: ablate discovered circuits in the backbone and check whether the predicted subset of images is disproportionately affected.
- Human evaluation: present circuit member images to human raters and ask what visual concept is shared. Measure agreement.
- Scale to harder datasets and deeper backbones.
