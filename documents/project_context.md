# CTLS Phase 1 — Project Context

## 1. What This Project Is Trying to Do

This is a mechanistic interpretability project. The goal is to discover **circuits** — recurring, reusable computational pathways — inside a pretrained neural network, without using class labels and without modifying the network.

A "circuit" in this context is a contiguous span of layers `[l_start, l_end]` that a specific group of images activates in the same functional way. If a set of images undergoes the same transformation sequence through layers 3–6, those images share a circuit at that span, regardless of whether they belong to the same class or not. The project's central claim is that networks don't process every input differently — they recycle stable, reusable computational patterns, and those patterns can be discovered purely from activation structure.

The method trains a **meta-encoder** to read the backbone's internal activations and produce a latent space — called circuit space or z-space — whose geometry reflects how functionally similar any two inputs are at each layer. Circuit discovery then operates directly in that latent space.

---

## 2. Architecture

### 2.1 Backbone (Frozen)

A pretrained ResNet18 with all parameters frozen. It processes inputs and produces activation trajectories. We hook into each BasicBlock at two locations:

- **Block output (post-relu, post-addition):** Used as input to the MetaEncoder's per-layer projectors. L2-normalized before being passed to the encoder.
- **bn2 output (pre-addition, non-skip branch):** The "flow" at that layer — what the block itself contributed, independent of the skip connection. This is the training signal (see Section 3).

For each BasicBlock, the residual structure is:
```
out = conv1 → bn1 → relu → conv2 → bn2    # ← hook here for flow
out += skip_connection
out = relu(out)                             # ← hook here for trajectory
```

The backbone hooks on `bn2` of all 8 BasicBlocks across ResNet18's 4 layer groups.

### 2.2 Flow Compression

The raw bn2 output is `[B, C_l, H_l, W_l]` — too large to use as a training target directly. It is compressed with a fixed (non-trainable) pipeline per layer:

```
F_l(x): [B, C_l, H_l, W_l]
    → AdaptiveMaxPool2d(4, 4)      # spatial grid, max pool
    → Flatten()
    → Linear(C_l * 16, D_flow)    # project to fixed dimension
    → flow_l(x): [B, D_flow]
```

`D_flow = 256` (configurable). `G = 4` grid. Max pooling is used instead of average pooling because the flow signal is sparse — a block makes large, localized activations at the few channels where it's doing circuit-relevant work, and average pooling would wash those peaks out.

This entire pipeline runs under `torch.no_grad()` — it is fixed target computation, not a trained component.

### 2.3 MetaEncoder

A RoPE transformer that takes the L2-normalized block-output activations and produces per-layer latent representations:

```
[h_1, h_2, ..., h_L]  →  [z_1, z_2, ..., z_L]
```

**Per-layer projectors:** Each layer `l` has its own dedicated projector:
```
p_l = LayerNorm(GELU(W_l · h_l_hat))    [B, projection_dim]
```
Because all projectors map to the same `projection_dim`, the encoder is backbone-agnostic with respect to layer widths.

**Transformer with RoPE:** The projected tokens are passed through a multi-head transformer. Rotary Position Embeddings inject layer depth information multiplicatively into the attention mechanism, giving the transformer an inductive bias that favors attending to nearby layers. This is appropriate because circuits tend to span contiguous layers.

The transformer outputs `z_1, ..., z_L` — per-layer latent representations. These are L2-normalized and are the only outputs used downstream.

### 2.4 Per-Layer Regressors

For the information loss, a separate MLP regressor exists for each layer `l`. It takes the element-wise product of two inputs' z-representations and predicts their co-activation target:

```
regressor_l(z_l(a) ⊙ z_l(b))  →  [N_pairs, D_flow]
```

Each regressor is a small MLP with its own output head sized to `D_flow`. They are part of `InfoLoss` and are saved in checkpoints.

---

## 3. Training Objective

### Current Objective: Normalized MSE (Information Loss Only)

The training objective is a single term — normalized MSE between predicted and true per-layer flow co-activations:

```
L = info_loss_weight * L_info
```

where `info_loss_weight = 5.0`.

**L_info** is computed as follows. For a batch of inputs with pair indices `(idx_a, idx_b)`:

1. Compute the flow co-activation target for each layer `l`:
   ```
   flow_coact_l = flow_l(a) ⊙ flow_l(b)    [N_pairs, D_flow]
   ```
   This is element-wise high where both inputs have large flow activations in the same channel — the direct signal for shared circuit use.

2. Predict this from z-space:
   ```
   pred_l = regressor_l(z_l(a) ⊙ z_l(b))   [N_pairs, D_flow]
   ```

3. Compute normalized MSE per layer (equivalent to 1 - R²):
   ```
   ss_res = ((pred_l - flow_coact_l) ** 2).sum()
   ss_tot = ((flow_coact_l - flow_coact_l.mean()) ** 2).sum().clamp(min=1e-8)
   loss_l = ss_res / ss_tot
   ```

4. Average over layers:
   ```
   L_info = mean over l of loss_l
   ```

The normalized MSE formulation is critical. Raw MSE is scale-dependent — when the target magnitude is small (which it is for L2-normalized vectors: std ≈ 1/√D_flow ≈ 0.06), raw MSE produces losses around 1.5e-5, while other terms can be thousands of times larger. Normalizing by `ss_tot` produces a scale-invariant loss that equals 1.0 at initialization (no better than predicting the mean) and approaches 0.0 at perfect reconstruction.

### What Was Removed: Geometry Loss

An earlier version of the training objective included a geometry loss term — a soft contrastive loss that encouraged z-vectors for pairs with high flow similarity to be geometrically close in z-space. This was removed for the following reason:

The geometry loss used cosine similarities between L2-normalized 256-dimensional flow vectors as soft targets. In 256-dimensional space, L2-normalized vectors concentrate on the unit hypersphere such that pairwise dot products have near-zero mean and std ≈ 1/√256 ≈ 0.06 — effectively all similarities collapse to ≈0 regardless of whether inputs share a circuit. This means the soft target distribution fed to the geometry loss's cross-entropy was approximately uniform from epoch 1 (entropy ≈ log(255) ≈ 5.54), producing zero gradient throughout training. The geometry loss was dead from the start.

Despite removing the geometry loss, empirical results (val_rho = 0.715) show that geometric consistency in z-space emerges naturally from the information loss — the encoder must organize z-space to make the regressor's predictions consistent across inputs, which produces geometric structure as a byproduct.

---

## 4. Circuit Discovery

### 4.1 Why Not Pairwise Similarity Clustering

An earlier design attempted to discover circuits by clustering pairwise dot products between z-vectors into 8-dimensional "span profiles." This failed due to the same concentration-of-measure problem: 128-dimensional L2-normalized z-vectors have pairwise dot products with std ≈ 1/√128 ≈ 0.088, so all profile vectors are near-zero regardless of true circuit membership. The clustering had nothing to work with.

### 4.2 Current Approach: Image-Centric UMAP + HDBSCAN

Instead of working in a scalar similarity space, the discovery pipeline works directly in the vector space of z-representations. For each candidate span `[l_start, l_end]`:

1. **Concatenate z-vectors** across span layers per image:
   ```
   X_i = [z_{l_start}(i), ..., z_{l_end}(i)]    [N, span_len * projection_dim]
   ```
   Each row is one image's representation of how it was processed across that span.

2. **UMAP dimensionality reduction** with cosine metric (appropriate because z-vectors are L2-normalized):
   ```
   X_reduced = UMAP(X, n_components=15, n_neighbors=15, metric='cosine')
   ```

3. **HDBSCAN clustering** on the reduced representation:
   ```
   labels = HDBSCAN(min_cluster_size=5, metric='euclidean').fit_predict(X_reduced)
   ```

4. **Canonicality filter:** Keep only clusters between 1% and 40% of total images. Below 1% = too small to be a recurring pattern; above 40% = degenerate mega-cluster.

This produces circuits as sets of images (not pairs), where each circuit is a group of images that UMAP+HDBSCAN found genuinely cluster together in z-space at that span.

### 4.3 Circuit Representation

Each discovered circuit is stored as:
- `span`: (l_start, l_end) inclusive
- `image_mask`: boolean array of length N indicating which images belong to the circuit
- `size`: number of images in the cluster
- `cluster_id`: HDBSCAN cluster label

Multi-circuit membership is correct behavior: one image can belong to circuits at multiple spans simultaneously, reflecting that it activates multiple distinct computational patterns at different depths.

### 4.4 Success Criteria

**C1 — Profile Reconstruction R²:** Mean R² (= 1 - normalized MSE) across all layers and held-out pairs ≥ 0.7. Tests whether z-space encodes sufficient information to reconstruct flow co-activation structure.

**C2 — Geometric Consistency:** Spearman ρ between z-space cosine similarity and flow cosine similarity ≥ 0.65 averaged across layers. Tests whether z-space geometry reflects flow similarity structure.

**C3 — Within-Span Similarity Elevation:** For each canonical circuit, mean pairwise z-cosine similarity among circuit members must exceed population mean by ≥ 1σ for that span. Tests whether discovered clusters are genuinely coherent in z-space.

**C4 — Circuit Diversity:** Circuits discovered across all spans must collectively cover ≥ 60% of all L layers. Tests that circuit space is not collapsing to a single depth region.

**C5 — Class Purity Bimodality:** Among all canonical circuits, the distribution of class purity scores (fraction of images from dominant class) must be bimodal — both class-agnostic circuits (purity < 0.3) and class-specific circuits (purity > 0.7) must exist. Tests that circuit space organizes computation at multiple levels of semantic abstraction.

---

## 5. Design Decisions and Changes Made During Development

**Flow targets instead of state similarity:** The original design used cosine similarity between accumulated block outputs (`h_l`) as the training signal. This is the wrong signal — `h_l` accumulates contributions from all prior layers via the skip connections, so similarity at `h_l` reflects shared history, not what that specific layer did. The correct signal is the block's contribution in isolation: the pre-skip bn2 output `F_l(x)`.

**Normalized MSE instead of raw MSE:** Raw MSE was 366,000× smaller than the geometry loss term at initialization due to the small magnitude of L2-normalized vector products. This made the two loss terms effectively incomparable and required artificial weighting. Normalized MSE is scale-invariant and meaningful on its own.

**Image-centric discovery instead of pair-centric:** Pairwise dot products between L2-normalized high-dimensional vectors suffer from concentration of measure in any space (flow space, z-space). The solution is to avoid derived scalars entirely and cluster the raw per-image z-vectors directly.

**Geometry loss removal:** The soft contrastive geometry loss was theoretically sound but empirically dead from epoch 1 due to near-uniform soft target distributions caused by concentration of measure in 256-d flow space. Removed rather than patched, since information loss alone yields ρ = 0.715 geometric consistency.

---

## 6. Known Limitations

**GAP discards spatial structure.** The block-output trajectory fed to the MetaEncoder is globally average-pooled from `[B, C, H, W]` to `[B, C]`. Circuits whose signature is spatially localized (e.g., an edge detector active in the top-left corner) are invisible to the encoder. The flow targets use a 4×4 max-pool grid, which partially preserves coarse spatial structure, but the encoder itself still operates on GAP'd activations.

**Concentration of measure in z-space.** Even in the learned z-space, similarity scores between L2-normalized vectors cluster around zero. The UMAP+HDBSCAN discovery works around this by operating in the vector space rather than the scalar similarity space, but the elevation scores for individual circuits are modest (1.0–2.25σ range). A better-trained model (higher R²) should produce tighter clusters with higher elevation.

**UMAP and HDBSCAN are stochastic.** Circuit discovery results can vary between runs. Circuit stability across random seeds has not been formally quantified.

**Ablation gap.** The key validity question — does the meta-encoder add anything beyond clustering raw backbone activations? — has not been run. This ablation is required before any strong claims about the method.

**CIFAR-10 only.** All training and evaluation has been on CIFAR-10 with ResNet18. Generalization to other datasets or backbones is untested.
