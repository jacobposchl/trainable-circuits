# Project Context

---

## 1. Objective and Scope

This project studies a frozen pretrained ResNet18 and asks whether reusable processing motifs can be recovered from internal activations without class labels and without modifying backbone weights.

The learned system does not change the backbone. It learns an analysis representation over backbone activations that is optimized to predict future residual updates. The primary output of the pipeline is a set of candidate circuits.

Definitions:

- A **candidate circuit** is a connected subset of `(layer, spatial cell)` nodes together with a subset of images for which those nodes show recurrent, statistically stable, above-baseline predictive engagement.
- A **validated circuit** is a candidate circuit that also passes held-out causal specificity tests.

This distinction is essential. Discovery from activations alone is not sufficient to claim mechanism. In this document, strong mechanistic language is reserved for validated circuits; earlier stages identify candidate mechanisms.

Core empirical claim:

If a latent token `z_{l,i}(x)` can predict the next residual update at the same spatial cell, and if subsets of images recur with similar externally defined future-flow descriptors across multiple connected nodes, then the backbone reuses structured processing motifs that are detectable from activation geometry.

Falsification conditions:

- If one-step prediction does not beat non-contextual baselines, the latent space is not capturing forward computation.
- If discovered candidate circuits are unstable across seeds and bootstrap resamples, the method is detecting clustering artifacts rather than reusable motifs.
- If candidate circuits do not show member-specific causal effects under matched interventions, they should not be interpreted as mechanisms.

---

## 2. Theoretical Framing

Residual networks process inputs by accumulating many local updates across depth. The object of interest is therefore not only "what is represented at layer `l`" but also "what local update is being computed next from the current state."

This motivates a flow-based view:

- Each spatial cell at each block is treated as a local unit of evidence.
- The block output `h_l` represents accumulated state up to depth `l`.
- The residual branch output `r_l` represents the incremental computation performed by block `l`.
- A useful analysis representation should encode enough information at `(l, i)` to predict `r_{l+1}` at the aligned cell.

Two cautionary points keep the framing rigorous:

- The pipeline constructs an **analysis graph**, not the backbone's literal computational graph. Depth-edge weights are prediction scores, not direct causal coefficients.
- Same-layer attention inside the meta-encoder is an analysis summary of learned dependencies. It is not, by itself, evidence of information flow inside the ResNet.

The project is therefore strongest if it first succeeds as a predictive representation-learning method and only then upgrades candidate circuits to validated circuits through held-out interventions.

---

## 3. Backbone and Observed Tensors

### 3.1 Backbone

The backbone `B` is a pretrained ResNet18 for CIFAR-10 with all parameters frozen. No gradients flow into `B` at any point.

Operational requirement:

- canonical experiment configs must set `backbone.weights_path` to a supervised CIFAR-10 checkpoint for the chosen ResNet architecture
- the code should fail loudly rather than silently constructing an untrained classifier head for intervention metrics

Let `h_0(x)` denote the output of the ResNet stem for image `x`. For BasicBlock `l = 1, ..., L` with `L = 8`:

```text
r_l(x) = F_l(h_{l-1}(x))
h_l(x) = ReLU(r_l(x) + s_l(h_{l-1}(x)))
```

where:

- `F_l` is the residual branch of block `l`
- `s_l` is the skip path, equal to identity except at downsampling blocks where it is the learned projection used by the ResNet

This notation is exact for both identity-skip and downsampling blocks.

### 3.2 Hooks

Two hooks are used:

- **State hook:** `h_l(x)` after skip-add and ReLU. This is the accumulated state at depth `l`.
- **Flow hook:** `r_l(x)` at the output of `bn2`, before skip-add and before ReLU. This is the isolated residual contribution of block `l`.

Token inputs are built from `h_l`. Prediction targets are built from `r_l`.

### 3.3 Tensor Shapes

For CIFAR-10 ResNet18:

```text
layer group 1 (blocks 1-2): H_l x W_l = 32 x 32, C_l = 64
layer group 2 (blocks 3-4): H_l x W_l = 16 x 16, C_l = 128
layer group 3 (blocks 5-6): H_l x W_l =  8 x  8, C_l = 256
layer group 4 (blocks 7-8): H_l x W_l =  4 x  4, C_l = 512
```

State tensors therefore satisfy:

```text
h_l(x) in R^{C_l x H_l x W_l}
r_l(x) in R^{C_l x H_l x W_l}
```

---

## 4. Spatiotemporal Tokenization

### 4.1 Fixed Spatial Grid

For each layer `l`, the state tensor `h_l(x)` is adaptively max-pooled to a fixed `G x G` grid with `G = 4`, giving `M = G^2 = 16` spatial cells per layer.

Let `P_{l,i}` denote grid cell `i` at layer `l`. The pooled state vector at that cell is:

```text
c_{l,i}(x) = AdaptiveMaxPool2d_G(h_l(x))[i] in R^{C_l}
```

The fixed grid is an indexing convention, not a claim of exact semantic alignment across depth. It ensures:

- constant token count across layers
- a simple aligned coordinate system across resolutions
- tractable `O((LM)^2)` self-attention with `L x M = 8 x 16 = 128` tokens per image

When resolution changes across layer groups, cell index `i` refers to the same normalized image region, not the same exact feature-map semantics.

### 4.2 Magnitude Feature

Activation magnitude is kept explicitly:

```text
s_{l,i}(x) = log(||c_{l,i}(x)||_2 + eps)
```

and concatenated to the pooled content vector:

```text
tilde_c_{l,i}(x) = [c_{l,i}(x); s_{l,i}(x)] in R^{C_l + 1}
```

This retains both direction and scale information before projection into the latent token space.

### 4.3 Position and Depth Encoding

Each grid cell `i` has normalized coordinates `(u_i, v_i) in [0, 1]^2`. A learned position encoder maps these coordinates to `E_pos(i) in R^d`. Each layer has a learned depth embedding `E_depth(l) in R^d`.

### 4.4 Full Token Embedding

Each layer has its own linear projection `W_{c,l}` because channel dimensions differ across layers. The input token is:

```text
u_{l,i}(x) = LayerNorm(W_{c,l} tilde_c_{l,i}(x) + E_pos(i) + E_depth(l)) in R^d
```

The full token set for image `x` is:

```text
U(x) = {u_{l,i}(x)} for l = 1, ..., L and i = 1, ..., M
```

---

## 5. Meta-Encoder

The meta-model `T_theta` is a transformer encoder over the `128` tokens of `U(x)`:

```text
Z(x) = T_theta(U(x)) = {z_{l,i}(x)}
```

where each latent token `z_{l,i}(x) in R^d`.

### 5.1 Attention Structure

Depth attention is causal:

```text
token (l, i) may attend to token (l', j) only if l' <= l
```

Within a fixed depth `l`, attention is fully bidirectional across spatial cells.

This mask imposes one explicit inductive bias: the latent token at depth `l` should summarize information available up to `l`, not future layers.

### 5.2 Positional Encoding Inside Attention

RoPE is applied only along depth order. Spatial position is already encoded additively in the input token. This separates depth order from spatial coordinates.

### 5.3 Output Normalization

The final token outputs are L2-normalized:

```text
z_{l,i}(x) = normalize(T_theta(U(x))[l, i]) in S^{d-1}
```

After this point, cosine similarity is the primary geometry used for training, evaluation, and discovery.

---

## 6. Flow Targets and External Trajectory Descriptors

### 6.1 One-Step Flow Target

For each block `l` and cell `i`, the local residual contribution is pooled from the flow hook:

```text
f_{l,i}(x) = AdaptiveMaxPool2d_G(r_l(x))[i] in R^{C_l}
```

To compare layers in a common space, each `f_{l,i}` is mapped to a fixed dimension `D_flow = 256` using a layer-specific fixed Gaussian projection `P_l` drawn once at initialization and then frozen:

```text
hat_f_{l,i}(x) = P_l f_{l,i}(x) in R^{D_flow}
tilde_f_{l,i}(x) = normalize(hat_f_{l,i}(x))
```

Because `C_l <= 512`, this compression is mild. The fixed projection is used only to standardize dimensionality; it does not inject trainable capacity.

### 6.2 Why the Target Is `r_l`, Not `h_l`

Predicting `h_{l+1}` from `z_{l,i}` would admit a degenerate copy-forward solution because `h_{l+1}` contains the skip path. Predicting `r_{l+1}` instead asks a sharper question:

```text
Does the token at (l, i) contain enough information to explain what the next block adds?
```

This is the backbone quantity most directly aligned with local forward computation.

### 6.3 External Future Descriptor

The original trajectory idea becomes rigorous only if future similarity is defined from backbone-native quantities, not from the live `z` space being optimized.

For each source node `(l, i)`, define the future-flow stack:

```text
g_{l,i}(x) = [tilde_f_{l,i}(x); tilde_f_{l+1,i}(x); ...; tilde_f_{L,i}(x)]
```

Since the stack length depends on source layer `l`, a layer-specific fixed projection `Q_l` maps it to a common trajectory dimension `d_traj = 256`:

```text
q_{l,i}(x) = normalize(Q_l g_{l,i}(x)) in R^{d_traj}
```

Properties of `q_{l,i}`:

- It depends only on frozen backbone residual signals.
- It preserves layer order because the future stack is concatenated, not averaged.
- It is defined separately at each node `(l, i)`, so "similar future" always means similarity among images at the same source node.

External trajectory similarity is then:

```text
s_flow((x, l, i), (x', l, i)) = cos(q_{l,i}(x), q_{l,i}(x'))
```

This quantity is used for evaluation, candidate-circuit discovery, and, in one optional configuration, trajectory supervision.

---

## 7. Training Objective

### 7.1 Prediction Loss

For `l = 1, ..., L - 1`, a layer-specific decoder `D_l` predicts the next residual update from `z_{l,i}`:

```text
p_{l,i}(x) = normalize(D_l(z_{l,i}(x)))
```

The one-step prediction loss is:

```text
L_pred = (1 / N(L - 1)M) sum_{x,l,i} ||p_{l,i}(x) - tilde_f_{l+1,i}(x)||_2^2
```

Since both vectors are unit-normalized, this is equivalent to cosine loss up to a constant factor.

### 7.2 Reconstruction Loss

A second decoder `R_l` predicts the same-layer residual target:

```text
rhat_{l,i}(x) = normalize(R_l(z_{l,i}(x)))
L_rec = (1 / NLM) sum_{x,l,i} ||rhat_{l,i}(x) - tilde_f_{l,i}(x)||_2^2
```

This keeps the latent space anchored to actual backbone content rather than becoming a purely predictive code.

### 7.3 Optional External Trajectory Alignment Loss

Two official configurations are evaluated:

- **Base configuration:** `lambda_traj = 0` for all epochs. `q_{l,i}` is used only for analysis and discovery.
- **Aligned configuration:** `lambda_traj > 0` is enabled only after the Base model has already reached acceptable validation prediction performance.

For the Aligned configuration, anchor positives are defined from external future similarity, not from the live `z` space.

For an anchor `a = (x, l, i)`, let `B_{l,i}` be the other batch elements at the same node `(l, i)`. Compute:

```text
w_{ab} = max(cos(q_a, q_b), 0)
```

Define the positive set `P(a)` as the top-`K` elements of `B_{l,i}` ranked by `w_{ab}` that also satisfy `w_{ab} >= gamma`. If `P(a)` is empty, the anchor is skipped.

Trajectory-loss hyperparameters are selected on the validation split from a fixed search grid:

- `K in {4, 8, 16}`
- `gamma in {0.2, 0.3, 0.4}`
- `tau in {0.05, 0.1, 0.2}`

The weighted supervised-contrastive loss is:

```text
L_traj = -(1 / |A|) sum_{a in A} (1 / sum_{b in P(a)} w_{ab})
         sum_{b in P(a)} w_{ab} log(
           exp(cos(z_a, z_b) / tau) /
           sum_{c in B_{l,i}, c != a} exp(cos(z_a, z_c) / tau)
         )
```

This loss is no longer circular because `q` is external to the learned latent space.

### 7.4 Full Objective

```text
L = lambda_pred L_pred + lambda_rec L_rec + lambda_traj L_traj
```

Default weights:

- `lambda_pred = 1.0`
- `lambda_rec = 0.2`
- `lambda_traj = 0.0` in the Base configuration
- `lambda_traj in {0.1, 0.2, 0.5}` in the Aligned configuration, selected on validation only

### 7.5 Training Phases

Training is staged for identifiability and clean ablation:

1. **Phase A:** optimize `L_pred` only.
2. **Phase B:** continue from Phase A and add `L_rec`.
3. **Phase C (Aligned only):** add `L_traj` only if the Phase B checkpoint already beats the strongest non-contextual baseline on validation for one-step prediction.

Phase C is accepted only if:

- validation external trajectory alignment improves relative to Phase B, and
- validation one-step prediction remains within one standard error of the Phase B score

Otherwise the Phase B checkpoint is retained as the final model.

Optimization details:

- optimizer: AdamW
- learning-rate schedule: cosine decay
- gradient clipping: `max_norm = 1.0`
- all backbone computations and all `f` / `q` targets run under `torch.no_grad()`

---

## 8. Candidate-Circuit Discovery

### 8.1 Analysis Graph

Discovery is performed on an analysis graph over backbone-aligned nodes.

Nodes:

```text
V = {(l, i) : l = 1, ..., L and i = 1, ..., M}
```

Structural adjacency is fixed and unweighted:

- **Depth adjacency:** `(l, i)` is adjacent to `(l + 1, i)`
- **Spatial adjacency:** `(l, i)` is adjacent to same-layer 4-neighbor cells `(l, j)`

These edges define connectivity only. They are not learned.

### 8.2 Predictive Engagement Score

For `l < L`, define the image-specific predictive engagement score:

```text
e_x(l, i) = cos(p_{l,i}(x), tilde_f_{l+1,i}(x))
```

This is a proxy for how well the latent token at `(l, i)` explains the next residual update. It is not itself a causal coefficient.

For descriptive visualization only, same-layer attention can also be summarized as:

```text
a_x(l, i, j) = symmetrized mean attention mass between tokens (l, i) and (l, j)
```

`a_x` is not used to define, filter, or validate circuits.

### 8.3 Discovery Split

Circuit discovery is run on a held-out **discovery split** that is not used to fit `T_theta` or the decoders. This prevents the clustering stage from being optimized on the same images used for model fitting.

### 8.4 Step 1: Node-Wise Clustering

For each node `(l, i)`:

- stack `q_{l,i}(x)` over discovery images `x`
- cluster with HDBSCAN using cosine distance
- keep only clusters whose size satisfies:

```text
n_min <= |C| <= n_max
```

with:

- `n_min = max(20, ceil(0.005 N_disc))`
- `n_max = floor(0.40 N_disc)`

For each retained cluster, estimate bootstrap stability by reclustering 20 bootstrap resamples of the discovery split and matching by maximum Jaccard overlap. Retain only clusters with mean bootstrap Jaccard at least `0.60`.

Each retained cluster is a tuple:

```text
u = ((l, i), C_u, stab_u)
```

### 8.5 Step 2: Merge Near-Duplicate Clusters Across Nodes

Retained node clusters are merged by image-set overlap.

Construct a graph whose vertices are retained clusters `u`. Put an edge between `u` and `v` if:

```text
Jaccard(C_u, C_v) >= theta_merge
```

with `theta_merge = 0.70`.

Each connected component defines one **circuit family**.

For each family `G`:

- choose the **medoid cluster** `u*` with highest mean Jaccard to the other clusters in `G`
- define the canonical image set `C_G = C_{u*}`
- define preliminary active nodes as the nodes of clusters `u in G` satisfying:

```text
Jaccard(C_u, C_G) >= theta_node
```

with `theta_node = 0.70`

### 8.6 Step 3: Connectivity and Engagement Filtering

The preliminary active-node set is intersected with the largest connected component under the fixed structural adjacency from Section 8.1. This ensures that a candidate circuit is a connected object rather than a disconnected bag of nodes.

For each active node `(l, i)` in family `G`, compute mean engagement over member images:

```text
bar_e_G(l, i) = mean_{x in C_G} e_x(l, i)
```

Also compute node-wise background statistics over the entire discovery split:

```text
mu_{l,i} = mean_x e_x(l, i)
sigma_{l,i} = std_x e_x(l, i)
```

An active node is called **engaged** if:

```text
bar_e_G(l, i) >= mu_{l,i} + sigma_{l,i}
```

A family is retained as a candidate circuit only if all of the following hold:

- at least two distinct layers are represented
- the largest connected component contains at least three nodes
- at least one depth-adjacent path of length at least two is engaged
- at least half of active nodes are engaged

### 8.7 Step 4: Circuit Representation

Each retained candidate circuit `G` is stored as:

- `image_set C_G`
- `representative_node v*_G` from the medoid cluster
- `active_nodes A_G`
- `engagement profile bar_e_G(l, i)` on active nodes
- `cluster stability stats`
- `node-wise trajectory centroids m_G(l, i) = normalize(mean_{x in C_G} q_{l,i}(x))` on active nodes
- `descriptive spatial summary a_G(l, i, j)` if needed for visualization
- `post hoc class purity`, computed only for analysis, never for discovery

Multi-circuit membership is allowed.

### 8.8 Held-Out Assignment

To evaluate interventions on held-out images, each candidate circuit defines an assignment rule.

For active node `v = (l, i)`, let:

```text
rho_G(v) = 10th percentile of cos(q_v(x), m_G(v)) over x in C_G
```

A held-out image `x` is assigned to candidate circuit `G` if:

- `cos(q_{v*_G}(x), m_G(v*_G)) >= rho_G(v*_G)`, and
- at least `50%` of active nodes satisfy their own centroid-similarity threshold

This assignment rule is fixed before looking at intervention outcomes.

---

## 9. Evaluation Criteria

Evaluation is split into primary confirmatory criteria and secondary descriptive criteria.

### 9.1 Primary Confirmatory Criteria

**P1 - Forward prediction beats non-contextual baselines.**

On held-out data, compare one-step prediction cosine against:

- `B_mean`: node-wise mean predictor
- `B_local`: per-layer MLP on pooled state `tilde_c_{l,i}` only, without transformer context
- `B_flow`: per-layer MLP on current flow target `tilde_f_{l,i}` only

The method passes P1 only if the improvement over the strongest baseline has 95% paired bootstrap confidence interval strictly greater than zero.

**P2 - Latent geometry aligns with external future similarity.**

At each node `(l, i)`, compute Spearman correlation between:

- pairwise latent similarities `cos(z_{l,i}(x), z_{l,i}(x'))`
- pairwise external future similarities `cos(q_{l,i}(x), q_{l,i}(x'))`

Compare this against the same correlation using raw pooled-state similarity and raw flow similarity. The method passes P2 only if mean improvement over the strongest baseline has 95% confidence interval greater than zero.

**P3 - Candidate circuits are stable.**

Run discovery over 5 random seeds and 20 bootstrap resamples. Match circuits by representative node and maximum centroid similarity. Discovery passes P3 only if:

- retained node clusters satisfy the per-cluster bootstrap threshold from Section 8.4, and
- matched candidate circuits show image-set Jaccard and active-node F1 above matched-null controls, with confidence intervals excluding zero improvement over null

**P4 - Candidate circuits show causal specificity.**

For each candidate circuit on held-out data, run the targeted intervention from Section 10.4. A candidate circuit becomes a **validated circuit** only if:

- member images show a larger intervention effect than matched non-members
- member images also show a larger effect than size-matched random-node controls
- the effect remains significant after multiple-comparison correction across tested circuits

The primary confirmatory effect metric is change in logit margin of the backbone's predicted class. Change in true-class logit is reported as a secondary label-dependent check.

### 9.2 Secondary Descriptive Criteria

These quantities are reported but are not necessary for success:

- same-layer reconstruction fidelity
- multi-step prediction decay with horizon
- active-node coverage over the `L x M` grid
- post hoc class purity distribution
- transfer to new datasets or architectures

Secondary criteria are descriptive because they can be informative even when they do not take a particular shape.

---

## 10. Experiments

### 10.1 Data Splits and Protocol

Use the standard CIFAR-10 train/test split, then partition the training set into:

- `40k` images for fitting `T_theta` and the decoders
- `5k` images for validation and model selection
- `5k` images as the discovery split for clustering and circuit construction

The `10k` standard test images remain untouched until final evaluation and interventions.

Labels are not used during representation learning or circuit discovery. Labels are used only for:

- post hoc descriptive analyses such as purity
- class-matched intervention controls
- optional label-dependent reporting such as true-class logit drop

### 10.2 Decoding and Representation Experiments

**A. Layer identity probe.**

Train a linear probe on `z_{l,i}` to predict `l`. This is a sanity check that depth information is preserved.

**B. Same-layer reconstruction.**

Evaluate cosine similarity between `rhat_{l,i}` and `tilde_f_{l,i}` on held-out data.

**C. One-step and multi-step prediction.**

Evaluate:

- one-step prediction of `tilde_f_{l+1,i}`
- multi-step prediction of `tilde_f_{l+k,i}` for `k = 2, 3, 4`

Multi-step prediction is always secondary to one-step prediction, which is the core supervised signal.

**D. Base vs Aligned configuration.**

Compare:

- Base: `L_pred + L_rec`
- Aligned: `L_pred + L_rec + L_traj` only after Phase B

The Aligned configuration is retained only if it improves P2 without violating the Phase C acceptance rule.

**E. Raw-activation baseline vs meta-encoder.**

Run the full discovery pipeline once with `q_{l,i}` built from backbone flow targets as defined above, and once with clustering performed directly on current-layer `tilde_f_{l,i}` only. This tests whether future-aware descriptors add value beyond direct local activation clustering.

**F. Pixel-space decoder (exploratory).**

Optionally train a separate decoder from `z_{l,i}` to the image patch corresponding to the receptive field of `(l, i)`. This is purely interpretive and does not affect any confirmatory metric.

### 10.3 Nulls and Sanity Checks

To ensure the method is not exploiting trivial shortcuts, run the following nulls:

- **Future-shuffle null:** shuffle next-layer targets across images within node `(l, i)` when scoring the learned decoder. Prediction performance should collapse relative to the unshuffled target pairing.
- **Depth-order null:** permute the order of future-flow blocks inside `g_{l,i}` before forming `q_{l,i}`. If P2 is unchanged, the trajectory descriptor is not using temporal order.
- **Node-shuffle null:** shuffle node labels during discovery. Stable circuits should disappear.

### 10.4 Causal Intervention

The intervention is targeted and local.

For a candidate circuit `G` with active node set `A_G`, and for a held-out image `x` assigned to `G`:

- at each active node `(l, i)`, zero only the residual-branch activations `r_l(x)` inside the feature-map region corresponding to cell `i`
- leave the skip path unchanged
- forward the modified activations through the frozen backbone and measure output change

This intervention is more specific than zeroing an entire layer or entire activation map and better matches the discovered object.

Controls:

- matched non-member images with the same predicted class and similar pre-intervention confidence
- size-matched random-node interventions at the same layers
- same-layer random-cell interventions with identical mask area

Primary effect metric:

```text
Delta_margin = margin_before - margin_after
```

where `margin` is the top-1 minus top-2 logit gap.

Secondary effect metric:

```text
Delta_true = true_class_logit_before - true_class_logit_after
```

Statistical test:

- paired bootstrap confidence intervals for effect sizes
- permutation test for member vs control differences
- Holm correction across tested circuits

### 10.5 Stability and Replication

Run the entire discovery procedure over 5 random seeds. In addition:

- bootstrap the discovery split 20 times
- match circuits by representative node and centroid similarity
- report image-set Jaccard, active-node F1, and intervention-effect consistency across runs

Only circuits that remain stable under this procedure should be emphasized.

### 10.6 Transfer Experiments

After the CIFAR-10 study:

- **Dataset transfer:** freeze the learned meta-encoder and evaluate prediction and discovery behavior on CIFAR-100 and STL-10.
- **Architecture transfer:** retrain the analysis model on ResNet34 and ResNet50 with the same protocol.

Transfer is exploratory. Failure to transfer does not invalidate the CIFAR-10 result, but success would strengthen the claim that the method captures general properties of residual processing.

---

## 11. Repository Workflow Alignment

The repository implements the experimental workflow with four CLI stages:

- `flow-train`
- `flow-evaluate`
- `flow-discover`
- `flow-intervene`

The first-class notebook suite mirrors these stages:

- `notebooks/nb01_training_and_representation_metrics.ipynb`
- `notebooks/nb02_candidate_circuit_discovery_and_stability.ipynb`
- `notebooks/nb03_interventions_and_qualitative_analysis.ipynb`

The standard artifact flow is:

- training checkpoints (`phase_b.pt`, `final.pt`)
- evaluation summary JSON
- candidate-circuit artifact JSON
- intervention summary JSON/CSV

This section is operational only. The scientific interpretation of the method remains defined by the sections above.

---

## 12. Design Decisions Reference

| Decision | Choice | Rationale |
|---|---|---|
| Backbone | Frozen pretrained ResNet18 | Keeps the scientific target fixed and prevents the analysis model from rewriting the mechanism it is meant to study |
| Token source | `h_l` (post-skip state) | Gives the encoder access to the accumulated state available at depth `l` |
| Prediction target | `r_{l+1}` (next residual branch) | Avoids the copy-forward degeneracy of predicting full block outputs |
| Grid | Fixed `4 x 4` across all layers | Keeps token count constant and attention tractable |
| Depth attention | Causal across depth, bidirectional within layer | Matches the directional prediction objective while preserving same-layer context |
| Trajectory descriptor | External future-flow descriptor `q_{l,i}` from frozen backbone targets | Removes the circularity of defining trajectory similarity from the live `z` space |
| Official training configs | Base and Aligned | Separates the core predictive objective from the optional trajectory-geometry regularizer |
| Discovery graph | Fixed backbone-aligned adjacency plus predictive engagement scores | Keeps connectivity structural and labels learned quantities as proxies rather than causal edges |
| Attention summaries | Visualization only | Avoids over-interpreting meta-encoder attention as backbone mechanism |
| Circuit validity standard | Stability plus held-out causal specificity | Discovery alone is insufficient for a mechanistic claim |

---

## 13. Limitations

This project has real limits even in its rigorous form:

- The fixed grid is a coarse spatial approximation and may merge multiple distinct computations inside one cell.
- Predictive engagement is still a proxy; it becomes mechanistically meaningful only after intervention.
- Future-flow descriptors are tied to the chosen backbone hooks. Different hooks could reveal different structure.
- Candidate circuits may still mix multiple sub-mechanisms if those sub-mechanisms co-occur on the same image subset.
- Transfer across datasets or architectures is not guaranteed because the analysis model is partly adapted to one backbone family.

These limitations do not invalidate the project, but they define the boundary of what a positive result would mean.
