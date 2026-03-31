"""
Trajectory animation for CTLS circuit analysis.

Animates how a single input progresses through the frozen backbone and
meta-encoder simultaneously:

  Left   — 3-D stacked activation heatmaps (one plane per backbone block,
            channel-max pooled to a 32×32 spatial map).  Layer index runs
            along the X axis (L1 left → L8 right).  The current layer is
            bright; past layers fade; future layers are ghost-dim.

  Right  — Circuit flow diagram.  At each layer K-means clusters images
            into circuit nodes (coloured rectangles, height ∝ population).
            All N image paths flow as dim coloured threads.  The query
            image's path lights up in white as the animation progresses,
            revealing which circuit it rode through the network.

  Top    — Raw query image on the left; predicted class label on the right
            (both static).

  Bottom — Softmax probability bar chart (static).

Typical usage
-------------
  from evaluation import precompute_circuit_flow, animate_trajectory
  from evaluation.circuit_analysis import CircuitAnalyzer, load_checkpoint
  from IPython.display import HTML

  backbone, meta_encoder, _ = load_checkpoint(config, ckpt_path, device)
  analyzer = CircuitAnalyzer(backbone, meta_encoder, val_loader, device)
  data = analyzer.collect_representations(max_samples=2000)

  circuit_data = precompute_circuit_flow(data["z_list"], data["labels"])

  anim = animate_trajectory(42, data, circuit_data, backbone, device)
  HTML(anim.to_jshtml())
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – registers 3d projection

from evaluation.circuit_analysis import CIFAR10_CLASSES, denormalize
from evaluation.circuit_viz import _COLOR_LIST

_BG = "#111118"


# --------------------------------------------------------------------------- #
# Public preprocessing helpers
# --------------------------------------------------------------------------- #

def collect_raw_activations(
    backbone,
    image: torch.Tensor,
    device: torch.device,
) -> list[np.ndarray]:
    """
    Return per-layer spatial activation heatmaps for a single image.

    Temporarily attaches forward hooks to ``backbone._block_modules`` to
    capture raw block outputs *before* compression.  Hooks are always removed
    after the call — no persistent side-effects on the backbone.

    Args:
        backbone: FrozenBackbone instance.
        image:    [1, 3, 32, 32] normalised tensor (any device).
        device:   Device to run the forward pass on.

    Returns:
        List of L numpy arrays, each [32, 32].  Derived as channel-max over
        C → bilinear upsample to 32×32.
    """
    raw: list[torch.Tensor] = []

    def _hook(module, inp, out):
        tensor = out[0] if isinstance(out, (tuple, list)) else out
        raw.append(tensor.detach().cpu())

    handles = [m.register_forward_hook(_hook) for m in backbone._block_modules]
    try:
        with torch.no_grad():
            backbone(image.to(device))
    finally:
        for h in handles:
            h.remove()

    heatmaps = []
    for feat in raw:                                # [1, C, H, W]
        hmap = feat[0].max(dim=0).values            # [H, W]
        hmap = F.interpolate(
            hmap.unsqueeze(0).unsqueeze(0),         # [1, 1, H, W]
            size=(32, 32),
            mode="bilinear",
            align_corners=False,
        )[0, 0].numpy()                             # [32, 32]
        heatmaps.append(hmap)

    return heatmaps


def get_softmax_probs(
    backbone,
    image: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """
    Run the full backbone forward and return softmax class probabilities.

    Args:
        backbone: FrozenBackbone instance.
        image:    [1, 3, 32, 32] normalised tensor.
        device:   Device to run on.

    Returns:
        [num_classes] float32 numpy array.
    """
    with torch.no_grad():
        logits = backbone.model(image.to(device))   # [1, num_classes]
    return torch.softmax(logits[0], dim=0).cpu().numpy()


def precompute_circuit_flow(
    z_list: list[torch.Tensor],
    labels: torch.Tensor | np.ndarray,
    n_clusters: int = 6,
    random_state: int = 42,
) -> dict:
    """
    Cluster images at each layer into circuit nodes for the flow diagram.

    At each layer the N z-vectors are grouped into ``n_clusters`` clusters
    via K-means.  Clusters are sorted by dominant class to give a stable
    vertical ordering.  The returned dict is passed directly to
    ``animate_trajectory``.

    Args:
        z_list:       L tensors, each [N, d].
        labels:       [N] integer class labels.
        n_clusters:   Number of circuit nodes per layer (default 6).
        random_state: K-means random seed.

    Returns:
        Dict with keys:
          cluster_labels      [N, L] int — cluster index per image per layer
          node_y_start        [L, K] float — bottom of each node in [0, 1]
          node_y_end          [L, K] float — top of each node in [0, 1]
          node_dominant_class [L, K] int   — most common class per node
          n_clusters          int
          n_layers            int
    """
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        raise ImportError("Install scikit-learn:  pip install scikit-learn")

    L  = len(z_list)
    N  = z_list[0].shape[0]
    K  = n_clusters
    labels_np = labels.numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)

    raw_labels = np.zeros((N, L), dtype=int)
    for l in range(L):
        z = z_list[l].numpy() if isinstance(z_list[l], torch.Tensor) else np.asarray(z_list[l])
        km = KMeans(n_clusters=K, random_state=random_state, n_init=5, max_iter=200)
        raw_labels[:, l] = km.fit_predict(z)

    # Per-layer: find dominant class, then sort clusters by that class so
    # images of the same class flow through similar vertical regions.
    node_dom_cls_raw  = np.zeros((L, K), dtype=int)
    for l in range(L):
        for k in range(K):
            mask = raw_labels[:, l] == k
            if mask.any():
                node_dom_cls_raw[l, k] = np.bincount(labels_np[mask], minlength=10).argmax()

    cluster_labels = np.zeros((N, L), dtype=int)
    node_dom_cls   = np.zeros((L, K), dtype=int)
    for l in range(L):
        order  = np.argsort(node_dom_cls_raw[l])   # ascending by class id
        remap  = np.empty(K, dtype=int)
        for new_k, old_k in enumerate(order):
            remap[old_k] = new_k
        cluster_labels[:, l] = remap[raw_labels[:, l]]
        node_dom_cls[l]      = node_dom_cls_raw[l][order]

    # Compute node y-spans: stack proportionally by cluster size, [0, 1]
    node_y_start = np.zeros((L, K))
    node_y_end   = np.zeros((L, K))
    for l in range(L):
        y = 0.0
        for k in range(K):
            size = int((cluster_labels[:, l] == k).sum())
            node_y_start[l, k] = y / N
            node_y_end[l, k]   = (y + size) / N
            y += size

    return {
        "cluster_labels":       cluster_labels,   # [N, L]
        "node_y_start":         node_y_start,     # [L, K]
        "node_y_end":           node_y_end,       # [L, K]
        "node_dominant_class":  node_dom_cls,     # [L, K]
        "n_clusters":           K,
        "n_layers":             L,
    }


def fit_trajectory_umap(
    z_list: list[torch.Tensor],
    n_neighbors: int = 15,
    random_state: int = 42,
) -> tuple[np.ndarray, Any]:
    """
    Fit a single global UMAP over all (image, layer) points.

    Stacks all L layers into one [N*L, d] matrix and fits one 2-D UMAP.
    Useful for standalone analysis; the main animation uses
    ``precompute_circuit_flow`` instead.

    Returns:
        coords:  [N, L, 2] float32
        reducer: fitted ``umap.UMAP`` object
    """
    try:
        import umap as umap_lib
    except ImportError:
        raise ImportError("Install umap-learn:  pip install umap-learn")

    L = len(z_list)
    N = z_list[0].shape[0]

    stacked = np.concatenate([
        z.numpy() if isinstance(z, torch.Tensor) else np.asarray(z)
        for z in z_list
    ], axis=0)

    reducer = umap_lib.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        metric="cosine",
        random_state=random_state,
        low_memory=False,
    )
    embedded = reducer.fit_transform(stacked)
    coords   = embedded.reshape(L, N, 2).transpose(1, 0, 2)
    return coords.astype(np.float32), reducer


# --------------------------------------------------------------------------- #
# Main animation entry point
# --------------------------------------------------------------------------- #

def animate_trajectory(
    image_idx: int,
    data: dict,
    circuit_data: dict,
    backbone,
    device: torch.device,
    class_names: list[str] | None = None,
    interval: int = 700,
    save_path: str | None = None,
    dpi: int = 120,
) -> FuncAnimation:
    """
    Build a layer-by-layer animation of a single image's forward pass.

    Args:
        image_idx:    Index into ``data['images']`` to animate.
        data:         Dict from ``CircuitAnalyzer.collect_representations()``.
        circuit_data: Dict from ``precompute_circuit_flow()``.
        backbone:     FrozenBackbone — used for raw activations and softmax.
        device:       Torch device for the backbone.
        class_names:  Class name strings (len = num_classes).  Defaults to
                      CIFAR-10 labels.
        interval:     Milliseconds per animation frame.
        save_path:    Save to this path (.gif requires Pillow, .mp4 requires
                      ffmpeg).  Animation is also returned regardless.
        dpi:          Figure DPI.

    Returns:
        ``matplotlib.animation.FuncAnimation`` object.
    """
    if class_names is None:
        class_names = CIFAR10_CLASSES

    images    = data["images"]
    labels    = data["labels"]
    L         = len(data["z_list"])

    labels_np    = labels.numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)
    image_tensor = images[image_idx : image_idx + 1]
    label_idx    = int(labels_np[image_idx])
    true_name    = class_names[label_idx] if label_idx < len(class_names) else str(label_idx)

    heatmaps     = collect_raw_activations(backbone, image_tensor, device)
    softmax_prob = get_softmax_probs(backbone, image_tensor, device)
    pred_idx     = int(np.argmax(softmax_prob))
    pred_name    = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)

    norm_hmaps = []
    for hmap in heatmaps:
        lo, hi = hmap.min(), hmap.max()
        norm_hmaps.append((hmap - lo) / (hi - lo + 1e-8))

    fig, ax_3d, ax_flow, ax_softmax = _make_figure(
        image_tensor[0], softmax_prob, pred_idx, pred_name, true_name, class_names, dpi
    )

    def update(frame: int):
        ax_3d.cla()
        ax_flow.cla()
        _draw_3d_stack(ax_3d, norm_hmaps, current_layer=frame)
        _draw_circuit_flow(
            ax_flow, circuit_data, labels_np, image_idx,
            current_layer=frame, class_names=class_names,
        )
        return []

    anim = FuncAnimation(fig, update, frames=L, interval=interval, blit=False)

    if save_path is not None:
        fps = max(1, round(1000 / interval))
        _save_animation(anim, save_path, dpi, fps)

    return anim


# --------------------------------------------------------------------------- #
# Internal drawing helpers
# --------------------------------------------------------------------------- #

def _make_figure(
    image: torch.Tensor,
    softmax_prob: np.ndarray,
    pred_idx: int,
    pred_name: str,
    true_name: str,
    class_names: list[str],
    dpi: int,
) -> tuple:
    """
    Construct the figure with its four regions.  Draws the two static panels
    (top strip and bottom softmax) immediately; returns animated axes.

    Returns:
        (fig, ax_3d, ax_flow, ax_softmax)
    """
    fig = plt.figure(figsize=(11.9, 8.5), dpi=dpi, facecolor=_BG)

    gs = gridspec.GridSpec(
        3, 2,
        figure=fig,
        height_ratios=[1.4, 5.2, 2.0],
        hspace=0.38,
        wspace=0.30,
    )

    ax_img_top  = fig.add_subplot(gs[0, 0])
    ax_text_top = fig.add_subplot(gs[0, 1])
    ax_3d       = fig.add_subplot(gs[1, 0], projection="3d")
    ax_flow     = fig.add_subplot(gs[1, 1])
    ax_softmax  = fig.add_subplot(gs[2, :])

    for ax in (ax_img_top, ax_text_top, ax_flow, ax_softmax):
        ax.set_facecolor(_BG)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444455")

    # Static: input image
    img_np = denormalize(image).permute(1, 2, 0).numpy()
    ax_img_top.imshow(np.clip(img_np, 0, 1), interpolation="nearest")
    ax_img_top.set_title("Input Image", fontsize=10, color="white", pad=3)
    ax_img_top.axis("off")

    # Static: prediction text
    ax_text_top.axis("off")
    correct = pred_name.lower() == true_name.lower()
    accent  = "#2ecc71" if correct else "#e74c3c"
    ax_text_top.text(
        0.5, 0.62, f"Predicted: {pred_name}",
        transform=ax_text_top.transAxes,
        fontsize=15, fontweight="bold",
        ha="center", va="center", color=accent,
    )
    ax_text_top.text(
        0.5, 0.28, f"True class: {true_name}",
        transform=ax_text_top.transAxes,
        fontsize=11, ha="center", va="center", color="#aaaacc",
    )

    # Static: softmax bars
    _draw_softmax(ax_softmax, softmax_prob, pred_idx, class_names)

    return fig, ax_3d, ax_flow, ax_softmax


def _draw_softmax(
    ax: plt.Axes,
    probs: np.ndarray,
    pred_idx: int,
    class_names: list[str],
) -> None:
    """Render softmax probability bar chart (called once, never updated)."""
    n    = len(probs)
    xs   = np.arange(n)
    cols = [_COLOR_LIST[i] if i < 10 else "steelblue" for i in range(n)]

    bars = ax.bar(xs, probs, color=cols, edgecolor="#333344",
                  linewidth=0.6, alpha=0.88)
    bars[pred_idx].set_edgecolor("white")
    bars[pred_idx].set_linewidth(2.0)

    names = (class_names[:n] if class_names else [str(i) for i in range(n)])
    ax.set_xticks(xs)
    ax.set_xticklabels(names, fontsize=8, rotation=35, ha="right", color="white")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Softmax prob.", fontsize=9, color="white")
    ax.set_title("Final Classification", fontsize=11, color="white", pad=4)
    ax.yaxis.set_tick_params(labelsize=8, labelcolor="white")
    ax.text(
        pred_idx, probs[pred_idx] + 0.03,
        f"{probs[pred_idx]:.2f}",
        ha="center", va="bottom", fontsize=9, fontweight="bold", color="white",
    )


def _draw_3d_stack(
    ax3d,
    heatmaps: list[np.ndarray],
    current_layer: int,
) -> None:
    """
    Render all L activation heatmap planes in 3-D space.

    Layer index maps to the X axis (L1 on the left, L8 on the right).
    Spatial content spans Y (width) and Z (height).  Alpha schedule:
      future layers  → 0.07 (ghost)
      past layers    → 0.22 (faded)
      current layer  → 1.00 (bright), with a white highlight border
    """
    L = len(heatmaps)
    H, W = heatmaps[0].shape
    ys = np.linspace(0, 1, W)
    zs = np.linspace(0, 1, H)
    Y_grid, Z_grid = np.meshgrid(ys, zs)
    hot = plt.get_cmap("hot")

    for l in range(L):
        alpha   = 1.00 if l == current_layer else (0.22 if l < current_layer else 0.07)
        X_layer = np.full_like(Y_grid, float(l))
        fc      = hot(heatmaps[l]).copy()
        fc[..., 3] = alpha

        ax3d.plot_surface(X_layer, Y_grid, Z_grid, facecolors=fc,
                          shade=False, linewidth=0, antialiased=False)

        if l == current_layer:
            bx = [l] * 5
            by = [0, 1, 1, 0, 0]
            bz = [0, 0, 1, 1, 0]
            ax3d.plot(bx, by, bz, color="white", linewidth=1.4, alpha=0.95)

    ax3d.set_xlim(-0.5, L - 0.5)
    ax3d.set_ylim(0, 1)
    ax3d.set_zlim(0, 1)
    ax3d.set_xticks(range(L))
    ax3d.set_xticklabels([f"L{l + 1}" for l in range(L)], fontsize=7, color="white")
    ax3d.set_yticklabels([])
    ax3d.set_zticklabels([])
    ax3d.set_xlabel("Layer", fontsize=9, color="white", labelpad=5)
    ax3d.set_ylabel("", labelpad=-10)
    ax3d.set_zlabel("", labelpad=-10)
    ax3d.set_title(f"Activations — Layer {current_layer + 1}/{L}",
                   fontsize=10, color="white", pad=6)
    ax3d.view_init(elev=20, azim=-65)

    try:
        ax3d.set_box_aspect([L * 0.35, 1, 1])
    except AttributeError:
        pass

    ax3d.set_facecolor(_BG)
    for pane in (ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("#333344")
    ax3d.grid(False)
    ax3d.tick_params(colors="white")


def _draw_circuit_flow(
    ax: plt.Axes,
    circuit_data: dict,
    labels: np.ndarray,
    image_idx: int,
    current_layer: int,
    class_names: list[str],
) -> None:
    """
    Render the circuit flow diagram for one animation frame.

    Each layer has K coloured rectangular nodes (height ∝ fraction of images
    in that cluster, colour = dominant class).  All N image paths are drawn
    as dim coloured threads behind the nodes.  The query image's path from
    L1 → current_layer is drawn in bright white, showing which circuit it
    has taken so far.

    Args:
        ax:            Matplotlib axes to draw on.
        circuit_data:  Dict from ``precompute_circuit_flow()``.
        labels:        [N] integer class labels.
        image_idx:     Which image to highlight.
        current_layer: Current animation frame (0-indexed).
        class_names:   Class name strings.
    """
    cluster_labels = circuit_data["cluster_labels"]    # [N, L]
    node_y_start   = circuit_data["node_y_start"]      # [L, K]
    node_y_end     = circuit_data["node_y_end"]        # [L, K]
    node_dom_cls   = circuit_data["node_dominant_class"]
    L              = circuit_data["n_layers"]
    K              = circuit_data["n_clusters"]
    N              = cluster_labels.shape[0]

    def _node_center(l: int, n: int) -> float:
        k = cluster_labels[n, l]
        return (node_y_start[l, k] + node_y_end[l, k]) / 2

    # --- Background: all N image paths as dim coloured threads ---
    all_segs   = []
    all_colors = []
    for n in range(N):
        ys = [_node_center(l, n) for l in range(L)]
        all_segs.append(list(zip(range(L), ys)))
        all_colors.append(_COLOR_LIST[int(labels[n]) % 10])

    lc = LineCollection(all_segs, colors=all_colors, alpha=0.04,
                        linewidth=0.5, zorder=1)
    ax.add_collection(lc)

    # --- Node rectangles at each layer ---
    node_half_w = 0.12
    for l in range(L):
        for k in range(K):
            y0 = node_y_start[l, k]
            y1 = node_y_end[l, k]
            if y1 - y0 < 1e-4:
                continue
            is_query = (cluster_labels[image_idx, l] == k)
            color    = _COLOR_LIST[int(node_dom_cls[l, k]) % 10]
            rect     = plt.Rectangle(
                (l - node_half_w, y0), 2 * node_half_w, y1 - y0,
                facecolor=color,
                edgecolor="white" if is_query else "#444455",
                linewidth=1.2 if is_query else 0.3,
                alpha=0.90 if is_query else 0.50,
                zorder=3,
            )
            ax.add_patch(rect)

    # --- Query image path up to current_layer ---
    qx = list(range(current_layer + 1))
    qy = [_node_center(l, image_idx) for l in qx]
    ax.plot(qx, qy, color="white", linewidth=2.5, alpha=1.0, zorder=10,
            solid_capstyle="round", solid_joinstyle="round")

    # Current position dot
    ax.scatter(current_layer, qy[-1], c="gold", s=90, zorder=11,
               edgecolors="white", linewidths=1.5)

    # --- Axis styling ---
    ax.set_xlim(-0.5, L - 0.5)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks(range(L))
    ax.set_xticklabels([f"L{l + 1}" for l in range(L)],
                       fontsize=8, color="white")
    ax.set_yticks([])
    ax.set_facecolor(_BG)
    ax.set_xlabel("Layer", fontsize=9, color="white")
    ax.set_title(f"Circuit Flow — Layer {current_layer + 1}/{L}",
                 fontsize=10, color="white", pad=4)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333344")
    ax.tick_params(axis="x", colors="white")

    # Compact class legend
    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=_COLOR_LIST[i % 10],
                   label=class_names[i] if i < len(class_names) else str(i),
                   markersize=5)
        for i in range(min(10, len(class_names)))
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=6,
              framealpha=0.25, markerscale=0.9, ncol=2,
              labelcolor="white", facecolor="#222233")


def _save_animation(anim: FuncAnimation, path: str, dpi: int, fps: int) -> None:
    """Save animation to .gif (Pillow) or .mp4 (ffmpeg)."""
    if path.endswith(".mp4"):
        from matplotlib.animation import FFMpegWriter
        writer = FFMpegWriter(fps=fps, bitrate=1800)
        anim.save(path, writer=writer, dpi=dpi)
    else:
        anim.save(path, writer="pillow", fps=fps, dpi=dpi)
    print(f"Saved → {path}")
