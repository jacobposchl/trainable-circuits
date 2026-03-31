"""
Trajectory animation for CTLS circuit analysis.

Animates how a single input progresses through the frozen backbone and
meta-encoder simultaneously:

  Left   — 3-D stacked activation heatmaps (one plane per backbone block,
            channel-max pooled to a 32×32 spatial map).  The current layer
            lights up full brightness; past layers fade; future layers are
            ghost-dim.

  Right  — Global UMAP embedding of all (image, layer) points.  Each frame
            shows the population at the current depth coloured by class.  The
            query image leaves a white comet trail as it moves through the
            space.

  Top    — Raw query image on the left; predicted class label on the right
            (both static).

  Bottom — Softmax probability bar chart (static).

Typical usage
-------------
  from evaluation import fit_trajectory_umap, animate_trajectory
  from evaluation.circuit_analysis import CircuitAnalyzer, load_checkpoint
  from IPython.display import HTML

  backbone, meta_encoder, _ = load_checkpoint(config, ckpt_path, device)
  analyzer = CircuitAnalyzer(backbone, meta_encoder, val_loader, device)
  data = analyzer.collect_representations(max_samples=2000)

  # Precompute once (~20-40 s on CPU)
  umap_coords, _ = fit_trajectory_umap(data["z_list"])

  anim = animate_trajectory(42, data, umap_coords, backbone, device)
  HTML(anim.to_jshtml())
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – registers 3d projection

from evaluation.circuit_analysis import CIFAR10_CLASSES, denormalize
from evaluation.circuit_viz import _COLOR_LIST

_BG = "#111118"   # figure background colour


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


def fit_trajectory_umap(
    z_list: list[torch.Tensor],
    n_neighbors: int = 15,
    random_state: int = 42,
) -> tuple[np.ndarray, Any]:
    """
    Fit a single global UMAP over all (image, layer) points.

    Stacks all L layers into one [N*L, d] matrix and fits one 2-D UMAP, so
    every image has a continuous L-step trajectory through a shared space.

    Args:
        z_list:       L tensors, each [N, d].
        n_neighbors:  UMAP ``n_neighbors`` parameter.
        random_state: Random seed for reproducibility.

    Returns:
        coords:  [N, L, 2] float32 — 2-D coordinate for every (image, layer).
        reducer: Fitted ``umap.UMAP`` object.  Keep it to transform new points
                 with ``reducer.transform()``, or discard.
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
    ], axis=0)                                      # [N*L, d]

    reducer = umap_lib.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        metric="cosine",
        random_state=random_state,
        low_memory=False,
    )
    embedded = reducer.fit_transform(stacked)       # [N*L, 2]

    # First N rows = layer 0, next N = layer 1, …
    coords = embedded.reshape(L, N, 2).transpose(1, 0, 2)  # [N, L, 2]
    return coords.astype(np.float32), reducer


# --------------------------------------------------------------------------- #
# Main animation entry point
# --------------------------------------------------------------------------- #

def animate_trajectory(
    image_idx: int,
    data: dict,
    umap_coords: np.ndarray,
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
        umap_coords:  [N, L, 2] array from ``fit_trajectory_umap()``.
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

    images    = data["images"]          # [N, 3, 32, 32]
    labels    = data["labels"]
    L         = len(data["z_list"])

    labels_np = labels.numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)

    image_tensor = images[image_idx : image_idx + 1]
    label_idx    = int(labels_np[image_idx])
    true_name    = class_names[label_idx] if label_idx < len(class_names) else str(label_idx)

    heatmaps     = collect_raw_activations(backbone, image_tensor, device)
    softmax_prob = get_softmax_probs(backbone, image_tensor, device)
    pred_idx     = int(np.argmax(softmax_prob))
    pred_name    = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)

    # Normalise each heatmap to [0, 1] for consistent colour scale
    norm_hmaps = []
    for hmap in heatmaps:
        lo, hi = hmap.min(), hmap.max()
        norm_hmaps.append((hmap - lo) / (hi - lo + 1e-8))

    fig, ax_3d, ax_umap, ax_softmax = _make_figure(
        image_tensor[0], softmax_prob, pred_idx, pred_name, true_name, class_names, dpi
    )

    def update(frame: int):
        ax_3d.cla()
        ax_umap.cla()
        _draw_3d_stack(ax_3d, norm_hmaps, current_layer=frame)
        _draw_umap_frame(
            ax_umap, umap_coords, labels_np, image_idx,
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
        (fig, ax_3d, ax_umap, ax_softmax)
    """
    fig = plt.figure(figsize=(14, 10), dpi=dpi, facecolor=_BG)

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
    ax_umap     = fig.add_subplot(gs[1, 1])
    ax_softmax  = fig.add_subplot(gs[2, :])

    # Style all axes backgrounds
    for ax in (ax_img_top, ax_text_top, ax_umap, ax_softmax):
        ax.set_facecolor(_BG)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444455")

    # --- Static: input image ---
    img_np = denormalize(image).permute(1, 2, 0).numpy()
    ax_img_top.imshow(np.clip(img_np, 0, 1), interpolation="nearest")
    ax_img_top.set_title("Input Image", fontsize=10, color="white", pad=3)
    ax_img_top.axis("off")

    # --- Static: prediction text ---
    ax_text_top.axis("off")
    correct = pred_name.lower() == true_name.lower()
    accent  = "#2ecc71" if correct else "#e74c3c"
    ax_text_top.text(
        0.5, 0.62,
        f"Predicted: {pred_name}",
        transform=ax_text_top.transAxes,
        fontsize=15, fontweight="bold",
        ha="center", va="center", color=accent,
    )
    ax_text_top.text(
        0.5, 0.28,
        f"True class: {true_name}",
        transform=ax_text_top.transAxes,
        fontsize=11, ha="center", va="center", color="#aaaacc",
    )

    # --- Static: softmax bars ---
    _draw_softmax(ax_softmax, softmax_prob, pred_idx, class_names)

    return fig, ax_3d, ax_umap, ax_softmax


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

    # Thicker border on predicted bar
    bars[pred_idx].set_edgecolor("white")
    bars[pred_idx].set_linewidth(2.0)

    names = (class_names[:n] if class_names else [str(i) for i in range(n)])
    ax.set_xticks(xs)
    ax.set_xticklabels(names, fontsize=8, rotation=35, ha="right", color="white")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Softmax prob.", fontsize=9, color="white")
    ax.set_title("Final Classification", fontsize=11, color="white", pad=4)
    ax.yaxis.set_tick_params(labelsize=8, labelcolor="white")

    # Confidence label above predicted bar
    ax.text(
        pred_idx, probs[pred_idx] + 0.03,
        f"{probs[pred_idx]:.2f}",
        ha="center", va="bottom", fontsize=9,
        fontweight="bold", color="white",
    )


def _draw_3d_stack(
    ax3d,
    heatmaps: list[np.ndarray],
    current_layer: int,
) -> None:
    """
    Render all L activation heatmap planes in 3-D space.

    Each plane sits at z = layer_index.  Alpha schedule:
      future layers  → 0.07 (ghost)
      past layers    → 0.22 (faded)
      current layer  → 1.00 (bright), with a white highlight border
    """
    L = len(heatmaps)
    H, W = heatmaps[0].shape
    xs = np.linspace(0, 1, W)
    ys = np.linspace(0, 1, H)
    X, Y = np.meshgrid(xs, ys)
    hot  = plt.get_cmap("hot")

    for l in range(L):
        alpha = 1.00 if l == current_layer else (0.22 if l < current_layer else 0.07)
        Z     = np.full_like(X, float(l))
        fc    = hot(heatmaps[l])            # [H, W, 4]
        fc    = fc.copy()
        fc[..., 3] = alpha

        ax3d.plot_surface(X, Y, Z, facecolors=fc, shade=False,
                          linewidth=0, antialiased=False)

        if l == current_layer:
            bx = [0, 1, 1, 0, 0]
            by = [0, 0, 1, 1, 0]
            bz = [l] * 5
            ax3d.plot(bx, by, bz, color="white", linewidth=1.4, alpha=0.95)

    ax3d.set_zlim(-0.5, L - 0.5)
    ax3d.set_xlim(0, 1)
    ax3d.set_ylim(0, 1)
    ax3d.set_zticks(range(L))
    ax3d.set_zticklabels([f"L{l + 1}" for l in range(L)], fontsize=7, color="white")
    ax3d.set_xticklabels([])
    ax3d.set_yticklabels([])
    ax3d.set_xlabel("", labelpad=-10)
    ax3d.set_ylabel("", labelpad=-10)
    ax3d.set_zlabel("Layer", fontsize=9, color="white", labelpad=5)
    ax3d.set_title(f"Activations — Layer {current_layer + 1}/{L}",
                   fontsize=10, color="white", pad=6)
    ax3d.view_init(elev=25, azim=-55)

    try:
        ax3d.set_box_aspect([1, 1, L * 0.35])
    except AttributeError:
        pass  # matplotlib < 3.3

    ax3d.set_facecolor(_BG)
    for pane in (ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("#333344")
    ax3d.grid(False)
    ax3d.tick_params(colors="white")


def _draw_umap_frame(
    ax: plt.Axes,
    umap_coords: np.ndarray,
    labels: np.ndarray,
    image_idx: int,
    current_layer: int,
    class_names: list[str],
) -> None:
    """
    Render one UMAP animation frame.

    Population: all N images at ``current_layer``, coloured by class.
    Trail: query image path from layer 0 → current_layer with comet fade.
    Current position: bright white dot with gold ring.
    """
    L = umap_coords.shape[1]

    # Background population at current layer
    pop = umap_coords[:, current_layer, :]          # [N, 2]
    for cls in range(10):
        mask = labels == cls
        if not mask.any():
            continue
        c = _COLOR_LIST[cls] if cls < 10 else "gray"
        ax.scatter(pop[mask, 0], pop[mask, 1], c=[c], s=4,
                   alpha=0.35, rasterized=True,
                   label=class_names[cls] if cls < len(class_names) else str(cls))

    # Comet trail for query image
    n_trail = current_layer + 1
    for i in range(n_trail):
        frac = (i + 1) / n_trail          # 0 = oldest, 1 = newest
        x, y = umap_coords[image_idx, i, 0], umap_coords[image_idx, i, 1]
        is_current = (i == n_trail - 1)

        if is_current:
            ax.scatter(x, y, c="white", s=130, alpha=1.0, zorder=6,
                       edgecolors="gold", linewidths=1.8)
        else:
            alpha_pt = 0.15 + 0.60 * frac
            size_pt  = 15  + 55  * frac
            ax.scatter(x, y, c="white", s=size_pt, alpha=alpha_pt,
                       zorder=5, edgecolors="none")

    # Trail line
    if current_layer > 0:
        tx = umap_coords[image_idx, :current_layer + 1, 0]
        ty = umap_coords[image_idx, :current_layer + 1, 1]
        ax.plot(tx, ty, color="white", linewidth=1.0, alpha=0.45, zorder=4)

    ax.set_facecolor(_BG)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"z-Space (UMAP) — Layer {current_layer + 1}/{L}",
                 fontsize=10, color="white", pad=4)

    # Compact class legend
    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=_COLOR_LIST[i] if i < 10 else "gray",
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
