"""
Circuit analysis utilities for Phase 1 meta-encoder validation.

Handles data collection from the frozen backbone + meta-encoder, pairwise
flow similarity profile computation, and class purity analysis for discovered
circuit clusters.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

_MEAN = torch.tensor([0.4914, 0.4822, 0.4465])
_STD  = torch.tensor([0.2470, 0.2435, 0.2616])


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """Normalised CIFAR-10 tensor -> [0, 1]. Accepts [C,H,W] or [B,C,H,W]."""
    mean = _MEAN.to(x.device)
    std  = _STD.to(x.device)
    if x.dim() == 4:
        mean = mean[None, :, None, None]
        std  = std[None, :, None, None]
    else:
        mean = mean[:, None, None]
        std  = std[:, None, None]
    return (x * std + mean).clamp(0, 1)


class CircuitAnalyzer:
    """
    Collects representations from the frozen backbone + trained meta-encoder
    and computes flow-based alignment profiles for downstream circuit discovery.
    """

    def __init__(
        self,
        backbone,
        meta_encoder,
        loader: DataLoader,
        device: torch.device,
    ):
        self.backbone     = backbone
        self.meta_encoder = meta_encoder
        self.loader       = loader
        self.device       = device

    @torch.no_grad()
    def collect_representations(self, max_samples: int = 10000) -> dict:
        """
        Collect trajectories, flow targets, per-layer z-vectors, images, labels.

        Returns dict with:
            trajectories: list of L tensors, each [N, D_flow] — compressed block
                          outputs (post-relu, post-skip), L2-normalised (CPU)
            flow_targets: list of L tensors, each [N, D_flow] — compressed bn2/bn3
                          outputs (pre-skip), L2-normalised (CPU).  Use these for
                          scalar profiles, co-activation targets, and discovery.
            z_list:       list of L tensors, each [N, d] (CPU)
            labels:       [N] integer class labels (CPU)
            images:       [N, 3, 32, 32] normalised images (CPU)
        """
        self.meta_encoder.eval()

        all_trajs:  list[list] | None = None
        all_flows:  list[list] | None = None
        all_z:      list[list] | None = None
        all_labels = []
        all_images = []
        n = 0

        for batch in self.loader:
            images = batch[0].to(self.device)
            labels = batch[-1]

            trajectory   = self.backbone(images)
            flow_targets = self.backbone._flow_targets
            z_list       = self.meta_encoder(trajectory)

            if all_trajs is None:
                all_trajs = [[] for _ in range(len(trajectory))]
                all_flows = [[] for _ in range(len(flow_targets))]
                all_z     = [[] for _ in range(len(z_list))]

            for l, h in enumerate(trajectory):
                all_trajs[l].append(h.cpu())
            for l, f in enumerate(flow_targets):
                all_flows[l].append(f.cpu())
            for l, z in enumerate(z_list):
                all_z[l].append(z.cpu())

            all_labels.append(labels.cpu())
            all_images.append(images.cpu())

            n += images.shape[0]
            if n >= max_samples:
                break

        L            = len(all_trajs)
        trajectories = [torch.cat(all_trajs[l], 0)[:max_samples] for l in range(L)]
        flow_targets = [torch.cat(all_flows[l],  0)[:max_samples] for l in range(L)]
        z_list       = [torch.cat(all_z[l],      0)[:max_samples] for l in range(L)]
        labels       = torch.cat(all_labels, 0)[:max_samples]
        images       = torch.cat(all_images, 0)[:max_samples]

        return {
            "trajectories": trajectories,
            "flow_targets": flow_targets,
            "z_list":       z_list,
            "labels":       labels,
            "images":       images,
        }

    @staticmethod
    def compute_all_profiles(
        flow_targets: list[torch.Tensor],
        chunk_size: int = 1000,
    ) -> torch.Tensor:
        """
        Compute full [N, N, L] pairwise flow cosine similarity matrix.

        Args:
            flow_targets: list of L tensors, each [N, D_flow], L2-normalised
            chunk_size:   max rows per chunk to keep memory bounded

        Returns:
            [N, N, L] pairwise per-layer flow cosine similarities
        """
        L = len(flow_targets)
        N = flow_targets[0].shape[0]

        profiles = torch.zeros(N, N, L)

        for l in range(L):
            f = flow_targets[l]   # [N, D_flow]
            for i in range(0, N, chunk_size):
                end = min(i + chunk_size, N)
                profiles[i:end, :, l] = f[i:end] @ f.t()

        return profiles

    @staticmethod
    def compute_pair_profiles(
        flow_targets: list[torch.Tensor],
        idx_a: torch.Tensor,
        idx_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute scalar flow cosine similarities for specific pairs.
        Used by SpanCentricDiscovery and geometric consistency evaluation.

        Args:
            flow_targets: list of L tensors, each [N, D_flow], L2-normalised
            idx_a, idx_b: [N_pairs] indices

        Returns:
            [N_pairs, L] per-layer flow cosine similarities
        """
        L       = len(flow_targets)
        N_pairs = idx_a.shape[0]
        profiles = torch.zeros(N_pairs, L)

        for l in range(L):
            f_a = flow_targets[l][idx_a]   # [N_pairs, D_flow]
            f_b = flow_targets[l][idx_b]
            profiles[:, l] = (f_a * f_b).sum(dim=-1)

        return profiles

    @staticmethod
    def compute_pair_rich_profiles(
        flow_targets: list[torch.Tensor],
        idx_a: torch.Tensor,
        idx_b: torch.Tensor,
    ) -> list[torch.Tensor]:
        """
        Compute flow co-activation vectors for specific pairs.
        Used as InfoLoss targets and for Criterion 1 evaluation.

        Args:
            flow_targets: list of L tensors, each [N, D_flow], L2-normalised
            idx_a, idx_b: [N_pairs] indices

        Returns:
            list of L tensors, each [N_pairs, D_flow]
        """
        return [flow_targets[l][idx_a] * flow_targets[l][idx_b]
                for l in range(len(flow_targets))]

    @staticmethod
    def compute_class_purity(
        pair_indices: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> float:
        """
        Compute class purity for a subset of pairs.

        Args:
            pair_indices: [N_pairs, 2] input indices per pair
            labels:       [N] class labels
            mask:         [N_pairs] boolean mask selecting cluster pairs

        Returns:
            Purity score in [0, 1]
        """
        selected     = pair_indices[mask]
        unique_inputs = selected.unique()
        input_labels  = labels[unique_inputs]

        if len(input_labels) == 0:
            return 0.0

        counts = input_labels.bincount()
        return float(counts.max()) / len(input_labels)


# --------------------------------------------------------------------------- #
# Checkpoint loading
# --------------------------------------------------------------------------- #

def load_checkpoint(
    config: dict,
    checkpoint_path: str,
    device: torch.device,
):
    """
    Build FrozenBackbone, MetaEncoder, and InfoLoss from a config dict and
    load weights from a checkpoint saved by Phase1Trainer.

    Args:
        config:          Parsed YAML config dict (same structure as phase1.yaml).
        checkpoint_path: Path to a .pt checkpoint file.
        device:          Target device for all models.

    Returns:
        (backbone, meta_encoder, info_loss) — all in eval mode on ``device``.
    """
    from models.backbone import FrozenBackbone
    from models.meta_encoder import MetaEncoder
    from losses.info_loss import InfoLoss

    mcfg = config["model"]
    ecfg = mcfg["meta_encoder"]
    rcfg = mcfg.get("regressor", {})
    fcfg = mcfg.get("flow_compression", {})

    backbone = FrozenBackbone(
        arch=mcfg["arch"],
        num_classes=mcfg.get("num_classes", 10),
        pretrained=mcfg.get("pretrained", True),
        grid_size=fcfg.get("grid_size", 4),
        flow_dim=fcfg.get("flow_dim", 256),
    ).to(device)

    meta_encoder = MetaEncoder(
        layer_dims=backbone.layer_dims,
        projection_dim=ecfg.get("projection_dim", 128),
        n_heads=ecfg.get("n_heads", 4),
        n_transformer_layers=ecfg.get("n_transformer_layers", 2),
        dropout=ecfg.get("dropout", 0.0),
    ).to(device)

    info_loss = InfoLoss(
        layer_dims=backbone.layer_dims,
        projection_dim=ecfg.get("projection_dim", 128),
        hidden_dim=rcfg.get("hidden_dim", 64),
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    meta_encoder.load_state_dict(ckpt["meta_encoder_state"])
    info_loss.load_state_dict(ckpt["info_loss_state"])

    backbone.eval()
    meta_encoder.eval()
    info_loss.eval()

    metrics = ckpt.get("val_metrics", {})
    r2  = metrics.get("r2",        float("nan"))
    rho = metrics.get("mean_rho",  float("nan"))
    print(f"Loaded checkpoint: epoch {ckpt['epoch']},  R²={r2:.4f},  ρ={rho:.4f}")

    return backbone, meta_encoder, info_loss
